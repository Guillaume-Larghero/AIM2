#!/usr/bin/env python3
"""
AIM2 — Disease Attractors in Chest X-Ray Generation
Attractor Loop — Full Test Set Run

Pipeline per sample:
  GT CXR ──LANCZOS──▶ 518×518 ──▶ MAIRA-2 ──▶ FINDINGS text
  GT CXR ──LANCZOS──▶ 512×512 ──▶ MedCLIP ──▶ anchor_embed  (256-dim L2)
  FINDINGS ──▶ FLUX.2-dev ──▶ gen_image (512×512)
  gen_image ──▶ MedCLIP ──▶ gen_embed  (256-dim L2)

  Attractor iterations (k times):
    gen_image ──LANCZOS──▶ 518×518 ──▶ MAIRA-2 ──▶ new_findings
    new_findings ──▶ FLUX.2-dev ──▶ new_gen_image (512×512)
    new_gen_image ──▶ MedCLIP ──▶ new_embed

Metrics tracked per iteration:
  image_cosine    = cosine_sim(anchor_embed, gen_embed_k)
  text_cosine     = cosine_sim(gt_text_embed, gen_text_embed_k)
  embed_l2        = ||anchor_embed - gen_embed_k||_2
  text_evolution  = cosine_sim(gen_text_embed_{k-1}, gen_text_embed_k)

Design decisions (confirmed from source):
  - FINDINGS only throughout (no IMPRESSION anywhere)
  - MAIRA-2 input: 518×518 LANCZOS (processor native size)
  - MedCLIP input: 512×512 LANCZOS + ImageNet normalisation
  - MedCLIP val transform: direct LANCZOS resize (get_val_transforms is broken
    for IMAGE_SIZE=512 — it does Resize(256)+CenterCrop(512) which is impossible)
  - FLUX.2 prompt: "Frontal chest X-ray. FINDINGS: {findings}"
  - FLUX.2 offload: enable_sequential_cpu_offload() for 44-80GB GPU compatibility
  - MAIRA-2 loaded on CPU, moved to GPU only during inference (~12s), then back
  - transformers==4.51.3 required (5.x breaks MAIRA-2 LLaVA forward)
  - num_additional_image_tokens=1 required for MAIRA-2 processor (DINO CLS token)
  - Resume support: skips studies with existing metrics.json

Usage:
  cd /n/groups/training/bmif203/AIM2
  python Experiments/attractor_loop/attractor_loop.py \\
      --n_samples 100 --n_iters 5 --split test \\
      --output_dir Experiments/attractor_loop/results/base_flux2
"""

import gc
import json
import logging
import os
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = "/n/groups/training/bmif203/AIM2"
DATA_CSV     = f"{BASE_DIR}/processed_data/processed_data.csv"
MEDCLIP_CKPT = f"{BASE_DIR}/CLIP/outputs/checkpoints/best_model.pth"
MODEL_ID     = "black-forest-labs/FLUX.2-dev"
HF_HOME      = "/n/scratch/users/g/gul075/.cache/huggingface"

os.environ["HF_HOME"]                = HF_HOME
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  IMAGE TRANSFORMS
# ══════════════════════════════════════════════════════════════════════════════

# MedCLIP: direct LANCZOS resize to 512px + ImageNet normalisation.
# NOTE: get_val_transforms(config) with IMAGE_SIZE=512 is BROKEN —
#       it does Resize(256) → CenterCrop(512) which is impossible (crop > resize).
#       The trainer.py was fixed to use this direct-resize approach at training time.
MEDCLIP_TRANSFORM = transforms.Compose([
    transforms.Resize(
        (512, 512),
        interpolation=transforms.InterpolationMode.LANCZOS,
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

def resize_for_maira(img: Image.Image) -> Image.Image:
    """518×518 LANCZOS — MAIRA-2 processor native resolution."""
    return img.resize((518, 518), Image.LANCZOS)


# ══════════════════════════════════════════════════════════════════════════════
#  MEDCLIP
# ══════════════════════════════════════════════════════════════════════════════

def load_medclip(device: torch.device):
    """Load MedCLIP from checkpoint with all config overrides applied."""
    logger.info(f"Loading MedCLIP from {MEDCLIP_CKPT}")
    if BASE_DIR not in sys.path:
        sys.path.insert(0, BASE_DIR)

    # Bio_ClinicalBERT ships only pytorch_model.bin (no safetensors).
    # transformers enforces torch >= 2.6 before calling torch.load (CVE-2025-32434).
    # aim_env has torch 2.5.1. These are trusted local weights — safe to bypass.
    # Patch both import_utils (source) AND modeling_utils (binds function at import).
    try:
        import transformers.utils.import_utils as _tu
        import transformers.modeling_utils as _tmu
        _noop = lambda: None
        _tu.check_torch_load_is_safe  = _noop
        _tmu.check_torch_load_is_safe = _noop
    except Exception:
        pass

    from CLIP.config.config import Config
    from CLIP.model.clip_model import MedicalCLIP

    config = Config()
    config.data.IMAGE_SIZE        = 512    # overridden from 224 at training time
    config.data.COMBINE_SECTIONS  = False  # FINDINGS only (USE_FINDINGS_ONLY=True)
    config.model.IMAGE_PRETRAINED = False  # loading weights from checkpoint

    model = MedicalCLIP(config)
    ckpt = torch.load(MEDCLIP_CKPT, map_location="cpu", weights_only=False)
    key = "model_state_dict" if "model_state_dict" in ckpt else "state_dict"
    model.load_state_dict(ckpt[key], strict=True)
    model = model.to(device).eval()

    logger.info(f"  MedCLIP loaded: epoch={ckpt.get('epoch','?')}  "
                f"val_loss={ckpt.get('val_loss','?')}")

    with torch.no_grad():
        dummy = torch.randn(1, 3, 512, 512).to(device)
        emb   = model.encode_image(dummy)
    assert emb.shape == (1, 256), f"Unexpected embed shape: {emb.shape}"
    assert abs(emb.norm(dim=1).item() - 1.0) < 0.01, "Embedding not L2-normalised"
    logger.info(f"  ✓ embed shape={tuple(emb.shape)}  norm={emb.norm(dim=1).item():.4f}")

    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    return model, config, tokenizer


@torch.no_grad()
def embed_image(model, pil_img: Image.Image, device: torch.device) -> torch.Tensor:
    """PIL RGB → (256,) L2-normalised float32 tensor on CPU."""
    tensor = MEDCLIP_TRANSFORM(pil_img.convert("RGB")).unsqueeze(0).to(device)
    return model.encode_image(tensor).squeeze(0).float().cpu()


@torch.no_grad()
def embed_text(model, tokenizer, text: str, device: torch.device) -> torch.Tensor:
    """FINDINGS string → (256,) L2-normalised float32 tensor on CPU."""
    enc = tokenizer(text, max_length=512, padding="max_length",
                    truncation=True, return_tensors="pt")
    emb = model.encode_text(
        enc["input_ids"].to(device),
        enc["attention_mask"].to(device),
    )
    return emb.squeeze(0).float().cpu()


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


# ══════════════════════════════════════════════════════════════════════════════
#  MAIRA-2
# ══════════════════════════════════════════════════════════════════════════════

def load_maira():
    """
    Load MAIRA-2 from local HF snapshot, keeping model on CPU.

    Key fixes applied:
    1. Local snapshot path — bypasses HF hub lookup which fails on compute nodes.
    2. num_additional_image_tokens=1 — MAIRA-2 DINO encoder has a CLS token.
       Without this, newer transformers raises a token/feature count mismatch.
       Confirmed fix: https://huggingface.co/microsoft/maira-2/discussions/6
    3. Model kept on CPU — L40S (44GB) cannot hold MAIRA (~15GB) + FLUX.2 Mistral
       (~24GB) simultaneously. Model is moved to GPU only during inference (~12s).
    4. transformers==4.51.3 required. 5.x restructured LlavaForConditionalGeneration
       and the custom Maira2ForConditionalGeneration forward breaks.
    """
    import glob
    logger.info("Loading MAIRA-2 (local snapshot, CPU)...")

    snapshot_pattern = os.path.join(
        HF_HOME, "hub", "models--microsoft--maira-2", "snapshots", "*"
    )
    snapshots = sorted(glob.glob(snapshot_pattern))
    if not snapshots:
        raise RuntimeError(
            f"MAIRA-2 snapshot not found. Run download_maira2.sh first.\n"
            f"Expected: {snapshot_pattern}"
        )
    model_path = snapshots[-1]
    logger.info(f"  Snapshot: {model_path}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float16,
    )
    model = model.eval()  # keep on CPU — moved to GPU only during inference

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        num_additional_image_tokens=1,  # DINO CLS token fix
    )
    logger.info("  ✓ MAIRA-2 ready (on CPU)")

    class MAIRAWrapper:
        """Thin wrapper exposing generate_report(image_path) → Report.findings."""

        def __init__(self, model, processor):
            self.model     = model
            self.processor = processor

        def generate_report(self, image_path, indication=None):
            frontal   = Image.open(image_path).convert("RGB")
            processed = self.processor.format_and_preprocess_reporting_input(
                current_frontal=frontal,
                current_lateral=None,
                prior_frontal=None,
                indication=indication,
                technique=None,
                comparison=None,
                prior_report=None,
                return_tensors="pt",
                get_grounding=False,
            )
            # Move model to GPU for inference, then back to CPU to free VRAM
            inf_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(inf_device)
            processed = {k: v.to(inf_device) for k, v in processed.items()}

            with torch.no_grad():
                output_ids = self.model.generate(
                    **processed,
                    max_new_tokens=300,
                    use_cache=True,
                )

            self.model.to("cpu")
            torch.cuda.empty_cache()

            prompt_len = processed["input_ids"].shape[-1]
            decoded    = self.processor.decode(
                output_ids[0][prompt_len:], skip_special_tokens=True
            ).lstrip()
            prediction = self.processor.convert_output_to_plaintext_or_grounded_sequence(
                decoded
            )
            findings = prediction if isinstance(prediction, str) else str(prediction)

            class _Report:
                pass
            r          = _Report()
            r.findings = findings.strip() or "No findings reported."
            return r

    return MAIRAWrapper(model, processor)


def run_maira(generator, pil_img: Image.Image) -> str:
    """PIL image → FINDINGS string via 518×518 LANCZOS input."""
    import tempfile
    img_518 = resize_for_maira(pil_img.convert("RGB"))
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
    img_518.save(tmp_path, format="JPEG", quality=95)
    try:
        report = generator.generate_report(image_path=tmp_path)
        return report.findings
    finally:
        os.remove(tmp_path)


# ══════════════════════════════════════════════════════════════════════════════
#  FLUX.2-dev
# ══════════════════════════════════════════════════════════════════════════════

def load_flux2():
    """
    Load base FLUX.2-dev with sequential CPU offload.

    enable_sequential_cpu_offload() streams layer-by-layer (peak VRAM ~0.5GB/layer)
    vs enable_model_cpu_offload() which loads full sub-models at once
    (Mistral=24GB, transformer=32GB) causing OOM on 44GB GPUs.
    Tradeoff: ~2-3x slower per inference step but fits on any GPU >= 16GB.
    """
    from diffusers import Flux2Pipeline
    logger.info("Loading FLUX.2-dev (base, no LoRA, sequential cpu offload)...")
    pipe = Flux2Pipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    pipe.enable_sequential_cpu_offload()
    logger.info("  ✓ FLUX.2-dev ready")
    return pipe


def run_flux2(
    pipe,
    findings_text: str,
    seed: int = 42,
    num_steps: int = 28,
    guidance_scale: float = 3.5,
    prompt_template: str = "Frontal chest X-ray. FINDINGS: {findings}",
) -> Image.Image:
    """
    FINDINGS text → 512×512 RGB PIL image.

    Prompt template notes:
      Base FLUX.2 (this run): "Frontal chest X-ray. FINDINGS: {findings}"
        Explicit modality cue helps the general-purpose base model.
      Future LoRA run: switch to "FINDINGS: {findings}"
        Must match build_conditioning_text() training distribution exactly.
    """
    prompt = prompt_template.format(findings=findings_text)
    result = pipe(
        prompt=prompt,
        height=512, width=512,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        max_sequence_length=128,
        generator=torch.Generator("cpu").manual_seed(seed),
    )
    return result.images[0]


# ══════════════════════════════════════════════════════════════════════════════
#  ATTRACTOR LOOP (single study)
# ══════════════════════════════════════════════════════════════════════════════

def run_loop(
    gt_image_path: str,
    study_id: str,
    gt_findings: str,
    medclip,
    tokenizer,
    maira,
    flux,
    device: torch.device,
    n_iters: int = 5,
    output_dir: str = ".",
    num_steps: int = 28,
    guidance_scale: float = 3.5,
    seed: int = 42,
    prompt_template: str = "Frontal chest X-ray. FINDINGS: {findings}",
) -> dict:
    study_dir = os.path.join(output_dir, study_id)
    os.makedirs(study_dir, exist_ok=True)

    gt_pil = Image.open(gt_image_path).convert("RGB")
    gt_pil.resize((512, 512), Image.LANCZOS).convert("L").save(
        os.path.join(study_dir, "gt_image.png"))
    with open(os.path.join(study_dir, "gt_findings.txt"), "w") as f:
        f.write(gt_findings)

    # ── Anchor embeddings ─────────────────────────────────────────────────────
    anchor_img  = embed_image(medclip, gt_pil, device)
    anchor_text = embed_text(medclip, tokenizer, gt_findings, device)
    logger.info(f"  [{study_id}] anchors: img_norm={anchor_img.norm():.4f}  "
                f"txt_norm={anchor_text.norm():.4f}")

    # ── Iteration 0: GT → MAIRA-2 → FLUX.2 ────────────────────────────────────
    logger.info(f"  [{study_id}] Iter 0: GT image → MAIRA-2...")
    t          = time.time()
    findings_0 = run_maira(maira, gt_pil)
    logger.info(f"  [{study_id}] Iter 0 MAIRA-2 ({time.time()-t:.1f}s)")
    logger.info(f"  [{study_id}] Findings 0: \"{findings_0[:120]}\"")

    logger.info(f"  [{study_id}] Iter 0: findings → FLUX.2 "
                f"(prompt: \"{prompt_template.format(findings=findings_0[:50])}...\")")
    t     = time.time()
    gen_0 = run_flux2(flux, findings_0, seed=seed, num_steps=num_steps,
                      guidance_scale=guidance_scale, prompt_template=prompt_template)
    logger.info(f"  [{study_id}] Iter 0 FLUX.2 ({time.time()-t:.1f}s)  size={gen_0.size}")

    gen_0.convert("L").save(os.path.join(study_dir, "gen_iter_000.png"))
    with open(os.path.join(study_dir, "findings_iter_000.txt"), "w") as f:
        f.write(findings_0)

    img_e0  = embed_image(medclip, gen_0, device)
    text_e0 = embed_text(medclip, tokenizer, findings_0, device)

    metrics = {
        "study_id":       study_id,
        "gt_findings":    gt_findings,
        "findings":       [findings_0],
        "image_cosine":   [cosine_sim(anchor_img, img_e0)],
        "text_cosine":    [cosine_sim(anchor_text, text_e0)],
        "embed_l2":       [float((anchor_img - img_e0).norm().item())],
        "text_evolution": [],
    }
    logger.info(
        f"  [{study_id}] Iter 0: "
        f"img_cos={metrics['image_cosine'][-1]:.4f}  "
        f"txt_cos={metrics['text_cosine'][-1]:.4f}  "
        f"l2={metrics['embed_l2'][-1]:.4f}"
    )

    # ── Attractor iterations ───────────────────────────────────────────────────
    current_img   = gen_0
    prev_text_emb = text_e0

    for k in range(1, n_iters + 1):
        logger.info(f"  [{study_id}] Iter {k}: gen_{k-1} → MAIRA-2...")
        t          = time.time()
        findings_k = run_maira(maira, current_img)
        logger.info(f"  [{study_id}] Iter {k} MAIRA-2 ({time.time()-t:.1f}s)")
        logger.info(f"  [{study_id}] Findings {k}: \"{findings_k[:120]}\"")

        logger.info(f"  [{study_id}] Iter {k}: findings → FLUX.2...")
        t     = time.time()
        gen_k = run_flux2(flux, findings_k, seed=seed + k, num_steps=num_steps,
                          guidance_scale=guidance_scale, prompt_template=prompt_template)
        logger.info(f"  [{study_id}] Iter {k} FLUX.2 ({time.time()-t:.1f}s)")

        gen_k.convert("L").save(
            os.path.join(study_dir, f"gen_iter_{k:03d}.png"))
        with open(os.path.join(study_dir, f"findings_iter_{k:03d}.txt"), "w") as f:
            f.write(findings_k)

        img_ek  = embed_image(medclip, gen_k, device)
        text_ek = embed_text(medclip, tokenizer, findings_k, device)
        txt_evo = cosine_sim(prev_text_emb, text_ek)

        metrics["findings"].append(findings_k)
        metrics["image_cosine"].append(cosine_sim(anchor_img, img_ek))
        metrics["text_cosine"].append(cosine_sim(anchor_text, text_ek))
        metrics["embed_l2"].append(float((anchor_img - img_ek).norm().item()))
        metrics["text_evolution"].append(txt_evo)

        logger.info(
            f"  [{study_id}] Iter {k}: "
            f"img_cos={metrics['image_cosine'][-1]:.4f}  "
            f"txt_cos={metrics['text_cosine'][-1]:.4f}  "
            f"l2={metrics['embed_l2'][-1]:.4f}  "
            f"txt_evo={txt_evo:.4f}"
        )

        current_img   = gen_k
        prev_text_emb = text_ek

    with open(os.path.join(study_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_samples",       type=int,   default=100)
    p.add_argument("--n_iters",         type=int,   default=5)
    p.add_argument("--num_steps",       type=int,   default=28)
    p.add_argument("--guidance_scale",  type=float, default=3.5)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--split",           type=str,   default="test")
    p.add_argument("--data_csv",        type=str,   default=DATA_CSV)
    p.add_argument("--output_dir",      type=str,
                   default=f"{BASE_DIR}/Experiments/attractor_loop/results/base_flux2")
    p.add_argument("--prompt_template", type=str,
                   default="Frontal chest X-ray. FINDINGS: {findings}",
                   help="Use 'FINDINGS: {findings}' when switching to LoRA model")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 60)
    logger.info("AIM2 Attractor Loop")
    logger.info("=" * 60)
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")
    if torch.cuda.is_available():
        gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)} ({gb:.0f} GB)")

    # ── Sample studies ─────────────────────────────────────────────────────────
    df     = pd.read_csv(args.data_csv, low_memory=False)
    df     = df[df["split"] == args.split]
    df     = df[df["has_findings"] == True]
    frontal = df[df["ViewPosition"].isin({"PA", "AP", "pa", "ap"})].copy()
    if frontal.empty:
        frontal = df.copy()
    frontal["_rank"] = frontal["ViewPosition"].map(
        lambda v: 0 if str(v).upper() == "PA" else 1)
    frontal = (frontal.sort_values("_rank")
                       .groupby("study_id", as_index=False).first()
                       .drop(columns=["_rank"]))
    sample = frontal.sample(
        n=min(args.n_samples, len(frontal)),
        random_state=args.seed,
    ).reset_index(drop=True)
    logger.info(f"\nSelected {len(sample)} studies from {args.split} split")

    # ── Resume support ─────────────────────────────────────────────────────────
    completed = {
        str(sid) for sid in sample["study_id"].astype(str)
        if os.path.exists(os.path.join(args.output_dir, str(sid), "metrics.json"))
    }
    if completed:
        logger.info(f"Resuming: {len(completed)} done, "
                    f"{len(sample)-len(completed)} remaining")

    # ── Load models ────────────────────────────────────────────────────────────
    logger.info("\nLoading models...")
    medclip, _, tokenizer = load_medclip(device)
    maira                 = load_maira()
    flux                  = load_flux2()

    # ── Run loop ───────────────────────────────────────────────────────────────
    all_metrics = []
    failed      = []

    for i, row in sample.iterrows():
        sid         = str(row["study_id"])
        img_path    = row["image_path"]
        gt_findings = str(row["findings"]) if pd.notna(row.get("findings")) else ""

        if sid in completed:
            with open(os.path.join(args.output_dir, sid, "metrics.json")) as f:
                all_metrics.append(json.load(f))
            continue

        if not gt_findings:
            logger.warning(f"Study {sid}: empty findings, skipping")
            continue
        if not os.path.exists(img_path):
            logger.error(f"Study {sid}: image not found at {img_path}")
            failed.append(sid)
            continue

        logger.info(f"\n{'─'*60}")
        logger.info(f"Study {i+1}/{len(sample)}: {sid}")
        logger.info(f"GT: \"{gt_findings[:120]}...\"")

        try:
            m = run_loop(
                gt_image_path  = img_path,
                study_id       = sid,
                gt_findings    = gt_findings,
                medclip        = medclip,
                tokenizer      = tokenizer,
                maira          = maira,
                flux           = flux,
                device         = device,
                n_iters        = args.n_iters,
                output_dir     = args.output_dir,
                num_steps      = args.num_steps,
                guidance_scale = args.guidance_scale,
                seed           = args.seed + i,
                prompt_template = args.prompt_template,
            )
            all_metrics.append(m)
        except Exception as e:
            logger.error(f"Study {sid} FAILED: {e}", exc_info=True)
            failed.append(sid)
            gc.collect()
            torch.cuda.empty_cache()

    # ── Summary ────────────────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"COMPLETE: {len(all_metrics)}/{len(sample)} studies OK")
    if failed:
        logger.error(f"Failed ({len(failed)}): {failed}")

    if all_metrics:
        summary = {}
        for k in range(args.n_iters + 1):
            cos = [m["image_cosine"][k] for m in all_metrics if len(m["image_cosine"]) > k]
            txt = [m["text_cosine"][k]  for m in all_metrics if len(m["text_cosine"])  > k]
            l2  = [m["embed_l2"][k]     for m in all_metrics if len(m["embed_l2"])     > k]
            if cos:
                summary[f"iter_{k}"] = {
                    "n":               len(cos),
                    "img_cosine_mean": float(np.mean(cos)),
                    "img_cosine_std":  float(np.std(cos)),
                    "txt_cosine_mean": float(np.mean(txt)),
                    "txt_cosine_std":  float(np.std(txt)),
                    "embed_l2_mean":   float(np.mean(l2)),
                    "embed_l2_std":    float(np.std(l2)),
                }

        logger.info(f"\n{'Iter':>5}  {'img_cos':>8}±{'std':>5}  "
                    f"{'txt_cos':>8}±{'std':>5}  {'l2':>8}±{'std':>5}")
        logger.info("─" * 60)
        for k in range(args.n_iters + 1):
            s = summary.get(f"iter_{k}", {})
            if s:
                logger.info(
                    f"{k:>5}  "
                    f"{s['img_cosine_mean']:>8.4f}±{s['img_cosine_std']:>5.3f}  "
                    f"{s['txt_cosine_mean']:>8.4f}±{s['txt_cosine_std']:>5.3f}  "
                    f"{s['embed_l2_mean']:>8.4f}±{s['embed_l2_std']:>5.3f}"
                )

        out = {
            "args":      vars(args),
            "n_studies": len(all_metrics),
            "failed":    failed,
            "summary":   summary,
            "per_study": all_metrics,
        }
        summary_path = os.path.join(args.output_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(out, f, indent=2)
        logger.info(f"\nResults: {args.output_dir}")
        logger.info(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()