#!/usr/bin/env python3
"""
AIM2 — Disease Attractors in Chest X-Ray Generation
Attractor Loop (Dry Run with Base FLUX.2-dev, No LoRA)

Pipeline per sample:
    GT CXR ──LANCZOS──▶ 518×518 ──▶ MAIRA-2 ──▶ FINDINGS text
    GT CXR ──LANCZOS──▶ 512×512 ──▶ MedCLIP ──▶ anchor_embed  (256-dim L2)
    FINDINGS ──▶ FLUX.2-dev ──▶ gen_image (512×512)
    gen_image ──▶ MedCLIP ──▶ gen_embed  (256-dim L2)

    Attractor iterations (K times):
        gen_image ──LANCZOS──▶ 518×518 ──▶ MAIRA-2 ──▶ new_findings
        new_findings ──▶ FLUX.2-dev ──▶ new_gen_image (512×512)
        new_gen_image ──▶ MedCLIP ──▶ new_embed

    Tracked metrics per step:
        image_drift  = 1 - cosine_sim(anchor_embed, gen_embed_k)
        text_drift   = 1 - cosine_sim(text_embed_k, text_embed_{k+1})
        embed_l2     = ||anchor_embed - gen_embed_k||_2

Design decisions (confirmed):
    - FINDINGS only throughout (no IMPRESSION anywhere)
    - MAIRA-2 input: 518×518 LANCZOS (processor native size)
    - MedCLIP input: 512×512 LANCZOS + ImageNet normalisation
    - MedCLIP config: IMAGE_SIZE=512, USE_FINDINGS_ONLY=True (combine_sections=False)
    - FLUX.2 prompt: "FINDINGS: <maira_output>"
    - FLUX.2 inference: guidance=3.5, steps=28, max_seq=128, cpu generator
    - MedCLIP val transform: Resize(512,LANCZOS)→ToTensor→Normalize(ImageNet)
      (NOT the broken get_val_transforms which does Resize(256)+CenterCrop(512))

Usage:
    cd /n/groups/training/bmif203/AIM2
    python Experiments/attractor_loop/attractor_loop.py \
        --n_samples 20 --n_iters 5 --output_dir Results/attractor_dry_run
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = "/n/groups/training/bmif203/AIM2"
DATA_CSV     = f"{BASE_DIR}/processed_data/processed_data.csv"
MEDCLIP_CKPT = f"{BASE_DIR}/CLIP/outputs/checkpoints/best_model.pth"
MAIRA_FILE   = f"{BASE_DIR}/MAIRA/maira.py"
MODEL_ID     = "black-forest-labs/FLUX.2-dev"
HF_HOME      = "/n/scratch/users/g/gul075/.cache/huggingface"

os.environ["HF_HOME"]                 = HF_HOME
os.environ["TOKENIZERS_PARALLELISM"]  = "false"

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  IMAGE TRANSFORMS
# ══════════════════════════════════════════════════════════════════════════════

# MedCLIP: 512px LANCZOS direct resize (the broken get_val_transforms was fixed
# to this during training — see trainer.py session fix)
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

# MAIRA-2: 518×518 LANCZOS (processor native size)
def resize_for_maira(img: Image.Image) -> Image.Image:
    return img.resize((518, 518), Image.LANCZOS)


# ══════════════════════════════════════════════════════════════════════════════
#  MEDCLIP LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_medclip(checkpoint_path: str, device: torch.device):
    """
    Load the trained MedCLIP model from checkpoint.

    The checkpoint was saved by CLIP/utils/checkpoint.py and contains:
        {"model_state_dict": ..., "config": ..., ...}
    We reconstruct the config programmatically to match training exactly.
    """
    logger.info(f"Loading MedCLIP from {checkpoint_path}")

    # Add the AIM2 repo root to sys.path so CLIP.* imports resolve
    if BASE_DIR not in sys.path:
        sys.path.insert(0, BASE_DIR)

    from CLIP.config.config import Config
    from CLIP.model.clip_model import MedicalCLIP

    # Reconstruct training config
    config = Config()
    config.data.IMAGE_SIZE         = 512        # overridden from 224 at training time
    config.data.COMBINE_SECTIONS   = False      # FINDINGS only
    config.model.IMAGE_ENCODER     = "vit_base_patch16_224"
    config.model.IMAGE_PRETRAINED  = False      # we load from checkpoint, not HF

    model = MedicalCLIP(config)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # The checkpoint key may be "model_state_dict" or "state_dict"
    state_key = "model_state_dict" if "model_state_dict" in ckpt else "state_dict"
    model.load_state_dict(ckpt[state_key], strict=True)
    model = model.to(device).eval()
    logger.info("MedCLIP loaded (epoch %s, val_loss=%s)",
                ckpt.get("epoch", "?"), ckpt.get("val_loss", "?"))
    return model, config


# ══════════════════════════════════════════════════════════════════════════════
#  MEDCLIP EMBEDDING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def embed_image(model, pil_img: Image.Image, device: torch.device) -> torch.Tensor:
    """
    PIL image → (256,) L2-normalized float32 tensor on CPU.
    Input image is expected as RGB; we apply MEDCLIP_TRANSFORM here.
    """
    img_rgb  = pil_img.convert("RGB")
    tensor   = MEDCLIP_TRANSFORM(img_rgb).unsqueeze(0).to(device)
    embed    = model.encode_image(tensor)          # (1, 256) L2-normalized
    return embed.squeeze(0).float().cpu()


@torch.no_grad()
def embed_text(
    model,
    tokenizer,
    text: str,
    device: torch.device,
    max_length: int = 512,
) -> torch.Tensor:
    """
    FINDINGS string → (256,) L2-normalized float32 tensor on CPU.
    Uses Bio_ClinicalBERT tokenizer with max_length=512.
    """
    encoded = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids      = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    embed = model.encode_text(input_ids, attention_mask)  # (1, 256) L2-normalized
    return embed.squeeze(0).float().cpu()


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


# ══════════════════════════════════════════════════════════════════════════════
#  MAIRA-2 WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

def load_maira(device: torch.device):
    """
    Load MAIRAReportGenerator from the existing MAIRA/maira.py wrapper.
    Uses local_files_only=True (model already downloaded).
    """
    logger.info("Loading MAIRA-2 from local cache...")
    if BASE_DIR not in sys.path:
        sys.path.insert(0, BASE_DIR)
    from MAIRA.maira import MAIRAReportGenerator
    generator = MAIRAReportGenerator(
        device=str(device),
        use_grounding=False,
    )
    logger.info("MAIRA-2 ready.")
    return generator


def run_maira(generator, pil_img: Image.Image) -> str:
    """
    PIL image → FINDINGS string.
    Resizes to 518×518 LANCZOS before passing to MAIRA-2 (processor native).
    Returns the raw findings text.
    """
    img_518 = resize_for_maira(pil_img.convert("RGB"))
    # Temporarily save to a temp file — MAIRA wrapper expects a file path
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
    img_518.save(tmp_path, format="JPEG", quality=95)
    try:
        report = generator.generate_report(image_path=tmp_path)
        findings = report.findings.strip()
    finally:
        os.remove(tmp_path)
    return findings if findings else "No findings reported."


# ══════════════════════════════════════════════════════════════════════════════
#  FLUX.2 WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

def load_flux2(device: torch.device):
    """
    Load base FLUX.2-dev pipeline (no LoRA).
    Uses CPU offload to fit within 80 GB alongside MAIRA-2 (freed before loading).
    """
    from diffusers import Flux2Pipeline
    logger.info("Loading FLUX.2-dev pipeline (base, no LoRA)...")
    pipe = Flux2Pipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()
    logger.info("FLUX.2-dev ready.")
    return pipe


def run_flux2(
    pipe,
    findings_text: str,
    seed: int = 42,
    guidance_scale: float = 3.5,
    num_inference_steps: int = 28,
    prompt_template: str = "Frontal chest X-ray. FINDINGS: {findings}",
) -> Image.Image:
    """
    FINDINGS text → 512×512 RGB PIL image.

    Prompt template notes:
    - Base FLUX.2 (this dry run): "Frontal chest X-ray. FINDINGS: {findings}"
      Adds modality + projection cue which helps the base general-purpose model.
    - Future LoRA (once fine-tuned): switch to "FINDINGS: {findings}"
      to exactly match the training distribution from build_conditioning_text().
    """
    prompt = prompt_template.format(findings=findings_text)
    result = pipe(
        prompt=prompt,
        height=512,
        width=512,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        max_sequence_length=128,
        generator=torch.Generator("cpu").manual_seed(seed),
    )
    return result.images[0]


# ══════════════════════════════════════════════════════════════════════════════
#  ATTRACTOR LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_attractor_loop(
    gt_image_path: str,
    study_id: str,
    gt_findings: str,
    medclip_model,
    medclip_tokenizer,
    maira_generator,
    flux_pipe,
    device: torch.device,
    n_iters: int = 5,
    output_dir: str = ".",
    guidance_scale: float = 3.5,
    num_inference_steps: int = 28,
    seed: int = 42,
    prompt_template: str = "Frontal chest X-ray. FINDINGS: {findings}",
) -> Dict:
    """
    Full attractor loop for one study.

    Returns a dict with per-iteration metrics:
        image_cosine[k]  = cosine_sim(anchor_img_embed, gen_img_embed at iter k)
        text_cosine[k]   = cosine_sim(gt_text_embed, gen_text_embed at iter k)
        embed_l2[k]      = L2 distance anchor vs gen at iter k
        text_evolution[k]= cosine_sim(text_embed[k-1], text_embed[k])
        findings[k]      = MAIRA-2 findings text at iter k
    """
    study_dir = os.path.join(output_dir, study_id)
    os.makedirs(study_dir, exist_ok=True)

    gt_pil = Image.open(gt_image_path).convert("RGB")

    # ── Anchor embeddings (from GT image and GT findings) ─────────────────────
    logger.info(f"[{study_id}] Computing anchor embeddings...")
    anchor_img_embed  = embed_image(medclip_model, gt_pil, device)
    anchor_text_embed = embed_text(medclip_model, medclip_tokenizer,
                                   gt_findings, device)

    # Save anchor image
    gt_pil_512 = gt_pil.resize((512, 512), Image.LANCZOS)
    gt_pil_512.convert("L").save(os.path.join(study_dir, "gt_image.png"))
    with open(os.path.join(study_dir, "gt_findings.txt"), "w") as f:
        f.write(gt_findings)

    # ── Iteration 0: GT image → MAIRA-2 → FLUX.2 ──────────────────────────────
    logger.info(f"[{study_id}] Iter 0: GT → MAIRA-2...")
    t0 = time.time()
    findings_0 = run_maira(maira_generator, gt_pil)
    logger.info(f"[{study_id}] Iter 0 MAIRA-2 ({time.time()-t0:.1f}s): {findings_0[:80]}...")

    logger.info(f"[{study_id}] Iter 0: MAIRA-2 findings → FLUX.2...")
    logger.info(f"[{study_id}] Prompt: {prompt_template.format(findings=findings_0[:60])}...")
    t0 = time.time()
    gen_image_0 = run_flux2(flux_pipe, findings_0,
                            seed=seed, guidance_scale=guidance_scale,
                            num_inference_steps=num_inference_steps)
    logger.info(f"[{study_id}] Iter 0 FLUX.2 ({time.time()-t0:.1f}s)")

    gen_image_0.convert("L").save(os.path.join(study_dir, "gen_iter_000.png"))
    with open(os.path.join(study_dir, "findings_iter_000.txt"), "w") as f:
        f.write(findings_0)

    gen_img_embed_0   = embed_image(medclip_model, gen_image_0, device)
    gen_text_embed_0  = embed_text(medclip_model, medclip_tokenizer,
                                   findings_0, device)

    # ── Metrics storage ────────────────────────────────────────────────────────
    metrics = {
        "study_id":        study_id,
        "gt_findings":     gt_findings,
        "n_iters":         n_iters,
        "findings":        [findings_0],
        "image_cosine":    [cosine_sim(anchor_img_embed, gen_img_embed_0)],
        "text_cosine":     [cosine_sim(anchor_text_embed, gen_text_embed_0)],
        "embed_l2":        [float((anchor_img_embed - gen_img_embed_0).norm().item())],
        "text_evolution":  [],   # similarity between consecutive findings embeddings
    }

    logger.info(
        f"[{study_id}] Iter 0: "
        f"img_cos={metrics['image_cosine'][-1]:.4f}  "
        f"txt_cos={metrics['text_cosine'][-1]:.4f}  "
        f"l2={metrics['embed_l2'][-1]:.4f}"
    )

    # ── Attractor iterations ───────────────────────────────────────────────────
    current_image   = gen_image_0
    prev_text_embed = gen_text_embed_0

    for k in range(1, n_iters + 1):
        logger.info(f"[{study_id}] Iter {k}: gen_image → MAIRA-2...")
        t0 = time.time()
        findings_k = run_maira(maira_generator, current_image)
        logger.info(f"[{study_id}] Iter {k} MAIRA-2 ({time.time()-t0:.1f}s): "
                    f"{findings_k[:80]}...")

        logger.info(f"[{study_id}] Iter {k}: MAIRA-2 findings → FLUX.2...")
        t0 = time.time()
        gen_image_k = run_flux2(flux_pipe, findings_k,
                                seed=seed + k,
                                guidance_scale=guidance_scale,
                                num_inference_steps=num_inference_steps,
                                prompt_template=prompt_template)
        logger.info(f"[{study_id}] Iter {k} FLUX.2 ({time.time()-t0:.1f}s)")

        gen_image_k.convert("L").save(
            os.path.join(study_dir, f"gen_iter_{k:03d}.png"))
        with open(os.path.join(study_dir, f"findings_iter_{k:03d}.txt"), "w") as f:
            f.write(findings_k)

        gen_img_embed_k  = embed_image(medclip_model, gen_image_k, device)
        gen_text_embed_k = embed_text(medclip_model, medclip_tokenizer,
                                      findings_k, device)

        text_evo = cosine_sim(prev_text_embed, gen_text_embed_k)

        metrics["findings"].append(findings_k)
        metrics["image_cosine"].append(cosine_sim(anchor_img_embed, gen_img_embed_k))
        metrics["text_cosine"].append(cosine_sim(anchor_text_embed, gen_text_embed_k))
        metrics["embed_l2"].append(float((anchor_img_embed - gen_img_embed_k).norm().item()))
        metrics["text_evolution"].append(text_evo)

        logger.info(
            f"[{study_id}] Iter {k}: "
            f"img_cos={metrics['image_cosine'][-1]:.4f}  "
            f"txt_cos={metrics['text_cosine'][-1]:.4f}  "
            f"l2={metrics['embed_l2'][-1]:.4f}  "
            f"txt_evo={text_evo:.4f}"
        )

        current_image   = gen_image_k
        prev_text_embed = gen_text_embed_k

    # Save metrics
    with open(os.path.join(study_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_samples",     type=int,   default=20,
                   help="Number of studies to run the attractor loop on")
    p.add_argument("--n_iters",       type=int,   default=5,
                   help="Number of attractor iterations per study")
    p.add_argument("--output_dir",    type=str,
                   default=f"{BASE_DIR}/Results/attractor_dry_run",
                   help="Where to save images and metrics")
    p.add_argument("--split",         type=str,   default="test",
                   help="Dataset split to sample from")
    p.add_argument("--guidance_scale",type=float, default=3.5)
    p.add_argument("--num_steps",     type=int,   default=28)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--data_csv",      type=str,   default=DATA_CSV)
    p.add_argument("--prompt_template", type=str,
                   default="Frontal chest X-ray. FINDINGS: {findings}",
                   help="Prompt format. Use {findings} as placeholder. "
                        "For LoRA inference, switch to 'FINDINGS: {findings}' "
                        "to match training distribution.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {torch.cuda.get_device_name(0)} ({gb:.0f} GB)")

    # ── Sample studies from test split ────────────────────────────────────────
    logger.info(f"Loading dataset from {args.data_csv}")
    df = pd.read_csv(args.data_csv, low_memory=False)
    df = df[df["split"] == args.split]
    df = df[df["has_findings"] == True]

    # One frontal view per study (prefer PA over AP)
    frontal_mask = df["ViewPosition"].isin({"PA", "AP", "pa", "ap"})
    df_frontal = df[frontal_mask].copy() if frontal_mask.any() else df.copy()
    df_frontal["_rank"] = df_frontal["ViewPosition"].map(
        lambda v: 0 if str(v).upper() == "PA" else 1)
    df_frontal = (df_frontal.sort_values("_rank")
                             .groupby("study_id", as_index=False)
                             .first()
                             .drop(columns=["_rank"]))

    df_sample = df_frontal.sample(
        n=min(args.n_samples, len(df_frontal)),
        random_state=args.seed,
    ).reset_index(drop=True)

    logger.info(f"Selected {len(df_sample)} studies from {args.split} split")

    # ── Load models ───────────────────────────────────────────────────────────
    # Load order matters for VRAM:
    # MAIRA-2 (~24 GB) + FLUX.2 (streamed via cpu_offload) + MedCLIP (~1 GB)
    # All fit on A100 80GB with cpu_offload on FLUX.2

    # MedCLIP first (smallest, ~1 GB)
    medclip_model, medclip_cfg = load_medclip(MEDCLIP_CKPT, device)
    medclip_tokenizer = AutoTokenizer.from_pretrained(
        medclip_cfg.model.TEXT_ENCODER)

    # MAIRA-2 (~24 GB on GPU)
    maira_gen = load_maira(device)

    # FLUX.2-dev (cpu_offload: transformer streamed through GPU)
    flux_pipe = load_flux2(device)

    # ── Run attractor loop ────────────────────────────────────────────────────
    all_metrics = []
    failed = []

    for i, row in df_sample.iterrows():
        study_id  = str(row["study_id"])
        img_path  = row["image_path"]
        gt_findings = row["findings"] if pd.notna(row.get("findings")) else ""

        if not gt_findings:
            logger.warning(f"Study {study_id}: no GT findings, skipping")
            continue

        if not os.path.exists(img_path):
            logger.warning(f"Study {study_id}: image not found at {img_path}, skipping")
            failed.append(study_id)
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Study {i+1}/{len(df_sample)}: {study_id}")
        logger.info(f"GT findings: {gt_findings[:120]}...")

        try:
            metrics = run_attractor_loop(
                gt_image_path       = img_path,
                study_id            = study_id,
                gt_findings         = gt_findings,
                medclip_model       = medclip_model,
                medclip_tokenizer   = medclip_tokenizer,
                maira_generator     = maira_gen,
                flux_pipe           = flux_pipe,
                device              = device,
                n_iters             = args.n_iters,
                output_dir          = args.output_dir,
                guidance_scale      = args.guidance_scale,
                num_inference_steps = args.num_steps,
                seed                = args.seed + i,
                prompt_template     = args.prompt_template,
            )
            all_metrics.append(metrics)

        except Exception as exc:
            logger.error(f"Study {study_id} failed: {exc}", exc_info=True)
            failed.append(study_id)
            continue

    # ── Aggregate and save results ────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"Completed {len(all_metrics)}/{len(df_sample)} studies")
    if failed:
        logger.warning(f"Failed: {failed}")

    if all_metrics:
        # Aggregate across studies
        summary = {}
        for k in range(args.n_iters + 1):
            cos_vals = [m["image_cosine"][k] for m in all_metrics
                        if len(m["image_cosine"]) > k]
            txt_vals = [m["text_cosine"][k] for m in all_metrics
                        if len(m["text_cosine"]) > k]
            l2_vals  = [m["embed_l2"][k] for m in all_metrics
                        if len(m["embed_l2"]) > k]
            if cos_vals:
                summary[f"iter_{k}"] = {
                    "n":              len(cos_vals),
                    "img_cosine_mean": float(np.mean(cos_vals)),
                    "img_cosine_std":  float(np.std(cos_vals)),
                    "txt_cosine_mean": float(np.mean(txt_vals)),
                    "txt_cosine_std":  float(np.std(txt_vals)),
                    "embed_l2_mean":   float(np.mean(l2_vals)),
                    "embed_l2_std":    float(np.std(l2_vals)),
                }

        # Print summary table
        logger.info("\nSUMMARY — mean metrics across studies:")
        logger.info(f"{'Iter':>5}  {'img_cos':>8}  {'txt_cos':>8}  {'embed_l2':>9}")
        logger.info("-" * 40)
        for k in range(args.n_iters + 1):
            s = summary.get(f"iter_{k}", {})
            if s:
                logger.info(
                    f"{k:>5}  {s['img_cosine_mean']:>8.4f}  "
                    f"{s['txt_cosine_mean']:>8.4f}  "
                    f"{s['embed_l2_mean']:>9.4f}"
                )

        # Save
        summary_path = os.path.join(args.output_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump({
                "args":      vars(args),
                "n_studies": len(all_metrics),
                "failed":    failed,
                "summary":   summary,
                "per_study": all_metrics,
            }, f, indent=2)

        logger.info(f"\nResults saved to: {args.output_dir}")
        logger.info(f"Summary JSON:     {summary_path}")


if __name__ == "__main__":
    main()