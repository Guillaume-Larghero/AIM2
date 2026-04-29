#!/usr/bin/env python3
"""
AIM2 — Disease Attractors in Chest X-Ray Generation (ChexGen variant)
Attractor Loop — Pilot / Full Test Set Run

Pipeline per study, per iteration k ∈ {0, 1, ..., n_iters}:
  current_image (PIL RGB 512×512)
    ├─ LANCZOS 518×518 ──▶ MAIRA-2 ──▶ findings_k         (CPU↔GPU swap)
    └─ MEDCLIP transform   ──▶ MedCLIP image encoder ──▶ img_embed_k  (256-d L2)

  findings_k
    ├─ T5Embedder.clean_caption (lowercase + strip URLs/punct) ──▶ cleaned
    │   └─ tokenize(max=120) ──▶ T5EncoderModel ──▶ caption_embs (1,120,4096) BF16
    │       └─ .float() ──▶ DiT.y_embedder ──▶ (1,1152) hidden
    ├─ p_sample_loop(model.forward_with_cfg, ...) ──▶ latent (1,4,64,64)
    │   └─ VAE decode ──▶ next image (1,3,512,512) in [-1,1]
    └─ MedCLIP text encoder (uses ORIGINAL findings, not cleaned) ──▶ text_embed_k

  Iteration k=0 is the bootstrap: GT image → MAIRA-2 → ChexGen.
  Iteration k≥1 is the autoregressive loop: gen_{k-1} → MAIRA-2 → ChexGen.

Metrics tracked per iteration (in metrics.json):
  image_cosine    = cos(anchor_img_embed_GT,  img_embed_k)
  text_cosine     = cos(anchor_text_embed_GT, text_embed_k)
  embed_l2        = ||anchor_img_embed_GT - img_embed_k||_2
  image_evolution = cos(img_embed_{k-1},  img_embed_k)        [k≥1]
  text_evolution  = cos(text_embed_{k-1}, text_embed_k)       [k≥1]

Pilot diagnostics tracked per iteration (additionally):
  - findings: original/cleaned char + word counts
  - T5 token counts (natural vs used), was_truncated flag, tokens_lost
  - DiT sampling wallclock, VAE decode wallclock
  - generated image value range
  - embedding norms (sanity check L2≈1.0)
  - peak GPU memory snapshot

Design decisions / gotchas confirmed against ChexGen source:
  - ChexGen weight  : finetune_impression_512.pth (IMPRESSION-conditioned, not FINDINGS).
                      We use it with FINDINGS prompts → documented OOD limitation.
  - T5 encoder      : DeepFloyd/t5-v1_1-xxl, BF16 default. token_num=120 (fixed by training).
                      MAIRA-2 FINDINGS routinely exceed 120 tokens → truncation IS expected.
  - DiT             : DiT_XL_2, depth=28, hidden=1152, patch=2, fp32, fp32_attn=True.
                      input_size=64 → latent (1,4,64,64) → VAE → 512×512.
                      pos_embed_scale=2.0 (NOT default 1.0).
                      learn_sigma=True → 8-channel output; CFG on first 3 channels only
                      (DiT codebase quirk inherited).
  - VAE             : stabilityai/sd-vae-ft-ema, latent scale 0.18215.
  - CFG             : default 4.0, num_steps default 100. Standard doubled-batch trick.
  - DDP             : NOT needed. We call model.forward_with_cfg directly (no .module).
  - Cast            : T5 outputs BF16, DiT y_embedder MLP is FP32 → explicit y.float() before
                      passing into forward_with_cfg.
  - MedCLIP         : 512×512 (matches ChexGen output → no resize). Bio_ClinicalBERT for text,
                      USE_FINDINGS_ONLY=True at training time. Embeds ORIGINAL findings, not
                      the lowercased cleaned ones.
  - MAIRA-2         : 518×518 LANCZOS input, get_grounding=False, max_new_tokens=300.
                      transformers==4.51.3 pinned. num_additional_image_tokens=1 for DINO CLS.
                      Kept on CPU, swapped to GPU per inference call.

Memory budget (L40S 48 GB):
  Resident GPU : T5-xxl BF16 (~11 GB) + DiT FP32 (~2.7 GB) + VAE (~0.3 GB) + MedCLIP (~1.3 GB)
                 ≈ 15 GB
  Transient    : MAIRA-2 swaps in (~14 GB) during ~13 s of report generation
  Peak         : ~30 GB → 18 GB headroom

Usage (pilot):
  python Experiments/attractor_loop/attractor_loop_chexgen.py \
      --n_samples 5 --n_iters 5 --num_steps 100 --cfg_scale 4.0 --seed 42 \
      --output_dir Experiments/attractor_loop/results/chexgen_pilot
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
import traceback
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
CHEXGEN_DIR  = f"{BASE_DIR}/ChexGen"
CHEXGEN_CFG  = f"{CHEXGEN_DIR}/configs/model.py"
CHEXGEN_CKPT = f"{CHEXGEN_DIR}/weights/finetune_impression_512.pth"
HF_HOME      = "/n/scratch/users/g/gul075/.cache/huggingface"

os.environ["HF_HOME"]                = HF_HOME
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── mmcv stub ─────────────────────────────────────────────────────────────────
# radiffuser.utils.logger does `from mmcv.utils.logging import logger_initialized`
# at import time. We never call get_root_logger from radiffuser anywhere in the
# inference path — but the import is eager. mmcv has a heavy CUDA-compiled build
# that's painful on O2; instead we pre-register a fake module hierarchy so the
# import succeeds without installing the real package.
import sys as _sys
import types as _types
if "mmcv" not in _sys.modules:
    _mmcv_logging = _types.ModuleType("mmcv.utils.logging")
    _mmcv_logging.logger_initialized = {}
    _mmcv_utils = _types.ModuleType("mmcv.utils")
    _mmcv_utils.logging = _mmcv_logging
    _mmcv = _types.ModuleType("mmcv")
    _mmcv.utils = _mmcv_utils
    _sys.modules["mmcv"]               = _mmcv
    _sys.modules["mmcv.utils"]         = _mmcv_utils
    _sys.modules["mmcv.utils.logging"] = _mmcv_logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Speed knobs from ChexGen sample.py
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True


# ══════════════════════════════════════════════════════════════════════════════
#  IMAGE TRANSFORMS
# ══════════════════════════════════════════════════════════════════════════════

# MedCLIP: direct LANCZOS resize to 512px + ImageNet normalisation.
# ChexGen output is already 512×512 → for generated images this resize is a no-op.
# GT images (~2735×2790 in MIMIC-CXR JPG) get downsampled identically.
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


def latent_image_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """
    VAE-decoded tensor (1,3,512,512) in [-1, 1] → PIL RGB 512×512.

    Mirrors torchvision.utils.save_image(normalize=True, value_range=(-1, 1))
    behaviour. Returns RGB even though radiographs are intrinsically 1-channel:
    MedCLIP and MAIRA-2 both expect 3-channel input.
    """
    assert image_tensor.shape == (1, 3, 512, 512), \
        f"unexpected VAE output shape: {tuple(image_tensor.shape)}"
    t = image_tensor.squeeze(0).clamp(-1.0, 1.0)
    t = (t + 1.0) / 2.0                                    # [0, 1]
    arr = (t * 255.0).byte().permute(1, 2, 0).cpu().numpy()  # (512, 512, 3) uint8
    return Image.fromarray(arr, mode="RGB")


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
    config.data.IMAGE_SIZE        = 512
    config.data.COMBINE_SECTIONS  = False  # FINDINGS only
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

    Key fixes (carried over from FLUX.2 attractor_loop.py):
    1. Local snapshot path — bypasses HF hub lookup which fails on compute nodes.
    2. num_additional_image_tokens=1 — MAIRA-2 DINO encoder has a CLS token.
       Without this, newer transformers raises a token/feature count mismatch.
    3. Model kept on CPU — moved to GPU only during inference (~13 s).
    4. transformers==4.51.3 required.
    """
    import glob
    logger.info("Loading MAIRA-2 (local snapshot, CPU)...")

    snapshot_pattern = os.path.join(
        HF_HOME, "hub", "models--microsoft--maira-2", "snapshots", "*"
    )
    snapshots = sorted(glob.glob(snapshot_pattern))
    if not snapshots:
        raise RuntimeError(
            f"MAIRA-2 snapshot not found.\nExpected: {snapshot_pattern}"
        )
    model_path = snapshots[-1]
    logger.info(f"  Snapshot: {model_path}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float16,
    ).eval()

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        num_additional_image_tokens=1,
    )
    logger.info("  ✓ MAIRA-2 ready (on CPU)")

    class MAIRAWrapper:
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
            inf_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(inf_device)
            processed = {k: v.to(inf_device) for k, v in processed.items()}
            with torch.no_grad():
                output_ids = self.model.generate(
                    **processed, max_new_tokens=300, use_cache=True,
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
            findings = findings.strip() or "No findings reported."

            class _R: pass
            r = _R(); r.findings = findings
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
        return generator.generate_report(image_path=tmp_path).findings
    finally:
        os.remove(tmp_path)


# ══════════════════════════════════════════════════════════════════════════════
#  ChexGen
# ══════════════════════════════════════════════════════════════════════════════

class ChexGenWrapper:
    """
    Self-contained ChexGen inference wrapper that mirrors the upstream
    tools/sample.py logic without the DDP launch wrapper.

    Components:
      - dit       : DiT_XL_2 (FP32) on GPU
      - vae       : SD-VAE-ft-ema (FP32) on GPU
      - t5        : T5Embedder (BF16) on GPU; provides clean_caption + tokenize + encode
      - diffusion : SpacedDiffusion at the requested num_sampling_steps
      - latent_size, token_num : pulled from the config
    """

    def __init__(self, dit, vae, t5_embedder, diffusion, latent_size, token_num, device):
        self.dit         = dit
        self.vae         = vae
        self.t5          = t5_embedder
        self.diffusion   = diffusion
        self.latent_size = latent_size
        self.token_num   = token_num
        self.device      = device

    def _count_natural_tokens(self, text: str) -> int:
        """Token count without truncation, after T5 cleaning."""
        cleaned = self.t5.text_preprocessing(text)
        toks = self.t5.tokenizer(
            cleaned, padding=False, truncation=False,
            add_special_tokens=True, return_tensors="pt",
        )
        return int(toks["input_ids"].shape[-1])

    @torch.no_grad()
    def generate(self, findings_text: str, *, seed: int, num_steps: int,
                 cfg_scale: float) -> tuple:
        """
        FINDINGS string → (PIL RGB 512×512, diag dict).

        Diagnostics returned cover everything we want to track in the pilot:
        cleaning, tokenization, truncation, latent shape, value range, timings.
        """
        diag = {
            "input": {
                "raw_chars":  len(findings_text),
                "raw_words":  len(findings_text.split()),
            },
            "t5": {},
            "chexgen": {
                "cfg_scale": float(cfg_scale),
                "num_steps": int(num_steps),
                "seed":      int(seed),
            },
        }

        # ── T5 cleaning + tokenization diagnostics ────────────────────────────
        cleaned = self.t5.text_preprocessing(findings_text)
        diag["input"]["cleaned"]       = cleaned
        diag["input"]["cleaned_chars"] = len(cleaned)

        natural_tokens = self._count_natural_tokens(findings_text)
        used_tokens    = min(natural_tokens, self.token_num)
        was_truncated  = natural_tokens > self.token_num
        diag["t5"]["natural_token_count"] = natural_tokens
        diag["t5"]["used_token_count"]    = used_tokens
        diag["t5"]["was_truncated"]       = bool(was_truncated)
        diag["t5"]["tokens_lost"]         = max(0, natural_tokens - self.token_num)

        # ── T5 encoding ───────────────────────────────────────────────────────
        t0 = time.time()
        caption_embs, emb_masks = self.t5.get_text_embeddings(
            [findings_text], token_nums=self.token_num,
        )
        # caption_embs: (1, 120, 4096) BF16  →  cast to FP32 for the FP32 DiT MLP
        caption_embs = caption_embs.float()
        diag["t5"]["wallclock_s"] = round(time.time() - t0, 3)
        diag["t5"]["embed_shape"] = list(caption_embs.shape)
        diag["t5"]["mask_shape"]  = list(emb_masks.shape)

        # CFG batch construction (mirror sample.py exactly)
        # caption_embs[:, None] : (1, 1, 120, 4096)
        # null_y                 : (1, 1, 120, 4096) — pre-learned null embedding
        # y_cfg                  : (2, 1, 120, 4096) — [cond ; null]
        caption_embs = caption_embs[:, None]
        null_y = self.dit.y_embedder.y_embedding[None].repeat(1, 1, 1)[:, None]
        y_cfg = torch.cat([caption_embs, null_y], dim=0)

        # Initial noise (same noise in both halves; forward_with_cfg re-duplicates anyway)
        gen = torch.Generator(device=self.device).manual_seed(seed)
        z = torch.randn(
            1, 4, self.latent_size, self.latent_size,
            device=self.device, generator=gen,
        ).repeat(2, 1, 1, 1)

        diag["chexgen"]["latent_shape"] = list(z.shape)

        # ── Diffusion sampling ────────────────────────────────────────────────
        model_kwargs = dict(y=y_cfg, cfg_scale=cfg_scale, mask=emb_masks)
        t0 = time.time()
        samples = self.diffusion.p_sample_loop(
            self.dit.forward_with_cfg,
            z.shape, z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=False,
            device=self.device,
        )
        torch.cuda.synchronize()
        diag["chexgen"]["sampling_wallclock_s"] = round(time.time() - t0, 3)

        # Take conditional half, decode through VAE
        samples = samples.chunk(2, dim=0)[0]                 # (1, 4, 64, 64)
        t0 = time.time()
        decoded = self.vae.decode(samples / 0.18215).sample  # (1, 3, 512, 512), [-1,1]
        torch.cuda.synchronize()
        diag["chexgen"]["vae_decode_wallclock_s"] = round(time.time() - t0, 3)
        diag["chexgen"]["decoded_shape"]          = list(decoded.shape)
        diag["chexgen"]["decoded_value_range"]    = [
            float(decoded.min().item()), float(decoded.max().item()),
        ]

        # PIL conversion (RGB; downstream may convert to grayscale for save)
        pil = latent_image_to_pil(decoded)

        # GPU memory snapshot
        if torch.cuda.is_available():
            diag["memory"] = {
                "peak_gb":   round(torch.cuda.max_memory_allocated() / 1e9, 2),
                "active_gb": round(torch.cuda.memory_allocated()    / 1e9, 2),
            }

        return pil, diag


def load_chexgen(device: torch.device, num_steps: int) -> ChexGenWrapper:
    """
    Load DiT + VAE + T5Embedder. Returns a ChexGenWrapper.

    We bypass tools/sample.py and DDP entirely. The radiffuser package is
    imported by adding ChexGen/ to sys.path; module side-effects register
    DiT_XL_2 with mmengine's MODELS Registry.
    """
    logger.info("Loading ChexGen (DiT + VAE + T5)...")
    if CHEXGEN_DIR not in sys.path:
        sys.path.insert(0, CHEXGEN_DIR)

    # Triggers @MODELS.register_module() decorators in dit.py
    from radiffuser.models import dit as _dit_mod  # noqa: F401
    from radiffuser.models.builder import build_model
    from radiffuser.models.t5     import T5Embedder
    from radiffuser.diffusion     import create_diffusion
    from radiffuser.utils         import find_model

    from mmengine import Config as MMConfig
    from diffusers.models import AutoencoderKL

    cfg = MMConfig.fromfile(CHEXGEN_CFG)
    model_cfg   = cfg.get("model")
    latent_size = int(model_cfg.get("input_size"))
    token_num   = int(model_cfg.get("token_num", 120))
    logger.info(f"  cfg.input_size={latent_size}  cfg.token_num={token_num}  "
                f"cfg.type={model_cfg.get('type')}")

    # ── DiT ───────────────────────────────────────────────────────────────────
    dit = build_model(model_cfg).to(device)
    sd  = find_model(CHEXGEN_CKPT, revised_keys={"module.": ""})
    missing, unexpected = dit.load_state_dict(sd, strict=False)
    if missing or unexpected:
        # Most likely culprits: 'pos_embed' or 'y_embedder.y_embedding' are
        # parameters that may or may not be in the released ckpt depending on
        # the training pipeline. They're either deterministic (sin-cos) or
        # frozen at init, so missing is non-fatal IF the checkpoint matches
        # the model architecture otherwise.
        logger.warning(f"  DiT load_state_dict missing keys ({len(missing)}): "
                       f"{missing[:5]}{'...' if len(missing) > 5 else ''}")
        logger.warning(f"  DiT load_state_dict unexpected keys ({len(unexpected)}): "
                       f"{unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    dit.float().eval()
    n_params = sum(p.numel() for p in dit.parameters())
    logger.info(f"  ✓ DiT loaded: {n_params/1e6:.0f}M params, FP32, "
                f"GPU mem ~{n_params*4/1e9:.1f} GB")

    # ── VAE ───────────────────────────────────────────────────────────────────
    vae_id = "stabilityai/sd-vae-ft-ema"
    vae = AutoencoderKL.from_pretrained(vae_id).to(device).eval()
    logger.info(f"  ✓ VAE loaded: {vae_id}")

    # ── T5 (BF16, GPU-resident) ───────────────────────────────────────────────
    # T5Embedder hardcodes its default cache to ~/.cache/IF_, which on O2 lives
    # in HOME (tight quota). We explicitly point it at scratch. The path
    # contains the model dir directly, NOT a 't5-v1_1-xxl' subdir — T5Embedder
    # appends that itself when local_cache=True.
    t5_cache_dir = "/n/scratch/users/g/gul075/.cache/IF_"
    t5 = T5Embedder(device=device, cache_dir=t5_cache_dir)
    logger.info(f"  ✓ T5Embedder loaded: dir_or_name={t5.dir_or_name}  "
                f"dtype={t5.torch_dtype}  device={t5.device}  "
                f"cache_dir={t5_cache_dir}")

    # ── Diffusion (respaced) ──────────────────────────────────────────────────
    diffusion = create_diffusion(str(num_steps))
    logger.info(f"  ✓ Diffusion: num_steps={num_steps}  "
                f"actual_timesteps={diffusion.num_timesteps}")

    return ChexGenWrapper(dit, vae, t5, diffusion, latent_size, token_num, device)


# ══════════════════════════════════════════════════════════════════════════════
#  SELF-TEST
# ══════════════════════════════════════════════════════════════════════════════

def run_self_test(chexgen, medclip, tokenizer, device):
    """
    Generate a known prompt end-to-end and assert all shapes and norms.
    Fails fast if any component is mis-wired.
    """
    logger.info("\n" + "─" * 60)
    logger.info("SELF-TEST: end-to-end smoke test of ChexGen → MedCLIP")
    logger.info("─" * 60)
    test_prompt = ("Lungs are clear. Heart is normal size probably exaggerated "
                   "by mediastinal fat. No pleural abnormality.")
    t0 = time.time()
    pil, diag = chexgen.generate(
        test_prompt, seed=42, num_steps=chexgen.diffusion.num_timesteps,
        cfg_scale=4.0,
    )
    logger.info(f"  ChexGen wallclock={time.time()-t0:.1f}s  "
                f"size={pil.size}  mode={pil.mode}")
    assert pil.size == (512, 512) and pil.mode == "RGB"

    img_e  = embed_image(medclip, pil, device)
    text_e = embed_text(medclip, tokenizer, test_prompt, device)
    assert img_e.shape  == (256,), f"img_embed shape: {img_e.shape}"
    assert text_e.shape == (256,), f"text_embed shape: {text_e.shape}"
    assert abs(img_e.norm().item()  - 1.0) < 0.01
    assert abs(text_e.norm().item() - 1.0) < 0.01

    sim = cosine_sim(img_e, text_e)
    logger.info(f"  ✓ Image embed: shape={tuple(img_e.shape)}  norm={img_e.norm():.4f}")
    logger.info(f"  ✓ Text embed:  shape={tuple(text_e.shape)}  norm={text_e.norm():.4f}")
    logger.info(f"  ✓ img↔text cosine (sanity, expect > 0): {sim:+.4f}")
    logger.info(f"  ✓ Diag: t5_natural={diag['t5']['natural_token_count']}  "
                f"used={diag['t5']['used_token_count']}  "
                f"trunc={diag['t5']['was_truncated']}  "
                f"sample={diag['chexgen']['sampling_wallclock_s']}s  "
                f"vae={diag['chexgen']['vae_decode_wallclock_s']}s  "
                f"range=[{diag['chexgen']['decoded_value_range'][0]:.2f}, "
                f"{diag['chexgen']['decoded_value_range'][1]:.2f}]")
    logger.info("SELF-TEST PASSED")
    logger.info("─" * 60 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
#  ATTRACTOR LOOP (single study)
# ══════════════════════════════════════════════════════════════════════════════

def run_loop(
    *,
    gt_image_path: str,
    study_id: str,
    gt_findings: str,
    medclip,
    tokenizer,
    maira,
    chexgen: ChexGenWrapper,
    device: torch.device,
    n_iters: int,
    output_dir: str,
    num_steps: int,
    cfg_scale: float,
    seed: int,
) -> dict:
    study_dir = os.path.join(output_dir, study_id)
    os.makedirs(study_dir, exist_ok=True)

    # ── GT image + findings ───────────────────────────────────────────────────
    gt_pil = Image.open(gt_image_path).convert("RGB")
    gt_pil_512 = gt_pil.resize((512, 512), Image.LANCZOS)
    gt_pil_512.convert("L").save(os.path.join(study_dir, "gt_image.png"))
    with open(os.path.join(study_dir, "gt_findings.txt"), "w") as f:
        f.write(gt_findings)

    # Anchors
    anchor_img  = embed_image(medclip, gt_pil, device)
    anchor_text = embed_text(medclip, tokenizer, gt_findings, device)
    np.save(os.path.join(study_dir, "anchor_img_embed.npy"),  anchor_img.numpy())
    np.save(os.path.join(study_dir, "anchor_text_embed.npy"), anchor_text.numpy())
    logger.info(f"  [{study_id}] anchors: "
                f"img_norm={anchor_img.norm():.4f}  txt_norm={anchor_text.norm():.4f}  "
                f"gt_pil_orig={gt_pil.size}")

    metrics = {
        "study_id":    study_id,
        "gt_findings": gt_findings,
        "gt_image_path": gt_image_path,
        "config": {
            "n_iters":   n_iters,
            "num_steps": num_steps,
            "cfg_scale": cfg_scale,
            "base_seed": seed,
        },
        "anchor_norms": {
            "img":  float(anchor_img.norm().item()),
            "text": float(anchor_text.norm().item()),
        },
        "iterations": [],   # detailed diagnostics
        # Compact lists for downstream analysis (mirror FLUX.2 metrics format)
        "findings":        [],
        "image_cosine":    [],
        "text_cosine":     [],
        "embed_l2":        [],
        "image_evolution": [],
        "text_evolution":  [],
    }

    current_img    = gt_pil           # iteration 0 input is GT
    prev_img_emb   = None
    prev_text_emb  = None

    for k in range(n_iters + 1):
        iter_seed = seed + k
        iter_diag = {"k": k, "iter_seed": iter_seed,
                     "input_image_source": "ground_truth" if k == 0 else f"gen_iter_{k-1:03d}",
                     "input_image_size":   list(current_img.size)}

        # ── MAIRA-2: image → findings ─────────────────────────────────────────
        t0 = time.time()
        findings_k = run_maira(maira, current_img)
        maira_dt = time.time() - t0
        iter_diag["maira"] = {
            "wallclock_s":  round(maira_dt, 2),
            "input_size":   [518, 518],
            "output_chars": len(findings_k),
            "output_words": len(findings_k.split()),
        }
        logger.info(f"  [{study_id}] iter {k}: MAIRA-2 ({maira_dt:.1f}s)  "
                    f"findings_chars={len(findings_k)}  "
                    f"head=\"{findings_k[:120]}{'...' if len(findings_k) > 120 else ''}\"")

        # ── ChexGen: findings → image ─────────────────────────────────────────
        try:
            gen_pil, gen_diag = chexgen.generate(
                findings_k, seed=iter_seed,
                num_steps=num_steps, cfg_scale=cfg_scale,
            )
        except Exception as e:
            logger.error(f"  [{study_id}] iter {k}: ChexGen FAILED: {e}")
            iter_diag["chexgen_error"] = str(e)
            metrics["iterations"].append(iter_diag)
            raise

        iter_diag.update(gen_diag)  # merge t5/chexgen/memory diagnostics

        # Persist artifacts
        gen_pil.convert("L").save(
            os.path.join(study_dir, f"gen_iter_{k:03d}.png"))
        with open(os.path.join(study_dir, f"findings_iter_{k:03d}.txt"), "w") as f:
            f.write(findings_k)

        # ── MedCLIP embeddings ────────────────────────────────────────────────
        img_ek  = embed_image(medclip, gen_pil, device)
        text_ek = embed_text(medclip, tokenizer, findings_k, device)
        np.save(os.path.join(study_dir, f"img_embed_iter_{k:03d}.npy"),  img_ek.numpy())
        np.save(os.path.join(study_dir, f"text_embed_iter_{k:03d}.npy"), text_ek.numpy())

        iter_diag["medclip"] = {
            "img_embed_norm":  float(img_ek.norm().item()),
            "text_embed_norm": float(text_ek.norm().item()),
            "embed_dim":       int(img_ek.shape[0]),
        }

        # ── Metrics ───────────────────────────────────────────────────────────
        img_cos  = cosine_sim(anchor_img,  img_ek)
        txt_cos  = cosine_sim(anchor_text, text_ek)
        l2_to_gt = float((anchor_img - img_ek).norm().item())
        img_evo  = cosine_sim(prev_img_emb,  img_ek)  if prev_img_emb  is not None else None
        txt_evo  = cosine_sim(prev_text_emb, text_ek) if prev_text_emb is not None else None

        iter_diag["metrics"] = {
            "image_cosine":    img_cos,
            "text_cosine":     txt_cos,
            "embed_l2":        l2_to_gt,
            "image_evolution": img_evo,
            "text_evolution":  txt_evo,
        }

        metrics["iterations"].append(iter_diag)
        metrics["findings"].append(findings_k)
        metrics["image_cosine"].append(img_cos)
        metrics["text_cosine"].append(txt_cos)
        metrics["embed_l2"].append(l2_to_gt)
        if img_evo is not None: metrics["image_evolution"].append(img_evo)
        if txt_evo is not None: metrics["text_evolution"].append(txt_evo)

        logger.info(
            f"  [{study_id}] iter {k}: ChexGen sample={gen_diag['chexgen']['sampling_wallclock_s']}s  "
            f"vae={gen_diag['chexgen']['vae_decode_wallclock_s']}s  "
            f"t5_tok={gen_diag['t5']['used_token_count']}/{gen_diag['t5']['natural_token_count']}"
            f"{' (TRUNC)' if gen_diag['t5']['was_truncated'] else ''}  "
            f"img_cos={img_cos:+.4f}  txt_cos={txt_cos:+.4f}  l2={l2_to_gt:.4f}  "
            f"img_evo={img_evo if img_evo is None else f'{img_evo:+.4f}'}  "
            f"txt_evo={txt_evo if txt_evo is None else f'{txt_evo:+.4f}'}"
        )

        # Roll forward
        current_img   = gen_pil
        prev_img_emb  = img_ek
        prev_text_emb = text_ek

    with open(os.path.join(study_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_samples",   type=int,   default=5,
                   help="Number of test studies to sample. Pilot default: 5. "
                        "Ignored if --use_all is set.")
    p.add_argument("--use_all",     action="store_true",
                   help="Iterate over ALL test studies (with FINDINGS) instead "
                        "of sampling n_samples. Use with --chunk_idx / --n_chunks "
                        "for SLURM job arrays.")
    p.add_argument("--chunk_idx",   type=int,   default=0,
                   help="Which chunk to run (0-indexed). Used only with --use_all.")
    p.add_argument("--n_chunks",    type=int,   default=1,
                   help="Total number of chunks the full set is split into.")
    p.add_argument("--n_iters",     type=int,   default=5,
                   help="Number of attractor iterations after iter 0. Total iters = n_iters+1.")
    p.add_argument("--num_steps",   type=int,   default=100,
                   help="ChexGen sampling steps. Default: 100 (their published).")
    p.add_argument("--cfg_scale",   type=float, default=4.0)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--data_csv",    type=str,   default=DATA_CSV)
    p.add_argument("--output_dir",  type=str,
                   default=f"{BASE_DIR}/Experiments/attractor_loop/results/chexgen_pilot")
    p.add_argument("--skip_self_test", action="store_true",
                   help="Skip the startup smoke test (saves ~1.5 min).")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 60)
    logger.info("AIM2 ChexGen Attractor Loop")
    logger.info("=" * 60)
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")
    if torch.cuda.is_available():
        gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)} ({gb:.0f} GB)")
        torch.cuda.reset_peak_memory_stats()

    # ── Sample studies ────────────────────────────────────────────────────────
    df = pd.read_csv(args.data_csv, low_memory=False)
    df = df[df["split"] == "test"]
    # Mirror the safer string-cast filter from preflight.
    if "has_findings" in df.columns:
        mask = df["has_findings"].astype(str).str.lower().isin(["true", "1", "1.0"])
        df = df[mask]
    df = df.groupby("study_id", as_index=False).first()

    if args.use_all:
        # Deterministic sort so chunk i always covers the same studies across
        # job-array runs, regardless of sampling-order randomness.
        df_sorted = df.sort_values("study_id").reset_index(drop=True)

        n_total = len(df_sorted)
        chunk_size = (n_total + args.n_chunks - 1) // args.n_chunks  # ceil div
        chunk_start = args.chunk_idx * chunk_size
        chunk_end   = min(chunk_start + chunk_size, n_total)
        sample = df_sorted.iloc[chunk_start:chunk_end].reset_index(drop=True)

        logger.info(f"\nUsing ALL test studies, chunk {args.chunk_idx+1}/{args.n_chunks}")
        logger.info(f"  Total studies (test + has_findings): {n_total}")
        logger.info(f"  Chunk range: [{chunk_start}, {chunk_end}) → {len(sample)} studies")
    else:
        sample = df.sample(
            n=min(args.n_samples, len(df)),
            random_state=args.seed,
        ).reset_index(drop=True)
        logger.info(f"\nSampled {len(sample)} studies from test split "
                    f"(out of {len(df)} unique test studies)")

    # ── Resume support ────────────────────────────────────────────────────────
    completed = {
        str(sid) for sid in sample["study_id"].astype(str)
        if os.path.exists(os.path.join(args.output_dir, str(sid), "metrics.json"))
    }
    if completed:
        logger.info(f"Resuming: {len(completed)} done, "
                    f"{len(sample)-len(completed)} remaining")

    # ── Load all models ───────────────────────────────────────────────────────
    logger.info("\nLoading models...")
    medclip, _, tokenizer = load_medclip(device)
    maira                 = load_maira()
    chexgen               = load_chexgen(device, num_steps=args.num_steps)

    # ── Self-test ─────────────────────────────────────────────────────────────
    if not args.skip_self_test:
        run_self_test(chexgen, medclip, tokenizer, device)

    # ── Run loop over studies ─────────────────────────────────────────────────
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
        logger.info(f"GT findings ({len(gt_findings)} chars): "
                    f"\"{gt_findings[:120]}{'...' if len(gt_findings) > 120 else ''}\"")

        try:
            # Per-study deterministic seed: SHA-256 of study_id + base seed.
            # Python's built-in hash() is randomized per-process by default
            # (PYTHONHASHSEED), so we use hashlib for cross-run reproducibility.
            # This is invariant to chunk ordering: a study run in chunk 3 of
            # an 8-chunk job gets the same seed as if run alone.
            # Note: existing 100 main-run studies used `seed + i * 1000` where
            # i was their position in the n=100 sample. Their existing data
            # has those seeds locked in metrics.json — we don't re-run them.
            # Any new studies will use this hash-based seed instead.
            import hashlib as _hashlib
            study_seed = (
                args.seed +
                int(_hashlib.sha256(sid.encode()).hexdigest()[:8], 16)
            ) & 0x7FFFFFFF

            m = run_loop(
                gt_image_path = img_path,
                study_id      = sid,
                gt_findings   = gt_findings,
                medclip       = medclip,
                tokenizer     = tokenizer,
                maira         = maira,
                chexgen       = chexgen,
                device        = device,
                n_iters       = args.n_iters,
                output_dir    = args.output_dir,
                num_steps     = args.num_steps,
                cfg_scale     = args.cfg_scale,
                seed          = study_seed,
            )
            all_metrics.append(m)
        except Exception as e:
            logger.error(f"Study {sid} FAILED: {e}")
            traceback.print_exc()
            failed.append(sid)
            gc.collect()
            torch.cuda.empty_cache()

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"COMPLETE: {len(all_metrics)}/{len(sample)} studies OK")
    if failed:
        logger.error(f"Failed ({len(failed)}): {failed}")

    if all_metrics:
        summary = {}
        n_total_iters = args.n_iters + 1
        for k in range(n_total_iters):
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
        # Truncation rate across all iterations
        n_calls = sum(len(m["iterations"]) for m in all_metrics)
        n_trunc = sum(
            1 for m in all_metrics for it in m["iterations"]
            if it.get("t5", {}).get("was_truncated", False)
        )
        summary["truncation_rate"] = round(n_trunc / max(n_calls, 1), 4)
        summary["truncation_count"] = f"{n_trunc}/{n_calls}"

        logger.info(f"\n{'Iter':>5}  {'img_cos':>8}±{'std':>5}  "
                    f"{'txt_cos':>8}±{'std':>5}  {'l2':>8}±{'std':>5}")
        logger.info("─" * 60)
        for k in range(n_total_iters):
            s = summary.get(f"iter_{k}", {})
            if s:
                logger.info(
                    f"{k:>5}  "
                    f"{s['img_cosine_mean']:>+8.4f}±{s['img_cosine_std']:>5.3f}  "
                    f"{s['txt_cosine_mean']:>+8.4f}±{s['txt_cosine_std']:>5.3f}  "
                    f"{s['embed_l2_mean']:>8.4f}±{s['embed_l2_std']:>5.3f}"
                )
        logger.info(f"\nT5 truncation rate: {summary['truncation_count']} "
                    f"({summary['truncation_rate']*100:.1f}%)")

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