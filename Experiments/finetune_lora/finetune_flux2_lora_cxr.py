#!/usr/bin/env python3
"""
LoRA Fine-tuning of FLUX.2-dev on MIMIC-CXR
============================================

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  FLUX.2 vs FLUX.1 — WHAT CHANGED (READ THIS FIRST)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Component           │ FLUX.1-dev (12B)           │ FLUX.2-dev (32B)
  ────────────────────┼────────────────────────────┼──────────────────────────────
  Transformer         │ FluxTransformer2DModel      │ Flux2Transformer2DModel
  Double-stream blks  │ 19                          │ 8  (less text-image mixing)
  Single-stream blks  │ 38                          │ 48 (more joint processing)
  Text encoder        │ CLIP-L (768) + T5-XXL(4096) │ Mistral Small 3.1 ONLY (24B)
  Pooled text dim     │ 768 (CLIP-L)                │ None (Mistral outputs seq only)
  Sequence text dim   │ 4096 (T5-XXL)               │ Mistral hidden dim
  VAE                 │ AutoencoderKL (16 channels)  │ AutoencoderKLFlux2 (32 channels)
  Latent token dim    │ 64 (16ch × 2×2 patch)        │ 128 (32ch × 2×2 patch)
  Guidance (train)    │ 3.5 passed as float          │ 1.0  ← critical change
  Guidance (infer)    │ 3.5                          │ 2–4 (3.5–4 recommended)
  Pipeline            │ FluxPipeline                 │ Flux2Pipeline
  VRAM (bf16, full)   │ ~34 GB steady-state          │ >80 GB without tricks
  Min. practical GPU  │ A100 40GB                    │ H100 80GB (or heavy quant)

  KEY INSIGHT — 32-CHANNEL VAE:
  The new AutoencoderKLFlux2 doubles latent channels from 16 → 32.
  This means the packing step now produces 128-dim tokens (not 64):
    (B, 32, H/8, W/8)  →  pack  →  (B, (H/16)*(W/16), 128)
  At resolution=1024: (B, 32, 128, 128) → (B, 4096, 128)

  KEY INSIGHT — SINGLE TEXT ENCODER:
  FLUX.2 drops both CLIP-L and T5-XXL and uses a single Mistral Small 3.1
  (~24B params, ~24 GB bf16). The pipeline's encode_prompt() handles this
  internally. For training, we PRE-CACHE all text embeddings before the
  main loop — loading a 24B model each step would be catastrophic.

  KEY INSIGHT — guidance_scale=1 AT TRAINING:
  Unlike FLUX.1-dev (guidance_scale=3.5 at training), FLUX.2 should be
  trained with guidance_scale=1. Using 3.5 during training when the model
  was distilled differently causes distribution mismatch and poor convergence.

  MEMORY STRATEGY FOR H100 80GB:
  ┌──────────────────────────────────────┬──────────┐
  │ Flux2Transformer2DModel (bf16)        │ ~64 GB   │
  │ AutoencoderKLFlux2 (bf16)            │ ~  3 GB  │
  │ Mistral Small 3.1 (bf16)             │ ~24 GB   │
  │                                      │ >80 GB!  │
  └──────────────────────────────────────┴──────────┘
  → Cache image latents to disk BEFORE training, then free VAE
  → Cache text embeddings to disk BEFORE training, then free Mistral
  → Training loop only holds transformer + LoRA adapters in GPU memory
  → This brings steady-state to ~64-68 GB — fits on H100 80GB ✓

  MEMORY STRATEGY FOR A100 40GB (your Harvard cluster):
  → Use 4-bit quantization (bitsandbytes NF4) for the transformer
  → Still cache latents + embeddings
  → rank=16 instead of 32 to save activation memory
  → Expect ~38-40 GB with QLoRA; very tight

  INSTALL REQUIREMENTS (diffusers main branch required):
  pip uninstall diffusers -y
  pip install git+https://github.com/huggingface/diffusers -U
  pip install "transformers>=4.50.0" "peft>=0.13.0" "accelerate>=0.34.0"
  pip install bitsandbytes safetensors pandas Pillow tqdm

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Usage (H100 80GB, recommended):
    accelerate launch --num_processes 1 --mixed_precision bf16 \\
        finetune_flux2_lora_cxr.py \\
        --data_csv /path/to/processed_data.csv \\
        --image_root /path/to/cxr_png \\
        --cache_dir /path/to/cache \\
        --model_id black-forest-labs/FLUX.2-dev \\
        --lora_rank 32 --resolution 1024 \\
        --train_batch_size 1 --gradient_accumulation_steps 16 \\
        --max_train_steps 15000

Usage (A100 40GB, quantized):
    Same + add: --use_4bit_quantization --lora_rank 16

Two-phase design:
  Phase 1 (cache_only):  Encode all images + prompts, save to disk.
                          VAE + Mistral are freed after this phase.
  Phase 2 (train):        LoRA training loop using cached tensors only.
                          Only the transformer is in GPU memory.
"""

import argparse
import copy
import json
import logging
import math
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    Flux2Pipeline,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,   # same formula — model-agnostic
)

from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict

# Try importing FLUX.2-specific classes (requires diffusers from main)
try:
    from diffusers import Flux2Transformer2DModel, AutoencoderKLFlux2
    HAVE_FLUX2_CLASSES = True
except ImportError:
    HAVE_FLUX2_CLASSES = False
    raise ImportError(
        "FLUX.2-dev requires diffusers from the main branch.\n"
        "Run: pip uninstall diffusers -y && "
        "pip install git+https://github.com/huggingface/diffusers -U"
    )

logger = get_logger(__name__, log_level="INFO")

# FLUX.2 VAE has 32 latent channels (vs 16 in FLUX.1/SD3.5)
FLUX2_LATENT_CHANNELS = 32
# Packing: 2×2 latent pixels → 1 token; token dim = 32*4 = 128
FLUX2_PACKED_TOKEN_DIM = 128  # 32 channels * 4 pixels (2x2 group)


# ══════════════════════════════════════════════════════════════════════
#  TEXT / IMAGE CACHE  (Phase 1 — run once before training)
# ══════════════════════════════════════════════════════════════════════

def build_conditioning_text(
    findings: str,
    indication: str = "",
    use_indication: bool = False,
    max_chars: int = 3000,  # Mistral 3.1 handles 32K tokens — much more room
) -> str:
    """
    Build the text prompt for FLUX.2-dev.

    FINDINGS ONLY — deliberately excludes IMPRESSION.

    Rationale: MAIRA-2 (our report generator in the attractor loop) generates
    FINDINGS only. MedCLIP (our embedding model) is trained on FINDINGS only.
    Training FLUX.2 on FINDINGS+IMPRESSION would create a train/inference
    distribution mismatch: at inference time the prompt is FINDINGS only, but
    the model was conditioned on longer FINDINGS+IMPRESSION strings.

    Mistral Small 3.1 handles ~32K tokens; MIMIC FINDINGS are ~65-200 tokens.
    max_chars=3000 is a backstop that will never be hit in practice.
    """
    parts = []
    if use_indication and indication and indication.strip():
        ind = indication.strip().replace("___", "patient")
        parts.append(f"INDICATION: {ind}")
    if findings and findings.strip():
        parts.append(f"FINDINGS: {findings.strip()}")
    if not parts:
        return "No findings reported."
    text = " ".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars]
    return text



# CheXpert label columns present in the processed_data.csv
CHEXPERT_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
    "Lung Opacity", "No Finding", "Pleural Effusion", "Pleural Other",
    "Pneumonia", "Pneumothorax", "Support Devices",
]


def _stratified_sample(
    df: pd.DataFrame,
    n: int,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Stratified sample from a multi-label CheXpert dataframe.

    Goal: equal coverage of all 14 CheXpert labels with no label prioritised.

    Algorithm:
      1. Compute per-label quota = n // n_labels (857 for n=12000, 14 labels).
      2. For each label, collect up to `quota` positive (value==1.0) rows,
         sampling randomly within the label's positive pool.
      3. Combine all collected rows; deduplicate by study_id.
         (Multi-label samples count once — they fulfil multiple quotas.)
      4. If |result| < n: fill remaining slots from rows not yet selected,
         chosen uniformly at random.
      5. If |result| > n: sample down to exactly n.

    No external libraries needed. Handles class imbalance naturally:
    rare labels (Fracture, Lung Lesion) may contribute fewer than quota
    samples — that's correct, their full positive pool is included.

    Logs coverage stats so you can verify the distribution.
    """
    rng = np.random.default_rng(seed)
    available_labels = [l for l in CHEXPERT_LABELS if l in df.columns]
    n_labels  = len(available_labels)
    quota     = n // n_labels

    collected_idx = set()

    for label in available_labels:
        positives = df.index[df[label] == 1.0].tolist()
        if not positives:
            continue
        k = min(len(positives), quota)
        chosen = rng.choice(positives, size=k, replace=False).tolist()
        collected_idx.update(chosen)

    # Fill remaining slots if under target
    if len(collected_idx) < n:
        remaining = [i for i in df.index if i not in collected_idx]
        fill_k    = min(n - len(collected_idx), len(remaining))
        fill_idx  = rng.choice(remaining, size=fill_k, replace=False).tolist()
        collected_idx.update(fill_idx)

    result = df.loc[sorted(collected_idx)].copy()

    # Sample down if over target (can happen when multi-label samples satisfy >1 quota)
    if len(result) > n:
        result = result.sample(n=n, random_state=seed).reset_index(drop=True)

    # ── Coverage report ──────────────────────────────────────────────
    logger.info(f"Stratified sample: {len(result)} / {len(df)} rows selected")
    for label in available_labels:
        if label not in result.columns:
            continue
        pos  = (result[label] == 1.0).sum()
        total_pos = (df[label] == 1.0).sum()
        logger.info(f"  {label:30s}: {pos:4d} positive "
                    f"({pos/len(result)*100:.1f}% of sample | "
                    f"{pos}/{total_pos} available positives used)")

    return result.reset_index(drop=True)


class MIMICCXRRawDataset(Dataset):
    """
    Lightweight dataset used ONLY for Phase 1 cache building.
    Returns raw PIL images and text strings — no tensor transforms.
    """

    def __init__(
        self,
        csv_path: str,
        image_root: Optional[str] = None,
        resolution: int = 1024,
        split: str = "train",
        use_indication: bool = False,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ):
        self.resolution = resolution
        self.use_indication = use_indication

        df = pd.read_csv(csv_path, low_memory=False)
        df["split"] = df["split"].replace({"validate": "val"})
        df = df[df["split"] == split].copy()

        # Filter to rows with FINDINGS only — mirrors MedCLIP dataset filter.
        # MAIRA-2 generates FINDINGS only; using IMPRESSION rows here would
        # train FLUX.2 on prompts that are never seen at inference time.
        has_findings = (
            df["has_findings"].astype(str).str.lower().isin(["true", "1", "1.0"])
        )
        df = df[has_findings].copy()

        for col in ["findings", "impression", "indication"]:
            df[col] = df.get(col, pd.Series([""] * len(df))).fillna("").astype(str)

        if image_root:
            df["resolved_path"] = df["image_path"].apply(
                lambda p: p if os.path.isabs(str(p)) else os.path.join(image_root, str(p))
            )
        else:
            df["resolved_path"] = df["image_path"].astype(str)

        if max_samples is not None and max_samples < len(df):
            df = _stratified_sample(df, max_samples, seed)

        self.df = df.reset_index(drop=True)
        logger.info(f"MIMICCXRRawDataset: {len(self.df)} samples | resolution={resolution}")

        # Image transform — same LANCZOS strategy as before
        self._img_transform = transforms.Compose([
            transforms.Resize(
                (resolution, resolution),
                interpolation=transforms.InterpolationMode.LANCZOS,
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = row["resolved_path"]
        study_id = str(row.get("study_id", idx))

        try:
            img = Image.open(img_path)
            if img.mode != "RGB":
                img = img.convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to load {img_path}: {e}. Using blank.")
            img = Image.new("RGB", (self.resolution, self.resolution), 128)

        pixel_values = self._img_transform(img)

        prompt = build_conditioning_text(
            findings=row["findings"],
            indication=row.get("indication", ""),
            use_indication=self.use_indication,
        )

        return {
            "pixel_values": pixel_values,  # (3, H, W), [-1, 1]
            "prompt": prompt,
            "study_id": study_id,
        }


@torch.no_grad()
def build_cache(
    raw_dataset: MIMICCXRRawDataset,
    pipe: Flux2Pipeline,
    cache_dir: str,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int = 4,
    num_workers: int = 4,
    max_sequence_length: int = 512,
    force_rebuild: bool = False,
):
    """
    Phase 1: Pre-encode all images (VAE) and prompts (Mistral) to disk.

    This is THE critical memory optimization for FLUX.2.  Mistral Small 3.1
    is ~24 GB and the FLUX.2 VAE needs GPU too. By caching before training:
    - Neither the VAE nor Mistral lives in GPU memory during the training loop
    - Training loop only needs: transformer (64 GB) + LoRA adapters + activations
    - This is feasible on a single H100 80 GB

    Cache structure on disk:
        cache_dir/
            latents/
                {study_id}.pt      ← (32, H/8, W/8) spatial, VAE-encoded
            embeddings/
                {study_id}.pt      ← dict of prompt_embeds + pooled_prompt_embeds

    Why not cache packed latents?
        Latents are packed during the training loop (after noise sampling).
        Caching spatial latents gives more flexibility (e.g., for crop augmentation
        in future experiments).
    """
    latent_dir = os.path.join(cache_dir, "latents")
    embed_dir  = os.path.join(cache_dir, "embeddings")
    os.makedirs(latent_dir, exist_ok=True)
    os.makedirs(embed_dir,  exist_ok=True)

    # Quick check: if all files exist and force_rebuild=False, skip
    if not force_rebuild:
        existing_latents = set(f.stem for f in Path(latent_dir).glob("*.pt"))
        existing_embeds  = set(f.stem for f in Path(embed_dir).glob("*.pt"))
        if len(existing_latents) >= len(raw_dataset) and len(existing_embeds) >= len(raw_dataset):
            logger.info(f"Cache already complete ({len(existing_latents)} items). Skipping rebuild.")
            return

    loader = DataLoader(
        raw_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    vae = pipe.vae
    vae.to(device)
    vae.eval()

    logger.info("Phase 1a — Encoding image latents with AutoencoderKLFlux2...")
    for batch in tqdm(loader, desc="Caching latents"):
        pixel_values = batch["pixel_values"].to(device=device, dtype=dtype)
        study_ids = batch["study_id"]

        latents = vae.encode(pixel_values).latent_dist.sample()
        # FLUX.2 VAE normalisation: shift then scale
        latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
        # Save as bf16: halves disk usage (36GB→18GB for 70K samples at 512px)
        latents = latents.bfloat16().cpu()

        for i, sid in enumerate(study_ids):
            out_path = os.path.join(latent_dir, f"{sid}.pt")
            if not os.path.exists(out_path) or force_rebuild:
                torch.save(latents[i], out_path)

    vae.cpu()
    torch.cuda.empty_cache()
    logger.info("Phase 1a — Latent caching complete. VAE offloaded to CPU.")

    logger.info("Phase 1b — Encoding text with Mistral Small 3.1...")
    # Use Flux2Pipeline's encode_prompt — this handles Mistral tokenization internally
    pipe.text_encoder.to(device)  # Mistral to GPU for encoding
    pipe.tokenizer.to(device) if hasattr(pipe.tokenizer, "to") else None

    for batch in tqdm(loader, desc="Caching embeddings"):
        prompts = batch["prompt"]
        study_ids = batch["study_id"]

        # Use the pipeline's built-in encode_prompt for Mistral
        # FLUX.2 encode_prompt returns (prompt_embeds,) — no pooled vector
        prompt_embeds = pipe.encode_prompt(
            prompt=list(prompts),
            device=device,
            max_sequence_length=max_sequence_length,
        )
        # prompt_embeds may be a tuple or tensor depending on diffusers version
        if isinstance(prompt_embeds, tuple):
            prompt_embeds = prompt_embeds[0]
        # Save as bf16: halves disk usage vs fp32.
        # At seq=256, hidden=5120: 184GB bf16 vs 367GB fp32 for 70K samples.
        prompt_embeds = prompt_embeds.bfloat16().cpu()

        for i, sid in enumerate(study_ids):
            out_path = os.path.join(embed_dir, f"{sid}.pt")
            if not os.path.exists(out_path) or force_rebuild:
                torch.save(prompt_embeds[i], out_path)

    pipe.text_encoder.cpu()
    torch.cuda.empty_cache()
    logger.info("Phase 1b — Embedding caching complete. Mistral offloaded to CPU.")


# ══════════════════════════════════════════════════════════════════════
#  CACHED DATASET  (Phase 2 — used during training loop)
# ══════════════════════════════════════════════════════════════════════

class CachedMIMICDataset(Dataset):
    """
    Training dataset that loads from disk cache rather than re-encoding.

    Each __getitem__ call loads:
      - latents: (32, H/8, W/8) — spatial, VAE-encoded, [-1,1]-normalised
      - prompt_embeds: (seq_len, hidden_dim) — Mistral text embeddings

    Everything else (noise, packing, timestep sampling) happens in the
    training loop, not here.
    """

    def __init__(
        self,
        cache_dir: str,
        csv_path: str,
        split: str = "train",
        max_samples: Optional[int] = None,
        seed: int = 42,
    ):
        latent_dir = os.path.join(cache_dir, "latents")
        embed_dir  = os.path.join(cache_dir, "embeddings")

        df = pd.read_csv(csv_path, low_memory=False)
        df["split"] = df["split"].replace({"validate": "val"})
        df = df[df["split"] == split].copy()

        # Keep only rows that have cached files
        df["study_id"] = df["study_id"].astype(str)
        df = df[
            df["study_id"].apply(
                lambda sid: (
                    os.path.exists(os.path.join(latent_dir, f"{sid}.pt")) and
                    os.path.exists(os.path.join(embed_dir, f"{sid}.pt"))
                )
            )
        ].copy()

        if max_samples is not None and max_samples < len(df):
            df = _stratified_sample(df, max_samples, seed)

        self.df = df.reset_index(drop=True)
        self.latent_dir = latent_dir
        self.embed_dir  = embed_dir
        logger.info(f"CachedMIMICDataset: {len(self.df)} cached samples")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        sid = str(self.df.iloc[idx]["study_id"])
        latents       = torch.load(os.path.join(self.latent_dir, f"{sid}.pt"), weights_only=True)
        prompt_embeds = torch.load(os.path.join(self.embed_dir,  f"{sid}.pt"), weights_only=True)
        return {
            "latents": latents,                   # (32, H/8, W/8)
            "prompt_embeds": prompt_embeds,        # (seq_len, hidden_dim)
            "study_id": sid,
        }


def collate_cached(examples):
    return {
        "latents":       torch.stack([e["latents"]       for e in examples]),
        "prompt_embeds": torch.stack([e["prompt_embeds"] for e in examples]),
        "study_ids":     [e["study_id"] for e in examples],
    }


# ══════════════════════════════════════════════════════════════════════
#  LATENT PACKING (updated for 32-channel FLUX.2 VAE)
# ══════════════════════════════════════════════════════════════════════

def pack_latents(
    latents: torch.Tensor,
    batch_size: int,
    num_channels: int,   # 32 for FLUX.2 (vs 16 for FLUX.1)
    height: int,
    width: int,
) -> torch.Tensor:
    """
    Pack 2D latent feature maps into a 1D token sequence.

    FLUX.2 has the same packing logic as FLUX.1, but since the VAE now
    outputs 32 channels (not 16), the packed token dimension is 128 (not 64):
        Input:  (B, 32, H, W)         where H = resolution/8
        Group:  2×2 latent pixels     → 1 patch token
        Output: (B, H/2 * W/2, 128)   where 128 = 32 channels × 4 pixels

    Example at resolution=1024:
        latents: (B, 32, 128, 128)
        packed:  (B, 4096, 128)   ← 4096 tokens of dim 128

    The 128-dim tokens are then projected to the model's hidden dim inside
    Flux2Transformer2DModel (x_embedder: Linear(128, d_model)).
    """
    latents = latents.view(batch_size, num_channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels * 4)
    return latents


def unpack_latents(
    latents: torch.Tensor,
    height: int,
    width: int,
    vae_scale_factor: int = 8,
) -> torch.Tensor:
    """Reverse of pack_latents. Reconstructs (B, 32, H//8, W//8)."""
    batch_size = latents.shape[0]
    h = height // vae_scale_factor
    w = width  // vae_scale_factor
    latents = latents.view(batch_size, h // 2, w // 2, -1, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(batch_size, -1, h, w)
    return latents


def prepare_latent_image_ids(
    batch_size: int,
    packed_h: int,   # resolution // 16
    packed_w: int,   # resolution // 16
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    2D RoPE position IDs for image patch tokens.
    Identical structure to FLUX.1 — dim 0 = 0 (image), dims 1-2 = (row, col).
    Returns: (B, packed_h * packed_w, 3)
    """
    ids = torch.zeros(packed_h, packed_w, 4)  # 4 axes for FLUX.2
    ids[..., 2] = torch.arange(packed_h)[:, None]
    ids[..., 3] = torch.arange(packed_w)[None, :]
    ids = ids.reshape(packed_h * packed_w, 4).unsqueeze(0).expand(batch_size, -1, -1)
    return ids.to(device=device, dtype=dtype)


def get_sigmas(
    timesteps: torch.Tensor,
    scheduler,
    n_dim: int = 3,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Look up sigmas from the scheduler for given timesteps."""
    sigmas = scheduler.sigmas.to(device=timesteps.device, dtype=dtype)
    schedule_timesteps = scheduler.timesteps.to(timesteps.device)
    step_indices = [
        (schedule_timesteps == t).nonzero(as_tuple=True)[0][0].item()
        for t in timesteps
    ]
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


# ══════════════════════════════════════════════════════════════════════
#  LoRA CONFIGURATION FOR FLUX.2
# ══════════════════════════════════════════════════════════════════════

# FLUX.2 block distribution: 8 double-stream + 48 single-stream.
# The single-stream blocks now dominate (73% of params vs 46% in FLUX.1).
# LoRA targets are structurally identical to FLUX.1 — same module names —
# but the 48 single-stream blocks mean significantly more parameters.
#
# Recommend targeting ALL single-stream blocks since they dominate FLUX.2:
#   - "proj_mlp" covers the MLP+attention fusion in single-stream blocks
# Double-stream LoRA (ff_context.*) is optional and can be dropped for VRAM savings.

# ── CONFIRMED MODULE NAMES (from Flux2Transformer2DModel.named_modules()) ──────
# Verified 2026-04-17 against black-forest-labs/FLUX.2-dev architecture.
# Key differences from FLUX.1:
#   - Double-stream context attn out: "to_add_out" (not "add_out_proj")
#   - Double-stream FF: "ff.linear_in/out" (not "ff.net.0.proj/ff.net.2")
#   - Double-stream context FF: "ff_context.linear_in/out" (not "ff_context.net.*")
#   - Single-stream block name: "single_transformer_blocks" (not "transformer_blocks")
#   - Single-stream fused proj: "attn.to_qkv_mlp_proj" (not "proj_mlp")
#   - Single-stream output: "attn.to_out" (new — wasn't targeted before)
#
# Expected LoRA pairs with these targets:
#   double-stream attention: 8 blocks × 8 modules = 64 pairs
#   double-stream FF:        8 blocks × 4 modules = 32 pairs
#   single-stream:           48 blocks × 2 modules = 96 pairs
#   TOTAL: 192 pairs = 384 LoRA weight tensors
# ──────────────────────────────────────────────────────────────────────────────

FLUX2_LORA_TARGET_MODULES = [
    # ── Double-stream: image-stream attention (8 blocks) ────────────
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",           # double-stream uses Linear wrapped as Sequential[0]
    # ── Double-stream: text (context) stream attention ───────────────
    "attn.add_q_proj",
    "attn.add_k_proj",
    "attn.add_v_proj",
    "attn.to_add_out",         # FLUX.2 name (was "add_out_proj" in FLUX.1 — WRONG)
    # ── Double-stream: feed-forward (image stream) ───────────────────
    "ff.linear_in",            # FLUX.2 name (was "ff.net.0.proj" — WRONG)
    "ff.linear_out",           # FLUX.2 name (was "ff.net.2" — WRONG)
    # ── Double-stream: feed-forward (text/context stream) ────────────
    "ff_context.linear_in",    # FLUX.2 name (was "ff_context.net.0.proj" — WRONG)
    "ff_context.linear_out",   # FLUX.2 name (was "ff_context.net.2" — WRONG)
    # ── Single-stream: fused QKV+MLP projection (48 blocks) ──────────
    "attn.to_qkv_mlp_proj",    # FLUX.2 name (was "proj_mlp" — WRONG)
    # NOTE: "attn.to_out.0" already covers single-stream output.
    # single_transformer_blocks.N.attn.to_out is a ModuleList([Linear, Dropout])
    # — targeting the ModuleList directly raises a PEFT ValueError.
    # "attn.to_out.0" matches the inner Linear in both double-stream and
    # single-stream blocks (48 single × 1 + 8 double × 1 = 56 matches).
]

# Lighter variant: attention only, no FF. Saves ~25% VRAM at cost of less adaptation.
FLUX2_LORA_TARGET_ATTN_ONLY = [
    "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
    "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", "attn.to_add_out",
    "attn.to_qkv_mlp_proj",  # to_out.0 already in list above
]


def create_lora_config(
    rank: int,
    alpha: Optional[float] = None,
    dropout: float = 0.0,
    attn_only: bool = False,
) -> LoraConfig:
    targets = FLUX2_LORA_TARGET_ATTN_ONLY if attn_only else FLUX2_LORA_TARGET_MODULES
    return LoraConfig(
        r=rank,
        lora_alpha=alpha if alpha is not None else rank,
        target_modules=targets,
        lora_dropout=dropout,
        bias="none",
    )


# ══════════════════════════════════════════════════════════════════════
#  VALIDATION PROMPTS
# ══════════════════════════════════════════════════════════════════════

VALIDATION_PROMPTS = [
    # Fixed FINDINGS-only prompts drawn from typical MIMIC test cases.
    # These deliberately exclude IMPRESSION — MAIRA-2 generates FINDINGS only,
    # so prompts at inference time will always be FINDINGS-only strings.
    (
        "FINDINGS: There is cardiomegaly with pulmonary vascular congestion and bilateral "
        "pleural effusions, right greater than left. No pneumothorax."
    ),
    (
        "FINDINGS: Lung volumes are low. No focal consolidation, pleural effusion, or "
        "pneumothorax. The cardiac silhouette is normal in size. Support devices in place."
    ),
    (
        "FINDINGS: Right lower lobe opacity consistent with consolidation. Mild cardiomegaly. "
        "No pneumothorax. Endotracheal tube and nasogastric tube present."
    ),
]


@torch.no_grad()
def run_validation(
    transformer,
    vae,
    pipe: Flux2Pipeline,
    accelerator: Accelerator,
    output_dir: str,
    step: int,
    resolution: int,
    dtype: torch.dtype,
    guidance_scale: float = 3.5,
    num_inference_steps: int = 28,
    max_sequence_length: int = 512,
):
    """
    Generate validation CXR images using the FLUX.2 pipeline with LoRA weights.

    Uses the Flux2Pipeline directly for inference (handles Mistral internally).
    The transformer is temporarily swapped for the LoRA-adapted version.
    """
    logger.info(f"Running validation at step {step}...")
    transformer.eval()
    device = accelerator.device

    val_dir = os.path.join(output_dir, "validation_samples", f"step_{step:06d}")
    os.makedirs(val_dir, exist_ok=True)

    # ── SAFE VALIDATION ──────────────────────────────────────────────
    # Approach: convert PEFT state dict to diffusers format, then use
    # pipe.load_lora_weights() on a fresh pipeline that already has
    # enable_model_cpu_offload() applied to its BASE transformer.
    #
    # Why NOT: pipe.transformer = get_peft_model(...) then enable_model_cpu_offload()
    #   → enable_model_cpu_offload calls remove_all_hooks() which calls
    #     delattr(PeftModel, "_hf_hook") — PeftModel has no _hf_hook → AttributeError.
    #
    # Key conversion (PEFT → diffusers):
    #   base_model.model.transformer_blocks.N.attn.to_q.lora_A.default.weight
    #   → transformer.transformer_blocks.N.attn.to_q.lora_A.weight
    from peft.utils import get_peft_model_state_dict as _get_state_dict
    from safetensors.torch import save_file as _save_file

    def _peft_to_diffusers_keys(state_dict):
        out = {}
        for k, v in state_dict.items():
            k = k.replace("base_model.model.", "transformer.")
            k = k.replace(".lora_A.default.", ".lora_A.")
            k = k.replace(".lora_B.default.", ".lora_B.")
            out[k] = v
        return out

    # Step 1: offload training transformer to CPU (~64 GB freed)
    unwrapped = accelerator.unwrap_model(transformer)
    unwrapped.to("cpu")
    torch.cuda.empty_cache()
    logger.info("Training transformer offloaded to CPU (~64 GB freed)")

    # Step 2: extract and convert LoRA weights
    lora_state = _get_state_dict(unwrapped)
    lora_diffusers = _peft_to_diffusers_keys(lora_state)
    _tmp_lora = os.path.join(val_dir, "_tmp_lora.safetensors")
    _save_file({k: v.contiguous() for k, v in lora_diffusers.items()}, _tmp_lora)
    logger.info(f"Saved {len(lora_diffusers)} LoRA tensors in diffusers format")

    # Step 3: load_lora_weights onto a pipe whose transformer has no PEFT wrapper
    # enable_model_cpu_offload() is safe here — transformer is plain nn.Module
    pipe.enable_model_cpu_offload()
    pipe.load_lora_weights(_tmp_lora)
    os.remove(_tmp_lora)

    for i, prompt in enumerate(VALIDATION_PROMPTS):
        result = pipe(
            prompt=prompt,
            height=resolution,
            width=resolution,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            max_sequence_length=max_sequence_length,
            generator=torch.Generator("cpu").manual_seed(42 + i),
        )
        img = result.images[0].convert("L")  # grayscale for CXR comparison
        img.save(os.path.join(val_dir, f"sample_{i:02d}.png"))
        with open(os.path.join(val_dir, f"sample_{i:02d}_prompt.txt"), "w") as f:
            f.write(prompt)

    # Step 3: restore training transformer to GPU
    unwrapped.to(device)
    torch.cuda.empty_cache()
    transformer.train()
    logger.info("Training transformer restored to GPU")


# ══════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
) -> float:
    """Compute mu for FLUX.2 dynamic timestep shifting. res=512 → mu=0.632."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def train(args):
    # ── Accelerator setup ────────────────────────────────────────────
    logging_dir = os.path.join(args.output_dir, "logs")
    project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard" if args.report_to == "tensorboard" else None,
        project_config=project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    weight_dtype = torch.bfloat16
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "no":
        weight_dtype = torch.float32

    device = accelerator.device

    # ── Phase 1: Load pipeline and build cache ───────────────────────
    logger.info(f"Loading FLUX.2-dev pipeline from: {args.model_id}")
    logger.info("(This loads Mistral Small 3.1 + FLUX.2 VAE — expect ~27 GB CPU RAM)")

    pipe = Flux2Pipeline.from_pretrained(
        args.model_id,
        torch_dtype=weight_dtype,
    )

    # Build cache if needed (or force rebuild)
    if accelerator.is_main_process:
        os.makedirs(args.cache_dir, exist_ok=True)
        raw_dataset = MIMICCXRRawDataset(
            csv_path=args.data_csv,
            image_root=args.image_root,
            resolution=args.resolution,
            split="train",
            use_indication=args.use_indication,
            max_samples=args.max_train_samples,
            seed=args.seed or 42,
        )
        build_cache(
            raw_dataset=raw_dataset,
            pipe=pipe,
            cache_dir=args.cache_dir,
            device=device,
            dtype=weight_dtype,
            batch_size=args.cache_batch_size,
            num_workers=args.dataloader_num_workers,
            max_sequence_length=args.max_sequence_length,
            force_rebuild=args.force_cache_rebuild,
        )

    accelerator.wait_for_everyone()  # all ranks wait for cache to be ready

    # ── Phase 2: Free text encoder + VAE, load transformer for LoRA ──
    # Extract transformer and scheduler from pipe before freeing everything
    scheduler = pipe.scheduler
    train_scheduler = copy.deepcopy(scheduler)
    image_seq_len = (args.resolution // 8 // 2) ** 2  # 1024 at res=512
    mu = calculate_shift(image_seq_len)
    logger.info(f"Scheduler mu={mu:.4f} for resolution={args.resolution}")
    train_scheduler.set_timesteps(
        train_scheduler.config.num_train_timesteps, device="cpu", mu=mu,
    )

    # Extract VAE for validation (kept on CPU, moved to GPU only at val time)
    vae = pipe.vae
    vae.requires_grad_(False)
    vae.eval()
    vae.to("cpu")

    # Free text encoder — we don't need it during training (all cached)
    del pipe.text_encoder
    if hasattr(pipe, "text_encoder_2"):
        del pipe.text_encoder_2
    torch.cuda.empty_cache()
    logger.info("Mistral text encoder freed from memory.")

    # Load transformer separately for LoRA wrapping
    logger.info("Loading Flux2Transformer2DModel for LoRA fine-tuning...")

    if args.use_4bit_quantization:
        # QLoRA: load transformer in NF4 (bitsandbytes) for A100 40GB
        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            raise ImportError("Install bitsandbytes for 4-bit quantization: pip install bitsandbytes")
        logger.warning(
            "4-bit quantization enabled (QLoRA mode). "
            "This is intended for A100 40GB. For H100 80GB, use full bf16."
        )
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=weight_dtype,
        )
        transformer = Flux2Transformer2DModel.from_pretrained(
            args.model_id, subfolder="transformer",
            quantization_config=bnb_config,
            torch_dtype=weight_dtype,
        )
    else:
        transformer = Flux2Transformer2DModel.from_pretrained(
            args.model_id, subfolder="transformer",
            torch_dtype=weight_dtype,
        )

    transformer.requires_grad_(False)

    # Confirm FLUX.2-dev guidance embedding
    has_guidance = getattr(transformer.config, "timestep_guidance_channels", 0) > 0
    if not has_guidance:
        logger.warning(
            "guidance_embeds=False: this may not be FLUX.2-dev. "
            "Expected black-forest-labs/FLUX.2-dev."
        )

    # ── Apply LoRA ───────────────────────────────────────────────────
    lora_config = create_lora_config(
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        attn_only=args.lora_attn_only,
    )
    # Enable gradient checkpointing BEFORE PEFT wrapping.
    # Must be called on the base diffusers model (ModelMixin.enable_gradient_checkpointing).
    # enable_input_require_grads / gradient_checkpointing_enable are transformers methods
    # and do NOT exist on Flux2Transformer2DModel — only enable_gradient_checkpointing()
    # (diffusers ModelMixin) is available, and it must be called before get_peft_model().
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        logger.info("Gradient checkpointing enabled (diffusers ModelMixin)")

    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()

    if args.resume_from_checkpoint and os.path.isdir(args.resume_from_checkpoint):
        ckpt_path = os.path.join(args.resume_from_checkpoint, "pytorch_lora_weights.safetensors")
        if os.path.exists(ckpt_path):
            logger.info(f"Resuming LoRA weights from: {ckpt_path}")
            from safetensors.torch import load_file
            state_dict = load_file(ckpt_path)
            set_peft_model_state_dict(transformer, state_dict)

    # ── Cached Dataset & DataLoader ──────────────────────────────────
    train_dataset = CachedMIMICDataset(
        cache_dir=args.cache_dir,
        csv_path=args.data_csv,
        split="train",
        max_samples=args.max_train_samples,
        seed=args.seed or 42,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_cached,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # ── Optimizer & LR scheduler ─────────────────────────────────────
    lora_params = [p for p in transformer.parameters() if p.requires_grad]
    logger.info(f"Trainable LoRA parameters: {sum(p.numel() for p in lora_params):,}")

    optimizer = torch.optim.AdamW(
        lora_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    # Precompute geometry for packed latents
    latent_size  = args.resolution // 8
    packed_h     = latent_size // 2
    packed_w     = latent_size // 2
    n_img_tokens = packed_h * packed_w

    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("════════════════════════════════════════")
    logger.info("  FLUX.2-dev LoRA Training Configuration")
    logger.info(f"  Model:             {args.model_id}")
    logger.info(f"  Dataset:           {len(train_dataset)} samples (cached)")
    logger.info(f"  Resolution:        {args.resolution} → latent {latent_size}×{latent_size}"
                f" → packed {n_img_tokens} tokens of dim {FLUX2_PACKED_TOKEN_DIM}")
    logger.info(f"  Max steps:         {args.max_train_steps}")
    logger.info(f"  Effective batch:   {total_batch_size}")
    logger.info(f"  LoRA rank:         {args.lora_rank}  (attn_only={args.lora_attn_only})")
    logger.info(f"  Learning rate:     {args.learning_rate}")
    logger.info(f"  Guidance (train):  {args.guidance_scale_train}  ← MUST be 1.0 for FLUX.2")
    logger.info(f"  4-bit QLoRA:       {args.use_4bit_quantization}")
    logger.info(f"  Mixed precision:   {args.mixed_precision}")
    logger.info("════════════════════════════════════════")

    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint and os.path.isdir(args.resume_from_checkpoint):
        ckpt_basename = os.path.basename(args.resume_from_checkpoint.rstrip("/"))
        try:
            global_step = int(ckpt_basename.split("-")[1])
        except (IndexError, ValueError):
            global_step = 0
        first_epoch = global_step // num_update_steps_per_epoch
        for _ in range(global_step):
            lr_scheduler.step()

    if accelerator.is_main_process:
        accelerator.init_trackers("flux2_cxr_lora", config=vars(args))

    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
        desc="Training FLUX.2-dev LoRA",
    )

    for epoch in range(first_epoch, num_train_epochs):
        transformer.train()
        train_loss = 0.0
        log_loss_accum = 0.0   # accumulates loss for --logging_steps window

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                bsz = batch["latents"].shape[0]

                # ── 1. Load cached latents + embeddings ───────────────
                # Latents: (B, 32, H/8, W/8)  — 32 channels for FLUX.2 VAE
                latents       = batch["latents"].to(device=device, dtype=weight_dtype)
                # Embeddings: (B, seq_len, hidden_dim)  — Mistral output
                prompt_embeds = batch["prompt_embeds"].to(device=device, dtype=weight_dtype)

                # ── 2. Pack latents (32-channel → 128-dim tokens) ─────
                # FLUX.2: (B, 32, H/8, W/8) → (B, n_img_tokens, 128)
                latents_packed = pack_latents(
                    latents, bsz, FLUX2_LATENT_CHANNELS, latent_size, latent_size
                )

                # ── 3. Prepare position IDs for RoPE ──────────────────
                img_ids = prepare_latent_image_ids(
                    bsz, packed_h, packed_w, device, weight_dtype
                )
                txt_ids = torch.zeros(
                    bsz, prompt_embeds.shape[1], 4,  # 4 axes for FLUX.2 RoPE
                    device=device, dtype=weight_dtype,
                )

                # ── 4. Sample flow-matching timesteps ─────────────────
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * train_scheduler.config.num_train_timesteps).long()
                indices = indices.clamp(0, train_scheduler.config.num_train_timesteps - 1)
                timesteps = train_scheduler.timesteps[indices].to(device)

                # ── 5. Add noise (rectified flow) ──────────────────────
                noise = torch.randn_like(latents_packed)
                sigmas = get_sigmas(
                    timesteps, train_scheduler, n_dim=3, dtype=weight_dtype
                )
                noisy_latents = (1.0 - sigmas) * latents_packed + sigmas * noise

                # ── 6. Guidance vector ────────────────────────────────
                # CRITICAL: Use guidance_scale_train = 1.0 for FLUX.2-dev
                # (NOT 3.5 like FLUX.1-dev — different distillation scheme)
                if has_guidance:
                    guidance_vec = torch.full(
                        (bsz,), args.guidance_scale_train,
                        device=device, dtype=weight_dtype,
                    )
                else:
                    guidance_vec = None

                # ── 7. Transformer forward pass ───────────────────────
                # FLUX.2 vs FLUX.1 differences:
                #   • hidden_states: 128-dim tokens (not 64)
                #   • encoder_hidden_states: Mistral embeddings
                #     (no pooled_projections for FLUX.2 — Mistral outputs
                #      sequence only, no pooled CLS token)
                #   • guidance = 1.0 at training (not 3.5)
                noise_pred = transformer(
                    hidden_states=noisy_latents,
                    encoder_hidden_states=prompt_embeds,
                    # NOTE: FLUX.2 may not use pooled_projections
                    # since Mistral doesn't produce a pooled CLS token.
                    # Remove this arg if it causes an error.
                    timestep=timesteps / 1000.0,
                    img_ids=img_ids,
                    txt_ids=txt_ids,
                    guidance=guidance_vec,
                    return_dict=False,
                )[0]
                # noise_pred: (B, n_img_tokens, 128)

                # ── 8. Flow matching loss (identical formula to FLUX.1) ─
                target = noise - latents_packed
                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme=args.weighting_scheme,
                    sigmas=sigmas,
                )
                loss = (
                    weighting.float() *
                    (noise_pred.float() - target.float()) ** 2
                ).mean()

                # ── 9. Backward ───────────────────────────────────────
                avg_loss = accelerator.gather(loss.repeat(bsz)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(lora_params, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                log_loss_accum += train_loss
                train_loss = 0.0

                if global_step % args.logging_steps == 0 and accelerator.is_main_process:
                    avg_log_loss = log_loss_accum / args.logging_steps
                    accelerator.log(
                        {"train_loss": avg_log_loss, "lr": lr_scheduler.get_last_lr()[0]},
                        step=global_step,
                    )
                    logger.info(
                        f"step={global_step:>6d}  loss={avg_log_loss:.4f}  lr={lr_scheduler.get_last_lr()[0]:.2e}"
                    )
                    log_loss_accum = 0.0

                if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                    save_checkpoint(
                        transformer, accelerator, args.output_dir,
                        global_step, args.checkpoints_total_limit
                    )

                if (
                    args.validation_steps > 0
                    and global_step % args.validation_steps == 0
                    and accelerator.is_main_process
                ):
                    # For validation, we need the full pipeline with Mistral
                    # Reload Flux2Pipeline for validation (expensive but only at val checkpoints)
                    logger.info("Loading full pipeline for validation (Mistral + VAE)...")
                    val_pipe = Flux2Pipeline.from_pretrained(
                        args.model_id, torch_dtype=weight_dtype
                    )
                    run_validation(
                        transformer=transformer,
                        vae=val_pipe.vae,
                        pipe=val_pipe,
                        accelerator=accelerator,
                        output_dir=args.output_dir,
                        step=global_step,
                        resolution=args.resolution,
                        dtype=weight_dtype,
                        guidance_scale=args.guidance_scale,
                        num_inference_steps=args.num_inference_steps,
                        max_sequence_length=args.max_sequence_length,
                    )
                    del val_pipe
                    torch.cuda.empty_cache()

            progress_bar.set_postfix({"loss": f"{avg_loss.item():.4f}", "step": global_step})

            if global_step >= args.max_train_steps:
                break

    # ── Final save ───────────────────────────────────────────────────
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = os.path.join(args.output_dir, "final_lora_weights")
        os.makedirs(final_dir, exist_ok=True)

        unwrapped = accelerator.unwrap_model(transformer)
        lora_state = get_peft_model_state_dict(unwrapped)

        from safetensors.torch import save_file
        save_file(lora_state, os.path.join(final_dir, "pytorch_lora_weights.safetensors"))
        logger.info(f"Final LoRA weights saved to: {final_dir}")

        inference_config = {
            "model_id": args.model_id,
            "lora_weights_path": os.path.join(final_dir, "pytorch_lora_weights.safetensors"),
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_attn_only": args.lora_attn_only,
            "resolution": args.resolution,
            "max_sequence_length": args.max_sequence_length,
            "guidance_scale_train": args.guidance_scale_train,
            "guidance_scale_inference": args.guidance_scale,
            "prompt_format": "FINDINGS: <findings>",
            "prompt_note": (
                "FINDINGS only — no IMPRESSION. "
                "MAIRA-2 generates FINDINGS only; MedCLIP is trained on FINDINGS only. "
                "Prompts must be FINDINGS-only strings for correct attractor loop behaviour."
            ),
            "architecture_notes": (
                "FLUX.2-dev: 32B DiT, Mistral Small 3.1 text encoder, "
                "32-channel VAE, 8 double-stream + 48 single-stream blocks. "
                "guidance_scale=1.0 at training, 3.5 recommended at inference."
            ),
        }
        with open(os.path.join(final_dir, "inference_config.json"), "w") as f:
            json.dump(inference_config, f, indent=2)

    accelerator.end_training()
    logger.info("Training complete.")


def save_checkpoint(transformer, accelerator, output_dir, step, total_limit):
    ckpt_dir = os.path.join(output_dir, f"checkpoint-{step:06d}")
    os.makedirs(ckpt_dir, exist_ok=True)
    unwrapped = accelerator.unwrap_model(transformer)
    lora_state = get_peft_model_state_dict(unwrapped)
    from safetensors.torch import save_file
    save_file(lora_state, os.path.join(ckpt_dir, "pytorch_lora_weights.safetensors"))
    logger.info(f"Saved checkpoint: {ckpt_dir}")
    if total_limit is not None:
        existing = sorted(
            [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[1]),
        )
        for d in existing[: len(existing) - total_limit]:
            shutil.rmtree(os.path.join(output_dir, d))


# ══════════════════════════════════════════════════════════════════════
#  INFERENCE HELPER
# ══════════════════════════════════════════════════════════════════════

def load_flux2_with_lora(
    model_id: str,
    lora_weights_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> Flux2Pipeline:
    """
    Load FLUX.2-dev with the CXR LoRA adapter for inference.

    ── WHAT IS SAVED vs WHAT IS LOADED ─────────────────────────────────
    Training saves ONLY the LoRA adapter matrices — not the full model.
    Each checkpoint contains a single file:
        pytorch_lora_weights.safetensors   (~50-150 MB for rank=32)

    This file holds the low-rank update matrices (A and B) for every
    targeted linear layer. At inference, you load the 32B base model
    from HuggingFace and inject the LoRA deltas on top:
        W_effective = W_base + (alpha/r) * B @ A

    The base model weights are NEVER duplicated or modified on disk.

    ── HOW TO LOAD IN YOUR GENERATION PIPELINE ─────────────────────────

        from finetune_flux2_lora_cxr import load_flux2_with_lora, build_conditioning_text

        # Load base model + inject LoRA (enable_model_cpu_offload handles >80GB)
        pipe = load_flux2_with_lora(
            model_id="black-forest-labs/FLUX.2-dev",
            lora_weights_path="Experiments/finetune_lora_flux2/outputs/checkpoint-005000/"
                               "pytorch_lora_weights.safetensors",
        )

        # At each loop iteration, call with FINDINGS only (MAIRA-2 output)
        maira2_findings = "Lung volumes are low. No focal consolidation..."
        prompt = build_conditioning_text(findings=maira2_findings)

        result = pipe(
            prompt=prompt,
            height=512, width=512,   # matches MedCLIP input — no resize needed
            num_inference_steps=28,
            guidance_scale=3.5,
            max_sequence_length=256,
        )
        generated_image = result.images[0]   # PIL Image, 512×512 RGB
        # Pass directly to MedCLIP image encoder (same 512px, no transform needed)
        # Resize to 518×518 only before passing to MAIRA-2 in the next loop iteration

    ── COMPARING CHECKPOINTS ────────────────────────────────────────────
    To compare different training steps, just swap the lora_weights_path:
        checkpoint-001000/pytorch_lora_weights.safetensors   (early)
        checkpoint-005000/pytorch_lora_weights.safetensors   (mid)
        final_lora_weights/pytorch_lora_weights.safetensors  (final)
    The base model is loaded once; only the small LoRA file changes.

    NOTE: Full-precision inference requires >80 GB VRAM (H100).
    For the A100 80GB nodes: enable_model_cpu_offload() streams layers
    from CPU RAM to GPU on demand — slower but fits in 80GB.
    """
    pipe = Flux2Pipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe.load_lora_weights(lora_weights_path)
    pipe.enable_model_cpu_offload()
    return pipe


# ══════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning of FLUX.2-dev on MIMIC-CXR"
    )
    # Data
    parser.add_argument("--data_csv",          type=str,  required=True)
    parser.add_argument("--image_root",        type=str,  default=None)
    parser.add_argument("--cache_dir",         type=str,  required=True,
                        help="Directory for pre-cached latents + embeddings.")
    parser.add_argument("--force_cache_rebuild", action="store_true",
                        help="Rebuild cache even if files already exist.")
    parser.add_argument("--cache_batch_size",  type=int,  default=4,
                        help="Batch size for Phase 1 caching (VAE + Mistral).")
    parser.add_argument("--use_indication",    action="store_true")
    parser.add_argument("--max_train_samples", type=int,  default=12000,
                        help="Stratified sample size. 12000 gives good pathology coverage "
                             "with ~29GB cache (vs ~288GB for full 100K).")
    parser.add_argument("--max_sequence_length", type=int, default=128,
                        help="Max Mistral token length. MIMIC FINDINGS are ~65-200 tokens; "
                             "256 covers >99%% of cases. 512 works but doubles cache size.")

    # Model
    parser.add_argument("--model_id",    type=str, default="black-forest-labs/FLUX.2-dev")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--use_4bit_quantization", action="store_true",
                        help="QLoRA with bitsandbytes NF4. Required for A100 40GB.")

    # LoRA
    parser.add_argument("--lora_rank",    type=int,   default=32,
                        help="16 for A100 40GB with QLoRA; 32 for H100 bf16.")
    parser.add_argument("--lora_alpha",   type=float, default=None)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_attn_only", action="store_true",
                        help="Only target attention layers (not FF). Saves ~25%% VRAM.")

    # Training
    parser.add_argument("--resolution",    type=int, default=512,
                        help="Output resolution. 512 matches MedCLIP input size exactly "
                             "(no extra resize needed in the attractor loop). "
                             "FLUX.2 needs resolution divisible by 16: 512÷16=32 ✓")
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--max_train_steps",  type=int, default=15000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate",  type=float, default=5e-5,
                        help="5e-5 for rank=32 on 32B model. 1e-4 causes over-adaptation.")
    parser.add_argument("--lr_scheduler",   type=str,   default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int,  default=500)
    parser.add_argument("--lr_num_cycles",  type=int,   default=1)
    parser.add_argument("--lr_power",       type=float, default=1.0)
    parser.add_argument("--max_grad_norm",  type=float, default=1.0)

    # Adam
    parser.add_argument("--adam_beta1",        type=float, default=0.9)
    parser.add_argument("--adam_beta2",        type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-4)
    parser.add_argument("--adam_epsilon",      type=float, default=1e-8)

    # Guidance — CRITICAL: use 1.0 for FLUX.2, NOT 3.5
    parser.add_argument("--guidance_scale_train", type=float, default=1.0,
                        help="Guidance at training. MUST be 1.0 for FLUX.2-dev.")
    parser.add_argument("--guidance_scale",       type=float, default=3.5,
                        help="Guidance at validation/inference (2–4 range).")

    # Flow matching
    parser.add_argument("--weighting_scheme", type=str,   default="logit_normal",
                        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"])
    parser.add_argument("--logit_mean",  type=float, default=0.0)
    parser.add_argument("--logit_std",   type=float, default=1.0)
    parser.add_argument("--mode_scale",  type=float, default=1.29)

    # Output
    parser.add_argument("--output_dir",           type=str, default="./flux2_cxr_lora")
    parser.add_argument("--checkpointing_steps",  type=int, default=1000,
                        help="Save a LoRA checkpoint every N optimizer steps. "
                             "At 12K samples, batch=1, accum=32: ~375 steps/epoch. "
                             "1000 steps ≈ every 2.7 epochs.")
    parser.add_argument("--logging_steps",           type=int, default=200,
                        help="Log loss to console and tensorboard every N optimizer steps.")
    parser.add_argument("--checkpoints_total_limit", type=int, default=15,
                        help="Keep this many recent checkpoints. "
                             "15 × 1000-step checkpoints covers the full 15K-step run.")
    parser.add_argument("--validation_steps",     type=int, default=2000)
    parser.add_argument("--num_inference_steps",  type=int, default=28)

    # System
    parser.add_argument("--mixed_precision",          type=str, default="bf16")
    parser.add_argument("--dataloader_num_workers",   type=int, default=8)
    parser.add_argument("--seed",                     type=int, default=42)
    parser.add_argument("--report_to",                type=str, default="tensorboard")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)