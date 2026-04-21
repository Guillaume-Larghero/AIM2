#!/usr/bin/env python3
"""
LoRA Fine-tuning of Stable Diffusion 3.5 Medium on MIMIC-CXR
=============================================================

Trains a LoRA adapter on the MMDiT transformer of SD3.5 Medium using
MIMIC-CXR image-report pairs.  Text encoders (T5-XXL, CLIP-L, CLIP-G)
are fully frozen — only the denoising transformer learns domain adaptation.

At inference, the loop becomes a clean 2-component dynamical system:
    report_t  ──► SD3.5-CXR-LoRA (text-only) ──► image_t
    image_t   ──► Medical CLIP ──► image FAISS ──► LLM ──► report_t+1

Usage (single GPU):
    python finetune_sd35_lora_cxr.py \
        --data_csv /path/to/processed_data.csv \
        --image_root /path/to/cxr_jpg \
        --model_id stabilityai/stable-diffusion-3.5-medium \
        --output_dir ./sd35_cxr_lora \
        --resolution 512 \
        --lora_rank 32 \
        --train_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --max_train_steps 15000 \
        --learning_rate 1e-4

Usage (multi-GPU with accelerate, recommended):
    accelerate launch --num_processes 2 finetune_sd35_lora_cxr.py [args]

SLURM example (2× A100 40GB):
    #SBATCH --gres=gpu:2 --mem=80G --time=24:00:00
    module load miniconda3 && conda activate aim_env
    accelerate launch --num_processes 2 finetune_sd35_lora_cxr.py [args]

Dependencies:
    pip install diffusers>=0.31.0 transformers>=4.45.0 peft>=0.13.0 \
                accelerate>=0.34.0 bitsandbytes xformers torch torchvision \
                pandas Pillow tqdm
"""

import argparse
import copy
import logging
import math
import os
import shutil
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)

from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict

from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)

logger = get_logger(__name__, log_level="INFO")


# ══════════════════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════════════════

def build_conditioning_text(
    findings: str,
    impression: str,
    indication: str = "",
    use_indication: bool = False,
    max_chars: int = 1800,   # T5-XXL handles ~512 tokens ≈ 1800-2000 chars; stay safe
) -> str:
    """
    Build the text prompt fed to the diffusion model.

    Format is chosen to match EXACTLY what the inference loop will use:
      "FINDINGS: <findings> IMPRESSION: <impression>"

    This is the single most important consistency requirement — training
    and inference must use identical prompt formatting.

    Args:
        findings: Radiology findings section (may be empty).
        impression: Radiology impression section (may be empty).
        indication: Clinical indication (optional; adds clinical context).
        use_indication: Whether to prepend indication.
        max_chars: Truncate at this character count to stay within T5 context.
    """
    parts = []

    # Indication is optional — useful clinically but adds noise if too sparse
    if use_indication and indication and indication.strip():
        ind = indication.strip()
        # Anonymize placeholders that appear frequently in MIMIC
        ind = ind.replace("___", "patient")
        parts.append(f"INDICATION: {ind}")

    if findings and findings.strip():
        parts.append(f"FINDINGS: {findings.strip()}")

    if impression and impression.strip():
        parts.append(f"IMPRESSION: {impression.strip()}")

    if not parts:
        # Should not happen after dataset filtering, but safe fallback
        return "Chest radiograph with no available report."

    text = " ".join(parts)

    # Hard truncate at character level as a backstop
    # (the tokenizer will handle proper token-level truncation)
    if len(text) > max_chars:
        text = text[:max_chars]

    return text


class MIMICCXRDataset(Dataset):
    """
    Dataset for MIMIC-CXR LoRA fine-tuning.

    Loads image-report pairs from the processed CSV.  Images are CXR JPEGs
    (grayscale in practice); we convert to RGB by repeating channels since
    SD3.5 expects 3-channel input.

    Filtering strategy (applied at __init__):
      - split == 'train'
      - has_findings OR has_impression (at least one text field)
      - image file exists on disk (validated lazily in __getitem__ for speed)

    Notes on the CSV format:
      - `findings` and `impression` can be NaN when has_findings/has_impression is False
      - `image_path` is the absolute path to the .jpg file
    """

    def __init__(
        self,
        csv_path: str,
        image_root: Optional[str] = None,
        resolution: int = 512,
        split: str = "train",
        use_indication: bool = False,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ):
        self.resolution = resolution
        self.use_indication = use_indication

        df = pd.read_csv(csv_path, low_memory=False)

        # Normalise split column (some datasets use 'validate')
        df["split"] = df["split"].replace({"validate": "val"})
        df = df[df["split"] == split].copy()
        logger.info(f"Samples in '{split}' split: {len(df)}")

        # Must have at least one text field
        has_text = (
            (df["has_findings"].astype(str).str.lower().isin(["true", "1", "1.0"])) |
            (df["has_impression"].astype(str).str.lower().isin(["true", "1", "1.0"]))
        )
        df = df[has_text].copy()
        logger.info(f"Samples with text after filtering: {len(df)}")

        # Fill NaN text fields with empty string
        for col in ["findings", "impression", "indication"]:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)
            else:
                df[col] = ""

        # Resolve image paths
        if image_root:
            # If image_path in CSV is absolute, use as-is; otherwise join with root
            df["resolved_path"] = df["image_path"].apply(
                lambda p: p if os.path.isabs(str(p)) else os.path.join(image_root, str(p))
            )
        else:
            df["resolved_path"] = df["image_path"].astype(str)

        if max_samples is not None and max_samples < len(df):
            df = df.sample(n=max_samples, random_state=seed).reset_index(drop=True)

        self.df = df.reset_index(drop=True)

        # Image transform pipeline: direct squash to (resolution, resolution).
        # No cropping of any kind — MIMIC images are preserved exactly as acquired.
        # Horizontal flip is intentionally omitted: laterality is clinically meaningful.
        self._img_transform = transforms.Compose([
            transforms.Resize(
                (resolution, resolution),
                interpolation=transforms.InterpolationMode.LANCZOS,
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        logger.info(f"MIMICCXRDataset: {len(self.df)} samples | resolution={resolution}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = row["resolved_path"]

        # ── Image ────────────────────────────────────────────────────
        try:
            img = Image.open(img_path)
            # MIMIC-CXR JPEGs are grayscale; repeat to 3 channels for SD3.5 VAE.
            # R=G=B=gray — no information is added or lost.
            if img.mode == "RGBA":
                img = img.convert("RGB")
            elif img.mode != "RGB":
                img = img.convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to load image at {img_path}: {e}. Using blank.")
            img = Image.new("RGB", (self.resolution, self.resolution), 128)

        pixel_values = self._img_transform(img)  # (3, resolution, resolution)

        # ── Text ─────────────────────────────────────────────────────
        prompt = build_conditioning_text(
            findings=row["findings"],
            impression=row["impression"],
            indication=row.get("indication", ""),
            use_indication=self.use_indication,
        )

        return {
            "pixel_values": pixel_values,           # (3, H, W), float32, range [-1, 1]
            "prompt": prompt,                        # raw string; tokenised in collate
            "study_id": str(row.get("study_id", idx)),
        }


def collate_fn(examples):
    pixel_values = torch.stack([e["pixel_values"] for e in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    prompts = [e["prompt"] for e in examples]
    study_ids = [e["study_id"] for e in examples]
    return {
        "pixel_values": pixel_values,
        "prompts": prompts,
        "study_ids": study_ids,
    }


# ══════════════════════════════════════════════════════════════════════
#  TEXT ENCODING
# ══════════════════════════════════════════════════════════════════════

def encode_prompt_sd3(
    prompt_batch: List[str],
    text_encoders,            # [clip_l, clip_g, t5]
    tokenizers,               # [tok_clip_l, tok_clip_g, tok_t5]
    device: torch.device,
    dtype: torch.dtype,
    max_sequence_length: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encode a batch of prompts using SD3's three text encoders.

    Returns:
        prompt_embeds: (B, 512, 4096)  — T5 only → fed to encoder_hidden_states
        pooled_prompt_embeds: (B, 2048) — CLIP-L + CLIP-G pooled → fed to pooled_projections

    SD3's context_embedder is Linear(4096, 1536) — it expects T5 dim only.
    CLIP sequence embeddings are NOT passed as encoder_hidden_states.
    Only the CLIP pooled outputs contribute, via pooled_projections.
    """
    clip_l, clip_g, t5 = text_encoders
    tok_l, tok_g, tok_t5 = tokenizers

    # ── CLIP-L pooled (77 tokens) ────────────────────────────────────
    text_inputs_l = tok_l(
        prompt_batch,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        out_l = clip_l(text_inputs_l.input_ids, output_hidden_states=True)
    pooled_l = out_l.text_embeds.to(dtype)                     # (B, 768)

    # ── CLIP-G pooled (77 tokens) ────────────────────────────────────
    text_inputs_g = tok_g(
        prompt_batch,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        out_g = clip_g(text_inputs_g.input_ids, output_hidden_states=True)
    pooled_g = out_g.text_embeds.to(dtype)                     # (B, 1280)

    # ── T5-XXL sequence embeddings (up to 512 tokens) ────────────────
    # This is the only input to encoder_hidden_states — dim 4096 matches
    # SD3Transformer2DModel.context_embedder: Linear(4096, 1536)
    text_inputs_t5 = tok_t5(
        prompt_batch,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        t5_embeds = t5(text_inputs_t5.input_ids)[0].to(dtype)  # (B, 512, 4096)

    # T5 sequence → encoder_hidden_states
    prompt_embeds = t5_embeds                                   # (B, 512, 4096)

    # Pooled CLIP-L + CLIP-G → pooled_projections  →  (B, 2048)
    pooled_prompt_embeds = torch.cat([pooled_l, pooled_g], dim=-1)

    return prompt_embeds, pooled_prompt_embeds


# ══════════════════════════════════════════════════════════════════════
#  LORA CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

# Target modules in SD3Transformer2DModel (MMDiT architecture).
# JointTransformerBlock structure:
#   attn  → JointAttention: to_q, to_k, to_v, to_out, add_{q,k,v}_proj, add_out_proj
#   ff    → FeedForward: net.0.proj, net.2
#   ff_context → same structure (for text stream)
# We target all attention projections across both streams.
SD3_LORA_TARGET_MODULES = [
    # Image stream attention
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    # Text stream attention (joint blocks only)
    "attn.add_q_proj",
    "attn.add_k_proj",
    "attn.add_v_proj",
    "attn.add_out_proj",
    # Feed-forward (image stream) — optional; remove if VRAM constrained
    "ff.net.0.proj",
    "ff.net.2",
]


def create_lora_config(rank: int, alpha: Optional[float] = None, dropout: float = 0.0):
    """Create PEFT LoRA config targeting SD3's MMDiT transformer blocks."""
    return LoraConfig(
        r=rank,
        lora_alpha=alpha if alpha is not None else rank,   # alpha=rank → effective scale=1
        target_modules=SD3_LORA_TARGET_MODULES,
        lora_dropout=dropout,
        bias="none",
    )


# ══════════════════════════════════════════════════════════════════════
#  VALIDATION SAMPLING
# ══════════════════════════════════════════════════════════════════════

VALIDATION_PROMPTS = [
    # Pulled from typical MIMIC test set impressions — fixed across runs for comparability
    (
        "FINDINGS: There is cardiomegaly with pulmonary vascular congestion and bilateral "
        "pleural effusions, right greater than left. No pneumothorax. "
        "IMPRESSION: Cardiomegaly with pulmonary edema and bilateral pleural effusions."
    ),
    (
        "FINDINGS: Lung volumes are low. No focal consolidation, pleural effusion, or "
        "pneumothorax. The cardiac silhouette is normal in size. Support devices in place. "
        "IMPRESSION: No acute cardiopulmonary abnormality."
    ),
    (
        "FINDINGS: There is a right lower lobe opacity consistent with consolidation. "
        "Mild cardiomegaly. No pneumothorax. Endotracheal tube and nasogastric tube present. "
        "IMPRESSION: Right lower lobe pneumonia. Cardiomegaly. Support devices in place."
    ),
]


@torch.no_grad()
def run_validation(
    transformer,
    vae,
    text_encoders,
    tokenizers,
    scheduler,
    accelerator,
    output_dir: str,
    step: int,
    resolution: int,
    dtype: torch.dtype,
    guidance_scale: float = 7.0,
    num_inference_steps: int = 28,
):
    """Generate sample images from fixed prompts and save for visual inspection."""
    logger.info(f"Running validation at step {step}...")

    transformer.eval()
    device = accelerator.device

    val_dir = os.path.join(output_dir, "validation_samples", f"step_{step:06d}")
    os.makedirs(val_dir, exist_ok=True)

    for i, prompt in enumerate(VALIDATION_PROMPTS):
        prompt_embeds, pooled_embeds = encode_prompt_sd3(
            [prompt], text_encoders, tokenizers, device, dtype
        )

        # Classifier-free guidance: encode empty prompt
        neg_embeds, neg_pooled = encode_prompt_sd3(
            [""], text_encoders, tokenizers, device, dtype
        )

        latent_channels = 16   # SD3.5 VAE latent channels
        latent_size = resolution // 8
        latents = torch.randn(
            1, latent_channels, latent_size, latent_size,
            device=device, dtype=dtype,
        )
        # FlowMatchEulerDiscreteScheduler: no init_noise_sigma — pure N(0,1) is correct

        scheduler.set_timesteps(num_inference_steps, device=device)

        for t in scheduler.timesteps:
            # CFG: duplicate latents
            latent_input = torch.cat([latents, latents], dim=0)
            t_batch = t.expand(2)

            pe = torch.cat([neg_embeds, prompt_embeds], dim=0)
            ppe = torch.cat([neg_pooled, pooled_embeds], dim=0)

            noise_pred = transformer(
                hidden_states=latent_input,
                timestep=t_batch,
                encoder_hidden_states=pe,
                pooled_projections=ppe,
                return_dict=False,
            )[0]

            noise_uncond, noise_cond = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # Decode: reverse VAE encoding transform
        latents_scaled = (latents / vae.config.scaling_factor) + vae.config.shift_factor
        img_tensor = vae.decode(latents_scaled.to(dtype)).sample
        img_tensor = (img_tensor / 2 + 0.5).clamp(0, 1)
        img_np = img_tensor[0].permute(1, 2, 0).float().cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)

        # Save as grayscale (convert RGB→L for visual comparison with real CXRs)
        pil_img = Image.fromarray(img_np).convert("L")
        pil_img.save(os.path.join(val_dir, f"sample_{i:02d}.png"))

        # Also save the prompt for reference
        with open(os.path.join(val_dir, f"sample_{i:02d}_prompt.txt"), "w") as f:
            f.write(prompt)

    transformer.train()
    logger.info(f"Saved validation samples to {val_dir}")


# ══════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════

def train(args):
    # ── Accelerator setup ───────────────────────────────────────────
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

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # ── Load models ─────────────────────────────────────────────────
    logger.info(f"Loading SD3.5 Medium from: {args.model_id}")

    # Tokenizers
    tokenizer_clip_l = CLIPTokenizer.from_pretrained(
        args.model_id, subfolder="tokenizer"
    )
    tokenizer_clip_g = CLIPTokenizer.from_pretrained(
        args.model_id, subfolder="tokenizer_2"
    )
    tokenizer_t5 = T5TokenizerFast.from_pretrained(
        args.model_id, subfolder="tokenizer_3"
    )

    # Text encoders — fully frozen, loaded in weight_dtype to save VRAM
    text_encoder_l = CLIPTextModelWithProjection.from_pretrained(
        args.model_id, subfolder="text_encoder",
        torch_dtype=weight_dtype,
    )
    text_encoder_g = CLIPTextModelWithProjection.from_pretrained(
        args.model_id, subfolder="text_encoder_2",
        torch_dtype=weight_dtype,
    )
    text_encoder_t5 = T5EncoderModel.from_pretrained(
        args.model_id, subfolder="text_encoder_3",
        torch_dtype=weight_dtype,
    )
    for enc in [text_encoder_l, text_encoder_g, text_encoder_t5]:
        enc.requires_grad_(False)
        enc.eval()

    # VAE — frozen
    vae = AutoencoderKL.from_pretrained(
        args.model_id, subfolder="vae",
        torch_dtype=weight_dtype,
    )
    vae.requires_grad_(False)
    vae.eval()

    # Noise scheduler
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.model_id, subfolder="scheduler"
    )
    # Deep-copy for the training loop.  run_validation() calls
    # scheduler.set_timesteps(num_inference_steps=28) which overwrites
    # scheduler.timesteps and scheduler.sigmas in-place (1000→28 elements).
    # The training loop indexes sigmas/timesteps with values up to 999 →
    # CUDA out-of-bounds assertion on the very next step after validation.
    # train_scheduler is never passed to validation and always retains
    # the full 1000-step arrays.
    train_scheduler = copy.deepcopy(scheduler)

    # Transformer (MMDiT) — this is the only trainable component
    transformer = SD3Transformer2DModel.from_pretrained(
        args.model_id, subfolder="transformer",
        torch_dtype=weight_dtype,
    )
    transformer.requires_grad_(False)   # will be re-enabled by PEFT

    # ── Apply LoRA ──────────────────────────────────────────────────
    lora_config = create_lora_config(
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()

    # Resume from checkpoint if requested
    if args.resume_from_checkpoint and os.path.isdir(args.resume_from_checkpoint):
        ckpt_path = os.path.join(args.resume_from_checkpoint, "pytorch_lora_weights.safetensors")
        if os.path.exists(ckpt_path):
            logger.info(f"Resuming LoRA from: {ckpt_path}")
            from safetensors.torch import load_file
            state_dict = load_file(ckpt_path)
            set_peft_model_state_dict(transformer, state_dict)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # ── Dataset & DataLoader ─────────────────────────────────────────
    train_dataset = MIMICCXRDataset(
        csv_path=args.data_csv,
        image_root=args.image_root,
        resolution=args.resolution,
        split="train",
        use_indication=args.use_indication,
        max_samples=args.max_train_samples,
        seed=args.seed or 42,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
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

    # Steps per epoch (accounting for grad accumulation)
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

    # ── Accelerate prepare ──────────────────────────────────────────
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    # Text encoders + VAE to device (not wrapped by accelerate — they're frozen)
    device = accelerator.device
    for enc in [text_encoder_l, text_encoder_g, text_encoder_t5]:
        enc.to(device)
    vae.to(device)

    # ── Training state ──────────────────────────────────────────────
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )
    logger.info("════════════════════════════════════════")
    logger.info("  Training configuration")
    logger.info(f"  Dataset:           {len(train_dataset)} samples")
    logger.info(f"  Epochs:            {num_train_epochs}")
    logger.info(f"  Max steps:         {args.max_train_steps}")
    logger.info(f"  Batch size/GPU:    {args.train_batch_size}")
    logger.info(f"  Grad accumulation: {args.gradient_accumulation_steps}")
    logger.info(f"  Effective batch:   {total_batch_size}")
    logger.info(f"  LoRA rank:         {args.lora_rank}")
    logger.info(f"  Learning rate:     {args.learning_rate}")
    logger.info(f"  Mixed precision:   {args.mixed_precision}")
    logger.info("════════════════════════════════════════")

    global_step = 0
    first_epoch = 0

    # ── Resume step/epoch counters ───────────────────────────────────
    # The LoRA weights are loaded earlier (before accelerator.prepare).
    # Here we restore global_step and first_epoch from the checkpoint
    # directory name, then fast-forward the LR scheduler so its cosine
    # position is correct for the resumed step.
    if args.resume_from_checkpoint and os.path.isdir(args.resume_from_checkpoint):
        ckpt_basename = os.path.basename(args.resume_from_checkpoint.rstrip("/"))
        try:
            global_step = int(ckpt_basename.split("-")[1])
        except (IndexError, ValueError):
            logger.warning(
                f"Could not parse step from checkpoint name '{ckpt_basename}'; "
                "global_step will start from 0."
            )
            global_step = 0
        first_epoch = global_step // num_update_steps_per_epoch
        logger.info(
            f"Resuming from step {global_step} (epoch {first_epoch}); "
            "fast-forwarding LR scheduler..."
        )
        for _ in range(global_step):
            lr_scheduler.step()

    if accelerator.is_main_process:
        accelerator.init_trackers("sd35_cxr_lora", config=vars(args))

    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
        desc="Training",
    )

    for epoch in range(first_epoch, num_train_epochs):
        transformer.train()
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                # ── Encode images to latent space ────────────────────
                pixel_values = batch["pixel_values"].to(dtype=weight_dtype)

                with torch.no_grad():
                    # SD3.5 VAE latent: encode, scale, shift
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
                    latents = latents.to(weight_dtype)

                # ── Encode text prompts ──────────────────────────────
                with torch.no_grad():
                    prompt_embeds, pooled_embeds = encode_prompt_sd3(
                        batch["prompts"],
                        [text_encoder_l, text_encoder_g, text_encoder_t5],
                        [tokenizer_clip_l, tokenizer_clip_g, tokenizer_t5],
                        device=device,
                        dtype=weight_dtype,
                        max_sequence_length=args.t5_max_length,
                    )

                # ── Sample timesteps (flow matching) ─────────────────
                # SD3 uses flow matching / rectified flow — timesteps ∈ [0, 1]
                # We use the logit-normal weighting from the SD3 paper to
                # bias sampling toward intermediate noise levels
                bsz = latents.shape[0]
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

                # ── Add noise (flow matching: linear interpolation) ───
                noise = torch.randn_like(latents)
                sigmas = train_scheduler.sigmas[indices].to(latents.device)
                # Reshape sigmas for broadcasting: (B, 1, 1, 1)
                sigmas = sigmas.view(-1, 1, 1, 1).to(weight_dtype)
                noisy_latents = (1 - sigmas) * latents + sigmas * noise

                # ── Forward pass ─────────────────────────────────────
                model_pred = transformer(
                    hidden_states=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_embeds,
                    return_dict=False,
                )[0]

                # ── Flow matching loss ────────────────────────────────
                # Target is the velocity field: v = noise - latent (rectified flow)
                target = noise - latents

                # Optionally weight by loss weighting (from SD3 paper)
                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme=args.weighting_scheme,
                    sigmas=sigmas,
                )
                loss = (weighting.float() * (model_pred.float() - target.float()) ** 2)
                loss = loss.mean()

                # ── Backward ─────────────────────────────────────────
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(lora_params, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # ── End of gradient accumulation step ────────────────────
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    accelerator.log(
                        {
                            "train_loss": train_loss,
                            "lr": lr_scheduler.get_last_lr()[0],
                            "epoch": epoch,
                        },
                        step=global_step,
                    )
                train_loss = 0.0

                # ── Checkpoint ───────────────────────────────────────
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_checkpoint(
                            transformer, accelerator, args.output_dir,
                            global_step, args.checkpoints_total_limit
                        )

                # ── Validation ───────────────────────────────────────
                if (
                    args.validation_steps > 0
                    and global_step % args.validation_steps == 0
                    and accelerator.is_main_process
                ):
                    run_validation(
                        accelerator.unwrap_model(transformer),
                        vae,
                        [text_encoder_l, text_encoder_g, text_encoder_t5],
                        [tokenizer_clip_l, tokenizer_clip_g, tokenizer_t5],
                        scheduler,
                        accelerator,
                        args.output_dir,
                        global_step,
                        args.resolution,
                        weight_dtype,
                        guidance_scale=args.guidance_scale,
                        num_inference_steps=args.num_inference_steps,
                    )

            progress_bar.set_postfix({
                "loss": f"{avg_loss.item():.4f}",
                "step": global_step,
            })

            if global_step >= args.max_train_steps:
                break

    # ── Final save ──────────────────────────────────────────────────
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = os.path.join(args.output_dir, "final_lora_weights")
        os.makedirs(final_dir, exist_ok=True)
        unwrapped = accelerator.unwrap_model(transformer)
        lora_state = get_peft_model_state_dict(unwrapped)

        from safetensors.torch import save_file
        save_file(lora_state, os.path.join(final_dir, "pytorch_lora_weights.safetensors"))
        logger.info(f"Final LoRA weights saved to: {final_dir}")

        # Also save inference config for reproducibility
        import json
        inference_config = {
            "model_id": args.model_id,
            "lora_weights_path": os.path.join(final_dir, "pytorch_lora_weights.safetensors"),
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "resolution": args.resolution,
            "t5_max_length": args.t5_max_length,
            "prompt_format": "FINDINGS: <findings> IMPRESSION: <impression>",
            "conditioning_strategy": "text_only",
            "note": (
                "Text encoders are NOT saved here — they are loaded from model_id. "
                "Only the LoRA adapter weights for the MMDiT transformer are stored."
            ),
        }
        with open(os.path.join(final_dir, "inference_config.json"), "w") as f:
            json.dump(inference_config, f, indent=2)

    accelerator.end_training()
    logger.info("Training complete.")


def save_checkpoint(transformer, accelerator, output_dir, step, total_limit):
    """Save LoRA checkpoint and prune old ones."""
    ckpt_dir = os.path.join(output_dir, f"checkpoint-{step:06d}")
    os.makedirs(ckpt_dir, exist_ok=True)

    unwrapped = accelerator.unwrap_model(transformer)
    lora_state = get_peft_model_state_dict(unwrapped)

    from safetensors.torch import save_file
    save_file(lora_state, os.path.join(ckpt_dir, "pytorch_lora_weights.safetensors"))
    logger.info(f"Saved checkpoint: {ckpt_dir}")

    # Prune old checkpoints
    if total_limit is not None:
        existing = sorted(
            [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[1]),
        )
        if len(existing) > total_limit:
            to_remove = existing[: len(existing) - total_limit]
            for d in to_remove:
                shutil.rmtree(os.path.join(output_dir, d))
                logger.info(f"Pruned checkpoint: {d}")


# ══════════════════════════════════════════════════════════════════════
#  INFERENCE HELPER  (integrate into your DiffusionImageGenerator)
# ══════════════════════════════════════════════════════════════════════

def load_sd35_with_lora(
    model_id: str,
    lora_weights_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> StableDiffusion3Pipeline:
    """
    Load SD3.5 Medium with the CXR LoRA adapter for inference.

    Drop-in usage in DiffusionImageGenerator.load_model():

        from finetune_sd35_lora_cxr import load_sd35_with_lora, build_conditioning_text

        self.pipe = load_sd35_with_lora(
            model_id="stabilityai/stable-diffusion-3.5-medium",
            lora_weights_path="/path/to/final_lora_weights/pytorch_lora_weights.safetensors",
            device=self.device,
        )

    Then generate with:

        prompt = build_conditioning_text(findings=report.findings,
                                         impression=report.impression)
        result = self.pipe(
            prompt=prompt,
            negative_prompt="",
            num_inference_steps=28,
            guidance_scale=7.0,
            height=512, width=512,
            max_sequence_length=512,   # T5 context length
        )
        image = result.images[0].convert("L")   # back to grayscale for CXR
    """
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_id, torch_dtype=dtype
    )
    pipe.load_lora_weights(lora_weights_path)
    pipe = pipe.to(device)

    if device == "cuda":
        pipe.enable_attention_slicing()
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    return pipe


# ══════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning of SD3.5 Medium on MIMIC-CXR"
    )

    # Data
    parser.add_argument("--data_csv", type=str, required=True,
                        help="Path to processed_data.csv")
    parser.add_argument("--image_root", type=str, default=None,
                        help="Optional root to prepend to relative image paths in CSV")
    parser.add_argument("--use_indication", action="store_true",
                        help="Include clinical indication in prompt (default: off)")
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Subsample training set (None = use all ~110K)")
    parser.add_argument("--t5_max_length", type=int, default=512,
                        help="Max T5 token length for text encoding (max=512)")

    # Model
    parser.add_argument("--model_id", type=str,
                        default="stabilityai/stable-diffusion-3.5-medium",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to a checkpoint directory to resume from")

    # LoRA
    parser.add_argument("--lora_rank", type=int, default=32,
                        help="LoRA rank r (32 is a good default; 64 for higher capacity)")
    parser.add_argument("--lora_alpha", type=float, default=None,
                        help="LoRA alpha (default: equal to rank → effective scale=1)")
    parser.add_argument("--lora_dropout", type=float, default=0.0,
                        help="LoRA dropout (small values like 0.05 help with small datasets)")

    # Training
    parser.add_argument("--resolution", type=int, default=768,
                        help="Training image resolution. 768 is recommended for A100 40GB.")
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--max_train_steps", type=int, default=15000,
                        help="Overrides num_train_epochs if set")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Effective batch = train_batch_size × grad_accum × n_gpus")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Reduces VRAM at cost of ~20% slower training")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="cosine",
                        choices=["linear", "cosine", "cosine_with_restarts",
                                 "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Adam
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-4)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)

    # Flow matching weighting (from SD3 paper; logit_normal is recommended)
    parser.add_argument("--weighting_scheme", type=str, default="logit_normal",
                        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"])
    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)
    parser.add_argument("--mode_scale", type=float, default=1.29)

    # Output & checkpointing
    parser.add_argument("--output_dir", type=str, default="./sd35_cxr_lora",
                        help="Directory for LoRA weights, logs, and validation samples")
    parser.add_argument("--checkpointing_steps", type=int, default=1000)
    parser.add_argument("--checkpoints_total_limit", type=int, default=7,
                        help="Keep only the last N checkpoints")
    parser.add_argument("--validation_steps", type=int, default=1000,
                        help="Generate validation images every N steps (0 = disabled)")
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--guidance_scale", type=float, default=7.0)

    # System
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                        choices=["no", "fp16", "bf16"],
                        help="bf16 recommended for A100; fp16 for older GPUs")
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report_to", type=str, default="tensorboard",
                        choices=["tensorboard", "wandb", "none"])

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)