#!/usr/bin/env python3
"""
train_lora.py
-------------
LoRA fine-tuning of SD 1.5 on MIMIC-CXR with Long-CLIP as text encoder (248 tokens).

Based on diffusers/examples/text_to_image/train_text_to_image_lora.py pattern.

Usage:
  accelerate launch DIFFUSION/train_lora.py \
    --manifest        dataset/mimic_cxr_manifest.csv \
    --image_root      dataset/physionet.org/files/mimic-cxr-jpg/2.1.0 \
    --report_root     dataset/physionet.org/files/mimic-cxr/2.0.0 \
    --output_dir      DIFFUSION/outputs/lora_longclip \
    --mixed_precision fp16
"""

import argparse
import logging
import math
import os
import re
import types

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler

from peft import LoraConfig, get_peft_model, get_peft_model_state_dict

from transformers import CLIPConfig, CLIPModel, CLIPProcessor

logger = get_logger(__name__, log_level="INFO")

# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    cli = parser.parse_args()

    with open(cli.config) as f:
        cfg = yaml.safe_load(f)

    # Convert dict → SimpleNamespace for attribute-style access
    args = types.SimpleNamespace(**cfg)
    return args


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

_SECTION_RE = re.compile(r"^\s*([A-Z][A-Z /]+):\s*", re.MULTILINE)

def _parse_report(text):
    headers = [(m.group(1).strip(), m.start(), m.end())
               for m in _SECTION_RE.finditer(text)]
    if not headers:
        return {"FULL": text.strip()}
    sections = {}
    for i, (name, _, start) in enumerate(headers):
        end = headers[i + 1][1] if i + 1 < len(headers) else len(text)
        sections[name] = text[start:end].strip()
    return sections


class MIMICCXRDataset(Dataset):
    def __init__(self, manifest_csv, image_root, report_root, tokenizer,
                 resolution=512, split="train", report_section="both", max_train_samples=None):
        df = pd.read_csv(manifest_csv)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.report_root = report_root
        self.tokenizer = tokenizer
        self.report_section = report_section
        self.image_root = image_root  # needed before the existence check below

        # Filter to rows whose image file actually exists on disk, stopping
        # as soon as max_train_samples are found (fast early-exit for debug runs).
        if max_train_samples is not None:
            logger.info(f"Scanning for {max_train_samples} downloaded images...")
            found = []
            for _, row in self.df.iterrows():
                if os.path.exists(self._image_path(row)):
                    found.append(row)
                    if len(found) == max_train_samples:
                        break
            self.df = pd.DataFrame(found).reset_index(drop=True)
            logger.info(f"Using {len(self.df)} images")

        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def _image_path(self, row):
        pid = str(int(row["subject_id"]))
        sid = int(row["study_id"])
        return os.path.join(
            self.image_root, "files",
            f"p{pid[:2]}", f"p{pid}", f"s{sid}", f"{row['dicom_id']}.jpg"
        )

    def _report_path(self, row):
        pid = str(int(row["subject_id"]))
        sid = int(row["study_id"])
        return os.path.join(
            self.report_root, "files",
            f"p{pid[:2]}", f"p{pid}", f"s{sid}.txt"
        )

    def _get_caption(self, row):
        path = self._report_path(row)
        if not os.path.exists(path):
            return "Chest radiograph."
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        sections = _parse_report(raw)

        if self.report_section == "both":
            parts = []
            if "FINDINGS" in sections:
                parts.append(sections["FINDINGS"])
            if "IMPRESSION" in sections:
                parts.append(sections["IMPRESSION"])
            return " ".join(parts) if parts else raw.strip()
        elif self.report_section == "findings":
            return sections.get("FINDINGS", sections.get("IMPRESSION", raw.strip()))
        else:  # impression
            return sections.get("IMPRESSION", sections.get("FINDINGS", raw.strip()))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Image: grayscale MIMIC → RGB for SD
        image = Image.open(self._image_path(row)).convert("RGB")
        pixel_values = self.transform(image)

        # Caption → tokenize
        caption = self._get_caption(row)
        input_ids = self.tokenizer(
            caption,
            max_length=248,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]

        return {"pixel_values": pixel_values, "input_ids": input_ids}


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "input_ids":    torch.stack([b["input_ids"]    for b in batch]),
    }


# --------------------------------------------------------------------------- #
# Min-SNR loss weighting
# --------------------------------------------------------------------------- #

def compute_snr(noise_scheduler, timesteps):
    """Compute SNR for each timestep (for Min-SNR weighting)."""
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas = alphas_cumprod[timesteps] ** 0.5
    sqrt_one_minus = (1 - alphas_cumprod[timesteps]) ** 0.5
    snr = (sqrt_alphas / sqrt_one_minus) ** 2
    return snr


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    args = parse_args()

    # Set the default CUDA device from config before Accelerate initialises.
    device_cfg = str(getattr(args, "device", "cuda"))
    if device_cfg.startswith("cuda"):
        torch.cuda.set_device(device_cfg)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=os.path.join(args.output_dir, args.logging_dir),
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Load Long-CLIP text encoder + tokenizer
    # ------------------------------------------------------------------ #
    device = accelerator.device
    logger.info(f"Device: {device}")
    dtype = torch.float16 if args.mixed_precision == "fp16" else (
            torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32)
    logger.info(f"Loading Long-CLIP from {args.longclip_model}")

    clip_config = CLIPConfig.from_pretrained(args.longclip_model)
    clip_config.text_config.max_position_embeddings = 248
    clip_model = CLIPModel.from_pretrained(
        args.longclip_model, torch_dtype=dtype, config=clip_config
    )
    clip_processor = CLIPProcessor.from_pretrained(
        args.longclip_model, padding="max_length", max_length=248
    )
    tokenizer    = clip_processor.tokenizer
    text_encoder = clip_model.text_model

    # ------------------------------------------------------------------ #
    # Load SD 1.5 components
    # ------------------------------------------------------------------ #
    logger.info(f"Loading SD 1.5 from {args.pretrained_model}")
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model, subfolder="scheduler")
    vae  = AutoencoderKL.from_pretrained(args.pretrained_model, subfolder="vae",  revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model, subfolder="unet", revision=args.revision)

    # Freeze VAE and text encoder — only UNet LoRA is trained
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # ------------------------------------------------------------------ #
    # Apply LoRA to UNet cross-attention layers
    # ------------------------------------------------------------------ #
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        bias="none",
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # ------------------------------------------------------------------ #
    # Dataset + dataloader
    # ------------------------------------------------------------------ #
    train_dataset = MIMICCXRDataset(
        manifest_csv=args.manifest,
        image_root=args.image_root,
        report_root=args.report_root,
        tokenizer=tokenizer,
        resolution=args.resolution,
        split="train",
        report_section=args.report_section,
        max_train_samples=getattr(args, "max_train_samples", None),
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    logger.info(f"Training samples: {len(train_dataset)}")

    # ------------------------------------------------------------------ #
    # Optimizer + scheduler
    # ------------------------------------------------------------------ #
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, unet.parameters()),
        lr=float(args.learning_rate),
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # ------------------------------------------------------------------ #
    # Accelerate prepare
    # ------------------------------------------------------------------ #
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    vae          = vae.to(device, dtype=dtype)
    text_encoder = text_encoder.to(device, dtype=dtype)

    if accelerator.is_main_process:
        accelerator.init_trackers("lora_mimic_cxr")

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    global_step = 0
    progress_bar = tqdm(
        range(args.max_train_steps),
        disable=not accelerator.is_local_main_process,
        desc="Steps",
    )

    # Resume from checkpoint
    if args.resume_from_checkpoint:
        accelerator.load_state(args.resume_from_checkpoint)
        global_step = int(os.path.basename(args.resume_from_checkpoint).split("-")[1])
        progress_bar.update(global_step)
        logger.info(f"Resumed from checkpoint: {args.resume_from_checkpoint} (step {global_step})")

    for epoch in range(math.ceil(args.max_train_steps / num_update_steps_per_epoch)):
        unet.train()
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):

                # Encode images to latents
                latents = vae.encode(
                    batch["pixel_values"].to(dtype=dtype)
                ).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (bsz,), device=latents.device
                ).long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Text conditioning via Long-CLIP
                encoder_hidden_states = text_encoder(
                    batch["input_ids"].to(device)
                )[0]  # [batch, 248, hidden_size]

                # Predict noise
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Loss (with optional Min-SNR weighting)
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type: {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                loss = loss.mean(dim=list(range(1, len(loss.shape))))  # [bsz]

                if args.snr_gamma > 0:
                    snr = compute_snr(noise_scheduler, timesteps)
                    msnr_weights = torch.clamp(snr, max=args.snr_gamma) / snr
                    loss = (loss * msnr_weights).mean()
                else:
                    loss = loss.mean()

                avg_loss = accelerator.gather(loss.repeat(bsz)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                    ckpt_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(ckpt_path)
                    logger.info(f"Saved checkpoint: {ckpt_path}")

                if global_step >= args.max_train_steps:
                    break

        if global_step >= args.max_train_steps:
            break

    # ------------------------------------------------------------------ #
    # Save final LoRA weights
    # ------------------------------------------------------------------ #
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet_unwrapped = accelerator.unwrap_model(unet)
        lora_state_dict = get_peft_model_state_dict(unet_unwrapped)

        save_path = os.path.join(args.output_dir, "pytorch_lora_weights.safetensors")
        from safetensors.torch import save_file
        save_file(lora_state_dict, save_path)
        logger.info(f"LoRA weights saved to {save_path}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
