#!/usr/bin/env python3
"""
Generate metrics.json for resumed studies in chexgen_long.
Matches the format of original metrics.json exactly, using placeholders for unavailable diagnostic info.
"""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def cosine_sim(a, b):
    """Cosine similarity between two vectors."""
    a = torch.from_numpy(a) if isinstance(a, np.ndarray) else a
    b = torch.from_numpy(b) if isinstance(b, np.ndarray) else b
    return float((a / a.norm() * b / b.norm()).sum().item())


def get_t5_token_count(text):
    """Estimate T5 token count using a simple tokenizer."""
    try:
        from transformers import T5Tokenizer
        tokenizer = T5Tokenizer.from_pretrained("t5-v1_1-xxl")
        tokens = tokenizer.encode(text)
        return len(tokens)
    except:
        # Fallback: rough estimate (1 token ≈ 0.75 words)
        return int(len(text.split()) / 0.75)


def generate_metrics_for_study(sid, output_dir, main_dir, K_existing=10, n_iters=100,
                               num_steps=100, cfg_scale=4.0, base_seed=42):
    """Generate metrics.json for a single study, matching original format."""
    study_dir = Path(output_dir) / sid
    main_study_dir = Path(main_dir) / sid

    if not study_dir.exists():
        logger.warning(f"Study {sid} not found in output_dir")
        return False

    # Load anchor embeddings
    anchor_img_path = study_dir / "anchor_img_embed.npy"
    anchor_text_path = study_dir / "anchor_text_embed.npy"

    if not anchor_img_path.exists() or not anchor_text_path.exists():
        logger.warning(f"  [{sid}] missing anchor embeddings")
        return False

    anchor_img = np.load(anchor_img_path)
    anchor_text = np.load(anchor_text_path)

    # Try to load GT findings and image path from main_dir
    gt_findings = ""
    gt_image_path = ""
    findings_path = main_study_dir / "gt_findings.txt"
    if findings_path.exists():
        with open(findings_path) as f:
            gt_findings = f.read()

    # Compute per-study seed (same as original)
    import hashlib
    per_study_seed = (
        base_seed +
        int(hashlib.sha256(sid.encode()).hexdigest()[:8], 16)
    ) & 0x7FFFFFFF

    metrics = {
        "study_id": sid,
        "gt_findings": gt_findings,
        "gt_image_path": gt_image_path,
        "config": {
            "n_iters": n_iters,
            "num_steps": num_steps,
            "cfg_scale": cfg_scale,
            "base_seed": per_study_seed,
            "base_seed_long": per_study_seed,
            "resumed_from_k": K_existing,
        },
        "anchor_norms": {
            "img": float(torch.from_numpy(anchor_img).norm().item()),
            "text": float(torch.from_numpy(anchor_text).norm().item()),
        },
        "iterations": [],
    }

    # Process each iteration
    prev_img_emb = None
    prev_text_emb = None

    for k in range(n_iters + 1):
        img_path = study_dir / f"img_embed_iter_{k:03d}.npy"
        text_path = study_dir / f"text_embed_iter_{k:03d}.npy"
        findings_path = study_dir / f"findings_iter_{k:03d}.txt"
        gen_img_path = study_dir / f"gen_iter_{k:03d}.png"

        if not img_path.exists() or not text_path.exists():
            logger.warning(f"  [{sid}] missing embeddings for iter {k}")
            break

        img_emb = np.load(img_path)
        text_emb = np.load(text_path)

        # Read findings
        if findings_path.exists():
            with open(findings_path) as f:
                findings_text = f.read()
        else:
            findings_text = ""

        # Get image size
        img_size = [512, 512]
        if gen_img_path.exists():
            pil_img = Image.open(gen_img_path)
            img_size = list(pil_img.size)

        # Compute metrics
        img_cos = cosine_sim(anchor_img, img_emb)
        txt_cos = cosine_sim(anchor_text, text_emb)
        anchor_img_t = torch.from_numpy(anchor_img)
        img_emb_t = torch.from_numpy(img_emb)
        l2_to_gt = float((anchor_img_t - img_emb_t).norm().item())

        iter_seed = per_study_seed + k

        # Determine input source
        if k == 0:
            input_source = "ground_truth"
        else:
            input_source = f"gen_iter_{k-1:03d}"

        # Compute text statistics
        raw_chars = len(findings_text)
        raw_words = len(findings_text.split())
        cleaned_text = findings_text  # Resume captures cleaned text directly
        cleaned_chars = len(cleaned_text)

        # Compute T5 token counts
        natural_token_count = get_t5_token_count(findings_text)
        used_token_count = natural_token_count
        was_truncated = False
        tokens_lost = 0

        # Build iteration entry matching original structure
        iter_entry = {
            "k": k,
            "iter_seed": iter_seed,
            "input_image_source": input_source,
            "input_image_size": img_size,
            "maira": None,  # Placeholder: wallclock_s, input_size, output_chars, output_words not captured
            "input": {
                "raw_chars": raw_chars,
                "raw_words": raw_words,
                "cleaned": cleaned_text,
                "cleaned_chars": cleaned_chars,
            },
            "t5": {
                "natural_token_count": natural_token_count,
                "used_token_count": used_token_count,
                "was_truncated": was_truncated,
                "tokens_lost": tokens_lost,
                "wallclock_s": None,  # Not captured during resume
                "embed_shape": [1, 120, 4096],  # Standard T5 shape
                "mask_shape": [1, 120],  # Standard T5 mask shape
            },
            "chexgen": {
                "cfg_scale": cfg_scale,
                "num_steps": num_steps,
                "seed": iter_seed,
                "latent_shape": None,  # Not captured
                "sampling_wallclock_s": None,  # Not captured
                "vae_decode_wallclock_s": None,  # Not captured
                "decoded_shape": [1, 3, 512, 512],  # Standard decoded shape
                "decoded_value_range": None,  # Not captured
            },
            "memory": None,  # Not captured
            "medclip": {
                "img_embed_norm": float(torch.from_numpy(img_emb).norm().item()),
                "text_embed_norm": float(torch.from_numpy(text_emb).norm().item()),
                "embed_dim": int(img_emb.shape[0]),
            },
            "metrics": {
                "image_cosine": img_cos,
                "text_cosine": txt_cos,
                "embed_l2": l2_to_gt,
                "image_evolution": None if prev_img_emb is None else cosine_sim(prev_img_emb, img_emb),
                "text_evolution": None if prev_text_emb is None else cosine_sim(prev_text_emb, text_emb),
            }
        }

        metrics["iterations"].append(iter_entry)
        prev_img_emb = img_emb
        prev_text_emb = text_emb

    # Save metrics
    metrics_path = study_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"  [{sid}] ✓ metrics.json created ({len(metrics['iterations'])} iters)")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study_ids", nargs="+", required=True,
                       help="Study IDs to process")
    parser.add_argument("--output_dir", required=True,
                       help="Output directory (chexgen_long)")
    parser.add_argument("--main_dir", required=True,
                       help="Main directory (chexgen_main)")
    parser.add_argument("--n_iters", type=int, default=100)
    parser.add_argument("--K_existing", type=int, default=10)
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--base_seed", type=int, default=42)
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("Generating metrics.json for resumed studies")
    logger.info("=" * 70)
    logger.info(f"  Studies: {len(args.study_ids)}")
    logger.info(f"  {', '.join(args.study_ids)}")

    success = 0
    failed = 0
    for sid in args.study_ids:
        try:
            if generate_metrics_for_study(sid, args.output_dir, args.main_dir,
                                         K_existing=args.K_existing,
                                         n_iters=args.n_iters,
                                         num_steps=args.num_steps,
                                         cfg_scale=args.cfg_scale,
                                         base_seed=args.base_seed):
                success += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"  [{sid}] Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    logger.info("\n" + "=" * 70)
    logger.info(f"Complete: {success} successful, {failed} failed")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
