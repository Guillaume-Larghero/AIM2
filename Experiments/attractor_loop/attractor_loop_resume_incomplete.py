#!/usr/bin/env python3
"""
AIM2 — Resume incomplete studies (K=10→100 extension).

Processes a small set of studies that didn't complete in the main long-horizon run.
Uses the same pipeline as attractor_loop_chexgen_long_roshan.py but operates on
a specified list of study IDs only.

USAGE:
    python attractor_loop_resume_incomplete.py \\
        --study_ids 54043642 55303396 53907259 54023727 54151404 54870443 55303241 \\
        --main_dir results/chexgen_main \\
        --output_dir results/chexgen_long
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from glob import glob

import numpy as np
import torch
from PIL import Image

# Add paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
AIM2_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
sys.path.insert(0, AIM2_DIR)
sys.path.insert(0, os.path.join(AIM2_DIR, "ChexGen"))
sys.path.insert(0, CURRENT_DIR)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_models(device="cuda", num_steps=100):
    """Load ChexGen and embedding models."""
    try:
        from attractor_loop_chexgen_roshan import (
            load_medclip, load_maira, load_chexgen,
            embed_image, embed_text, run_maira,
        )
        logger.info(f"Loading models on device: {device}...")
        medclip, _, tokenizer = load_medclip(device)
        maira = load_maira()
        chexgen = load_chexgen(device, num_steps=num_steps)

        return {
            "medclip": medclip,
            "tokenizer": tokenizer,
            "maira": maira,
            "chexgen": chexgen,
            "embed_image": embed_image,
            "embed_text": embed_text,
            "run_maira": run_maira,
            "device": device,
        }
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise


def process_study(sid, main_dir, output_dir, models, K_existing=10, n_iters=100,
                  num_steps=100, cfg_scale=4.0, base_seed=42):
    """Process a single study from K_existing to n_iters."""

    study_output = Path(output_dir) / sid
    study_output.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing study {sid}...")

    try:
        # Copy iter 0..K_existing from main
        main_study = Path(main_dir) / sid
        if not main_study.exists():
            logger.warning(f"  Study {sid} not found in main_dir, skipping")
            return False

        logger.info(f"  Copying iter 0-{K_existing} from chexgen_main...")
        for k in range(K_existing + 1):
            for ext in ["png", "npy", "txt"]:
                src_pattern = str(main_study / f"*iter_{k:03d}*.{ext}")
                for src in glob(src_pattern):
                    dst = study_output / Path(src).name
                    if not dst.exists():
                        import shutil
                        shutil.copy2(src, dst)

        # Also copy anchor embeddings if they exist
        for fname in ["anchor_img_embed.npy", "anchor_text_embed.npy",
                     "gt_image.png", "gt_findings.txt"]:
            src = main_study / fname
            dst = study_output / fname
            if src.exists() and not dst.exists():
                import shutil
                shutil.copy2(src, dst)

        # Check what's already there
        existing_ks = set()
        for k in range(K_existing + 1, n_iters + 1):
            if (study_output / f"gen_iter_{k:03d}.png").exists():
                existing_ks.add(k)

        ks_to_process = [k for k in range(K_existing + 1, n_iters + 1)
                        if k not in existing_ks]

        if not ks_to_process:
            logger.info(f"  Study {sid} already complete (iter {K_existing+1}-{n_iters})")
            return True

        logger.info(f"  Processing iter {ks_to_process[0]}-{ks_to_process[-1]}...")

        # Load current image and embeddings
        current_img = Image.open(study_output / f"gen_iter_{K_existing:03d}.png")
        prev_img_emb = np.load(study_output / f"img_embed_iter_{K_existing:03d}.npy")
        prev_text_emb = np.load(study_output / f"text_embed_iter_{K_existing:03d}.npy")

        # Generate iterations
        for k in ks_to_process:
            logger.debug(f"    Generating iter {k}...")

            # Update seed per iteration
            iter_seed = base_seed + k

            # Run MAIRA (text generation)
            findings_text = models["run_maira"](models["maira"], current_img)

            # Generate image
            gen_img, _ = models["chexgen"].generate(
                findings_text,
                seed=iter_seed,
                num_steps=num_steps,
                cfg_scale=cfg_scale,
            )

            # Embed generated image
            img_emb = models["embed_image"](models["medclip"], gen_img, models["device"])

            # Embed findings
            text_emb = models["embed_text"](models["medclip"], models["tokenizer"],
                                             findings_text, models["device"])

            # Save outputs
            gen_img.save(study_output / f"gen_iter_{k:03d}.png")
            np.save(study_output / f"img_embed_iter_{k:03d}.npy", img_emb.numpy())
            np.save(study_output / f"text_embed_iter_{k:03d}.npy", text_emb.numpy())
            with open(study_output / f"findings_iter_{k:03d}.txt", "w") as f:
                f.write(findings_text)

            # Update for next iteration
            current_img = gen_img
            prev_img_emb = img_emb
            prev_text_emb = text_emb

        logger.info(f"  ✓ Study {sid} complete")
        return True

    except Exception as e:
        logger.error(f"  ✗ Error processing {sid}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study_ids", nargs="+", required=True,
                       help="Study IDs to process")
    parser.add_argument("--main_dir", required=True,
                       help="Path to chexgen_main results")
    parser.add_argument("--output_dir", required=True,
                       help="Output directory for chexgen_long results")
    parser.add_argument("--K_existing", type=int, default=10,
                       help="Existing K value from main_dir (default: 10)")
    parser.add_argument("--n_iters", type=int, default=100,
                       help="Final iteration count (default: 100)")
    parser.add_argument("--num_steps", type=int, default=100,
                       help="Diffusion steps (default: 100)")
    parser.add_argument("--cfg_scale", type=float, default=4.0,
                       help="Classifier-free guidance scale (default: 4.0)")
    parser.add_argument("--base_seed", type=int, default=42,
                       help="Base random seed (default: 42)")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("AIM2 — Resume incomplete studies (K expansion)")
    logger.info("=" * 70)
    logger.info(f"  Study IDs to process: {len(args.study_ids)}")
    logger.info(f"  {', '.join(args.study_ids)}")
    logger.info(f"  K_existing: {args.K_existing}")
    logger.info(f"  n_iters: {args.n_iters}")

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"  Device: {device}")

    # Load models
    models = load_models(device=device, num_steps=args.num_steps)

    # Process studies
    results = {"success": 0, "failed": 0, "already_complete": 0}
    for sid in args.study_ids:
        try:
            success = process_study(
                sid,
                args.main_dir,
                args.output_dir,
                models,
                K_existing=args.K_existing,
                n_iters=args.n_iters,
                num_steps=args.num_steps,
                cfg_scale=args.cfg_scale,
                base_seed=args.base_seed,
            )
            if success:
                results["success"] += 1
            else:
                results["failed"] += 1
        except Exception as e:
            logger.error(f"Failed to process {sid}: {e}")
            results["failed"] += 1

    logger.info("\n" + "=" * 70)
    logger.info("Resume complete")
    logger.info("=" * 70)
    logger.info(f"  Successful: {results['success']}")
    logger.info(f"  Failed: {results['failed']}")


if __name__ == "__main__":
    main()
