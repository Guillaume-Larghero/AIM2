#!/usr/bin/env python3
"""
AIM2 — ChexGen Sampling-Step Sweep Ablation.

Tests whether anchor-information loss is an artifact of insufficient
diffusion sampling steps, or a genuine property of the coupled map.

Hypothesis under test:
  More denoising steps → cleaner image generation → better anchor
  preservation. If MI loss is similar across num_steps ∈ {25, 50, 100},
  the loss is intrinsic to the coupled map and NOT a sampling artifact.

Design:
  • 20 anchors selected from main run summary at quantiles 0.05..0.95
    of iter-0 image cosine (same as Lyapunov v2 and CFG sweep).
  • Each anchor run at num_steps ∈ {25, 50}. (num_steps=100 = main run.)
  • 6 iterations, CFG = 4.0 (the main-run setting).
  • Single seed per (anchor, num_steps).

Total trajectories: 20 anchors × 2 step values = 40 trajectories.

Compute per trajectory scales linearly with num_steps:
  • 25 steps:  ~13 sec/iter × 6 iters = ~80 sec/traj
  • 50 steps:  ~25 sec/iter × 6 iters = ~150 sec/traj
Total compute: 20×80 + 20×150 = 4600 sec ≈ 77 min
Plus model load (~3 min). Wallclock ~85 min.

NOTE: ChexGenWrapper takes num_steps at LOAD time (we verified this in
the original load_chexgen helper). Therefore we must reload chexgen for
each step value. To minimize churn, the script processes one step value
fully before swapping.
"""

import argparse
import gc
import json
import logging
import os
import sys
import traceback
from glob import glob

import numpy as np
import pandas as pd
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from attractor_loop_chexgen import (
    load_medclip,
    load_maira,
    load_chexgen,
    run_loop,
)


def select_anchors_local(main_dir: str, n_anchors: int):
    """Pick n_anchors at evenly-spaced quantiles of iter-0 image cosine.
    Returns: list of dicts [{"study_id": str, "iter0_cos": float}, ...]."""
    summary_path = os.path.join(main_dir, "summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Main run summary not found: {summary_path}")
    with open(summary_path) as f:
        summary = json.load(f)

    studies = summary["per_study"]
    valid = [(s["study_id"], s["image_cosine"][0]) for s in studies
             if len(s.get("image_cosine", [])) > 0]
    sids = np.array([s for s, _ in valid])
    cos  = np.array([c for _, c in valid])

    q_lo = 0.05 if n_anchors >= 10 else 0.10
    q_hi = 0.95 if n_anchors >= 10 else 0.90
    qs = np.linspace(q_lo, q_hi, n_anchors)
    targets = np.quantile(cos, qs)

    chosen_idx = []
    used = set()
    for t in targets:
        order = np.argsort(np.abs(cos - t))
        for i in order:
            if int(i) not in used:
                chosen_idx.append(int(i))
                used.add(int(i))
                break

    return [{"study_id": str(sids[i]), "iter0_cos": float(cos[i])}
            for i in chosen_idx]

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

BASE_DIR = "/n/groups/training/bmif203/AIM2"
DATA_CSV = f"{BASE_DIR}/processed_data/processed_data.csv"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_anchors",   type=int, default=20)
    p.add_argument("--step_values", type=int, nargs="+", default=[25, 50],
                   help="num_steps values to sweep. 100 already in main run.")
    p.add_argument("--n_iters",     type=int, default=5)
    p.add_argument("--cfg_scale",   type=float, default=4.0,
                   help="Hold CFG fixed at main-run value during step sweep.")
    p.add_argument("--base_seed",   type=int, default=40000,
                   help="Distinct from main run, Lyapunov (20000), CFG (30000).")
    p.add_argument("--main_dir",    type=str,
                   default=f"{BASE_DIR}/Experiments/attractor_loop/results/chexgen_main")
    p.add_argument("--output_dir",  type=str,
                   default=f"{BASE_DIR}/Experiments/attractor_loop/results/chexgen_step_sweep")
    p.add_argument("--data_csv",    type=str, default=DATA_CSV)
    p.add_argument("--skip_self_test", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("=" * 60)
    logger.info("AIM2 Step-Count Sweep Ablation")
    logger.info("=" * 60)
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")

    chosen = select_anchors_local(args.main_dir, n_anchors=args.n_anchors)
    logger.info(f"\nSelected {len(chosen)} anchors at quantiles 0.05..0.95")
    for i, c in enumerate(chosen):
        logger.info(f"  [{i:2d}] {c['study_id']}  iter0_img_cos={c['iter0_cos']:+.4f}")

    df = pd.read_csv(args.data_csv, low_memory=False)
    df = df[df["split"] == "test"]
    if "has_findings" in df.columns:
        mask = df["has_findings"].astype(str).str.lower().isin(["true", "1", "1.0"])
        df = df[mask]
    df = df.groupby("study_id", as_index=False).first()
    df["study_id"] = df["study_id"].astype(str)
    df = df.set_index("study_id")

    # ── Load lightweight models once ──────────────────────────────────────────
    logger.info("\nLoading MedCLIP and MAIRA...")
    medclip, _, tokenizer = load_medclip(device)
    maira                 = load_maira()

    all_results = []
    failed = []

    for step_idx, num_steps in enumerate(args.step_values):
        # Reload chexgen for each step value (it takes num_steps at __init__).
        logger.info(f"\nLoading ChexGen with num_steps={num_steps}...")
        chexgen = load_chexgen(device, num_steps=num_steps)

        step_dir = os.path.join(args.output_dir, f"steps_{num_steps:03d}")
        os.makedirs(step_dir, exist_ok=True)
        logger.info(f"\n{'='*60}")
        logger.info(f"num_steps = {num_steps}  ({step_idx+1}/{len(args.step_values)})")
        logger.info(f"  output: {step_dir}")
        logger.info(f"{'='*60}")

        for anchor_idx, c in enumerate(chosen):
            sid = str(c["study_id"])
            if sid not in df.index:
                logger.warning(f"  [skip] {sid}: not in data CSV")
                continue
            row = df.loc[sid]
            img_path    = row["image_path"]
            gt_findings = str(row["findings"]) if pd.notna(row.get("findings")) else ""

            if not gt_findings or not os.path.exists(img_path):
                logger.warning(f"  [skip] {sid}: missing data")
                continue

            anchor_dir = os.path.join(step_dir, sid)
            os.makedirs(anchor_dir, exist_ok=True)

            seed = args.base_seed + 1000 * step_idx + anchor_idx

            metrics_path = os.path.join(anchor_dir, "metrics.json")
            if os.path.exists(metrics_path):
                logger.info(f"  [{sid}] already done at steps={num_steps}, skipping")
                with open(metrics_path) as f:
                    all_results.append({"num_steps": num_steps, "anchor": sid,
                                        **json.load(f)})
                continue

            logger.info(f"\n  {'─'*55}")
            logger.info(f"  [{anchor_idx+1}/{len(chosen)}] steps={num_steps}  anchor={sid}")
            logger.info(f"  iter0_cos quantile target: {c['iter0_cos']:.4f}")
            logger.info(f"  seed: {seed}")

            try:
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
                    output_dir    = step_dir,
                    num_steps     = num_steps,
                    cfg_scale     = args.cfg_scale,
                    seed          = seed,
                )
                all_results.append({"num_steps": num_steps, "anchor": sid, **m})
            except Exception as e:
                logger.error(f"  FAILED steps={num_steps} anchor={sid}: {e}")
                traceback.print_exc()
                failed.append((num_steps, sid))
                gc.collect()
                torch.cuda.empty_cache()

        # Free chexgen before reloading at next step value
        del chexgen
        gc.collect()
        torch.cuda.empty_cache()

    # ── Aggregate ──────────────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"COMPLETE: {len(all_results)} trajectories OK, {len(failed)} failed")

    if all_results:
        sweep_summary = {"per_trajectory": all_results, "per_steps": {}}
        K_total = max(len(r.get("image_cosine", [])) for r in all_results)
        for ns in args.step_values:
            cell = [r for r in all_results if r["num_steps"] == ns]
            if not cell:
                continue
            stats = {"n_trajectories": len(cell)}
            for k in range(K_total):
                ic = [r["image_cosine"][k] for r in cell
                      if len(r.get("image_cosine", [])) > k]
                tc = [r["text_cosine"][k]  for r in cell
                      if len(r.get("text_cosine",  [])) > k]
                if ic:
                    stats[f"iter_{k}"] = {
                        "image_cos_mean": float(np.mean(ic)),
                        "image_cos_std":  float(np.std(ic)),
                        "text_cos_mean":  float(np.mean(tc)),
                        "text_cos_std":   float(np.std(tc)),
                    }
            sweep_summary["per_steps"][f"{ns:03d}"] = stats

        out_path = os.path.join(args.output_dir, "sweep_summary.json")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(sweep_summary, f, indent=2, default=str)
        logger.info(f"\nWrote sweep summary → {out_path}")

        # Compact comparison table
        logger.info("\nPer-step image cosine to anchor at iter k:")
        logger.info(f"  Iter   " + "  ".join(f"steps={ns}" for ns in args.step_values))
        for k in range(K_total):
            row = f"  {k:>4} "
            for ns in args.step_values:
                key = f"{ns:03d}"
                if key in sweep_summary["per_steps"] and \
                   f"iter_{k}" in sweep_summary["per_steps"][key]:
                    s = sweep_summary["per_steps"][key][f"iter_{k}"]
                    row += f"  {s['image_cos_mean']:+.3f}±{s['image_cos_std']:.2f}"
                else:
                    row += "  --"
            logger.info(row)


if __name__ == "__main__":
    main()