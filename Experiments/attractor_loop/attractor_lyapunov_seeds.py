#!/usr/bin/env python3
"""
AIM2 ChexGen — Lyapunov B1: seed-replicate experiment

Spawn K=10 trajectories from each of N=5 anchors, varying only the ChexGen
noise seed at each iteration. Used downstream to compute:
  - Finite-time stochastic Lyapunov exponent (trajectory divergence rate)
  - Basin of attraction radius (cluster final endpoints, measure spread)

This script is a thin shim around attractor_loop_chexgen.py's loop logic.
We import its loaders + run_loop and call it K*N times. Each (anchor, seed)
trajectory writes to its own subdirectory, structurally identical to the
main run, so the analysis script reads them with the same code path.

Anchor selection:
  - Pulled from the existing chexgen_main/ run.
  - Strategy: span the iter-0 img_cos range. Pick 5 quantiles {0.1, 0.3,
    0.5, 0.7, 0.9} of the empirical iter-0 cosine distribution. Captures
    high-fidelity (ends in same basin?) vs. low-fidelity (where do they
    drift to?) anchors.
  - Override with --anchor_ids if you want specific studies.

Output layout:
  results/chexgen_lyapunov/
    anchor_<study_id>/
      seed_<j>/                          j ∈ {0..K-1}
        gt_image.png, gt_findings.txt    (only saved once per anchor)
        anchor_img_embed.npy
        anchor_text_embed.npy
        gen_iter_000.png … gen_iter_005.png
        findings_iter_000.txt …
        img_embed_iter_000.npy …
        text_embed_iter_000.npy …
        metrics.json
    summary.json
"""

import argparse
import gc
import json
import logging
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

BASE_DIR = "/n/groups/training/bmif203/AIM2"
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, f"{BASE_DIR}/Experiments/attractor_loop")

# Import loaders + loop from the main script. This means we get the mmcv
# shim, all the dimension-audited code paths, and identical metric formats.
from attractor_loop_chexgen import (
    load_medclip,
    load_maira,
    load_chexgen,
    run_loop,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def select_anchors(main_dir, n_anchors, mode="quantile"):
    """
    Pick anchors from the main run.

    mode='quantile': pick 5 anchors at equally-spaced quantiles of iter-0
    img_cosine. Captures range from worst to best initial fidelity.
    """
    summary_path = os.path.join(main_dir, "summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Main run summary not found: {summary_path}")
    with open(summary_path) as f:
        summary = json.load(f)

    studies = summary["per_study"]
    iter0_cos = np.array(
        [s["image_cosine"][0] for s in studies if len(s.get("image_cosine", [])) > 0]
    )
    sids = np.array(
        [s["study_id"] for s in studies if len(s.get("image_cosine", [])) > 0]
    )

    if mode == "quantile":
        # Wider range than the original 0.1..0.9 — captures more extreme
        # anchors when n_anchors is large.
        q_lo = 0.05 if n_anchors >= 10 else 0.10
        q_hi = 0.95 if n_anchors >= 10 else 0.90
        qs = np.linspace(q_lo, q_hi, n_anchors)
        targets = np.quantile(iter0_cos, qs)
        chosen_idx = []
        used = set()
        for t in targets:
            order = np.argsort(np.abs(iter0_cos - t))
            for i in order:
                if i not in used:
                    chosen_idx.append(i)
                    used.add(i)
                    break
        chosen = [(str(sids[i]), float(iter0_cos[i])) for i in chosen_idx]
    else:
        raise ValueError(f"unknown mode: {mode}")

    logger.info(f"Selected {len(chosen)} anchors:")
    for sid, c in chosen:
        logger.info(f"  study {sid}  iter0_img_cos={c:+.4f}")
    return [sid for sid, _ in chosen]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_anchors",   type=int, default=5)
    p.add_argument("--n_seeds",     type=int, default=10)
    p.add_argument("--n_iters",     type=int, default=5,
                   help="Iterations per trajectory (paper uses 5 for Lyapunov).")
    p.add_argument("--num_steps",   type=int, default=100)
    p.add_argument("--cfg_scale",   type=float, default=4.0)
    p.add_argument("--base_seed",   type=int, default=10000,
                   help="Different from main run's seed=42 to avoid noise reuse.")
    p.add_argument("--data_csv",    type=str,
                   default=f"{BASE_DIR}/processed_data/processed_data.csv")
    p.add_argument("--main_dir",    type=str,
                   default=f"{BASE_DIR}/Experiments/attractor_loop/results/chexgen_main",
                   help="Source of anchor selection (must contain summary.json).")
    p.add_argument("--output_dir",  type=str,
                   default=f"{BASE_DIR}/Experiments/attractor_loop/results/chexgen_lyapunov")
    p.add_argument("--anchor_ids",  nargs="+", default=None,
                   help="Explicit list of study_ids to override quantile selection.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 60)
    logger.info("AIM2 ChexGen Lyapunov Seed-Replicate Experiment")
    logger.info("=" * 60)
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")

    # ── Pick anchors ──────────────────────────────────────────────────────────
    if args.anchor_ids:
        anchors = args.anchor_ids
        logger.info(f"\nUsing user-specified anchors: {anchors}")
    else:
        anchors = select_anchors(args.main_dir, args.n_anchors)

    # ── Read metadata for those anchors ───────────────────────────────────────
    df = pd.read_csv(args.data_csv, low_memory=False)
    df = df[df["split"] == "test"]
    df = df.groupby("study_id", as_index=False).first()
    df["study_id"] = df["study_id"].astype(str)
    df = df.set_index("study_id")

    # ── Load all models once ──────────────────────────────────────────────────
    logger.info("\nLoading models (one-time)...")
    medclip, _, tokenizer = load_medclip(device)
    maira                 = load_maira()
    chexgen               = load_chexgen(device, num_steps=args.num_steps)

    # ── Run K trajectories per anchor ─────────────────────────────────────────
    summary = {"args": vars(args), "anchors": anchors, "trajectories": []}

    n_total = len(anchors) * args.n_seeds
    n_done  = 0
    n_fail  = 0

    for a_idx, sid in enumerate(anchors):
        if sid not in df.index:
            logger.error(f"Anchor {sid} not found in data_csv (expected in test split). Skipping.")
            continue

        row = df.loc[sid]
        img_path    = row["image_path"]
        gt_findings = str(row["findings"]) if pd.notna(row.get("findings")) else ""

        anchor_dir = os.path.join(args.output_dir, f"anchor_{sid}")
        os.makedirs(anchor_dir, exist_ok=True)

        logger.info(f"\n{'━'*60}")
        logger.info(f"Anchor {a_idx+1}/{len(anchors)}: study {sid}")
        logger.info(f"  image: {img_path}")
        logger.info(f"  findings ({len(gt_findings)} chars): "
                    f"\"{gt_findings[:120]}{'...' if len(gt_findings) > 120 else ''}\"")
        logger.info(f"  spawning {args.n_seeds} seed-replicate trajectories...")

        for j in range(args.n_seeds):
            seed_dir = os.path.join(anchor_dir, f"seed_{j:02d}")
            metrics_path = os.path.join(seed_dir, "metrics.json")
            if os.path.exists(metrics_path):
                logger.info(f"  [{sid}] seed {j}: cached, skipping")
                with open(metrics_path) as f:
                    summary["trajectories"].append(json.load(f))
                n_done += 1
                continue

            # Different seed for each (anchor, replicate). Spaced widely.
            traj_seed = args.base_seed + a_idx * 100_000 + j * 10_000

            logger.info(f"  [{sid}] seed {j} (rng={traj_seed}): launching")
            try:
                m = run_loop(
                    gt_image_path = img_path,
                    study_id      = f"anchor_{sid}/seed_{j:02d}",
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
                    seed          = traj_seed,
                )
                m["lyapunov"] = {"anchor_sid": sid, "replicate_idx": j,
                                 "trajectory_seed": traj_seed}
                summary["trajectories"].append(m)
                n_done += 1
            except Exception as e:
                logger.error(f"  [{sid}] seed {j} FAILED: {e}")
                traceback.print_exc()
                n_fail += 1
                gc.collect(); torch.cuda.empty_cache()

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"COMPLETE: {n_done}/{n_total} trajectories OK  ({n_fail} failed)")
    logger.info(f"{'='*60}")

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Wrote summary → {args.output_dir}/summary.json")


if __name__ == "__main__":
    main()