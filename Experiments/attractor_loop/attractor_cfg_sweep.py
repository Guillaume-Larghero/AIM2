#!/usr/bin/env python3
"""
AIM2 — CFG Sweep Ablation.

Tests whether anchor-information loss in the MAIRA-2 ↔ ChexGen loop is
robust to classifier-free guidance (CFG) strength.

Hypothesis under test:
  Lower CFG → less prior dominance from ChexGen → more anchor information
  preserved per iteration. If MI loss is similar across CFG ∈ {2, 4, 7},
  the loss is a robust property of the coupled map, not a CFG artifact.

Design:
  • 20 anchors selected from main-run summary at quantiles 0.05..0.95 of
    iter-0 image cosine (same selection as Lyapunov v2).
  • Each anchor run at CFG ∈ {2.0, 7.0}. (CFG=4.0 is the main run.)
  • 6 iterations (matching Lyapunov), num_steps=100.
  • Single seed per (anchor, CFG) — we're not testing within-cell variance,
    we're testing across-CFG mean shift.

Total trajectories: 20 anchors × 2 CFG = 40 trajectories
Per-trajectory: ~50 sec → ~33 min compute + 3 min model load → ~40 min.

Output: results/chexgen_cfg_sweep/cfg_<X>/<anchor_sid>/<seed_NNN>/

Reuses load_medclip / load_maira / load_chexgen / run_loop from main script.
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

    Returns: list of dicts [{"study_id": str, "iter0_cos": float}, ...].

    Inlined here (not imported from attractor_lyapunov_seeds) so the schema
    is explicit and decoupled from any future changes there.
    """
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
    p.add_argument("--n_anchors",   type=int, default=20,
                   help="Number of anchors to evaluate per CFG value.")
    p.add_argument("--cfg_values",  type=float, nargs="+", default=[2.0, 7.0],
                   help="CFG values to sweep. CFG=4.0 already in main run, "
                        "so default is just the off-design points.")
    p.add_argument("--n_iters",     type=int, default=5,
                   help="Iterations per trajectory (6 total including iter 0).")
    p.add_argument("--num_steps",   type=int, default=100)
    p.add_argument("--base_seed",   type=int, default=30000,
                   help="Base seed for the sweep. Distinct from main run "
                        "(per-study hash) and Lyapunov (20000).")
    p.add_argument("--main_dir",    type=str,
                   default=f"{BASE_DIR}/Experiments/attractor_loop/results/chexgen_main")
    p.add_argument("--output_dir",  type=str,
                   default=f"{BASE_DIR}/Experiments/attractor_loop/results/chexgen_cfg_sweep")
    p.add_argument("--data_csv",    type=str, default=DATA_CSV)
    p.add_argument("--skip_self_test", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("=" * 60)
    logger.info("AIM2 CFG Sweep Ablation")
    logger.info("=" * 60)
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")

    # ── Select anchors from main-run summary ──────────────────────────────────
    chosen = select_anchors_local(args.main_dir, n_anchors=args.n_anchors)
    logger.info(f"\nSelected {len(chosen)} anchors at quantiles 0.05..0.95")
    for i, c in enumerate(chosen):
        logger.info(f"  [{i:2d}] {c['study_id']}  iter0_img_cos={c['iter0_cos']:+.4f}")

    # ── Load image_path lookup from CSV ───────────────────────────────────────
    df = pd.read_csv(args.data_csv, low_memory=False)
    df = df[df["split"] == "test"]
    if "has_findings" in df.columns:
        mask = df["has_findings"].astype(str).str.lower().isin(["true", "1", "1.0"])
        df = df[mask]
    df = df.groupby("study_id", as_index=False).first()
    df["study_id"] = df["study_id"].astype(str)
    df = df.set_index("study_id")

    # ── Load models once, run all (anchor × CFG) combinations ─────────────────
    logger.info("\nLoading models...")
    medclip, _, tokenizer = load_medclip(device)
    maira                 = load_maira()
    # chexgen wrapper takes num_steps at load time; CFG is per-call
    chexgen               = load_chexgen(device, num_steps=args.num_steps)

    # ── Run sweep ─────────────────────────────────────────────────────────────
    all_results = []
    failed = []

    for cfg_idx, cfg in enumerate(args.cfg_values):
        cfg_dir = os.path.join(args.output_dir, f"cfg_{cfg:.1f}")
        os.makedirs(cfg_dir, exist_ok=True)
        logger.info(f"\n{'='*60}")
        logger.info(f"CFG = {cfg:.1f}  ({cfg_idx+1}/{len(args.cfg_values)})")
        logger.info(f"  output: {cfg_dir}")
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
                logger.warning(f"  [skip] {sid}: empty findings or missing image")
                continue

            # Per-(cfg, anchor) deterministic seed: hash + base + cfg index
            anchor_dir = os.path.join(cfg_dir, sid)
            os.makedirs(anchor_dir, exist_ok=True)

            # Seed is independent across CFG values so we don't share noise
            # schedules between CFG levels — each level gets its own
            # within-trajectory stochasticity.
            seed = args.base_seed + 1000 * cfg_idx + anchor_idx

            metrics_path = os.path.join(anchor_dir, "metrics.json")
            if os.path.exists(metrics_path):
                logger.info(f"  [{sid}] already done at CFG={cfg:.1f}, skipping")
                with open(metrics_path) as f:
                    all_results.append({"cfg": cfg, "anchor": sid,
                                        **json.load(f)})
                continue

            logger.info(f"\n  {'─'*55}")
            logger.info(f"  [{anchor_idx+1}/{len(chosen)}] CFG={cfg:.1f}  anchor={sid}")
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
                    output_dir    = cfg_dir,         # per-cfg subdir
                    num_steps     = args.num_steps,
                    cfg_scale     = cfg,             # the swept variable
                    seed          = seed,
                )
                all_results.append({"cfg": cfg, "anchor": sid, **m})
            except Exception as e:
                logger.error(f"  FAILED CFG={cfg:.1f} anchor={sid}: {e}")
                traceback.print_exc()
                failed.append((cfg, sid))
                gc.collect()
                torch.cuda.empty_cache()

    # ── Aggregate per-CFG summary ─────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"COMPLETE: {len(all_results)} trajectories OK, {len(failed)} failed")
    if failed:
        logger.warning(f"Failed combos: {failed}")

    if all_results:
        # Build per-CFG aggregate stats
        sweep_summary = {"per_trajectory": all_results, "per_cfg": {}}
        K_total = max(len(r.get("image_cosine", [])) for r in all_results)
        for cfg in args.cfg_values:
            cell = [r for r in all_results if abs(r["cfg"] - cfg) < 1e-6]
            if not cell:
                continue
            cfg_stats = {"n_trajectories": len(cell)}
            for k in range(K_total):
                ic = [r["image_cosine"][k] for r in cell
                      if len(r.get("image_cosine", [])) > k]
                tc = [r["text_cosine"][k]  for r in cell
                      if len(r.get("text_cosine",  [])) > k]
                if ic:
                    cfg_stats[f"iter_{k}"] = {
                        "image_cos_mean": float(np.mean(ic)),
                        "image_cos_std":  float(np.std(ic)),
                        "text_cos_mean":  float(np.mean(tc)),
                        "text_cos_std":   float(np.std(tc)),
                    }
            sweep_summary["per_cfg"][f"{cfg:.1f}"] = cfg_stats

        out_path = os.path.join(args.output_dir, "sweep_summary.json")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(sweep_summary, f, indent=2, default=str)
        logger.info(f"\nWrote sweep summary → {out_path}")

        # Quick text comparison
        logger.info("\nPer-CFG image cosine to anchor at iter k:")
        logger.info(f"  Iter   " + "  ".join(f"CFG={c:.1f}" for c in args.cfg_values))
        for k in range(K_total):
            row = f"  {k:>4} "
            for cfg in args.cfg_values:
                key = f"{cfg:.1f}"
                if key in sweep_summary["per_cfg"] and \
                   f"iter_{k}" in sweep_summary["per_cfg"][key]:
                    s = sweep_summary["per_cfg"][key][f"iter_{k}"]
                    row += f"  {s['image_cos_mean']:+.3f}±{s['image_cos_std']:.2f}"
                else:
                    row += "  --"
            logger.info(row)


if __name__ == "__main__":
    main()