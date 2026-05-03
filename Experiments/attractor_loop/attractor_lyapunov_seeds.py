#!/usr/bin/env python3
"""
AIM2 ChexGen — Lyapunov Long-Horizon: K=100 seed-replicate experiment

Extends the original K=10 Lyapunov experiment to K=100 to support the
paper's Block K story: characterize whether per-anchor divergence
λ̄_a > 0 PERSISTS past K=10 (driving the K=30→K=100 entropy bounce-back
predicted by the two-timescale framework).

Setup:
  20 anchors × 10 seeds = 200 trajectories, each K=100 iterations.

Output layout (NEW directory chexgen_lyapunov_long):
  results/chexgen_lyapunov_long/
    anchor_<study_id>/
      seed_<jj>/                          jj ∈ {0..09}
        gt_image.png, gt_findings.txt
        anchor_img_embed.npy
        anchor_text_embed.npy
        gen_iter_000.png … gen_iter_100.png
        findings_iter_000.txt …
        img_embed_iter_000.npy …
        text_embed_iter_000.npy …
        metrics.json
    summary.json

Resume strategy (per-seed, robust):
  Each seed_<jj>/ is independently resumable:
   • If metrics.json exists AND img_embed files cover requested K → cached, skipped
   • If partial (some iters done) → metrics.json removed, run_loop re-entered;
     run_loop's internal per-iter resume picks up from the last completed iter
   • If empty → run from scratch
  This makes the script SLURM-array friendly and crash-safe.

Anchor selection — IMPORTANT:
  Defaults to quantile-based selection from main_dir/summary.json (matching the
  original Lyapunov experiment's design). If you want to reproduce the
  EXACT 20 anchors used in the K=10 Lyapunov full run, pass --anchor_ids
  with the study IDs from chexgen_lyapunov_full's summary.json.

Compute estimate (K=100 from scratch, no pre-stage):
  ~13 sec per iter on L40S × 100 iters × 200 trajectories = ~72 GPU-hr
  As SLURM array (1 task per anchor, 10 seeds × 100 iters = 1000 calls each):
  ~3.6 hr wallclock per task; with 8 concurrent GPUs across gpu_yu/gpu_quad,
  total wall ≈ 9 hr.
"""

import argparse
import gc
import json
import logging
import os
import sys
import traceback
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch

BASE_DIR = "/n/groups/training/bmif203/AIM2"
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, f"{BASE_DIR}/Experiments/attractor_loop")

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


# ──────────────────────────────────────────────────────────────────────────────
#  Anchor selection (from main_dir's summary.json)
# ──────────────────────────────────────────────────────────────────────────────

def select_anchors(main_dir, n_anchors, mode="quantile"):
    """Pick anchors from the main run by iter-0 image-cosine quantiles."""
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
                    chosen_idx.append(i); used.add(i); break
        chosen = [(str(sids[i]), float(iter0_cos[i])) for i in chosen_idx]
    else:
        raise ValueError(f"unknown mode: {mode}")

    logger.info(f"Selected {len(chosen)} anchors:")
    for sid, c in chosen:
        logger.info(f"  study {sid}  iter0_img_cos={c:+.4f}")
    return [sid for sid, _ in chosen]


def load_anchor_ids_from_existing_run(existing_dir):
    """Recover the anchor list from chexgen_lyapunov_full so we extend
    the SAME anchors (not new quantile picks)."""
    if not os.path.isdir(existing_dir):
        return None
    summary_path = os.path.join(existing_dir, "summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path) as f:
                s = json.load(f)
            if "anchors" in s:
                logger.info(f"Inheriting {len(s['anchors'])} anchors from "
                            f"existing run: {existing_dir}/summary.json")
                return list(s["anchors"])
        except Exception as e:
            logger.warning(f"Could not parse {summary_path}: {e}")
    # Fallback: discover from directory layout
    anchor_dirs = sorted(glob(os.path.join(existing_dir, "anchor_*")))
    if anchor_dirs:
        ids = [os.path.basename(d).replace("anchor_", "") for d in anchor_dirs]
        logger.info(f"Inherited {len(ids)} anchors from directory layout in "
                    f"{existing_dir}")
        return ids
    return None


# ──────────────────────────────────────────────────────────────────────────────
#  Per-seed resume helpers
# ──────────────────────────────────────────────────────────────────────────────

def count_completed_iters(seed_dir):
    """Highest iter for which both img_embed and text_embed files exist.
    Returns -1 if none done."""
    if not os.path.isdir(seed_dir):
        return -1
    img_files = sorted(glob(os.path.join(seed_dir, "img_embed_iter_*.npy")))
    if not img_files:
        return -1
    # Parse iter numbers and require contiguity (no missing iters)
    iters = sorted(int(os.path.basename(f).split("_")[-1].split(".")[0])
                   for f in img_files)
    last = -1
    for k in iters:
        txt_f = os.path.join(seed_dir, f"text_embed_iter_{k:03d}.npy")
        if k == last + 1 and os.path.exists(txt_f):
            last = k
        else:
            break
    return last


def prestage_existing_data(prestage_dir, output_dir, anchors, n_seeds):
    """Optionally copy (or symlink) iter files from chexgen_lyapunov_full into
    chexgen_lyapunov_long so run_loop can resume from K=10 instead of from
    scratch. Removes metrics.json so the wrapper re-enters run_loop.

    Use only if you trust run_loop's per-iter resume to leave existing
    embeddings untouched. Default = OFF (recommend running from scratch
    for robustness; the K=0..10 redundancy is ~10% of compute and
    cross-validates Block B's K=10 numbers)."""
    import shutil
    if not prestage_dir or not os.path.isdir(prestage_dir):
        logger.info(f"  No pre-stage dir at {prestage_dir} (or skipped)")
        return 0
    n_staged = 0
    for sid in anchors:
        for j in range(n_seeds):
            src = os.path.join(prestage_dir, f"anchor_{sid}", f"seed_{j:02d}")
            dst = os.path.join(output_dir, f"anchor_{sid}", f"seed_{j:02d}")
            if not os.path.isdir(src):
                continue
            os.makedirs(dst, exist_ok=True)
            for f in os.listdir(src):
                if f == "metrics.json":
                    continue  # skip — wrapper would short-circuit
                src_f = os.path.join(src, f); dst_f = os.path.join(dst, f)
                if os.path.isfile(src_f) and not os.path.exists(dst_f):
                    shutil.copy2(src_f, dst_f)
            n_staged += 1
    logger.info(f"  Pre-staged {n_staged} seed-trajectories from {prestage_dir}")
    return n_staged


# ──────────────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--n_anchors",   type=int, default=20)
    p.add_argument("--n_seeds",     type=int, default=10)
    p.add_argument("--n_iters",     type=int, default=100,
                   help="Iterations per trajectory (K=100 for paper).")
    p.add_argument("--num_steps",   type=int, default=100)
    p.add_argument("--cfg_scale",   type=float, default=4.0)
    p.add_argument("--base_seed",   type=int, default=20000,
                   help="MUST match chexgen_lyapunov_full base_seed if "
                        "extending those exact trajectories.")
    p.add_argument("--data_csv",    type=str,
                   default=f"{BASE_DIR}/processed_data/processed_data.csv")
    p.add_argument("--main_dir",    type=str,
                   default=f"{BASE_DIR}/Experiments/attractor_loop/results/chexgen_main",
                   help="Source of anchor selection (must contain summary.json).")
    p.add_argument("--output_dir",  type=str,
                   default=f"{BASE_DIR}/Experiments/attractor_loop/results/chexgen_lyapunov_long")
    p.add_argument("--anchor_ids",  nargs="+", default=None,
                   help="Explicit list of study_ids to override quantile selection.")
    p.add_argument("--inherit_anchors_from", type=str, default=None,
                   help="Existing chexgen_lyapunov_full dir; if set, the same "
                        "anchors used there are reused (recommended for "
                        "extending the original 20-anchor set to K=100).")
    p.add_argument("--prestage_from", type=str, default=None,
                   help="Existing chexgen_lyapunov_full dir; if set, copy "
                        "iters 0..K_existing into output_dir before running. "
                        "Saves ~10%% compute IF run_loop respects existing "
                        "iter files. Off by default (run from scratch is "
                        "robust).")
    # SLURM array support
    p.add_argument("--anchor_chunk_idx", type=int, default=-1,
                   help="If ≥0, only process this anchor index from the "
                        "selected list. Used for SLURM-array parallelism "
                        "(set --anchor_chunk_idx=$SLURM_ARRAY_TASK_ID).")
    p.add_argument("--seed_chunk_idx", type=int, default=-1,
                   help="If ≥0, only process this single seed (further "
                        "subdivision below the anchor level).")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 60)
    logger.info("AIM2 ChexGen Lyapunov Long-Horizon (K=100)")
    logger.info("=" * 60)
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")

    # ── Pick anchors ─────────────────────────────────────────────────────────
    if args.anchor_ids:
        anchors = args.anchor_ids
        logger.info(f"\nUsing explicit anchors: {anchors}")
    elif args.inherit_anchors_from:
        anchors = load_anchor_ids_from_existing_run(args.inherit_anchors_from)
        if anchors is None:
            logger.warning(f"Could not inherit anchors from "
                           f"{args.inherit_anchors_from}; falling back to "
                           f"quantile selection from {args.main_dir}")
            anchors = select_anchors(args.main_dir, args.n_anchors)
    else:
        anchors = select_anchors(args.main_dir, args.n_anchors)

    # SLURM array: filter to one anchor per task if requested
    if args.anchor_chunk_idx >= 0:
        if args.anchor_chunk_idx >= len(anchors):
            logger.info(f"anchor_chunk_idx={args.anchor_chunk_idx} >= "
                        f"len(anchors)={len(anchors)}; nothing to do, exiting.")
            return
        anchors = [anchors[args.anchor_chunk_idx]]
        logger.info(f"\nSLURM-array mode: this task processes ONLY anchor "
                    f"index {args.anchor_chunk_idx} → {anchors[0]}")

    # ── Optional pre-stage from existing K=10 run ────────────────────────────
    if args.prestage_from:
        prestage_existing_data(args.prestage_from, args.output_dir,
                                anchors, args.n_seeds)

    # ── Read metadata for those anchors ──────────────────────────────────────
    df = pd.read_csv(args.data_csv, low_memory=False)
    df = df[df["split"] == "test"]
    df = df.groupby("study_id", as_index=False).first()
    df["study_id"] = df["study_id"].astype(str)
    df = df.set_index("study_id")

    # ── Pre-flight: are there any seeds left to run? ─────────────────────────
    pending = []
    for a_idx, sid in enumerate(anchors):
        for j in range(args.n_seeds):
            if args.seed_chunk_idx >= 0 and j != args.seed_chunk_idx:
                continue
            seed_dir = os.path.join(args.output_dir, f"anchor_{sid}",
                                     f"seed_{j:02d}")
            existing_K = count_completed_iters(seed_dir)
            metrics_path = os.path.join(seed_dir, "metrics.json")
            metrics_complete = (
                os.path.exists(metrics_path) and existing_K >= args.n_iters
            )
            if not metrics_complete:
                pending.append((a_idx, sid, j, existing_K))
    logger.info(f"\nPending work: {len(pending)} seed-trajectories "
                f"(of {len(anchors)} × {args.n_seeds} = "
                f"{len(anchors)*args.n_seeds} total)")
    if not pending:
        logger.info("Everything cached — nothing to do.")
        return

    # ── Load all models once ─────────────────────────────────────────────────
    logger.info("\nLoading models (one-time)...")
    medclip, _, tokenizer = load_medclip(device)
    maira                 = load_maira()
    chexgen               = load_chexgen(device, num_steps=args.num_steps)

    # ── Run trajectories ─────────────────────────────────────────────────────
    summary = {"args": vars(args), "anchors": anchors, "trajectories": []}
    n_done = 0; n_fail = 0; n_skip = 0

    for a_idx, sid in enumerate(anchors):
        if sid not in df.index:
            logger.error(f"Anchor {sid} not found in data_csv. Skipping.")
            continue

        row = df.loc[sid]
        img_path    = row["image_path"]
        gt_findings = str(row["findings"]) if pd.notna(row.get("findings")) else ""

        anchor_dir = os.path.join(args.output_dir, f"anchor_{sid}")
        os.makedirs(anchor_dir, exist_ok=True)

        logger.info(f"\n{'━'*60}")
        logger.info(f"Anchor {a_idx+1}/{len(anchors)}: study {sid}")
        logger.info(f"  image: {img_path}")
        logger.info(f"  spawning {args.n_seeds} seed-replicate trajectories "
                    f"to K={args.n_iters}...")

        for j in range(args.n_seeds):
            if args.seed_chunk_idx >= 0 and j != args.seed_chunk_idx:
                continue

            seed_dir = os.path.join(anchor_dir, f"seed_{j:02d}")
            metrics_path = os.path.join(seed_dir, "metrics.json")

            # Smart per-seed resume
            existing_K = count_completed_iters(seed_dir)
            if os.path.exists(metrics_path) and existing_K >= args.n_iters:
                logger.info(f"  [{sid}] seed {j}: cached at K={existing_K}, "
                            f"skipping")
                try:
                    with open(metrics_path) as f:
                        summary["trajectories"].append(json.load(f))
                except Exception:
                    pass
                n_skip += 1; continue

            if existing_K >= 0 and os.path.exists(metrics_path):
                logger.info(f"  [{sid}] seed {j}: existing K={existing_K} "
                            f"< requested {args.n_iters}; removing "
                            f"metrics.json to allow extension")
                try: os.remove(metrics_path)
                except Exception: pass

            # Different seed for each (anchor, replicate). Spaced widely.
            traj_seed = args.base_seed + a_idx * 100_000 + j * 10_000

            extend_msg = (f" (extending from K={existing_K})"
                          if existing_K >= 0 else " (from scratch)")
            logger.info(f"  [{sid}] seed {j} (rng={traj_seed}): launching "
                        f"to K={args.n_iters}{extend_msg}")
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
                                 "trajectory_seed": traj_seed,
                                 "n_iters_completed": args.n_iters,
                                 "extended_from_K": existing_K}
                summary["trajectories"].append(m)
                n_done += 1
            except Exception as e:
                logger.error(f"  [{sid}] seed {j} FAILED: {e}")
                traceback.print_exc()
                n_fail += 1
                gc.collect(); torch.cuda.empty_cache()

    # ── Summary ──────────────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"COMPLETE  done={n_done}  cached={n_skip}  failed={n_fail}")
    logger.info(f"{'='*60}")

    # Per-task summary file (named by chunk if SLURM-array mode)
    suffix = ""
    if args.anchor_chunk_idx >= 0:
        suffix = f"_anchor{args.anchor_chunk_idx:02d}"
    if args.seed_chunk_idx >= 0:
        suffix += f"_seed{args.seed_chunk_idx:02d}"
    summary_path = os.path.join(args.output_dir, f"summary{suffix}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Wrote summary → {summary_path}")


if __name__ == "__main__":
    main()