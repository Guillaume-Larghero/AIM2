#!/usr/bin/env python3
"""
AIM2 — Block L: Latent space alignment (image-text embedding coherence).

============================================================================
QUESTION
============================================================================
As the attractor loop iterates, do image embeddings and text embeddings
remain aligned (high cosine similarity) or diverge into independent modes?

This tests whether the "elevator-music attractor" is a COUPLED attractor
(both modalities converge to the same degenerate mode) or DECOUPLED
(text and image have separate attractor basins).

============================================================================
HYPOTHESIS
============================================================================
Iteration 0 (ground truth): Real images + real reports are semantically
                           aligned. Expect cos_sim ≈ 0.60–0.75.

Early iterations (1–10):   ChexGen + MAIRA still coherent. Both generate
                           artifacts in tandem. Expect high alignment.

Mid iterations (20–50):    Image and text start degrading. May diverge:
                           • Coupled: both get stuck on same mode
                             → cos_sim increases (reconvergence)
                           • Decoupled: text bias ≠ image artifact
                             → cos_sim decreases (divergence)

Late iterations (50–100):  Steady-state attractor regime. Alignment reveals
                           mode structure:
                           • HIGH cos_sim → coupled attractors
                           • LOW cos_sim → independent attractors

============================================================================
INPUT
============================================================================
  Trajectory directory (chexgen_long with K=100)
  Per-study embeddings: img_embed_iter_NNN.npy, text_embed_iter_NNN.npy

============================================================================
OUTPUT
============================================================================
  block_L_results.json       — cos_sim statistics per iteration
  figures/L_*.png            — cosine similarity trajectories + distributions
  tables/L_*.tsv             — per-study cos_sim values
"""

import argparse
import json
import logging
import os
import sys
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def cosine_similarity(vec1, vec2):
    """Cosine similarity in [0, 1] (1 = identical, 0 = orthogonal)."""
    if vec1.shape != vec2.shape:
        raise ValueError(f"Shape mismatch: {vec1.shape} vs {vec2.shape}")
    # cosine() returns distance in [0, 2], convert to similarity [0, 1]
    return 1.0 - min(cosine(vec1, vec2), 2.0)


def load_trajectory_sids(trajectory_dir, max_studies=None):
    """List study IDs with valid directories."""
    dirs = sorted([
        d.name for d in Path(trajectory_dir).iterdir()
        if d.is_dir() and d.name.isdigit()
    ])
    if max_studies:
        dirs = dirs[:max_studies]
    return dirs


def load_embeddings_at_iter(trajectory_dir, sids, iter_k, verbose=True):
    """Load image and text embeddings for all sids at iteration k.

    Returns:
        img_embs: dict[sid -> np.array (768,)]
        txt_embs: dict[sid -> np.array (768,)]
        valid_sids: list of sids where both embeddings loaded
    """
    img_embs, txt_embs = {}, {}
    valid_sids = []

    for sid in sids:
        img_path = f"{trajectory_dir}/{sid}/img_embed_iter_{iter_k:03d}.npy"
        txt_path = f"{trajectory_dir}/{sid}/text_embed_iter_{iter_k:03d}.npy"

        if not os.path.exists(img_path) or not os.path.exists(txt_path):
            if verbose and sid == sids[0]:
                logger.warning(f"  Missing embeddings at iter {iter_k}")
            continue

        try:
            img_embs[sid] = np.load(img_path, allow_pickle=False)
            txt_embs[sid] = np.load(txt_path, allow_pickle=False)
            valid_sids.append(sid)
        except Exception as e:
            logger.warning(f"  Failed to load {sid} iter {iter_k}: {e}")
            continue

    return img_embs, txt_embs, valid_sids


def compute_cosine_similarities(img_embs, txt_embs, sids):
    """Compute cos_sim for each study."""
    cos_sims = {}
    for sid in sids:
        try:
            cs = cosine_similarity(img_embs[sid], txt_embs[sid])
            cos_sims[sid] = float(cs)
        except Exception as e:
            logger.warning(f"  Failed to compute cos_sim for {sid}: {e}")
            continue
    return cos_sims


def compute_statistics(cos_sims):
    """Compute summary statistics over a collection of cosine similarities."""
    if not cos_sims:
        return None

    vals = np.array(list(cos_sims.values()))
    return {
        "n_valid": int(len(vals)),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "median": float(np.median(vals)),
        "p25": float(np.percentile(vals, 25)),
        "p75": float(np.percentile(vals, 75)),
        "p10": float(np.percentile(vals, 10)),
        "p90": float(np.percentile(vals, 90)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Main analysis
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--trajectory_dir", required=True,
                        help="Path to chexgen_long results")
    parser.add_argument("--out_dir", required=True,
                        help="Output directory for JSON results")
    parser.add_argument("--figures_dir", default=None,
                        help="Output directory for figures (default: same as out_dir)")
    parser.add_argument("--probe_iters", default=None,
                        help="Comma-separated iterations to probe (default: auto-detect)")
    parser.add_argument("--max_studies", type=int, default=-1,
                        help="Limit number of studies (-1 = all)")
    parser.add_argument("--pdf", action="store_true",
                        help="Also save PDF figures")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.figures_dir is None:
        args.figures_dir = args.out_dir
    os.makedirs(args.figures_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("AIM2 Block L — Latent space alignment (image-text embedding coherence)")
    logger.info("=" * 70)
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")

    # ── Load study IDs ───────────────────────────────────────────────────────
    sids = load_trajectory_sids(args.trajectory_dir,
                                 max_studies=args.max_studies if args.max_studies > 0 else None)
    logger.info(f"\n  Loaded {len(sids)} study directories")

    # ── Determine probe iterations ───────────────────────────────────────────
    if args.probe_iters:
        probe_iters = [int(x) for x in args.probe_iters.split(",")]
    else:
        # Auto-detect by finding available embedding files
        any_sid = sids[0]
        emb_files = sorted(glob(os.path.join(args.trajectory_dir, any_sid,
                                              "img_embed_iter_*.npy")))
        if emb_files:
            max_k = max(int(os.path.basename(f).split("_")[-1].split(".")[0])
                       for f in emb_files)
            if max_k <= 11:
                probe_iters = [0, 1, 2, 5, 10]
            else:
                probe_iters = [0, 1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 35, 40, 45,
                              50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
        else:
            probe_iters = [0, 1, 5, 10, 100]
        logger.info(f"  Auto-selected probe_iters: {probe_iters}")

    # ── Main analysis loop ───────────────────────────────────────────────────
    results = {
        "args": vars(args),
        "n_studies": len(sids),
        "probe_iters": probe_iters,
        "cosine_similarities": {},
    }

    per_study_data = {k: {} for k in probe_iters}

    logger.info(f"\n  Computing cosine similarities...")
    for k in probe_iters:
        logger.info(f"\n  Iteration {k}:")
        img_embs, txt_embs, valid_sids = load_embeddings_at_iter(
            args.trajectory_dir, sids, k, verbose=(k == probe_iters[0]))

        if not valid_sids:
            logger.warning(f"    No valid embeddings at iter {k}, skipping")
            continue

        cos_sims = compute_cosine_similarities(img_embs, txt_embs, valid_sids)
        stats = compute_statistics(cos_sims)

        results["cosine_similarities"][k] = {
            "per_study": cos_sims,
            "statistics": stats,
        }
        per_study_data[k] = cos_sims

        if stats:
            logger.info(f"    N={stats['n_valid']}")
            logger.info(f"    mean={stats['mean']:.4f}, std={stats['std']:.4f}")
            logger.info(f"    median={stats['median']:.4f}, "
                       f"P25–P75=[{stats['p25']:.4f}, {stats['p75']:.4f}]")

    # ── Compute per-study trajectories (for studies with all iters) ─────────
    sids_all_iters = set.intersection(*[
        set(results["cosine_similarities"][k]["per_study"].keys())
        for k in probe_iters if k in results["cosine_similarities"]
    ])
    logger.info(f"\n  {len(sids_all_iters)} studies have embeddings at all probe iters")

    results["per_study_trajectories"] = {}
    for sid in sorted(sids_all_iters):
        traj = [results["cosine_similarities"][k]["per_study"][sid] for k in probe_iters]
        results["per_study_trajectories"][sid] = traj

    # ── Save results JSON ────────────────────────────────────────────────────
    out_json = os.path.join(args.out_dir, "block_L_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults → {out_json}")

    # ── Generate figures ─────────────────────────────────────────────────────
    logger.info(f"\nGenerating figures...")
    make_figures(results, args.figures_dir, args.pdf)

    # ── Summary ──────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("BLOCK L SUMMARY")
    logger.info("=" * 70)
    if probe_iters:
        k0_stats = results["cosine_similarities"][probe_iters[0]]["statistics"]
        kmax_stats = results["cosine_similarities"][probe_iters[-1]]["statistics"]
        logger.info(f"  Iteration {probe_iters[0]} (GT):")
        logger.info(f"    mean cos_sim = {k0_stats['mean']:.4f}")
        logger.info(f"    median = {k0_stats['median']:.4f}, "
                   f"P25–P75 = [{k0_stats['p25']:.4f}, {k0_stats['p75']:.4f}]")
        logger.info(f"\n  Iteration {probe_iters[-1]} (final):")
        logger.info(f"    mean cos_sim = {kmax_stats['mean']:.4f}")
        logger.info(f"    median = {kmax_stats['median']:.4f}, "
                   f"P25–P75 = [{kmax_stats['p25']:.4f}, {kmax_stats['p75']:.4f}]")

        delta_mean = kmax_stats['mean'] - k0_stats['mean']
        direction = "↑ (reconvergence)" if delta_mean > 0 else "↓ (divergence)"
        logger.info(f"\n  Change: {direction}")
        logger.info(f"    Δ(mean) = {delta_mean:+.4f}")


def make_figures(results, figures_dir, save_pdf=False):
    """Generate visualization figures."""
    os.makedirs(figures_dir, exist_ok=True)

    probe_iters = results["probe_iters"]
    cos_sims_per_iter = {k: results["cosine_similarities"][k]["statistics"]["mean"]
                         for k in probe_iters if k in results["cosine_similarities"]}

    if not cos_sims_per_iter:
        logger.warning("  No cosine similarity data to plot")
        return

    ks = sorted(cos_sims_per_iter.keys())
    means = [cos_sims_per_iter[k] for k in ks]

    # Extract std and percentiles
    stds = []
    p25s = []
    p75s = []
    for k in ks:
        stats = results["cosine_similarities"][k]["statistics"]
        stds.append(stats["std"])
        p25s.append(stats["p25"])
        p75s.append(stats["p75"])

    stds = np.array(stds)
    p25s = np.array(p25s)
    p75s = np.array(p75s)
    means = np.array(means)

    # ── Figure 1: Mean cosine similarity trajectory ──────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))

    # Error band (std)
    ax.fill_between(ks, means - stds, means + stds, alpha=0.2, color="blue",
                    label="± 1 std")
    # IQR band (P25–P75)
    ax.fill_between(ks, p25s, p75s, alpha=0.15, color="blue",
                    label="P25–P75")
    # Main curve
    ax.plot(ks, means, marker="o", ms=6, lw=2.0, color="darkblue",
           label="Mean cosine similarity")

    ax.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("Iteration $K$", fontsize=11)
    ax.set_ylabel("Cosine similarity (image-text embeddings)", fontsize=11)
    ax.set_title("Image-text embedding alignment across iterations", fontsize=12)
    ax.set_xlim(ks[0] - 2, ks[-1] + 2)

    # Tight y-scale based on data range
    y_min = float(np.nanmin(means) - np.nanmax(stds) * 1.1)
    y_max = float(np.nanmax(means) + np.nanmax(stds) * 1.1)
    ax.set_ylim(max(0, y_min - 0.02), min(1, y_max + 0.02))

    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    png_path = os.path.join(figures_dir, "fig_L_cosim_trajectory.png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    if save_pdf:
        fig.savefig(png_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  → {os.path.basename(png_path)}")

    # ── Figure 2: Distribution at K=0 vs K=max ─────────────────────────────
    if len(ks) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for ax_idx, k in enumerate([ks[0], ks[-1]]):
            ax = axes[ax_idx]
            cos_sims_k = list(results["cosine_similarities"][k]["per_study"].values())

            ax.hist(cos_sims_k, bins=20, alpha=0.7, color="steelblue", edgecolor="black")
            ax.axvline(np.mean(cos_sims_k), color="red", ls="-", lw=2, label="mean")
            ax.axvline(np.median(cos_sims_k), color="orange", ls="--", lw=2, label="median")

            ax.set_xlabel("Cosine similarity", fontsize=10)
            ax.set_ylabel("Count", fontsize=10)
            ax.set_title(f"Iteration {k} (N={len(cos_sims_k)})", fontsize=11)
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")

            # Tight x-scale based on data
            if cos_sims_k:
                x_min = min(cos_sims_k) - 0.05
                x_max = max(cos_sims_k) + 0.05
                ax.set_xlim(max(0, x_min), min(1, x_max))

        fig.tight_layout()
        png_path = os.path.join(figures_dir, "fig_L_distributions.png")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        if save_pdf:
            fig.savefig(png_path.replace(".png", ".pdf"), bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  → {os.path.basename(png_path)}")


if __name__ == "__main__":
    main()
