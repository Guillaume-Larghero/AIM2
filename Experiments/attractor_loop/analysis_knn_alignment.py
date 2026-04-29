#!/usr/bin/env python3
"""
AIM2 — Geometric Trajectory Analysis (Block I).

Posthoc analysis of three complementary geometric properties along
trajectories:

  1. kNN distance to training manifold per iteration
     "Are trajectories drifting INTO or OUT OF the training distribution
      over time?"

  2. Mean intra-pairwise distance (MIPD) per iteration
     "Does the cohort spread out or contract over iterations?"

  3. Displacement alignment per iteration step
     "Are all trajectories drifting in the SAME direction (coherent
      collapse) or random directions (diffuse drift)?"

These complement Block A (per-trajectory drift) by giving us a population-
level geometric picture of the dynamics.

OUTPUTS:
  analysis/I_geometry/
  ├── I_geometry.npz                — raw arrays, all metrics, both modalities
  ├── I_per_iter_metrics.csv        — long-format table for plotting
  └── I_geometry.pdf                — 6-panel figure (3 metrics × 2 modalities)

USAGE:
  python analysis_knn_alignment.py \\
      --main_dir   .../results/chexgen_main \\
      --ref_dir    .../reference_embeddings \\
      --out_dir    .../analysis \\
      --k_nn 10
"""

import argparse
import json
import logging
import os
import sys
from glob import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  Metric helpers
# ══════════════════════════════════════════════════════════════════════════════

def fit_knn(train_embs: np.ndarray, k: int = 10):
    """Fit a kNN index on training embeddings using cosine metric.

    sklearn's NearestNeighbors with metric='cosine' computes
    1 - cos(a, b), so distances live in [0, 2].
    """
    knn = NearestNeighbors(n_neighbors=k, metric="cosine",
                            algorithm="brute", n_jobs=-1)
    knn.fit(train_embs)
    return knn


def knn_distance_to_train(test_embs: np.ndarray, knn) -> np.ndarray:
    """Return the mean cosine distance to the k nearest training points,
    per test embedding. Output: (N,) vector."""
    dists, _ = knn.kneighbors(test_embs)
    return dists.mean(axis=1)  # (N,)


def mean_intra_pairwise_distance(embs: np.ndarray, max_n: int = 2000,
                                  rng: np.random.Generator = None):
    """Mean cosine distance within a cohort. Subsamples for speed."""
    if rng is None:
        rng = np.random.default_rng(42)
    if len(embs) > max_n:
        idx = rng.choice(len(embs), max_n, replace=False)
        E = embs[idx]
    else:
        E = embs
    if len(E) < 2:
        return float("nan"), float("nan")
    E_n = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    sim = E_n @ E_n.T
    iu = np.triu_indices(len(E), k=1)
    d = 1.0 - sim[iu]
    return float(d.mean()), float(d.std())


def displacement_alignment(embs_t0: np.ndarray, embs_t1: np.ndarray,
                            max_n: int = 1000,
                            rng: np.random.Generator = None):
    """Mean pairwise cosine of displacement vectors Δ_i = e_i^{t+1} − e_i^{t}.

    Interpretation:
      → 1.0  : all trajectories drift in the same direction (coherent collapse)
      → 0.0  : displacement directions are random (diffuse drift)
      < 0.0  : displacements antagonistic (unlikely in our setting)
    """
    if rng is None:
        rng = np.random.default_rng(42)
    disp = embs_t1 - embs_t0  # (N, D)
    norms = np.linalg.norm(disp, axis=1, keepdims=True) + 1e-12
    disp_n = disp / norms
    if len(disp_n) > max_n:
        idx = rng.choice(len(disp_n), max_n, replace=False)
        disp_sub = disp_n[idx]
    else:
        disp_sub = disp_n
    sim = disp_sub @ disp_sub.T
    iu = np.triu_indices(len(disp_sub), k=1)
    pairwise_sim = sim[iu]
    return {
        "mean_alignment":   float(pairwise_sim.mean()),
        "std_alignment":    float(pairwise_sim.std()),
        "median_alignment": float(np.median(pairwise_sim)),
        "mean_step_size":   float(norms.mean()),
        "std_step_size":    float(norms.std()),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Loaders (mirrors analysis_training_clusters.py — kept inline for portability)
# ══════════════════════════════════════════════════════════════════════════════

def load_trajectories(main_dir: str):
    metric_files = sorted(glob(os.path.join(main_dir, "*", "metrics.json")))
    logger.info(f"Found {len(metric_files)} trajectories in {main_dir}")

    Z_img_list, Z_txt_list, study_ids = [], [], []
    K_canonical = None

    for f in metric_files:
        sdir = os.path.dirname(f)
        sid = os.path.basename(sdir)
        img_files = sorted(glob(os.path.join(sdir, "img_embed_iter_*.npy")))
        txt_files = sorted(glob(os.path.join(sdir, "text_embed_iter_*.npy")))
        if not img_files or len(img_files) != len(txt_files):
            continue
        try:
            z_img = np.stack([np.load(p) for p in img_files])
            z_txt = np.stack([np.load(p) for p in txt_files])
        except Exception:
            continue
        if K_canonical is None:
            K_canonical = z_img.shape[0]
        if z_img.shape[0] != K_canonical or z_txt.shape[0] != K_canonical:
            continue
        Z_img_list.append(z_img); Z_txt_list.append(z_txt); study_ids.append(sid)

    if not Z_img_list:
        logger.error(f"No complete trajectories loaded from {main_dir}")
        sys.exit(1)

    Z_img = np.stack(Z_img_list); Z_txt = np.stack(Z_txt_list)
    logger.info(f"  Loaded {len(study_ids)} trajectories, K={K_canonical}")
    return Z_img, Z_txt, study_ids


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--main_dir", required=True)
    p.add_argument("--ref_dir",  required=True)
    p.add_argument("--out_dir",  required=True)
    p.add_argument("--k_nn",     type=int, default=10,
                   help="k for kNN distance to training manifold.")
    p.add_argument("--mipd_max", type=int, default=2000,
                   help="Subsample size for MIPD (O(N²) memory).")
    p.add_argument("--align_max", type=int, default=1000,
                   help="Subsample size for displacement alignment.")
    return p.parse_args()


def run_modality(modality: str, ref_embs: np.ndarray,
                 trajectories: np.ndarray, args, rng):
    N, K, D = trajectories.shape
    logger.info(f"\n[{modality.upper()}] {N} trajectories × {K} iters × {D}-d")

    # ── Fit kNN on training embeddings ────────────────────────────────────────
    logger.info(f"  Fitting kNN(k={args.k_nn}) on {len(ref_embs)} ref embeddings...")
    knn = fit_knn(ref_embs, k=args.k_nn)

    # ── Per-iter metrics ──────────────────────────────────────────────────────
    knn_mean = np.zeros(K)
    knn_std  = np.zeros(K)
    mipd_mean = np.zeros(K)
    mipd_std  = np.zeros(K)
    align_mean   = np.full(K, np.nan)  # K-1 valid values; pad iter-0 with NaN
    align_std    = np.full(K, np.nan)
    align_median = np.full(K, np.nan)
    step_size    = np.full(K, np.nan)

    for k in range(K):
        d_knn = knn_distance_to_train(trajectories[:, k, :], knn)
        knn_mean[k] = float(d_knn.mean())
        knn_std[k]  = float(d_knn.std())

        m, s = mean_intra_pairwise_distance(trajectories[:, k, :],
                                              max_n=args.mipd_max, rng=rng)
        mipd_mean[k] = m
        mipd_std[k]  = s

        if k > 0:
            a = displacement_alignment(trajectories[:, k - 1, :],
                                        trajectories[:, k, :],
                                        max_n=args.align_max, rng=rng)
            align_mean[k]   = a["mean_alignment"]
            align_std[k]    = a["std_alignment"]
            align_median[k] = a["median_alignment"]
            step_size[k]    = a["mean_step_size"]

        logger.info(f"    iter {k:2d}: "
                    f"kNN={knn_mean[k]:.3f}±{knn_std[k]:.3f}  "
                    f"MIPD={mipd_mean[k]:.3f}±{mipd_std[k]:.3f}  "
                    f"align={align_mean[k] if k>0 else float('nan'):.3f}")

    return {
        "knn_mean":     knn_mean,
        "knn_std":      knn_std,
        "mipd_mean":    mipd_mean,
        "mipd_std":     mipd_std,
        "align_mean":   align_mean,
        "align_std":    align_std,
        "align_median": align_median,
        "step_size":    step_size,
    }


def main():
    args = parse_args()
    rng = np.random.default_rng(42)

    cache_dir = os.path.join(args.out_dir, "cache")
    fig_dir   = os.path.join(args.out_dir, "figures")
    table_dir = os.path.join(args.out_dir, "tables")
    for d in (cache_dir, fig_dir, table_dir):
        os.makedirs(d, exist_ok=True)

    logger.info("=" * 60)
    logger.info("AIM2 Geometric Trajectory Analysis (Block I)")
    logger.info("=" * 60)
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")

    # Reference embeddings
    ref_img = np.load(os.path.join(args.ref_dir, "img_embeds.npy"))
    ref_txt = np.load(os.path.join(args.ref_dir, "txt_embeds.npy"))
    logger.info(f"\nReference: img {ref_img.shape}  txt {ref_txt.shape}")

    # Trajectories
    Z_img, Z_txt, study_ids = load_trajectories(args.main_dir)

    # Run both modalities
    results = {
        "image": run_modality("image", ref_img, Z_img, args, rng),
        "text":  run_modality("text",  ref_txt, Z_txt, args, rng),
    }

    K = Z_img.shape[1]
    iters = np.arange(K)

    # ── Save raw arrays ───────────────────────────────────────────────────────
    np.savez(os.path.join(cache_dir, "I_geometry.npz"),
             study_ids=np.array(study_ids),
             K=K, k_nn=args.k_nn,
             **{f"{m}_{k}": v for m, d in results.items() for k, v in d.items()})

    # ── Long-format CSV ───────────────────────────────────────────────────────
    rows = []
    for modality, res in results.items():
        for k in range(K):
            rows.append({
                "modality":     modality,
                "iter":         k,
                "knn_mean":     float(res["knn_mean"][k]),
                "knn_std":      float(res["knn_std"][k]),
                "mipd_mean":    float(res["mipd_mean"][k]),
                "mipd_std":     float(res["mipd_std"][k]),
                "align_mean":   float(res["align_mean"][k]),
                "align_std":    float(res["align_std"][k]),
                "align_median": float(res["align_median"][k]),
                "step_size":    float(res["step_size"][k]),
            })
    pd.DataFrame(rows).to_csv(os.path.join(table_dir, "I_per_iter_metrics.csv"),
                                index=False)
    logger.info(f"\nWrote {table_dir}/I_per_iter_metrics.csv")

    # ── Figure I: 6-panel grid (3 metrics × 2 modalities) ─────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex=True)

    for col, modality in enumerate(("image", "text")):
        res = results[modality]
        color = "C0" if modality == "image" else "C1"

        # Row 0: kNN distance to training
        ax = axes[0, col]
        ax.plot(iters, res["knn_mean"], "o-", color=color, lw=2, ms=5)
        ax.fill_between(iters,
                        res["knn_mean"] - res["knn_std"],
                        res["knn_mean"] + res["knn_std"],
                        color=color, alpha=0.18)
        ax.set_ylabel(f"Mean cosine dist\nto training k={args.k_nn}-NN")
        ax.set_title(f"{modality.title()} side")
        ax.grid(True, alpha=0.3)

        # Row 1: MIPD
        ax = axes[1, col]
        ax.plot(iters, res["mipd_mean"], "s-", color=color, lw=2, ms=5)
        ax.fill_between(iters,
                        res["mipd_mean"] - res["mipd_std"],
                        res["mipd_mean"] + res["mipd_std"],
                        color=color, alpha=0.18)
        ax.set_ylabel("Mean intra-pairwise\ncosine distance")
        ax.grid(True, alpha=0.3)

        # Row 2: displacement alignment
        ax = axes[2, col]
        valid = ~np.isnan(res["align_mean"])
        ax.plot(iters[valid], res["align_mean"][valid], "^-",
                color=color, lw=2, ms=5)
        ax.fill_between(iters[valid],
                        res["align_mean"][valid] - res["align_std"][valid],
                        res["align_mean"][valid] + res["align_std"][valid],
                        color=color, alpha=0.18)
        ax.axhline(0, color="k", lw=0.6, alpha=0.7)
        ax.set_ylabel("Displacement\nalignment (cos)")
        ax.set_xlabel("Iteration $k$")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.0)

    plt.suptitle("Block I — Geometric trajectory analysis "
                 "(kNN to train manifold, intra-cohort spread, drift coherence)",
                 fontsize=11)
    plt.tight_layout()
    fig_path = os.path.join(fig_dir, "I_geometry.pdf")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figure → {fig_path}")

    # Top-line summary
    summary = {"args": vars(args)}
    for modality, res in results.items():
        summary[modality] = {
            "knn_iter_0":    float(res["knn_mean"][0]),
            "knn_iter_K":    float(res["knn_mean"][-1]),
            "knn_change":    float(res["knn_mean"][-1] - res["knn_mean"][0]),
            "mipd_iter_0":   float(res["mipd_mean"][0]),
            "mipd_iter_K":   float(res["mipd_mean"][-1]),
            "mipd_change":   float(res["mipd_mean"][-1] - res["mipd_mean"][0]),
            "align_iter_1":  float(res["align_mean"][1]) if K > 1 else None,
            "align_iter_K":  float(res["align_mean"][-1]),
            "align_mean_over_run": float(np.nanmean(res["align_mean"])),
        }
    with open(os.path.join(args.out_dir, "I_geometry_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"\nSummary → {args.out_dir}/I_geometry_summary.json")


if __name__ == "__main__":
    main()