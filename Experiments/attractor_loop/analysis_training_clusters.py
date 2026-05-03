#!/usr/bin/env python3
"""
AIM2 — Training-Cluster Trajectory Analysis (Block H).

Posthoc analysis run AFTER attractor_loop_chexgen full run completes.

QUESTION ANSWERED:
  "Into which regions of the original training distribution do trajectories
   collapse over iterations, and how fast?"

This is fundamentally different from Block C (which clusters on iter-K
endpoints). Here we cluster on the FULL TRAINING distribution (55,691 MedCLIP
train+val embeddings, already cached in reference_embeddings/), then ask at
every trajectory iteration which training cluster each point belongs to.

THIS DIRECTLY ENABLES THE HINTZE COMPARISON:
  Hintze et al. observed convergence to "12 dominant motifs." We can
  quantify the analogous phenomenon: per-iteration assignment entropy
  starts high (test cohort spans many training clusters), then collapses
  if trajectories funnel into a few generic modes.

OUTPUTS:
  analysis/H_training_clusters/
  ├── H_training_clusters.npz       — cluster fits, per-iter assignments, transitions
  ├── H_assignment_entropy.csv      — entropy per iter, image and text modalities
  ├── H_dominant_clusters.csv       — top-K clusters by iter-K population
  ├── H_transition_matrices.npz     — per-iter Markov transition matrices
  └── H_*.pdf                       — figures: entropy curve, transition heatmaps

USAGE:
  python analysis_training_clusters.py \\
      --main_dir   .../results/chexgen_main \\
      --ref_dir    .../reference_embeddings \\
      --out_dir    .../analysis \\
      --n_clusters 15 \\
      --modality   image     # or "text"
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
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def cosine_distance_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Cosine distance between rows of A and rows of B. Inputs need not be
    L2-normalized — we normalize internally for safety."""
    A_n = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B_n = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return 1.0 - A_n @ B_n.T


def fit_training_clusters(train_embs: np.ndarray, n_clusters: int = 15,
                           random_state: int = 42, max_silhouette_n: int = 10000):
    """Fit k-means on training embeddings.

    For 55K points in 256-d, k-means takes ~30 sec. Silhouette is O(N²) so
    we subsample for that single number.
    """
    logger.info(f"  Fitting k-means with K={n_clusters} on {len(train_embs)} training embs...")
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    train_assignments = km.fit_predict(train_embs)
    centroids = km.cluster_centers_
    # Renormalize centroids back to unit sphere for cosine similarity
    centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)

    # Silhouette on subsample for speed (cosine metric)
    if len(train_embs) > max_silhouette_n:
        idx = np.random.default_rng(42).choice(len(train_embs), max_silhouette_n,
                                                replace=False)
        sil = float(silhouette_score(train_embs[idx], train_assignments[idx],
                                     metric="cosine"))
    else:
        sil = float(silhouette_score(train_embs, train_assignments, metric="cosine"))

    cluster_counts = np.bincount(train_assignments, minlength=n_clusters)
    max_entropy = float(np.log2(n_clusters))

    return {
        "centroids":          centroids,
        "n_clusters":         n_clusters,
        "train_assignments":  train_assignments,
        "train_silhouette":   sil,
        "train_cluster_counts": cluster_counts,
        "max_entropy":        max_entropy,
    }


def assign_to_clusters(test_embs: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Assign each test embedding to the nearest centroid by cosine distance."""
    dists = cosine_distance_matrix(test_embs, centroids)
    return dists.argmin(axis=1)


def shannon_entropy(assignments: np.ndarray, n_clusters: int) -> float:
    """Shannon entropy of cluster-assignment distribution, in bits."""
    counts = np.bincount(assignments, minlength=n_clusters).astype(float)
    probs = counts / max(counts.sum(), 1.0)
    nz = probs[probs > 0]
    return float(-np.sum(nz * np.log2(nz)))


def transition_matrix(assignments_t: np.ndarray, assignments_tp1: np.ndarray,
                       n_clusters: int) -> np.ndarray:
    """Row-stochastic transition matrix T where T[i,j] = P(j at t+1 | i at t).

    Rows with zero count remain zero (no smoothing) — they appear as black
    rows in the heatmap and indicate clusters with no trajectories at iter t.
    """
    T = np.zeros((n_clusters, n_clusters))
    for i_from, i_to in zip(assignments_t, assignments_tp1):
        T[i_from, i_to] += 1
    row_sum = T.sum(axis=1, keepdims=True)
    T = np.divide(T, np.maximum(row_sum, 1.0),
                   out=np.zeros_like(T), where=row_sum > 0)
    return T


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--main_dir",   required=True,
                   help="Per-study directory with metrics.json + embeddings (chexgen_main).")
    p.add_argument("--ref_dir",    required=True,
                   help="Reference embeddings dir (preflight output): "
                        "ref_img_embeddings.npy, ref_txt_embeddings.npy.")
    p.add_argument("--out_dir",    required=True,
                   help="Where to write H_* outputs (typically the analysis/ dir).")
    p.add_argument("--n_clusters", type=int, default=15,
                   help="k for k-means on training embeddings.")
    p.add_argument("--modality",   choices=["image", "text", "both"], default="both")
    p.add_argument("--random_state", type=int, default=42)
    return p.parse_args()


def load_trajectories(main_dir: str):
    """Load all trajectory embeddings and study IDs from main run output.

    Per-study layout in main_dir (created by attractor_loop_chexgen.py):
      <sid>/anchor_img_embed.npy           (256,)  GT iter-0 image
      <sid>/anchor_text_embed.npy          (256,)  GT iter-0 text
      <sid>/img_embed_iter_000.npy ...     (256,)  per-iteration image
      <sid>/text_embed_iter_000.npy ...    (256,)  per-iteration text
      <sid>/metrics.json                          (used as completion marker)
    """
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
            z_img = np.stack([np.load(p) for p in img_files])  # (K, 256)
            z_txt = np.stack([np.load(p) for p in txt_files])  # (K, 256)
        except Exception as e:
            logger.warning(f"  skipping {sid}: failed to load embeds ({e})")
            continue

        if K_canonical is None:
            K_canonical = z_img.shape[0]
        if z_img.shape[0] != K_canonical or z_txt.shape[0] != K_canonical:
            # Skip studies that didn't complete all iters (incomplete chunks)
            continue

        Z_img_list.append(z_img)
        Z_txt_list.append(z_txt)
        study_ids.append(sid)

    if not Z_img_list:
        logger.error(f"No complete trajectories loaded from {main_dir}")
        sys.exit(1)

    Z_img = np.stack(Z_img_list)  # (N, K, 256)
    Z_txt = np.stack(Z_txt_list)
    logger.info(f"  Loaded {len(study_ids)} complete trajectories, "
                f"K={K_canonical}, D={Z_img.shape[-1]}")
    return Z_img, Z_txt, study_ids


def run_modality(modality: str, train_embs: np.ndarray,
                 trajectories: np.ndarray, study_ids,
                 args, fig_dir: str, table_dir: str, cache_dir: str):
    """Run training-cluster analysis for one modality (image or text).

    trajectories: (N, K, D) array.
    """
    N, K, D = trajectories.shape
    logger.info(f"\n[{modality.upper()}] {N} trajectories × {K} iterations × {D}-d")

    # Fit clusters on training data
    fit = fit_training_clusters(train_embs, n_clusters=args.n_clusters,
                                 random_state=args.random_state)
    logger.info(f"  Train silhouette ({modality}): {fit['train_silhouette']:.4f}")
    logger.info(f"  Train cluster sizes: "
                f"{sorted(fit['train_cluster_counts'].tolist(), reverse=True)}")

    # Per-iter cluster assignments for trajectories
    assignments = np.zeros((N, K), dtype=np.int64)
    for k in range(K):
        assignments[:, k] = assign_to_clusters(trajectories[:, k, :], fit["centroids"])

    # Per-iter entropy + concentration
    entropy_per_iter   = np.zeros(K)
    norm_entropy_per_iter = np.zeros(K)
    cluster_counts_per_iter = np.zeros((K, args.n_clusters), dtype=np.int64)
    n_dominant_50 = np.zeros(K, dtype=np.int64)  # #clusters needed to cover 50% of mass
    n_dominant_90 = np.zeros(K, dtype=np.int64)

    for k in range(K):
        entropy_per_iter[k] = shannon_entropy(assignments[:, k], args.n_clusters)
        norm_entropy_per_iter[k] = entropy_per_iter[k] / max(fit["max_entropy"], 1e-12)
        counts = np.bincount(assignments[:, k], minlength=args.n_clusters)
        cluster_counts_per_iter[k] = counts
        sorted_counts = np.sort(counts)[::-1]
        cumfrac = np.cumsum(sorted_counts) / max(counts.sum(), 1)
        n_dominant_50[k] = int(np.searchsorted(cumfrac, 0.50) + 1)
        n_dominant_90[k] = int(np.searchsorted(cumfrac, 0.90) + 1)
        logger.info(f"    iter {k:2d}: entropy={entropy_per_iter[k]:.3f} "
                    f"(norm={norm_entropy_per_iter[k]:.3f})  "
                    f"clusters@50%={n_dominant_50[k]:2d}  @90%={n_dominant_90[k]:2d}")

    # Transition matrices: K-1 of them, T[i,j] = P(j at k+1 | i at k)
    Ts = np.zeros((K - 1, args.n_clusters, args.n_clusters))
    for k in range(K - 1):
        Ts[k] = transition_matrix(assignments[:, k], assignments[:, k + 1],
                                   args.n_clusters)

    # ── Save tables ───────────────────────────────────────────────────────────
    df_entropy = pd.DataFrame({
        "iter":              np.arange(K),
        "entropy":           entropy_per_iter,
        "normalized_entropy": norm_entropy_per_iter,
        "max_entropy":       fit["max_entropy"],
        "n_dominant_clusters_50pct": n_dominant_50,
        "n_dominant_clusters_90pct": n_dominant_90,
    })
    df_entropy.to_csv(os.path.join(table_dir,
                                    f"H_assignment_entropy_{modality}.csv"),
                      index=False)
    logger.info(f"  Wrote {table_dir}/H_assignment_entropy_{modality}.csv")

    # Top-K clusters by iter-K population (the "attractor modes")
    top_indices = np.argsort(cluster_counts_per_iter[-1])[::-1]
    top_rows = []
    for rank, ci in enumerate(top_indices):
        top_rows.append({
            "rank":           rank + 1,
            "cluster_id":     int(ci),
            "iter_K_count":   int(cluster_counts_per_iter[-1, ci]),
            "iter_K_pct":     float(cluster_counts_per_iter[-1, ci] / N * 100),
            "iter_0_count":   int(cluster_counts_per_iter[0, ci]),
            "iter_0_pct":     float(cluster_counts_per_iter[0, ci] / N * 100),
            "train_count":    int(fit["train_cluster_counts"][ci]),
            "train_pct":      float(fit["train_cluster_counts"][ci]
                                    / fit["train_cluster_counts"].sum() * 100),
        })
    pd.DataFrame(top_rows).to_csv(
        os.path.join(table_dir, f"H_dominant_clusters_{modality}.csv"), index=False)

    # ── Save raw arrays ───────────────────────────────────────────────────────
    np.savez(os.path.join(cache_dir, f"H_training_clusters_{modality}.npz"),
             centroids=fit["centroids"],
             train_assignments=fit["train_assignments"],
             train_silhouette=fit["train_silhouette"],
             train_cluster_counts=fit["train_cluster_counts"],
             trajectory_assignments=assignments,  # (N, K)
             study_ids=np.array(study_ids),
             entropy_per_iter=entropy_per_iter,
             normalized_entropy_per_iter=norm_entropy_per_iter,
             cluster_counts_per_iter=cluster_counts_per_iter,
             transition_matrices=Ts,
             n_clusters=args.n_clusters,
             max_entropy=fit["max_entropy"])
    logger.info(f"  Wrote {cache_dir}/H_training_clusters_{modality}.npz")

    # ── Figure: entropy curve + dominance + transition heatmaps ───────────────
    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(2, 3)

    # Top-left: entropy collapse
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(np.arange(K), entropy_per_iter, "o-", color="C0", lw=2, ms=5,
            label="Trajectory entropy")
    ax.axhline(fit["max_entropy"], ls="--", color="k", lw=0.8, alpha=0.7,
               label=f"Max ({fit['max_entropy']:.2f}={ args.n_clusters }-uniform)")
    ax.set_xlabel("Iteration $k$")
    ax.set_ylabel("Cluster-assignment entropy (bits)")
    ax.set_title(f"{modality.title()}: collapse into\ndominant training modes")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Top-middle: number of dominant clusters
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(np.arange(K), n_dominant_50, "o-", color="C1", lw=2, ms=5,
            label="50% mass")
    ax.plot(np.arange(K), n_dominant_90, "s-", color="C3", lw=2, ms=5,
            label="90% mass")
    ax.set_xlabel("Iteration $k$")
    ax.set_ylabel("Number of clusters")
    ax.set_title("Cluster concentration")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Top-right: stacked area of top-K cluster populations across iters
    ax = fig.add_subplot(gs[0, 2])
    top_n_show = min(8, args.n_clusters)
    top_iter_K = np.argsort(cluster_counts_per_iter[-1])[::-1][:top_n_show]
    bottom = np.zeros(K)
    cmap = plt.cm.tab20.colors
    for rank, ci in enumerate(top_iter_K):
        ys = cluster_counts_per_iter[:, ci] / N
        ax.fill_between(np.arange(K), bottom, bottom + ys,
                        color=cmap[rank % len(cmap)],
                        label=f"C{ci}", alpha=0.85)
        bottom += ys
    ax.fill_between(np.arange(K), bottom, np.ones(K),
                    color="lightgrey", alpha=0.5, label="other")
    ax.set_xlabel("Iteration $k$")
    ax.set_ylabel("Trajectory fraction")
    ax.set_title(f"Top-{top_n_show} iter-$K$ clusters")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=6, ncol=2, loc="lower right")

    # Bottom row: transition matrices for k=0→1, mid, last
    show_idxs = sorted(set([0, max((K - 1) // 2 - 1, 0), K - 2]))
    for col, ti in enumerate(show_idxs[:3]):
        ax = fig.add_subplot(gs[1, col])
        im = ax.imshow(Ts[ti], cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xlabel(f"cluster at $k{{=}}{ti+1}$")
        ax.set_ylabel(f"cluster at $k{{=}}{ti}$" if col == 0 else "")
        ax.set_title(f"Transition $k{{=}}{ti}\\to{ti+1}$")
        if col == 2:
            plt.colorbar(im, ax=ax, fraction=0.046, label=r"$P(j|i)$")

    plt.tight_layout()
    fig_path = os.path.join(fig_dir, f"H_training_clusters_{modality}.pdf")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Figure → {fig_path}")

    # Summary stats for top-line numbers
    return {
        "modality":               modality,
        "n_trajectories":         int(N),
        "K_iterations":           int(K),
        "n_clusters":             int(args.n_clusters),
        "max_entropy":            float(fit["max_entropy"]),
        "train_silhouette":       float(fit["train_silhouette"]),
        "entropy_iter_0":         float(entropy_per_iter[0]),
        "entropy_iter_K":         float(entropy_per_iter[-1]),
        "entropy_drop":           float(entropy_per_iter[0] - entropy_per_iter[-1]),
        "normalized_entropy_iter_K": float(norm_entropy_per_iter[-1]),
        "n_dominant_50pct_iter_0": int(n_dominant_50[0]),
        "n_dominant_50pct_iter_K": int(n_dominant_50[-1]),
        "n_dominant_90pct_iter_K": int(n_dominant_90[-1]),
    }


def main():
    args = parse_args()

    cache_dir = os.path.join(args.out_dir, "cache")
    fig_dir   = os.path.join(args.out_dir, "figures")
    table_dir = os.path.join(args.out_dir, "tables")
    for d in (cache_dir, fig_dir, table_dir):
        os.makedirs(d, exist_ok=True)

    logger.info("=" * 60)
    logger.info("AIM2 Training-Cluster Trajectory Analysis (Block H)")
    logger.info("=" * 60)
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")

    # ── Load reference embeddings (training distribution) ─────────────────────
    ref_img_path = os.path.join(args.ref_dir, "img_embeds.npy")
    ref_txt_path = os.path.join(args.ref_dir, "txt_embeds.npy")
    if not os.path.exists(ref_img_path) or not os.path.exists(ref_txt_path):
        logger.error(f"Reference embeddings not found in {args.ref_dir}. "
                     f"Expected img_embeds.npy and txt_embeds.npy "
                     f"(produced by preflight_embed_corpus.py).")
        sys.exit(1)
    ref_img = np.load(ref_img_path)
    ref_txt = np.load(ref_txt_path)
    logger.info(f"\nReference image embeddings: {ref_img.shape}")
    logger.info(f"Reference text embeddings:  {ref_txt.shape}")

    # ── Load trajectories ─────────────────────────────────────────────────────
    Z_img, Z_txt, study_ids = load_trajectories(args.main_dir)

    # ── Run analysis per modality ─────────────────────────────────────────────
    summary = {"args": vars(args)}

    if args.modality in ("image", "both"):
        s = run_modality("image", ref_img, Z_img, study_ids,
                          args, fig_dir, table_dir, cache_dir)
        summary["image"] = s

    if args.modality in ("text", "both"):
        s = run_modality("text", ref_txt, Z_txt, study_ids,
                          args, fig_dir, table_dir, cache_dir)
        summary["text"] = s

    # ── Top-line JSON summary ─────────────────────────────────────────────────
    with open(os.path.join(args.out_dir, "H_training_clusters_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"\nSummary → {args.out_dir}/H_training_clusters_summary.json")
    logger.info("Done.")


if __name__ == "__main__":
    main()