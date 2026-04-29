#!/usr/bin/env python3
"""
AIM2 — Local Trajectory Persistence (Block J).

Posthoc analysis answering the question:

  "Once trajectories converge to the attractor, do they STAY in their local
   neighborhood (sticky regions, fine-grained structure) or do they
   random-walk through the bounded attractor (true stochastic mixing)?"

This is a more direct test of the stochastic-mixing hypothesis than
Block I's displacement alignment. Block I told us trajectories drift in
uncorrelated directions; Block J asks whether trajectories that are
NEAR EACH OTHER stay near each other, or shuffle their neighborhoods
every iteration.

THREE METRICS, all computed on the already-existing N=1081 trajectories:

  1. STEP SIZE vs COHORT SPREAD
     Per iteration k, compute mean trajectory step size Δ_k = ||z_k - z_{k-1}||
     and compare to the cohort-wide MIPD at iteration k. The ratio
     Δ_k / MIPD_k tells us whether individual trajectories are stepping
     across the entire attractor each iteration (≈ √2, full random walk)
     or settling into specific points (<<1).

  2. TRAJECTORY AUTOCORRELATION (within-trajectory)
     For each trajectory, compute the cosine similarity between embeddings
     at iteration k and iteration k+lag for lags 1..5. If individual
     trajectories are sticky (stay in place), autocorrelation stays high
     even at lag=5. If they random-walk, autocorrelation decays toward
     the cohort-mean cosine value.

  3. kNN NEIGHBORHOOD PERSISTENCE (between-trajectory)
     At iteration k, find each trajectory's 10 nearest neighbors in the
     cohort. At iteration k+1, find the new 10 nearest neighbors. Compute
     the Jaccard overlap. If neighborhoods are persistent (same buddies
     iteration after iteration), Jaccard stays high. If trajectories
     shuffle neighbors, Jaccard drops to the random baseline (~10/N).

  This is the smoking gun: high kNN persistence = "buddies stay buddies"
  = sticky local structure even if no k-means clusters. Low kNN
  persistence = pure stochastic mixing.

OUTPUTS:
  analysis/J_persistence/
  ├── J_per_iter_metrics.csv     — long-format per-iter metrics
  ├── J_persistence.npz          — raw arrays
  └── J_persistence.pdf          — 3×2 figure (3 metrics × 2 modalities)

USAGE:
  python analysis_local_persistence.py \\
      --main_dir   .../results/chexgen_main \\
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
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def normalize(X: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization. Defensive — embeddings should already be
    L2-normed but this guarantees it for cosine similarity computations."""
    n = np.linalg.norm(X, axis=-1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def cohort_mipd(embs: np.ndarray, max_n: int = 2000,
                 rng: np.random.Generator = None):
    """Mean intra-pairwise cosine distance over a cohort. Subsamples if needed."""
    if rng is None:
        rng = np.random.default_rng(42)
    if len(embs) > max_n:
        idx = rng.choice(len(embs), max_n, replace=False)
        E = embs[idx]
    else:
        E = embs
    if len(E) < 2:
        return float("nan")
    E_n = normalize(E)
    sim = E_n @ E_n.T
    iu = np.triu_indices(len(E), k=1)
    d = 1.0 - sim[iu]
    return float(d.mean())


# ══════════════════════════════════════════════════════════════════════════════
#  Three core metrics
# ══════════════════════════════════════════════════════════════════════════════

def step_size_vs_mipd(traj: np.ndarray, mipd_max: int = 2000,
                       rng: np.random.Generator = None):
    """For each iteration k>=1, compute:
       - mean step size Δ_k (cosine distance) per trajectory
       - cohort MIPD at iter k
       - the ratio Δ_k / MIPD_k

    Interpretation:
      ratio ≈ √2 ≈ 1.41:    full random walk on a unit sphere
      ratio = 1:             step size = cohort spread
      ratio << 1:            trajectories barely move (sticky)
      ratio > 1:             trajectories step ACROSS the cohort each iter

    Returns:
      step_mean[K]:   mean per-trajectory step size at each iter (NaN at k=0)
      step_std[K]:    std of step sizes
      mipd[K]:        cohort MIPD at each iter
      ratio[K]:       step_mean / mipd
    """
    if rng is None:
        rng = np.random.default_rng(42)
    N, K, D = traj.shape

    step_mean = np.full(K, np.nan)
    step_std  = np.full(K, np.nan)
    mipd      = np.zeros(K)
    ratio     = np.full(K, np.nan)

    traj_n = normalize(traj.reshape(-1, D)).reshape(N, K, D)

    for k in range(K):
        mipd[k] = cohort_mipd(traj_n[:, k, :], max_n=mipd_max, rng=rng)
        if k > 0:
            sims = (traj_n[:, k, :] * traj_n[:, k - 1, :]).sum(axis=1)
            steps = 1.0 - sims          # per-trajectory cosine distance
            step_mean[k] = float(steps.mean())
            step_std[k]  = float(steps.std())
            if mipd[k] > 1e-12:
                ratio[k] = step_mean[k] / mipd[k]

    return {"step_mean": step_mean, "step_std": step_std,
            "mipd": mipd, "ratio": ratio}


def trajectory_autocorrelation(traj: np.ndarray, lags: list = (1, 2, 3, 5)):
    """For each trajectory, compute cosine similarity between embeddings
    separated by lag iterations, averaged over starting iterations and
    over the cohort.

    autocorr[lag][k_start] = mean over trajectories of
                             cos(z_{k_start}, z_{k_start + lag})

    A trajectory that stays in place has autocorr ≈ 1 at all lags.
    A trajectory that random-walks will have autocorr decay toward the
    cohort-baseline mean cosine.

    We compute both:
      - mean autocorr per (lag, k_start)
      - baseline = mean cosine between random pairs of trajectories at iter 0
        (this is what autocorr should approach if trajectories shuffle
        completely)
    """
    N, K, D = traj.shape
    traj_n = normalize(traj.reshape(-1, D)).reshape(N, K, D)

    # Baseline: mean cosine between random different-trajectory pairs at iter 0
    # (i.e. the prior distribution of cohort similarity, no trajectory link)
    rng = np.random.default_rng(42)
    pair_idx = rng.choice(N, size=(2000, 2), replace=True)
    pair_idx = pair_idx[pair_idx[:, 0] != pair_idx[:, 1]]
    baseline = float((traj_n[pair_idx[:, 0], 0, :] *
                       traj_n[pair_idx[:, 1], 0, :]).sum(axis=1).mean())

    results = {}
    for lag in lags:
        per_kstart = []
        for k_start in range(K - lag):
            sims = (traj_n[:, k_start, :] * traj_n[:, k_start + lag, :]).sum(axis=1)
            per_kstart.append(float(sims.mean()))
        results[f"lag_{lag}"] = np.array(per_kstart)
    results["baseline_random_pair_cos"] = baseline
    return results


def knn_neighborhood_persistence(traj: np.ndarray, k_nn: int = 10):
    """At each iteration k, find each trajectory's k_nn nearest neighbors in the
    cohort. Compute Jaccard overlap with the neighbor sets at iteration k+1.

    Returns:
      jaccard_mean[K-1]: mean Jaccard overlap between iter-k and iter-(k+1)
                         neighbor sets, averaged over trajectories
      jaccard_std[K-1]:  std
      random_baseline:   expected Jaccard if neighbor sets are independent random
                         draws of size k_nn from N (this is the null against which
                         to compare)
    """
    N, K, D = traj.shape
    traj_n = normalize(traj.reshape(-1, D)).reshape(N, K, D)

    neighbor_sets = []  # list of sets per iter
    for k in range(K):
        knn = NearestNeighbors(n_neighbors=k_nn + 1, metric="cosine",
                                algorithm="brute", n_jobs=-1)
        knn.fit(traj_n[:, k, :])
        # query=cohort: returns each point as its own first NN, drop it
        _, idx = knn.kneighbors(traj_n[:, k, :])
        sets_k = [set(idx[i, 1:]) for i in range(N)]   # exclude self
        neighbor_sets.append(sets_k)

    jaccard_mean = np.zeros(K - 1)
    jaccard_std  = np.zeros(K - 1)
    for k in range(K - 1):
        ja = np.zeros(N)
        for i in range(N):
            a, b = neighbor_sets[k][i], neighbor_sets[k + 1][i]
            inter = len(a & b)
            union = len(a | b)
            ja[i] = inter / max(union, 1)
        jaccard_mean[k] = float(ja.mean())
        jaccard_std[k]  = float(ja.std())

    # Random-baseline Jaccard for two random k_nn-subsets of N-1 candidates
    # E[|A ∩ B|] = k_nn^2 / (N-1) under uniform sampling without replacement
    # (good approximation for k_nn << N)
    expected_inter = (k_nn * k_nn) / (N - 1)
    expected_union = 2 * k_nn - expected_inter
    random_baseline = float(expected_inter / max(expected_union, 1))

    return {
        "jaccard_mean":   jaccard_mean,
        "jaccard_std":    jaccard_std,
        "random_baseline": random_baseline,
        "k_nn":           k_nn,
        "N":              N,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Trajectory loader (mirrors the other Block scripts)
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
    p.add_argument("--out_dir",  required=True)
    p.add_argument("--k_nn",     type=int, default=10)
    p.add_argument("--mipd_max", type=int, default=2000)
    p.add_argument("--lags",     type=int, nargs="+", default=[1, 2, 3, 5])
    return p.parse_args()


def run_modality(modality: str, traj: np.ndarray, args):
    N, K, D = traj.shape
    logger.info(f"\n[{modality.upper()}] {N} × {K} × {D}")

    rng = np.random.default_rng(42)

    logger.info("  Computing step-size vs MIPD ...")
    a = step_size_vs_mipd(traj, mipd_max=args.mipd_max, rng=rng)
    logger.info("  Computing trajectory autocorrelation ...")
    b = trajectory_autocorrelation(traj, lags=args.lags)
    logger.info(f"  Computing kNN(k={args.k_nn}) persistence ...")
    c = knn_neighborhood_persistence(traj, k_nn=args.k_nn)

    # Log per-iter values for inspection
    logger.info(f"    iter  step_size  MIPD     ratio    knn_jacc")
    for k in range(K):
        smean = a["step_mean"][k]
        rat   = a["ratio"][k]
        ja    = c["jaccard_mean"][k - 1] if 0 < k <= len(c["jaccard_mean"]) else float("nan")
        logger.info(f"    {k:3d}   "
                    f"{smean if np.isnan(smean) else f'{smean:.3f}':>7}   "
                    f"{a['mipd'][k]:.3f}   "
                    f"{rat if np.isnan(rat) else f'{rat:.3f}':>6}   "
                    f"{ja if np.isnan(ja) else f'{ja:.3f}':>7}")
    logger.info(f"    Autocorr at lag 1: mean over k = {b['lag_1'].mean():.3f}")
    logger.info(f"    Autocorr at lag 5: mean over k = {b['lag_5'].mean():.3f}")
    logger.info(f"    Random-pair baseline cosine: {b['baseline_random_pair_cos']:.3f}")
    logger.info(f"    kNN Jaccard mean (over k): {c['jaccard_mean'].mean():.3f}")
    logger.info(f"    kNN random baseline:       {c['random_baseline']:.3f}")

    return {"step": a, "autocorr": b, "knn": c}


def main():
    args = parse_args()
    cache_dir = os.path.join(args.out_dir, "cache")
    fig_dir   = os.path.join(args.out_dir, "figures")
    table_dir = os.path.join(args.out_dir, "tables")
    for d in (cache_dir, fig_dir, table_dir):
        os.makedirs(d, exist_ok=True)

    logger.info("=" * 60)
    logger.info("AIM2 Local Trajectory Persistence (Block J)")
    logger.info("=" * 60)
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")

    Z_img, Z_txt, study_ids = load_trajectories(args.main_dir)

    results = {
        "image": run_modality("image", Z_img, args),
        "text":  run_modality("text",  Z_txt, args),
    }

    K = Z_img.shape[1]
    iters = np.arange(K)

    # ── Save raw arrays ───────────────────────────────────────────────────────
    flat = {"K": K, "k_nn": args.k_nn, "study_ids": np.array(study_ids)}
    for modality, r in results.items():
        for sub_block, sub in r.items():
            for key, val in sub.items():
                flat[f"{modality}_{sub_block}_{key}"] = np.array(val)
    np.savez(os.path.join(cache_dir, "J_persistence.npz"), **flat)

    # ── Long-format CSV ───────────────────────────────────────────────────────
    rows = []
    for modality, r in results.items():
        for k in range(K):
            row = {
                "modality":      modality,
                "iter":          k,
                "step_mean":     float(r["step"]["step_mean"][k]),
                "step_std":      float(r["step"]["step_std"][k]),
                "mipd":          float(r["step"]["mipd"][k]),
                "ratio":         float(r["step"]["ratio"][k]),
            }
            if 0 < k <= len(r["knn"]["jaccard_mean"]):
                row["knn_jaccard"] = float(r["knn"]["jaccard_mean"][k - 1])
                row["knn_jaccard_std"] = float(r["knn"]["jaccard_std"][k - 1])
            else:
                row["knn_jaccard"]     = float("nan")
                row["knn_jaccard_std"] = float("nan")
            rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(table_dir, "J_per_iter_metrics.csv"),
                                index=False)
    logger.info(f"\nWrote {table_dir}/J_per_iter_metrics.csv")

    # ── Figure: 3 rows × 2 modality cols ──────────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(11, 10), sharex=True)

    for col, modality in enumerate(("image", "text")):
        r = results[modality]
        color = "C0" if modality == "image" else "C1"

        # Row 0: step size vs MIPD (paired curves) and ratio annotation
        ax = axes[0, col]
        ax.plot(iters, r["step"]["step_mean"], "o-", color=color, lw=2, ms=5,
                label=r"step size  $\Delta_k$")
        ax.plot(iters, r["step"]["mipd"], "s--", color="C7", lw=1.8, ms=4,
                label=r"cohort MIPD")
        ax.fill_between(iters,
                        r["step"]["step_mean"] - r["step"]["step_std"],
                        r["step"]["step_mean"] + r["step"]["step_std"],
                        color=color, alpha=0.15)
        ax.set_ylabel("Cosine distance")
        ax.set_title(f"{modality.title()}: per-traj step size vs cohort spread")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

        # Inset / annotation: ratio at iter K
        rat_K = r["step"]["ratio"][-1]
        ax.text(0.97, 0.05, rf"$\Delta_K/{{\rm MIPD}}_K = {rat_K:.2f}$",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=10, bbox=dict(boxstyle="round", fc="w", alpha=0.85))

        # Row 1: trajectory autocorrelation by lag
        ax = axes[1, col]
        cmap = plt.cm.viridis
        for li, lag in enumerate(args.lags):
            ac = r["autocorr"][f"lag_{lag}"]
            cc = cmap(li / max(len(args.lags) - 1, 1))
            ks = np.arange(len(ac))
            ax.plot(ks, ac, "o-", color=cc, lw=1.8, ms=4,
                    label=f"lag={lag}")
        ax.axhline(r["autocorr"]["baseline_random_pair_cos"],
                    ls=":", color="k", lw=1,
                    label=f"random-pair baseline ({r['autocorr']['baseline_random_pair_cos']:.2f})")
        ax.set_ylabel(r"Cosine sim along trajectory")
        ax.set_title(f"{modality.title()}: trajectory autocorrelation")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

        # Row 2: kNN Jaccard persistence
        ax = axes[2, col]
        ja = r["knn"]["jaccard_mean"]
        ja_std = r["knn"]["jaccard_std"]
        ks = np.arange(1, len(ja) + 1)        # transitions are k-1 -> k
        ax.plot(ks, ja, "^-", color=color, lw=2, ms=5,
                label=f"observed kNN Jaccard")
        ax.fill_between(ks, ja - ja_std, ja + ja_std, color=color, alpha=0.15)
        ax.axhline(r["knn"]["random_baseline"], ls=":", color="k", lw=1,
                    label=f"random baseline = {r['knn']['random_baseline']:.3f}")
        ax.axhline(1.0, ls="--", color="C2", lw=0.7, alpha=0.5,
                    label="perfect persistence (1.0)")
        ax.set_xlabel("Iteration $k$")
        ax.set_ylabel("kNN Jaccard")
        ax.set_title(f"{modality.title()}: neighborhood persistence (k={args.k_nn})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.02, 1.02)

    plt.suptitle("Block J — Local trajectory persistence "
                  "(do trajectories random-walk on the attractor or stick to local regions?)",
                  fontsize=11)
    plt.tight_layout()
    fig_path = os.path.join(fig_dir, "J_persistence.pdf")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figure → {fig_path}")

    # ── Top-line JSON summary ─────────────────────────────────────────────────
    summary = {"args": vars(args)}
    for modality, r in results.items():
        K = len(r["step"]["step_mean"])
        summary[modality] = {
            "step_size_iter_K":         float(r["step"]["step_mean"][-1]),
            "mipd_iter_K":              float(r["step"]["mipd"][-1]),
            "step_to_mipd_ratio_iter_K": float(r["step"]["ratio"][-1]),
            "step_to_mipd_ratio_late":  float(np.nanmean(r["step"]["ratio"][-5:])),
            "autocorr_lag_1_late":      float(r["autocorr"]["lag_1"][-3:].mean()),
            "autocorr_lag_5_mean":      float(r["autocorr"]["lag_5"].mean()) if len(r["autocorr"]["lag_5"]) else None,
            "random_pair_baseline":     float(r["autocorr"]["baseline_random_pair_cos"]),
            "knn_jaccard_late":         float(np.mean(r["knn"]["jaccard_mean"][-5:])),
            "knn_jaccard_random_baseline": float(r["knn"]["random_baseline"]),
            "knn_jaccard_normalized_late": float(
                (np.mean(r["knn"]["jaccard_mean"][-5:]) - r["knn"]["random_baseline"]) /
                (1.0 - r["knn"]["random_baseline"])
            ),
        }
    with open(os.path.join(args.out_dir, "J_persistence_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"\nSummary → {args.out_dir}/J_persistence_summary.json")
    logger.info("Done.")


if __name__ == "__main__":
    main()