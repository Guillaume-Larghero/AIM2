#!/usr/bin/env python3
"""
AIM2 — MI estimator calibration on synthetic Gaussian data.

Validates the KSG mutual-information estimator (sklearn
mutual_info_regression with shared-PCA-20 pre-projection, kNN=5) at the
exact (N, d) regime used in the paper, by running it on synthetic two-channel
Gaussian data with KNOWN mutual information.

The headline of the paper is "image MI drops 5.64 -> 0.17 nats in one
iteration"; reviewers can fairly ask whether the 0.17 floor is a real
residual signal or the estimator's bias floor in d=256, N=1008. This
script answers that by reporting estimator output at known I(X;Y) values
spanning the plausible range of the estimate.

OUTPUTS:
    out_dir/mi_calibration.json   — true_MI / estimated_MI pairs
    out_dir/mi_calibration.csv    — same data in flat form for plotting

USAGE:
    python mi_calibration.py --out_dir /path/to/analysis/MI_calibration
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("mi_calibration")


# ════════════════════════════════════════════════════════════════════════════
#  GAUSSIAN MI HELPERS
# ════════════════════════════════════════════════════════════════════════════
#
# For two scalar Gaussians (X, Y) with correlation rho, the analytical
# mutual information in nats is:
#     I(X;Y) = -0.5 * log(1 - rho^2)
# To produce a target I, invert:
#     rho = sqrt(1 - exp(-2 * I))
#
# We extend to d-dimensional Gaussians with diagonal correlation: each of
# the d coordinate pairs (X_i, Y_i) has the same scalar correlation rho.
# Then I(X; Y) = d * I_per_dim, since coordinates are independent.
# We CONSTRUCT samples this way so the per-dimension target is
# I_per_dim = I_total / d, but the paper's estimator is a MEAN univariate MI
# over PCA components, which approximates I_per_dim. We therefore report
# the per-dim true value as the natural ground truth for the estimator.

def make_paired_gaussian(N, d, rho, rng):
    """Generate (N, d) X and Y, with diagonal correlation rho between
    matched coordinates and zero correlation between unmatched coordinates.
    """
    Z = rng.standard_normal((N, d, 2))  # (N, d, [X, Y]) before correlation
    X = Z[:, :, 0]
    eps = Z[:, :, 1]
    # Y = rho * X + sqrt(1 - rho^2) * eps  -> per-coord correlation rho with X
    Y = rho * X + np.sqrt(1.0 - rho**2) * eps
    return X, Y


def rho_for_target_per_dim_mi(I_per_dim_nats):
    """Given target per-dimension MI in nats, return the correlation rho."""
    return float(np.sqrt(1.0 - np.exp(-2.0 * I_per_dim_nats)))


# ════════════════════════════════════════════════════════════════════════════
#  ESTIMATOR (mirror of attractor_analysis.py block E)
# ════════════════════════════════════════════════════════════════════════════
#
# Same code path as the paper:
#   1. Concatenate X and Y, fit PCA-20 on the union (mirrors "fit on union of
#      all trajectory points").
#   2. Project both X and Y through that PCA basis.
#   3. For each PCA dimension, compute univariate MI(X[:,d], Y[:,d]) with
#      sklearn's KSG implementation at n_neighbors=knn.
#   4. Report mean over PCA dimensions.
#
# Note: this is a univariate-mean MI, not a multivariate MI. It is the same
# proxy used in the paper, so the calibration is fair to the headline number.

def estimate_mi_paper_style(X, Y, n_components=20, knn=5, seed=42):
    """Estimate MI in the same way attractor_analysis.py block E does."""
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import mutual_info_regression

    # Fit shared PCA on union of trajectory points (here: X stacked with Y)
    union = np.vstack([X, Y])
    n_comp = min(n_components, union.shape[0] - 1, union.shape[1])
    pca = PCA(n_components=n_comp, random_state=seed).fit(union)

    X_p = pca.transform(X)
    Y_p = pca.transform(Y)

    mis = []
    for d_idx in range(X_p.shape[1]):
        try:
            mi = mutual_info_regression(
                X_p[:, d_idx:d_idx + 1], Y_p[:, d_idx],
                n_neighbors=knn, random_state=seed,
            )[0]
            mis.append(float(mi))
        except Exception:
            pass
    return float(np.mean(mis)) if mis else float("nan"), mis


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                  description=__doc__)
    p.add_argument("--out_dir", required=True,
                    help="Output directory for mi_calibration.json/csv")
    p.add_argument("--N", type=int, default=1008,
                    help="Sample size (default matches paper cohort N=1008)")
    p.add_argument("--d", type=int, default=256,
                    help="Ambient dimension (default matches paper d=256)")
    p.add_argument("--n_pca", type=int, default=20,
                    help="PCA components (default matches paper)")
    p.add_argument("--knn", type=int, default=5,
                    help="kNN parameter for KSG (default matches paper)")
    p.add_argument("--n_repeats", type=int, default=10,
                    help="Number of independent draws per condition (for SE)")
    p.add_argument("--target_MI_per_dim",
                    default="0.0,0.05,0.1,0.2,0.5,1.0,2.0,3.0,5.6",
                    help="Comma-separated target per-dimension MI values in nats. "
                         "Default spans the plausible estimator range.")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    logger.info("=" * 60)
    logger.info("MI estimator calibration on synthetic Gaussian")
    logger.info("=" * 60)
    logger.info(f"  N = {args.N}, d = {args.d}, n_pca = {args.n_pca}, knn = {args.knn}")
    logger.info(f"  n_repeats per target = {args.n_repeats}")

    target_MIs = [float(x) for x in args.target_MI_per_dim.split(",")]
    logger.info(f"  Target per-dimension MI values (nats): {target_MIs}")

    rng = np.random.default_rng(42)

    rows = []
    for I_target in target_MIs:
        rho = rho_for_target_per_dim_mi(I_target) if I_target > 0 else 0.0
        logger.info(f"\n  Target per-dim MI = {I_target:.3f} nats  =>  rho = {rho:.4f}")
        estimates = []
        for r in range(args.n_repeats):
            X, Y = make_paired_gaussian(args.N, args.d, rho, rng)
            mi_est, mis_per_pc = estimate_mi_paper_style(
                X, Y, n_components=args.n_pca, knn=args.knn, seed=42 + r)
            estimates.append(mi_est)
            logger.info(f"    rep {r:2d}: estimated mean per-PC MI = {mi_est:+.4f}")
        est_arr = np.array(estimates)
        row = {
            "target_per_dim_MI_nats":   I_target,
            "true_rho":                 rho,
            "n_repeats":                args.n_repeats,
            "estimated_MI_mean":        float(np.mean(est_arr)),
            "estimated_MI_std":         float(np.std(est_arr, ddof=1)) if len(est_arr) > 1 else 0.0,
            "estimated_MI_min":         float(np.min(est_arr)),
            "estimated_MI_max":         float(np.max(est_arr)),
            "estimates":                [float(v) for v in estimates],
        }
        rows.append(row)
        logger.info(f"    target {I_target:.3f} nats: estimated mean = "
                    f"{row['estimated_MI_mean']:+.4f} ± {row['estimated_MI_std']:.4f} "
                    f"(min={row['estimated_MI_min']:+.4f}, max={row['estimated_MI_max']:+.4f})")

    # ── Save JSON ────────────────────────────────────────────────────────────
    out_json = os.path.join(args.out_dir, "mi_calibration.json")
    with open(out_json, "w") as f:
        json.dump({
            "N": args.N,
            "d": args.d,
            "n_pca": args.n_pca,
            "knn": args.knn,
            "n_repeats": args.n_repeats,
            "rows": rows,
        }, f, indent=2)
    logger.info(f"\n  Results -> {out_json}")

    # ── Save CSV (flat) ──────────────────────────────────────────────────────
    out_csv = os.path.join(args.out_dir, "mi_calibration.csv")
    with open(out_csv, "w") as f:
        f.write("target_per_dim_MI_nats,true_rho,estimated_MI_mean,"
                "estimated_MI_std,estimated_MI_min,estimated_MI_max\n")
        for row in rows:
            f.write(f"{row['target_per_dim_MI_nats']:.6f},"
                    f"{row['true_rho']:.6f},"
                    f"{row['estimated_MI_mean']:.6f},"
                    f"{row['estimated_MI_std']:.6f},"
                    f"{row['estimated_MI_min']:.6f},"
                    f"{row['estimated_MI_max']:.6f}\n")
    logger.info(f"  Results -> {out_csv}")

    # ── Headline summary ─────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("CALIBRATION SUMMARY")
    logger.info("=" * 60)
    floor = next((r for r in rows if r["target_per_dim_MI_nats"] == 0.0), None)
    if floor:
        logger.info(f"  Estimator floor at I=0: "
                    f"{floor['estimated_MI_mean']:+.4f} ± {floor['estimated_MI_std']:.4f} nats")
        logger.info(f"  Paper iter-1 image MI = 0.17 nats (and this script provides")
        logger.info(f"  the estimator floor for direct comparison)")
    logger.info("\nDone.")


if __name__ == "__main__":
    main()