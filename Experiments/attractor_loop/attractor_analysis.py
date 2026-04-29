#!/usr/bin/env python3
"""
AIM2 ChexGen — Main Analysis: Attractor Dynamics in Generative AI Loops
========================================================================

Reads:
  results/chexgen_main/<study_id>/{anchor,img,text}_embed*.npy + metrics.json
  results/chexgen_lyapunov/anchor_<sid>/seed_<j>/*.npy  (optional)
  reference_embeddings/{img,txt}_embeds.npy + umap_*.pkl + meta.csv

Produces:
  figures/   — publication PDFs
  tables/    — LaTeX-ready CSVs with key statistics
  cache/     — intermediate analysis artifacts (clustering, projections, …)
  analysis_results.json — all scalar metrics, signed off and cited in paper

Block structure mirrors the analysis plan:

  A. Trajectory geometry      (convergence, tortuosity, modality coupling)
  B. Lyapunov stability       (system-wide + per-anchor basin radius)
  C. Multi-basin structure    (clustering, ergodic component characterization)
  D. Phase portrait           (2D UMAP, vector field, basin separatrix)
  E. High-dim analysis        (effective dim, spectral, persistent homology, MI)
  F. Clinical interpretation  (CheXpert profiling, bias quantification)

Each block runs independently; pass --blocks to subset (e.g. --blocks A C F).
"""

import argparse
import json
import logging
import os
import pickle
import sys
import warnings
from collections import Counter, defaultdict
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch
from scipy import stats
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = "/n/groups/training/bmif203/AIM2"
DEFAULT_MAIN     = f"{BASE_DIR}/Experiments/attractor_loop/results/chexgen_main"
DEFAULT_LYAPUNOV = f"{BASE_DIR}/Experiments/attractor_loop/results/chexgen_lyapunov"
DEFAULT_REFEMB   = f"{BASE_DIR}/Experiments/attractor_loop/reference_embeddings"
DEFAULT_OUT      = f"{BASE_DIR}/Experiments/attractor_loop/analysis"

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Publication style
plt.rcParams.update({
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "font.family":        "serif",
    "font.size":          9,
    "axes.labelsize":     9,
    "axes.titlesize":     10,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "lines.linewidth":    1.2,
})


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_main_run(main_dir):
    """
    Returns:
      trajectories: list of dicts, one per study, each with keys:
        study_id, anchor_img, anchor_text, img_traj (K+1, 256), txt_traj (K+1, 256),
        findings (list of K+1 strings), metrics (dict from metrics.json)
    """
    summary_path = os.path.join(main_dir, "summary.json")
    with open(summary_path) as f:
        summary = json.load(f)

    trajs = []
    for s in summary["per_study"]:
        sid = s["study_id"]
        sdir = os.path.join(main_dir, sid)
        if not os.path.isdir(sdir):
            continue

        anchor_img = np.load(os.path.join(sdir, "anchor_img_embed.npy"))
        anchor_txt = np.load(os.path.join(sdir, "anchor_text_embed.npy"))

        # Determine trajectory length from existing embed files
        img_files = sorted(glob(os.path.join(sdir, "img_embed_iter_*.npy")))
        K = len(img_files)
        img_traj = np.stack([np.load(f) for f in img_files])
        txt_traj = np.stack([
            np.load(os.path.join(sdir, f"text_embed_iter_{k:03d}.npy"))
            for k in range(K)
        ])

        trajs.append({
            "study_id":   sid,
            "anchor_img": anchor_img,                             # (256,)
            "anchor_txt": anchor_txt,                             # (256,)
            "img_traj":   img_traj,                               # (K, 256)
            "txt_traj":   txt_traj,                               # (K, 256)
            "findings":   s.get("findings", []),
            "metrics":    s,
        })
    logger.info(f"Loaded {len(trajs)} trajectories from {main_dir}")
    if trajs:
        logger.info(f"  Trajectory length: {trajs[0]['img_traj'].shape[0]} iters")
        logger.info(f"  Embedding dim: {trajs[0]['img_traj'].shape[1]}")
    return trajs


def load_lyapunov_run(lyap_dir):
    """Returns dict {anchor_sid: list of trajectory dicts (one per seed replicate)}."""
    if not os.path.isdir(lyap_dir):
        logger.warning(f"Lyapunov dir not found: {lyap_dir} — Block B1 will be skipped")
        return {}

    summary_path = os.path.join(lyap_dir, "summary.json")
    if not os.path.exists(summary_path):
        logger.warning(f"Lyapunov summary not found at {summary_path}")
        return {}

    out = defaultdict(list)
    anchors = sorted(glob(os.path.join(lyap_dir, "anchor_*")))
    for anchor_path in anchors:
        sid = os.path.basename(anchor_path).replace("anchor_", "")
        seed_dirs = sorted(glob(os.path.join(anchor_path, "seed_*")))
        for sd in seed_dirs:
            try:
                anchor_img = np.load(os.path.join(sd, "anchor_img_embed.npy"))
                anchor_txt = np.load(os.path.join(sd, "anchor_text_embed.npy"))
                img_files = sorted(glob(os.path.join(sd, "img_embed_iter_*.npy")))
                K = len(img_files)
                img_traj = np.stack([np.load(f) for f in img_files])
                txt_traj = np.stack([
                    np.load(os.path.join(sd, f"text_embed_iter_{k:03d}.npy"))
                    for k in range(K)
                ])
                out[sid].append({
                    "seed_dir":   sd,
                    "anchor_img": anchor_img,
                    "anchor_txt": anchor_txt,
                    "img_traj":   img_traj,
                    "txt_traj":   txt_traj,
                })
            except FileNotFoundError as e:
                logger.warning(f"  partial trajectory skipped: {sd}  ({e})")

    n_total = sum(len(v) for v in out.values())
    logger.info(f"Loaded Lyapunov: {len(out)} anchors × ~{n_total/max(len(out),1):.0f} seeds "
                f"= {n_total} trajectories")
    return dict(out)


def load_reference_embeddings(ref_dir):
    """Loads cached corpus embeddings + UMAP reducers + meta."""
    needed = ["img_embeds.npy", "txt_embeds.npy", "meta.csv",
              "umap_img.pkl", "umap_txt.pkl", "valid_masks.npz"]
    for n in needed:
        if not os.path.exists(os.path.join(ref_dir, n)):
            raise FileNotFoundError(
                f"Reference embedding artifact missing: {n}\n"
                f"Run preflight_embed_corpus.py first.")

    img_e = np.load(os.path.join(ref_dir, "img_embeds.npy"))
    txt_e = np.load(os.path.join(ref_dir, "txt_embeds.npy"))
    meta  = pd.read_csv(os.path.join(ref_dir, "meta.csv"))
    masks = np.load(os.path.join(ref_dir, "valid_masks.npz"))
    with open(os.path.join(ref_dir, "umap_img.pkl"), "rb") as f:
        umap_img = pickle.load(f)
    with open(os.path.join(ref_dir, "umap_txt.pkl"), "rb") as f:
        umap_txt = pickle.load(f)

    img_2d = np.load(os.path.join(ref_dir, "umap_img_2d.npy")) \
             if os.path.exists(os.path.join(ref_dir, "umap_img_2d.npy")) else None
    txt_2d = np.load(os.path.join(ref_dir, "umap_txt_2d.npy")) \
             if os.path.exists(os.path.join(ref_dir, "umap_txt_2d.npy")) else None

    logger.info(f"Loaded reference: {len(meta)} studies, "
                f"{masks['img_valid'].sum()} valid img, "
                f"{masks['txt_valid'].sum()} valid txt")
    return {"img_e":    img_e,    "txt_e":    txt_e,
            "img_valid": masks["img_valid"], "txt_valid": masks["txt_valid"],
            "meta":     meta,
            "umap_img": umap_img, "umap_txt": umap_txt,
            "img_2d":   img_2d,   "txt_2d":   txt_2d}


# ══════════════════════════════════════════════════════════════════════════════
#  BLOCK A — TRAJECTORY GEOMETRY
# ══════════════════════════════════════════════════════════════════════════════

def block_A_geometry(trajs, out_dir):
    """
    Convergence diagnostics + tortuosity + modality coupling.
    Outputs: figures/A1_convergence.pdf, A2_tortuosity.pdf,
             A3_modality_coupling.pdf, tables/A_geometry.csv,
             cache/A_geometry.npz
    """
    logger.info("\n" + "═" * 60)
    logger.info("BLOCK A — Trajectory Geometry")
    logger.info("═" * 60)

    N = len(trajs)
    K = trajs[0]["img_traj"].shape[0]   # number of iterations including iter 0

    # Build (N, K, 256) tensors (image and text)
    Z_img = np.stack([t["img_traj"] for t in trajs])  # (N, K, 256)
    Z_txt = np.stack([t["txt_traj"] for t in trajs])  # (N, K, 256)
    A_img = np.stack([t["anchor_img"] for t in trajs])  # (N, 256)
    A_txt = np.stack([t["anchor_txt"] for t in trajs])

    # ── Step sizes Δ_k = ||z_k - z_{k-1}|| ────────────────────────────────────
    # Stochastic systems: Δ_k stays bounded around a non-zero value at convergence
    # (= radius of attractor stochastic ball)
    delta_img = np.linalg.norm(Z_img[:, 1:] - Z_img[:, :-1], axis=-1)  # (N, K-1)
    delta_txt = np.linalg.norm(Z_txt[:, 1:] - Z_txt[:, :-1], axis=-1)

    # ── Anchor distance d_k = ||z_k - z_0_GT|| ────────────────────────────────
    # Note: z_0_GT is the GT image's embed (the anchor), NOT the iter-0 generated.
    d_anchor_img = np.linalg.norm(Z_img - A_img[:, None, :], axis=-1)  # (N, K)
    d_anchor_txt = np.linalg.norm(Z_txt - A_txt[:, None, :], axis=-1)

    # ── Tortuosity T = L/D ────────────────────────────────────────────────────
    # L = sum of step sizes; D = endpoint displacement from iter-0 generated.
    L_img = delta_img.sum(axis=1)
    D_img = np.linalg.norm(Z_img[:, -1] - Z_img[:, 0], axis=-1)
    T_img = L_img / np.maximum(D_img, 1e-9)

    L_txt = delta_txt.sum(axis=1)
    D_txt = np.linalg.norm(Z_txt[:, -1] - Z_txt[:, 0], axis=-1)
    T_txt = L_txt / np.maximum(D_txt, 1e-9)

    # ── Modality coupling: per-iter Pearson corr of step sizes ────────────────
    rho = np.array([
        stats.pearsonr(delta_img[:, k], delta_txt[:, k]).statistic
        for k in range(K - 1)
    ])

    # ── Save numerics ─────────────────────────────────────────────────────────
    cache_dir = os.path.join(out_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(os.path.join(cache_dir, "A_geometry.npz"),
             Z_img=Z_img, Z_txt=Z_txt, A_img=A_img, A_txt=A_txt,
             delta_img=delta_img, delta_txt=delta_txt,
             d_anchor_img=d_anchor_img, d_anchor_txt=d_anchor_txt,
             T_img=T_img, T_txt=T_txt, rho=rho)

    # ── Tabulate per-iteration stats ──────────────────────────────────────────
    rows = []
    for k in range(K):
        rows.append({
            "iter": k,
            "anchor_l2_img_mean":   float(d_anchor_img[:, k].mean()),
            "anchor_l2_img_std":    float(d_anchor_img[:, k].std()),
            "anchor_l2_txt_mean":   float(d_anchor_txt[:, k].mean()),
            "anchor_l2_txt_std":    float(d_anchor_txt[:, k].std()),
            "delta_img_mean":       float(delta_img[:, k-1].mean()) if k > 0 else 0,
            "delta_img_std":        float(delta_img[:, k-1].std())  if k > 0 else 0,
            "delta_txt_mean":       float(delta_txt[:, k-1].mean()) if k > 0 else 0,
            "delta_txt_std":        float(delta_txt[:, k-1].std())  if k > 0 else 0,
            "modality_coupling_rho": float(rho[k-1])  if k > 0 else np.nan,
        })
    df_A = pd.DataFrame(rows)
    table_dir = os.path.join(out_dir, "tables")
    os.makedirs(table_dir, exist_ok=True)
    df_A.to_csv(os.path.join(table_dir, "A_geometry.csv"), index=False)
    logger.info(f"  Geometry table → tables/A_geometry.csv")

    # ── Figure A1: step size and anchor distance vs iteration ─────────────────
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    iters_full = np.arange(K)
    iters_step = np.arange(1, K)

    ax = axes[0]
    for arr, lbl, c in [(d_anchor_img, "Image", "C0"),
                        (d_anchor_txt, "Text",  "C1")]:
        m = arr.mean(0)
        s = arr.std(0)
        ax.plot(iters_full, m, color=c, label=lbl, marker="o", ms=3)
        ax.fill_between(iters_full, m - s, m + s, color=c, alpha=0.18)
    ax.set_xlabel("Iteration $k$")
    ax.set_ylabel(r"Anchor distance $\|z_k - z_0^{\rm GT}\|_2$")
    ax.set_title("Anchor distance vs. iteration")
    ax.legend(loc="lower right")

    ax = axes[1]
    for arr, lbl, c in [(delta_img, "Image", "C0"),
                        (delta_txt, "Text",  "C1")]:
        m = arr.mean(0)
        s = arr.std(0)
        ax.plot(iters_step, m, color=c, label=lbl, marker="o", ms=3)
        ax.fill_between(iters_step, m - s, m + s, color=c, alpha=0.18)
    ax.set_xlabel("Iteration $k$")
    ax.set_ylabel(r"Step size $\|z_k - z_{k-1}\|_2$")
    ax.set_title("Step size vs. iteration")
    ax.legend(loc="upper right")

    plt.tight_layout()
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, "A1_convergence.pdf"))
    plt.close(fig)
    logger.info(f"  Figure → figures/A1_convergence.pdf")

    # ── Figure A2: tortuosity ─────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.5))
    bins = np.linspace(1.0, max(T_img.max(), T_txt.max(), 5.0), 30)
    ax.hist(T_img, bins=bins, alpha=0.55, label=f"Image ($\\overline{{T}}={T_img.mean():.2f}$)", color="C0")
    ax.hist(T_txt, bins=bins, alpha=0.55, label=f"Text ($\\overline{{T}}={T_txt.mean():.2f}$)",  color="C1")
    ax.axvline(1.0, color="k", ls="--", lw=0.6)
    ax.set_xlabel("Tortuosity $T = L/D$")
    ax.set_ylabel("Count")
    ax.set_title("Trajectory tortuosity")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "A2_tortuosity.pdf"))
    plt.close(fig)
    logger.info(f"  Figure → figures/A2_tortuosity.pdf")

    # ── Figure A3: modality coupling ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.plot(iters_step, rho, marker="o", ms=4, color="C2")
    ax.axhline(0, color="k", ls="--", lw=0.6)
    ax.set_xlabel("Iteration $k$")
    ax.set_ylabel(r"Cross-modal coupling $\rho_k$")
    ax.set_title(r"Image-text step coupling $\rho_k = {\rm corr}(\Delta_k^{\rm img},\Delta_k^{\rm txt})$")
    ax.set_ylim(-0.5, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "A3_modality_coupling.pdf"))
    plt.close(fig)
    logger.info(f"  Figure → figures/A3_modality_coupling.pdf")

    return {"Z_img": Z_img, "Z_txt": Z_txt, "A_img": A_img, "A_txt": A_txt,
            "delta_img": delta_img, "delta_txt": delta_txt,
            "d_anchor_img": d_anchor_img, "d_anchor_txt": d_anchor_txt,
            "T_img": T_img, "T_txt": T_txt, "rho": rho,
            "df_A": df_A}


# ══════════════════════════════════════════════════════════════════════════════
#  BLOCK B — LYAPUNOV STABILITY
# ══════════════════════════════════════════════════════════════════════════════

def block_B_lyapunov(trajs, lyap_data, basinB, out_dir):
    """
    B1: System-wide Lyapunov from main run + per-anchor from seed replicates.
    B2: Basin radius estimation.
    B3: Empirical Lyapunov function check.
    B4: First-passage time (skipped in v1, listed in TODO).
    """
    logger.info("\n" + "═" * 60)
    logger.info("BLOCK B — Lyapunov Stability")
    logger.info("═" * 60)

    fig_dir   = os.path.join(out_dir, "figures")
    cache_dir = os.path.join(out_dir, "cache")
    table_dir = os.path.join(out_dir, "tables")

    Z = np.stack([t["img_traj"] for t in trajs])  # (N, K, 256)
    N, K, D = Z.shape

    # ── B1a: System-wide finite-time Lyapunov from main-run pairs ─────────────
    # Without same-anchor replicates we use the natural pairwise distance ratio:
    # for each pair (i,j) of *different* studies, ratio of their distances at
    # iter k vs. iter 0. Negative log-mean ⇒ contraction toward shared attractor.
    # This is a proxy: it conflates "shared attractor" with "same basin".
    # The per-anchor measurement from replicates (B1b) is the rigorous one.
    logger.info("  B1a: system-wide divergence rate from pairwise distance ratios")
    lambda_sys = np.full(K, np.nan)
    pairs = np.array([(i, j) for i in range(N) for j in range(i+1, N)])
    if len(pairs) > 0:
        d0 = np.linalg.norm(Z[pairs[:, 0], 0] - Z[pairs[:, 1], 0], axis=-1)
        for k in range(K):
            dk = np.linalg.norm(Z[pairs[:, 0], k] - Z[pairs[:, 1], k], axis=-1)
            ratios = dk / np.maximum(d0, 1e-9)
            ratios = ratios[(ratios > 0) & np.isfinite(ratios)]
            if len(ratios) > 0:
                lambda_sys[k] = float(np.mean(np.log(ratios)) / max(k, 1))

    logger.info(f"    λ_sys at k=K-1: {lambda_sys[-1]:+.4f}  "
                f"({'contractive' if lambda_sys[-1] < 0 else 'expansive'})")

    # ── B1b: per-anchor Lyapunov from seed replicates ─────────────────────────
    lyap_per_anchor = {}
    if lyap_data:
        logger.info("  B1b: per-anchor Lyapunov from seed replicates")
        for sid, replicates in lyap_data.items():
            if len(replicates) < 2:
                continue
            R = np.stack([r["img_traj"] for r in replicates])  # (J, K_l, 256)
            J, K_l, _ = R.shape

            # Pairwise distance among replicates at each iter
            lambdas = np.zeros(K_l)
            for k in range(K_l):
                pd_k = pdist(R[:, k, :])
                if k == 0:
                    pd_0 = pd_k.copy()
                # log expansion ratio
                if k > 0:
                    ratios = pd_k / np.maximum(pd_0, 1e-9)
                    ratios = ratios[(ratios > 0) & np.isfinite(ratios)]
                    if len(ratios) > 0:
                        lambdas[k] = float(np.mean(np.log(ratios)) / k)

            lyap_per_anchor[sid] = lambdas
            logger.info(f"    anchor {sid}: λ at final = {lambdas[-1]:+.4f}")

    # ── B2: basin radius via cluster spread (deferred until C runs) ───────────
    # Reads from basinB if Block C already provided clustering.
    # We do a simple first-pass estimate here using endpoint dispersion; full
    # version updates after Block C.
    endpoint_dispersion = float(np.std(np.linalg.norm(Z[:, -1, :], axis=-1)))
    pairwise_endpoints = pdist(Z[:, -1, :])
    basin_radius_p95 = float(np.percentile(pairwise_endpoints, 95))
    basin_radius_p50 = float(np.percentile(pairwise_endpoints, 50))
    logger.info(f"  B2: endpoint dispersion (system-wide):")
    logger.info(f"      median pairwise = {basin_radius_p50:.4f}")
    logger.info(f"      95-pct pairwise = {basin_radius_p95:.4f}")

    # ── B3: Empirical Lyapunov function V(z) = ||z - c||² ────────────────────
    # c = global endpoint centroid
    c_global = Z[:, -1, :].mean(0)
    V_traj = np.linalg.norm(Z - c_global[None, None, :], axis=-1) ** 2  # (N, K)
    monotone_count = 0
    for i in range(N):
        if np.all(np.diff(V_traj[i]) <= 1e-6):
            monotone_count += 1
    monotone_frac = monotone_count / N
    logger.info(f"  B3: V(z)=||z-c||² monotone-decreasing on {monotone_count}/{N} "
                f"({100*monotone_frac:.1f}%) trajectories")

    # ── Save ──────────────────────────────────────────────────────────────────
    np.savez(os.path.join(cache_dir, "B_lyapunov.npz"),
             lambda_sys=lambda_sys,
             c_global=c_global, V_traj=V_traj,
             basin_radius_p50=basin_radius_p50,
             basin_radius_p95=basin_radius_p95)

    # ── Figure B: Lyapunov panel ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    ks = np.arange(K)

    ax = axes[0]
    ax.plot(ks, lambda_sys, marker="o", ms=4, color="C3", label="System-wide (pairwise)")
    if lyap_per_anchor:
        for sid, lambdas in lyap_per_anchor.items():
            # Skip k=0 entry (always 0 by construction)
            xs = np.arange(1, len(lambdas))
            ax.plot(xs, lambdas[1:], alpha=0.6, lw=0.8,
                    label=f"anchor {sid[:8]}")
    ax.axhline(0, color="k", ls="--", lw=0.6)
    ax.set_xlabel("Iteration $k$")
    ax.set_ylabel(r"$\lambda_k$")
    ax.set_title("Finite-time Lyapunov exponent")
    ax.legend(loc="best", fontsize=7)

    ax = axes[1]
    ax.plot(ks, V_traj.mean(0), color="C4", marker="o", ms=4)
    ax.fill_between(ks,
                    V_traj.mean(0) - V_traj.std(0),
                    V_traj.mean(0) + V_traj.std(0),
                    alpha=0.2, color="C4")
    ax.set_xlabel("Iteration $k$")
    ax.set_ylabel(r"$V(z) = \|z - c\|^2$")
    ax.set_title(f"Lyapunov function (monotone on {100*monotone_frac:.0f}% of traj.)")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "B_lyapunov.pdf"))
    plt.close(fig)
    logger.info(f"  Figure → figures/B_lyapunov.pdf")

    return {"lambda_sys": lambda_sys, "lyap_per_anchor": lyap_per_anchor,
            "V_traj": V_traj, "monotone_frac": monotone_frac,
            "basin_radius_p95": basin_radius_p95,
            "c_global": c_global}


# ══════════════════════════════════════════════════════════════════════════════
#  BLOCK C — MULTI-BASIN STRUCTURE
# ══════════════════════════════════════════════════════════════════════════════

def _cluster_in_space(X, k_range, metric_for_silhouette, label_for_log):
    """Helper: run silhouette + gap + HDBSCAN k-means on a representation X.

    For raw 256-d we pass L2-normalized X with metric='cosine' on silhouette;
    for PCA configs we pass the PCA projection with metric='euclidean'.

    Returns dict with sil_scores, gap_scores, n_hdb_clusters, n_noise,
    best_K, best_K_gap, basin_labels, km_centers.
    """
    sil_scores, gap_scores = {}, {}

    for k_try in range(k_range[0], k_range[1] + 1):
        if k_try >= X.shape[0]: break
        km = KMeans(n_clusters=k_try, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        if len(np.unique(labels)) > 1:
            sil_scores[k_try] = float(
                silhouette_score(X, labels, metric=metric_for_silhouette)
            )

    rng = np.random.default_rng(42)
    n_null = 20
    bbox_min, bbox_max = X.min(axis=0), X.max(axis=0)
    for k_try in range(k_range[0], k_range[1] + 1):
        if k_try >= X.shape[0]: break
        km = KMeans(n_clusters=k_try, random_state=42, n_init=10).fit(X)
        Wk_log = np.log(km.inertia_ + 1e-12)
        Wk_null_logs = []
        for _ in range(n_null):
            null_data = rng.uniform(bbox_min, bbox_max, size=X.shape)
            km_null = KMeans(n_clusters=k_try, random_state=42, n_init=5).fit(null_data)
            Wk_null_logs.append(np.log(km_null.inertia_ + 1e-12))
        gap_scores[k_try] = float(np.mean(Wk_null_logs) - Wk_log)

    best_K = max(sil_scores, key=sil_scores.get) if sil_scores else 2
    best_K_gap = max(gap_scores, key=gap_scores.get) if gap_scores else 2

    n_hdb_clusters, n_noise, hdb_labels = 0, X.shape[0], np.full(X.shape[0], -1)
    try:
        import hdbscan
        # Use euclidean metric — for L2-normalized vectors this is monotone
        # with cosine, and for PCA representations it's the natural choice.
        clusterer = hdbscan.HDBSCAN(min_cluster_size=10, metric="euclidean",
                                     cluster_selection_method="eom")
        hdb_labels = clusterer.fit_predict(X)
        n_hdb_clusters = int((np.unique(hdb_labels) >= 0).sum())
        n_noise = int((hdb_labels == -1).sum())
    except ImportError:
        logger.warning(f"  [{label_for_log}] hdbscan not installed — skipping density check")

    km_final = KMeans(n_clusters=best_K, random_state=42, n_init=10).fit(X)
    basin_labels = km_final.labels_

    logger.info(f"  [{label_for_log}] Silhouette: {sil_scores}")
    logger.info(f"  [{label_for_log}] Best K = {best_K} (sil={sil_scores.get(best_K, np.nan):.4f})")
    logger.info(f"  [{label_for_log}] Gap: {gap_scores}")
    logger.info(f"  [{label_for_log}] Best K (gap) = {best_K_gap} "
                f"(gap={gap_scores.get(best_K_gap, np.nan):.4f})")
    logger.info(f"  [{label_for_log}] HDBSCAN: {n_hdb_clusters} clusters, "
                f"{n_noise}/{X.shape[0]} ({100*n_noise/X.shape[0]:.0f}%) noise")

    return {
        "sil_scores":     sil_scores,
        "gap_scores":     gap_scores,
        "best_K":         best_K,
        "best_K_gap":     best_K_gap,
        "n_hdb_clusters": n_hdb_clusters,
        "n_noise":        n_noise,
        "hdb_labels":     hdb_labels,
        "basin_labels":   basin_labels,
        "km_centers":     km_final.cluster_centers_,
    }


def block_C_basins(trajs, A_results, out_dir, k_range=(2, 8)):
    """
    Cluster iter-K endpoints to identify ergodic components / basins.

    Runs THREE configurations side-by-side as sensitivity analyses:
      (1) raw 256-d L2-normalized cosine    ← canonical, downstream-feeding
      (2) PCA-20 euclidean                  ← legacy / sensitivity
      (3) PCA-50 euclidean                  ← additional sensitivity check

    The "raw" config feeds basin labels to Blocks D and F (most defensible —
    no dimensionality-reduction assumption). PCA configs are reported as
    sensitivity checks.

    Reviewer-defense rationale:
      • In high-d cosine-normalized spaces, k-means + cosine silhouette is
        valid (Aggarwal 2001 concentration-of-distances applies to
        unconstrained data, less to data on the unit sphere).
      • PCA-20 captures >90% variance but truncates structure that may
        matter; PCA-50 is the conservative middle ground. If all three
        agree qualitatively (weak silhouette, monotone gap, high HDBSCAN
        noise), the conclusion is robust to the dimensionality choice.
    """
    logger.info("\n" + "═" * 60)
    logger.info("BLOCK C — Multi-Basin Structure (3 configs)")
    logger.info("═" * 60)

    fig_dir   = os.path.join(out_dir, "figures")
    cache_dir = os.path.join(out_dir, "cache")
    table_dir = os.path.join(out_dir, "tables")

    Z = A_results["Z_img"]    # (N, K, 256)
    N, K, D = Z.shape
    endpoints_full = Z[:, -1, :]  # (N, 256)

    # ── Config 1: raw 256-d cosine (CANONICAL) ────────────────────────────────
    # MedCLIP outputs are already L2-normalized but we re-normalize to be safe.
    # With L2-normalized vectors, euclidean distance is monotone with cosine
    # distance, so euclidean k-means + cosine silhouette is consistent.
    logger.info("\n--- Config 1: raw 256-d cosine ---")
    norms = np.linalg.norm(endpoints_full, axis=1, keepdims=True)
    endpoints_norm = endpoints_full / np.maximum(norms, 1e-12)
    cfg_raw = _cluster_in_space(
        endpoints_norm, k_range,
        metric_for_silhouette="cosine", label_for_log="raw256-cos",
    )

    # ── Config 2: PCA-20 euclidean ────────────────────────────────────────────
    logger.info("\n--- Config 2: PCA-20 euclidean ---")
    from sklearn.decomposition import PCA
    n_pca20 = min(20, N - 1)
    pca20 = PCA(n_components=n_pca20, random_state=42).fit(endpoints_full)
    endpoints_pca20 = pca20.transform(endpoints_full)
    var20 = float(pca20.explained_variance_ratio_.sum())
    logger.info(f"  PCA-20 variance explained: {var20*100:.1f}%")
    cfg_pca20 = _cluster_in_space(
        endpoints_pca20, k_range,
        metric_for_silhouette="euclidean", label_for_log="PCA20-euc",
    )

    # ── Config 3: PCA-50 euclidean ────────────────────────────────────────────
    logger.info("\n--- Config 3: PCA-50 euclidean ---")
    n_pca50 = min(50, N - 1)
    pca50 = PCA(n_components=n_pca50, random_state=42).fit(endpoints_full)
    endpoints_pca50 = pca50.transform(endpoints_full)
    var50 = float(pca50.explained_variance_ratio_.sum())
    logger.info(f"  PCA-50 variance explained: {var50*100:.1f}%")
    cfg_pca50 = _cluster_in_space(
        endpoints_pca50, k_range,
        metric_for_silhouette="euclidean", label_for_log="PCA50-euc",
    )

    # ── Canonical basin assignment: raw 256-d cosine ─────────────────────────
    basin_labels = cfg_raw["basin_labels"]
    best_K       = cfg_raw["best_K"]
    best_K_gap   = cfg_raw["best_K_gap"]
    sil_scores   = cfg_raw["sil_scores"]
    gap_scores   = cfg_raw["gap_scores"]
    n_hdbscan_clusters = cfg_raw["n_hdb_clusters"]
    n_noise      = cfg_raw["n_noise"]
    hdb_labels   = cfg_raw["hdb_labels"]
    # K-means centroids in L2-normalized space; renormalize to unit sphere
    centers_raw = cfg_raw["km_centers"]
    cn = np.linalg.norm(centers_raw, axis=1, keepdims=True)
    basin_centers = centers_raw / np.maximum(cn, 1e-12)

    logger.info(f"\n  CANONICAL: raw 256-d cosine, K={best_K}, "
                f"silhouette={sil_scores.get(best_K, np.nan):.4f}, "
                f"HDBSCAN noise={100*n_noise/N:.0f}%")

    # ── Per-basin statistics (in original 256-d space) ───────────────────────
    basin_stats = []
    for b in range(best_K):
        members = np.where(basin_labels == b)[0]
        if len(members) == 0: continue
        pts = endpoints_full[members]
        cov = np.cov(pts.T)
        eigs = np.linalg.eigvalsh(cov)
        eigs = eigs[eigs > 1e-12]
        PR = (eigs.sum() ** 2) / (eigs ** 2).sum() if len(eigs) else 0
        d_c = np.linalg.norm(pts - basin_centers[b], axis=1)
        basin_stats.append({
            "basin": b,
            "n_members":     int(len(members)),
            "centroid_norm": float(np.linalg.norm(basin_centers[b])),
            "mean_radius":   float(d_c.mean()),
            "p95_radius":    float(np.percentile(d_c, 95)),
            "PR_eff_dim":    float(PR),
            "members":       list(members.astype(int)),
        })
        logger.info(f"    basin {b}: n={len(members):3d}  PR={PR:.2f}  "
                    f"meanR={d_c.mean():.3f}  p95R={np.percentile(d_c, 95):.3f}")

    # ── Sliced-W2 to iter-K distribution (256-d, no PCA) ─────────────────────
    rng = np.random.default_rng(42)
    n_proj = 100
    rnd_vecs = rng.normal(size=(n_proj, D))
    rnd_vecs /= np.linalg.norm(rnd_vecs, axis=1, keepdims=True)
    W2_to_final = np.zeros(K)
    final_dist = endpoints_full
    for k in range(K):
        cur = Z[:, k, :]
        sw = 0.0
        for v in rnd_vecs:
            a = np.sort(cur @ v); b = np.sort(final_dist @ v)
            sw += float(np.mean((a - b) ** 2))
        W2_to_final[k] = np.sqrt(sw / n_proj)
    logger.info(f"  Sliced W₂ to iter-K endpoint dist.: "
                f"{W2_to_final[0]:.3f} → {W2_to_final[-1]:.3f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    np.savez(os.path.join(cache_dir, "C_basins.npz"),
             # canonical (raw 256-d cosine)
             basin_labels=basin_labels,
             basin_centers=basin_centers,
             sil_scores=np.array(list(sil_scores.values())),
             sil_K=np.array(list(sil_scores.keys())),
             gap_scores=np.array(list(gap_scores.values())),
             gap_K=np.array(list(gap_scores.keys())),
             hdb_labels=hdb_labels,
             n_hdbscan_clusters=n_hdbscan_clusters,
             n_noise=n_noise,
             W2_to_final=W2_to_final,
             best_K=best_K,
             best_K_gap=best_K_gap,
             # PCA-20 sensitivity
             pca20_sil_K=np.array(list(cfg_pca20["sil_scores"].keys())),
             pca20_sil_v=np.array(list(cfg_pca20["sil_scores"].values())),
             pca20_gap_K=np.array(list(cfg_pca20["gap_scores"].keys())),
             pca20_gap_v=np.array(list(cfg_pca20["gap_scores"].values())),
             pca20_best_K=cfg_pca20["best_K"],
             pca20_n_hdb=cfg_pca20["n_hdb_clusters"],
             pca20_n_noise=cfg_pca20["n_noise"],
             pca20_var_explained=var20,
             # PCA-50 sensitivity
             pca50_sil_K=np.array(list(cfg_pca50["sil_scores"].keys())),
             pca50_sil_v=np.array(list(cfg_pca50["sil_scores"].values())),
             pca50_gap_K=np.array(list(cfg_pca50["gap_scores"].keys())),
             pca50_gap_v=np.array(list(cfg_pca50["gap_scores"].values())),
             pca50_best_K=cfg_pca50["best_K"],
             pca50_n_hdb=cfg_pca50["n_hdb_clusters"],
             pca50_n_noise=cfg_pca50["n_noise"],
             pca50_var_explained=var50)
    pd.DataFrame(basin_stats).drop(columns=["members"]).to_csv(
        os.path.join(table_dir, "C_basins.csv"), index=False)
    with open(os.path.join(cache_dir, "C_basin_members.json"), "w") as f:
        json.dump({
            "study_ids":      [t["study_id"] for t in trajs],
            "basin_labels":   basin_labels.tolist(),
            "basin_centers_norms": [float(np.linalg.norm(c)) for c in basin_centers],
        }, f, indent=2)

    # Configuration-comparison table for reviewer-defense
    cmp_rows = []
    for cfg_name, cfg in [("raw_256d_cos", cfg_raw),
                            ("pca20_euc",    cfg_pca20),
                            ("pca50_euc",    cfg_pca50)]:
        cmp_rows.append({
            "config":         cfg_name,
            "best_K_sil":     cfg["best_K"],
            "best_sil":       cfg["sil_scores"].get(cfg["best_K"], np.nan),
            "best_K_gap":     cfg["best_K_gap"],
            "best_gap":       cfg["gap_scores"].get(cfg["best_K_gap"], np.nan),
            "hdb_n_clusters": cfg["n_hdb_clusters"],
            "hdb_pct_noise":  100 * cfg["n_noise"] / N,
        })
    pd.DataFrame(cmp_rows).to_csv(
        os.path.join(table_dir, "C_config_comparison.csv"), index=False)

    # ── Figure: 3 configs × 4 metrics = 3×4 grid ─────────────────────────────
    fig, axes = plt.subplots(3, 4, figsize=(14, 8.5))
    cfgs_for_plot = [("raw 256-d cos",    cfg_raw,   "C0"),
                       ("PCA-20 euclid",    cfg_pca20, "C1"),
                       ("PCA-50 euclid",    cfg_pca50, "C2")]
    for row_i, (name, cfg, color) in enumerate(cfgs_for_plot):
        ax = axes[row_i, 0]
        ks = np.array(list(cfg["sil_scores"].keys()))
        vs = np.array(list(cfg["sil_scores"].values()))
        ax.plot(ks, vs, marker="o", ms=5, color=color)
        ax.axvline(cfg["best_K"], ls="--", lw=0.7, color="k",
                    label=f"best K={cfg['best_K']}")
        ax.set_xlabel(r"$K$"); ax.set_ylabel("Silhouette")
        ax.set_title(f"{name}: silhouette", fontsize=10)
        ax.legend(fontsize=8)

        ax = axes[row_i, 1]
        gks = np.array(list(cfg["gap_scores"].keys()))
        gvs = np.array(list(cfg["gap_scores"].values()))
        ax.plot(gks, gvs, marker="s", ms=5, color=color)
        ax.set_xlabel(r"$K$"); ax.set_ylabel("Gap")
        ax.set_title(f"{name}: gap statistic", fontsize=10)

        ax = axes[row_i, 2]
        ax.bar([0, 1], [cfg["n_hdb_clusters"], cfg["n_noise"]],
                color=[color, "C7"])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["#clusters", "#noise"])
        ax.set_title(f"{name}: HDBSCAN ({100*cfg['n_noise']/N:.0f}% noise)",
                      fontsize=10)

        ax = axes[row_i, 3]
        counts_b = np.bincount(cfg["basin_labels"], minlength=cfg["best_K"])
        ax.bar(np.arange(cfg["best_K"]), counts_b,
                color=plt.cm.tab10(np.arange(cfg["best_K"]) % 10))
        ax.set_xlabel("basin id"); ax.set_ylabel("studies")
        ax.set_title(f"{name}: K={cfg['best_K']} basin sizes", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "C_basins.pdf"))
    plt.close(fig)
    logger.info(f"  Figure → figures/C_basins.pdf")

    return {
        "basin_labels":  basin_labels,
        "basin_centers": basin_centers,
        "best_K":        best_K,
        "best_K_gap":    best_K_gap,
        "sil_scores":    sil_scores,
        "gap_scores":    gap_scores,
        "n_hdbscan_clusters": n_hdbscan_clusters,
        "n_noise":       n_noise,
        "W2_to_final":   W2_to_final,
        "basin_stats":   basin_stats,
        "configs": {
            "raw_256d_cos": cfg_raw,
            "pca20_euc":    cfg_pca20,
            "pca50_euc":    cfg_pca50,
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
#  BLOCK D — PHASE PORTRAITS
# ══════════════════════════════════════════════════════════════════════════════

def block_D_phase_portrait(trajs, ref, A_results, C_results, out_dir):
    """
    Phase portrait in 2D UMAP fitted on training corpus.
    Two figures: image-space portrait + text-space portrait.
    """
    logger.info("\n" + "═" * 60)
    logger.info("BLOCK D — Phase Portraits (2D UMAP)")
    logger.info("═" * 60)

    fig_dir = os.path.join(out_dir, "figures")
    cache_dir = os.path.join(out_dir, "cache")

    Z_img = A_results["Z_img"]
    Z_txt = A_results["Z_txt"]
    A_img = A_results["A_img"]
    A_txt = A_results["A_txt"]
    basin_labels = C_results["basin_labels"]

    N, K, D = Z_img.shape

    # ── Project trajectories into 2D via cached UMAP reducers ─────────────────
    logger.info("  Projecting trajectories through fitted UMAP...")
    flat_img = Z_img.reshape(-1, D)
    flat_txt = Z_txt.reshape(-1, D)

    proj_img = ref["umap_img"].transform(flat_img).reshape(N, K, 2)
    proj_txt = ref["umap_txt"].transform(flat_txt).reshape(N, K, 2)
    proj_anchor_img = ref["umap_img"].transform(A_img)  # (N, 2)
    proj_anchor_txt = ref["umap_txt"].transform(A_txt)

    np.savez(os.path.join(cache_dir, "D_phase_portrait.npz"),
             proj_img=proj_img, proj_txt=proj_txt,
             proj_anchor_img=proj_anchor_img, proj_anchor_txt=proj_anchor_txt)

    def plot_portrait(proj_traj, proj_anchor, ref_2d, ref_valid, name, basin_labels, fname):
        fig, ax = plt.subplots(figsize=(5.5, 5.0))
        # Background: training-corpus density
        valid = ref_valid & np.isfinite(ref_2d).all(axis=1)
        ax.hexbin(ref_2d[valid, 0], ref_2d[valid, 1],
                  gridsize=80, cmap="Greys", mincnt=2, alpha=0.45)

        cmap = plt.cm.tab10
        for i in range(N):
            color = cmap(basin_labels[i] % 10)
            x = proj_traj[i, :, 0]
            y = proj_traj[i, :, 1]
            ax.plot(x, y, "-", color=color, lw=0.6, alpha=0.55)
            # Arrowheads at each iteration midpoint
            for k in range(1, K):
                ax.annotate("", xy=(x[k], y[k]), xytext=(x[k-1], y[k-1]),
                            arrowprops=dict(arrowstyle="->", color=color,
                                            alpha=0.45, lw=0.5))
            # Mark start/end
            ax.plot(proj_anchor[i, 0], proj_anchor[i, 1], "o", color=color,
                    ms=5, mec="k", mew=0.5, alpha=0.9)
            ax.plot(x[-1], y[-1], "*", color=color, ms=9, mec="k", mew=0.5)

        ax.set_xlabel(f"UMAP-1 ({name})")
        ax.set_ylabel(f"UMAP-2 ({name})")
        ax.set_title(f"Phase portrait — {name} space\n"
                     f"(grey: training-corpus density, "
                     f"○ anchors, ★ iter-{K-1} endpoints, color: basin)")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, fname))
        plt.close(fig)
        logger.info(f"  Figure → figures/{fname}")

    plot_portrait(
        proj_img, proj_anchor_img, ref["img_2d"], ref["img_valid"],
        "image", basin_labels, "D1_phase_portrait_image.pdf",
    )
    plot_portrait(
        proj_txt, proj_anchor_txt, ref["txt_2d"], ref["txt_valid"],
        "text", basin_labels, "D2_phase_portrait_text.pdf",
    )

    # ── Vector field in 2D (image only) ───────────────────────────────────────
    # Bin space into a grid, average displacement vectors, plot as quiver
    logger.info("  Computing 2D vector field (image space)...")
    proj_flat = proj_img.reshape(-1, 2)
    deltas    = proj_img[:, 1:] - proj_img[:, :-1]
    deltas    = deltas.reshape(-1, 2)                  # (N*(K-1), 2)
    starts    = proj_img[:, :-1].reshape(-1, 2)

    n_bins = 18
    x_edges = np.linspace(proj_flat[:, 0].min(), proj_flat[:, 0].max(), n_bins + 1)
    y_edges = np.linspace(proj_flat[:, 1].min(), proj_flat[:, 1].max(), n_bins + 1)
    X = (x_edges[:-1] + x_edges[1:]) / 2
    Y = (y_edges[:-1] + y_edges[1:]) / 2
    U = np.zeros((n_bins, n_bins))
    V = np.zeros((n_bins, n_bins))
    counts = np.zeros((n_bins, n_bins), dtype=int)

    for s, d in zip(starts, deltas):
        ix = np.searchsorted(x_edges, s[0]) - 1
        iy = np.searchsorted(y_edges, s[1]) - 1
        if 0 <= ix < n_bins and 0 <= iy < n_bins:
            U[iy, ix] += d[0]
            V[iy, ix] += d[1]
            counts[iy, ix] += 1
    valid = counts > 1
    U[valid] /= counts[valid]
    V[valid] /= counts[valid]
    U[~valid] = np.nan
    V[~valid] = np.nan

    fig, ax = plt.subplots(figsize=(6.0, 5.5))
    valid_ref = ref["img_valid"] & np.isfinite(ref["img_2d"]).all(axis=1)
    ax.hexbin(ref["img_2d"][valid_ref, 0], ref["img_2d"][valid_ref, 1],
              gridsize=80, cmap="Greys", mincnt=2, alpha=0.4)
    XX, YY = np.meshgrid(X, Y)
    mag = np.sqrt(U ** 2 + V ** 2)
    ax.quiver(XX, YY, U, V, mag, cmap="viridis", scale_units="xy",
              angles="xy", scale=1.0, width=0.004, alpha=0.85)
    # Overlay basin centers
    centers_2d = ref["umap_img"].transform(C_results["basin_centers"])
    for b, c2 in enumerate(centers_2d):
        ax.plot(c2[0], c2[1], "X", color=plt.cm.tab10(b % 10),
                ms=15, mec="k", mew=1.0)
        ax.annotate(f"B{b}", xy=(c2[0], c2[1]), xytext=(7, 7),
                    textcoords="offset points", fontsize=10, fontweight="bold")
    ax.set_xlabel("UMAP-1 (image)")
    ax.set_ylabel("UMAP-2 (image)")
    ax.set_title("Mean displacement vector field — image space\n"
                 "(arrows: average step direction at each grid cell; ✕: basin centers)")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "D3_vector_field_image.pdf"))
    plt.close(fig)
    logger.info(f"  Figure → figures/D3_vector_field_image.pdf")

    return {"proj_img": proj_img, "proj_txt": proj_txt,
            "proj_anchor_img": proj_anchor_img,
            "proj_anchor_txt": proj_anchor_txt}


# ══════════════════════════════════════════════════════════════════════════════
#  BLOCK E — HIGH-DIMENSIONAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def block_E_high_dim(A_results, out_dir):
    """
    Effective dimensionality, spectral analysis, mutual information vs k.
    (Persistent homology requires gudhi/ripser — we skip in the default run
    and leave a stub for the supplementary analysis.)
    """
    logger.info("\n" + "═" * 60)
    logger.info("BLOCK E — High-Dimensional Analysis")
    logger.info("═" * 60)

    fig_dir = os.path.join(out_dir, "figures")
    cache_dir = os.path.join(out_dir, "cache")

    Z_img = A_results["Z_img"]   # (N, K, 256)
    Z_txt = A_results["Z_txt"]
    N, K, D = Z_img.shape

    # ── E1: Per-iteration participation ratio ─────────────────────────────────
    PR_img = np.zeros(K)
    PR_txt = np.zeros(K)
    for k in range(K):
        for arr, dst in [(Z_img[:, k], PR_img), (Z_txt[:, k], PR_txt)]:
            cov = np.cov(arr.T)
            eigs = np.linalg.eigvalsh(cov)
            eigs = eigs[eigs > 1e-12]
            if len(eigs):
                dst[k] = (eigs.sum() ** 2) / (eigs ** 2).sum()
    logger.info(f"  PR image: {PR_img[0]:.1f} → {PR_img[-1]:.1f}")
    logger.info(f"  PR text : {PR_txt[0]:.1f} → {PR_txt[-1]:.1f}")

    # ── E2: SVD of the trajectory tensor (image only for headline) ────────────
    # Center per-iteration, stack, take SVD across studies
    Z_centered = Z_img - Z_img.mean(0, keepdims=True)
    Z_flat = Z_centered.reshape(N, K * D)
    U, s, Vt = np.linalg.svd(Z_flat, full_matrices=False)
    eig_spectrum = s ** 2 / (s ** 2).sum()
    cum_var = np.cumsum(eig_spectrum)
    rank95 = int(np.argmax(cum_var >= 0.95)) + 1
    logger.info(f"  SVD: rank for 95% variance = {rank95}/{min(N, K*D)}")

    # ── E3: Mutual information between z_0 and z_k via Kraskov estimator ──────
    # FIX 1: PCA must be fit on the UNION of all trajectory points across all
    # iterations, not just z_0. Otherwise z_k drifts into a slightly different
    # subspace and its projection loses variance, artificially deflating MI.
    # Fitting on the union gives both modalities a fair representation.
    #
    # MI in 256-d with N=100 is statistically unreliable (curse of dimensionality:
    # KSG variance grows exponentially with d for fixed N). Standard practice:
    # project to a low-d PCA subspace first. We use 20 components, capturing
    # ~95% variance based on Block E2's spectrum.
    #
    # We use sklearn.feature_selection.mutual_info_regression which implements
    # KSG correctly (proper open-ball convention) and is much faster than a
    # hand-rolled radius_neighbors loop.
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import mutual_info_regression

    # Fit PCA on the union of all trajectory points (per modality)
    n_components_mi = 20
    Z_img_flat = Z_img.reshape(-1, D)  # (N*K, 256)
    Z_txt_flat = Z_txt.reshape(-1, D)
    pca_img_mi = PCA(n_components=min(n_components_mi, Z_img_flat.shape[0]-1),
                      random_state=42).fit(Z_img_flat)
    pca_txt_mi = PCA(n_components=min(n_components_mi, Z_txt_flat.shape[0]-1),
                      random_state=42).fit(Z_txt_flat)
    logger.info(f"  PCA-{n_components_mi} fit on union of trajectory points: "
                f"img var={pca_img_mi.explained_variance_ratio_.sum()*100:.1f}%  "
                f"txt var={pca_txt_mi.explained_variance_ratio_.sum()*100:.1f}%")

    def mi_via_shared_pca(X0_full, Xk_full, pca_fitted):
        """Project both arrays through the same PCA basis, then compute mean
        univariate MI(X0[:,d], Xk[:,d]) over PCA dimensions.

        This is a 1-d-per-axis MI proxy, NOT the full multivariate MI.
        Used because full multivariate MI in high d is poorly estimated at
        N=100. The trend over k is what matters (decreasing → forgetting),
        which the proxy preserves.
        """
        if X0_full.shape[0] < n_components_mi + 5:
            return np.nan
        X0_p = pca_fitted.transform(X0_full)
        Xk_p = pca_fitted.transform(Xk_full)
        mis = []
        for d_idx in range(X0_p.shape[1]):
            try:
                mi = mutual_info_regression(
                    X0_p[:, d_idx:d_idx+1], Xk_p[:, d_idx],
                    n_neighbors=3, random_state=42,
                )[0]
                mis.append(mi)
            except Exception:
                pass
        return float(np.mean(mis)) if mis else np.nan

    logger.info("  Computing MI(z_0, z_k) via shared-PCA(20) + KSG...")
    MI_img_per_k = np.zeros(K)
    MI_txt_per_k = np.zeros(K)
    for k in range(K):
        MI_img_per_k[k] = mi_via_shared_pca(Z_img[:, 0], Z_img[:, k], pca_img_mi)
        MI_txt_per_k[k] = mi_via_shared_pca(Z_txt[:, 0], Z_txt[:, k], pca_txt_mi)
        logger.info(f"    k={k}: MI_img={MI_img_per_k[k]:+.3f}  MI_txt={MI_txt_per_k[k]:+.3f}")

    # ── Save & figure ─────────────────────────────────────────────────────────
    np.savez(os.path.join(cache_dir, "E_high_dim.npz"),
             PR_img=PR_img, PR_txt=PR_txt,
             eig_spectrum=eig_spectrum, rank95=rank95,
             MI_img_per_k=MI_img_per_k, MI_txt_per_k=MI_txt_per_k)

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.5))
    ks = np.arange(K)

    ax = axes[0]
    ax.plot(ks, PR_img, marker="o", ms=4, color="C0", label="Image")
    ax.plot(ks, PR_txt, marker="s", ms=4, color="C1", label="Text")
    ax.set_xlabel("Iteration $k$")
    ax.set_ylabel("Effective dim. (PR)")
    ax.set_title("Effective dimensionality")
    ax.legend()

    ax = axes[1]
    n_show = min(20, len(eig_spectrum))
    ax.semilogy(np.arange(1, n_show + 1), eig_spectrum[:n_show],
                marker="o", ms=4, color="C7")
    ax.axhline(0.01, color="k", ls="--", lw=0.6, alpha=0.5)
    ax.set_xlabel("Component")
    ax.set_ylabel("Variance fraction (log)")
    ax.set_title(f"Spectral decay (rank-95={rank95})")

    ax = axes[2]
    ax.plot(ks, MI_img_per_k, marker="o", ms=4, color="C0", label="Image")
    ax.plot(ks, MI_txt_per_k, marker="s", ms=4, color="C1", label="Text")
    ax.axhline(0, color="k", ls="--", lw=0.6)
    ax.set_xlabel("Iteration $k$")
    ax.set_ylabel(r"$I(z_0;z_k)$ (nats)")
    ax.set_title("Mutual information w/ anchor")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "E_high_dim.pdf"))
    plt.close(fig)
    logger.info(f"  Figure → figures/E_high_dim.pdf")

    return {"PR_img": PR_img, "PR_txt": PR_txt,
            "eig_spectrum": eig_spectrum, "rank95": rank95,
            "MI_img_per_k": MI_img_per_k, "MI_txt_per_k": MI_txt_per_k}


# ══════════════════════════════════════════════════════════════════════════════
#  BLOCK F — CLINICAL INTERPRETATION
# ══════════════════════════════════════════════════════════════════════════════

def _profile_permutation_test(profile_dist, basin_labels, n_permutations=1000,
                                rng_seed=42):
    """Permutation test on patient pathology-profile distances.

    Args:
      profile_dist: (N, N) symmetric matrix of pairwise Jaccard distances on
                    CheXpert label vectors.
      basin_labels: (N,) basin id per patient.
      n_permutations: number of label shuffles.

    Returns dict with T_observed (mean d_diff − mean d_same — positive = signal),
    T_null (perm distribution), p_value (one-sided), cohen_d (effect size on
    same vs diff distance distributions), mean_d_same, mean_d_diff.
    """
    rng = np.random.default_rng(rng_seed)
    N = len(basin_labels)
    iu = np.triu_indices(N, k=1)
    pair_dists = profile_dist[iu]
    pair_same  = (basin_labels[iu[0]] == basin_labels[iu[1]])

    def stat_for_labels(same_mask):
        if same_mask.sum() == 0 or (~same_mask).sum() == 0:
            return 0.0, 0.0, 0.0
        d_same = pair_dists[same_mask]
        d_diff = pair_dists[~same_mask]
        return (float(d_diff.mean() - d_same.mean()),
                float(d_same.mean()),
                float(d_diff.mean()))

    T_obs, mean_same, mean_diff = stat_for_labels(pair_same)
    pooled_var = ((pair_dists[pair_same].var() + pair_dists[~pair_same].var()) / 2
                   if pair_same.sum() > 1 and (~pair_same).sum() > 1 else 1e-9)
    cohen_d = float(T_obs / np.sqrt(max(pooled_var, 1e-12)))

    T_null = np.zeros(n_permutations)
    for p in range(n_permutations):
        shuffled = rng.permutation(basin_labels)
        same_shuffled = (shuffled[iu[0]] == shuffled[iu[1]])
        T_null[p], _, _ = stat_for_labels(same_shuffled)
    p_value = float((T_null >= T_obs).mean())

    return {
        "T_observed":  T_obs,
        "T_null":      T_null,
        "p_value":     p_value,
        "cohen_d":     cohen_d,
        "mean_d_same": mean_same,
        "mean_d_diff": mean_diff,
    }


def block_F_clinical(trajs, A_results, C_results, out_dir, data_csv,
                       n_permutations=1000):
    """
    Profile basins by CheXpert label distribution + faithfulness regression.
    Requires processed_data.csv with CheXpert columns.

    PRIMARY TEST: profile-distance permutation test (Option D).
    Per-label χ² (the previous version) violates multi-label independence
    because patients have COMBINATIONS of pathologies, not single labels.
    This version computes pairwise Jaccard distance between patient
    14-bit CheXpert vectors and tests whether same-basin patients have
    systematically smaller profile distance than across basins. The null
    is a basin-label permutation, which handles multi-label correlation
    structure correctly.

    Faithfulness regression (R²) is kept since it's already
    multi-label-correct.
    Stable-subgroup analysis is kept and now uses the new permutation test.
    """
    logger.info("\n" + "═" * 60)
    logger.info("BLOCK F — Clinical Interpretation (profile-distance test)")
    logger.info("═" * 60)

    fig_dir = os.path.join(out_dir, "figures")
    cache_dir = os.path.join(out_dir, "cache")
    table_dir = os.path.join(out_dir, "tables")

    CHEXPERT_LABELS = [
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
        "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
        "Lung Opacity", "No Finding", "Pleural Effusion", "Pleural Other",
        "Pneumonia", "Pneumothorax", "Support Devices",
    ]

    df = pd.read_csv(data_csv, low_memory=False)
    df["study_id"] = df["study_id"].astype(str)
    df = df.drop_duplicates(subset=["study_id"], keep="first").set_index("study_id")
    available = [c for c in CHEXPERT_LABELS if c in df.columns]
    if not available:
        logger.warning("  No CheXpert columns found — skipping Block F")
        return None
    logger.info(f"  Found CheXpert labels: {len(available)}/{len(CHEXPERT_LABELS)}")

    # ── Build per-patient label vectors aligned with trajectory order ────────
    basin_labels = C_results["basin_labels"]
    n_basins = C_results["best_K"]
    Y = np.zeros((len(trajs), len(available)), dtype=np.uint8)
    has_labels = np.zeros(len(trajs), dtype=bool)
    for i, t in enumerate(trajs):
        sid = str(t["study_id"])
        if sid not in df.index: continue
        row = df.loc[sid]
        for j, lbl in enumerate(available):
            v = row[lbl]
            # CheXpert convention: 1 = positive, others (0/-1/NaN) = negative
            Y[i, j] = 1 if (pd.notna(v) and float(v) == 1.0) else 0
        has_labels[i] = True

    # Filter to patients with ≥1 positive label — Jaccard is undefined when
    # both vectors are all-zero. All-negative patients would otherwise be
    # treated as artificially "identical" and bias the test.
    has_any_pos = has_labels & (Y.sum(axis=1) > 0)
    n_with = int(has_any_pos.sum())
    n_zero = int(has_labels.sum() - n_with)
    logger.info(f"  Patients with ≥1 CheXpert positive: {n_with}/{has_labels.sum()} "
                f"(dropped {n_zero} all-negative)")

    Y_use = Y[has_any_pos]
    basin_use = basin_labels[has_any_pos]

    # ── Pairwise Jaccard distance on label profiles ──────────────────────────
    logger.info(f"  Computing pairwise Jaccard distances "
                f"({n_with}×{n_with} matrix)...")
    profile_dist_condensed = pdist(Y_use, metric="jaccard")
    profile_dist = squareform(profile_dist_condensed)

    # ── Permutation test: full cohort (with-positives subset) ────────────────
    logger.info(f"  Permutation test (n_perm={n_permutations}) — full cohort...")
    full_test = _profile_permutation_test(
        profile_dist, basin_use, n_permutations=n_permutations, rng_seed=42,
    )
    logger.info(f"    T_observed         = {full_test['T_observed']:+.4f}")
    logger.info(f"    Mean-d same-basin  = {full_test['mean_d_same']:.4f}")
    logger.info(f"    Mean-d diff-basin  = {full_test['mean_d_diff']:.4f}")
    logger.info(f"    Cohen's d          = {full_test['cohen_d']:+.3f}")
    logger.info(f"    Permutation p      = {full_test['p_value']:.4f}")

    # ── Faithfulness regression (kept — already multi-label-correct) ─────────
    rows = []
    for i, t in enumerate(trajs):
        if not has_labels[i]: continue
        c0 = float(np.dot(t["img_traj"][0], t["anchor_img"]))
        rows.append([t["study_id"], c0, *Y[i].tolist()])
    df_reg = pd.DataFrame(rows, columns=["study_id", "iter0_img_cos", *available])

    from sklearn.linear_model import LinearRegression
    X = df_reg[available].values
    y = df_reg["iter0_img_cos"].values
    if len(y) > len(available):
        lr = LinearRegression().fit(X, y)
        coef = lr.coef_
        r2 = float(lr.score(X, y))
    else:
        coef = np.full(len(available), np.nan)
        r2 = float("nan")
    logger.info(f"  Faithfulness regression R² = {r2:.4f}")

    # ── Stable-trajectory subgroup permutation test ──────────────────────────
    Z_img_local = A_results["Z_img"]
    A_img_local = A_results["A_img"]
    d_anchor = np.linalg.norm(Z_img_local - A_img_local[:, None, :], axis=-1)
    K_local = d_anchor.shape[1]
    half = K_local // 2
    early_steps = np.diff(d_anchor[:, :half], axis=1).mean(axis=1)
    late_steps  = np.diff(d_anchor[:, half:], axis=1).mean(axis=1)
    stable_mask = (np.abs(late_steps) < np.abs(early_steps)) & (early_steps > 0)
    n_stable = int(stable_mask.sum())
    logger.info(f"  Stable-trajectory subgroup: {n_stable}/{len(trajs)} "
                f"({100*n_stable/len(trajs):.0f}%)")

    stable_test = None
    n_sw = 0
    stable_and_with = stable_mask & has_any_pos
    n_sw = int(stable_and_with.sum())
    if n_sw >= 30 and len(np.unique(basin_labels[stable_and_with])) > 1:
        # Map stable_and_with onto the rows of the precomputed profile_dist
        stable_idx_in_used = np.where(has_any_pos)[0]
        stable_to_use = np.isin(stable_idx_in_used,
                                  np.where(stable_and_with)[0])
        profile_dist_stable = profile_dist[np.ix_(stable_to_use, stable_to_use)]
        basin_stable_use    = basin_use[stable_to_use]
        logger.info(f"  Permutation test — stable subgroup ({n_sw})...")
        stable_test = _profile_permutation_test(
            profile_dist_stable, basin_stable_use,
            n_permutations=n_permutations, rng_seed=43,
        )
        logger.info(f"    T_observed       = {stable_test['T_observed']:+.4f}")
        logger.info(f"    Cohen's d        = {stable_test['cohen_d']:+.3f}")
        logger.info(f"    Permutation p    = {stable_test['p_value']:.4f}")
    else:
        logger.info(f"  Skipping stable-subset permutation test "
                    f"(n_sw={n_sw} too small or single-basin)")

    # ── Per-basin label rates (kept for the heatmap visualization) ───────────
    label_rates = np.zeros((n_basins, len(available)))
    counts = np.zeros(n_basins)
    for i in range(len(trajs)):
        if not has_labels[i]: continue
        b = basin_labels[i]
        for j in range(len(available)):
            if Y[i, j] == 1:
                label_rates[b, j] += 1
        counts[b] += 1
    label_rates = np.divide(label_rates, np.maximum(counts[:, None], 1.0),
                              out=np.zeros_like(label_rates),
                              where=counts[:, None] > 0)

    # ── Save ─────────────────────────────────────────────────────────────────
    pd.DataFrame(label_rates, columns=available,
                 index=[f"basin_{b}" for b in range(n_basins)]).to_csv(
        os.path.join(table_dir, "F_basin_chexpert_rates.csv"))
    df_reg.to_csv(os.path.join(table_dir, "F_faithfulness_data.csv"),
                   index=False)
    pd.DataFrame({
        "label": available,
        "coef":  coef,
    }).to_csv(os.path.join(table_dir, "F_faithfulness_coefs.csv"), index=False)

    np.savez(os.path.join(cache_dir, "F_clinical.npz"),
             label_rates              = label_rates,
             counts                   = counts,
             T_observed_full          = full_test["T_observed"],
             T_null_full              = full_test["T_null"],
             p_value_full             = full_test["p_value"],
             cohen_d_full             = full_test["cohen_d"],
             mean_d_same_full         = full_test["mean_d_same"],
             mean_d_diff_full         = full_test["mean_d_diff"],
             n_permutations           = n_permutations,
             T_observed_stable        = (stable_test["T_observed"]
                                          if stable_test else np.nan),
             T_null_stable            = (stable_test["T_null"]
                                          if stable_test else np.array([])),
             p_value_stable           = (stable_test["p_value"]
                                          if stable_test else np.nan),
             cohen_d_stable           = (stable_test["cohen_d"]
                                          if stable_test else np.nan),
             n_stable                 = n_stable,
             n_stable_with_positives  = n_sw,
             n_with_positives         = n_with,
             n_dropped_all_negative   = n_zero,
             r2_faithful              = r2,
             reg_coefs                = coef,
             stable_mask              = stable_mask)

    # ── Figure F: heatmap + permutation null + faithfulness ──────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4),
                              gridspec_kw={"width_ratios": [2.4, 1.2, 1.2]})

    # Panel 1: basin × CheXpert heatmap with test stats in title
    ax = axes[0]
    im = ax.imshow(label_rates, aspect="auto", cmap="RdPu", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(available)))
    ax.set_xticklabels(available, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(np.arange(n_basins))
    ax.set_yticklabels([f"Basin {b}\n(n={int(counts[b])})"
                          for b in range(n_basins)])
    ttl = "Basin × CheXpert positive rate"
    ttl += (f"\nperm. test (full): T={full_test['T_observed']:+.3f}, "
             f"p={full_test['p_value']:.3f}, d={full_test['cohen_d']:+.2f}")
    if stable_test:
        ttl += (f"\nstable n={n_sw}: T={stable_test['T_observed']:+.3f}, "
                 f"p={stable_test['p_value']:.3f}, d={stable_test['cohen_d']:+.2f}")
    ax.set_title(ttl, fontsize=9)
    plt.colorbar(im, ax=ax, label="positive rate")

    # Panel 2: permutation null vs observed (full cohort)
    ax = axes[1]
    ax.hist(full_test["T_null"], bins=40, color="C7", alpha=0.7,
            edgecolor="white", label="null")
    ax.axvline(full_test["T_observed"], color="C3", lw=2.2,
                label=f"observed T={full_test['T_observed']:+.3f}")
    ax.set_xlabel("T  =  mean d_diff − mean d_same")
    ax.set_ylabel("Permutations")
    ax.set_title(f"Permutation null (full)\n"
                  f"p={full_test['p_value']:.3f}, "
                  f"d={full_test['cohen_d']:+.2f}",
                  fontsize=10)
    ax.legend(fontsize=9)

    # Panel 3: faithfulness coefficients
    ax = axes[2]
    order = np.argsort(coef)
    ax.barh(np.arange(len(available)), coef[order],
             color=["C3" if c < 0 else "C2" for c in coef[order]])
    ax.set_yticks(np.arange(len(available)))
    ax.set_yticklabels([available[i] for i in order], fontsize=8)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("β (iter-0 cosine ~ label)")
    ax.set_title(f"Faithfulness (R²={r2:.3f})", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "F_clinical.pdf"))
    plt.close(fig)
    logger.info(f"  Figure → figures/F_clinical.pdf")

    return {
        "label_rates":        label_rates,
        "perm_full":          full_test,
        "perm_stable":        stable_test,
        "n_stable":           n_stable,
        "n_stable_with_pos":  n_sw,
        "n_with_positives":   n_with,
        "n_dropped_all_neg":  n_zero,
        "r2_faithful":        r2,
        "reg_coefs":          coef,
        "labels":             available,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--main_dir",     type=str, default=DEFAULT_MAIN)
    p.add_argument("--lyapunov_dir", type=str, default=DEFAULT_LYAPUNOV)
    p.add_argument("--ref_dir",      type=str, default=DEFAULT_REFEMB)
    p.add_argument("--out_dir",      type=str, default=DEFAULT_OUT)
    p.add_argument("--data_csv",     type=str,
                   default=f"{BASE_DIR}/processed_data/processed_data.csv")
    p.add_argument("--blocks",       nargs="+", default=list("ABCDEF"))
    p.add_argument("--n_permutations", type=int, default=1000,
                   help="Permutations for Block F profile-distance test.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("AIM2 ChexGen — Attractor Analysis")
    logger.info("=" * 60)
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")

    # ── Load all data ─────────────────────────────────────────────────────────
    trajs = load_main_run(args.main_dir)
    lyap_data = load_lyapunov_run(args.lyapunov_dir)
    try:
        ref = load_reference_embeddings(args.ref_dir)
    except FileNotFoundError as e:
        logger.warning(f"{e}\n  Block D will be skipped.")
        ref = None

    results = {}

    if "A" in args.blocks:
        results["A"] = block_A_geometry(trajs, args.out_dir)
    if "B" in args.blocks:
        results["B"] = block_B_lyapunov(trajs, lyap_data, None, args.out_dir)
    if "C" in args.blocks:
        if "A" not in results:
            results["A"] = block_A_geometry(trajs, args.out_dir)
        results["C"] = block_C_basins(trajs, results["A"], args.out_dir)
    if "D" in args.blocks:
        if ref is None:
            logger.warning("Block D requires reference embeddings; skipping")
        else:
            if "A" not in results:
                results["A"] = block_A_geometry(trajs, args.out_dir)
            if "C" not in results:
                results["C"] = block_C_basins(trajs, results["A"], args.out_dir)
            results["D"] = block_D_phase_portrait(trajs, ref, results["A"],
                                                   results["C"], args.out_dir)
    if "E" in args.blocks:
        if "A" not in results:
            results["A"] = block_A_geometry(trajs, args.out_dir)
        results["E"] = block_E_high_dim(results["A"], args.out_dir)
    if "F" in args.blocks:
        if "A" not in results:
            results["A"] = block_A_geometry(trajs, args.out_dir)
        if "C" not in results:
            results["C"] = block_C_basins(trajs, results["A"], args.out_dir)
        results["F"] = block_F_clinical(trajs, results["A"], results["C"],
                                         args.out_dir, args.data_csv,
                                         n_permutations=args.n_permutations)

    # ── Save scalar results to JSON ───────────────────────────────────────────
    summary = {"args": vars(args), "n_trajectories": len(trajs)}
    if "A" in results:
        summary["A"] = {
            "img_anchor_l2_iter0_mean": float(results["A"]["d_anchor_img"][:, 0].mean()),
            "img_anchor_l2_iterK_mean": float(results["A"]["d_anchor_img"][:, -1].mean()),
            "txt_anchor_l2_iter0_mean": float(results["A"]["d_anchor_txt"][:, 0].mean()),
            "txt_anchor_l2_iterK_mean": float(results["A"]["d_anchor_txt"][:, -1].mean()),
            "tortuosity_img_mean":      float(results["A"]["T_img"].mean()),
            "tortuosity_txt_mean":      float(results["A"]["T_txt"].mean()),
        }
    if "B" in results:
        summary["B"] = {
            "lambda_sys_final":  float(results["B"]["lambda_sys"][-1]),
            "monotone_V_frac":   float(results["B"]["monotone_frac"]),
            "basin_radius_p95":  float(results["B"]["basin_radius_p95"]),
            "lyapunov_per_anchor": {
                k: list(map(float, v)) for k, v in results["B"]["lyap_per_anchor"].items()
            },
        }
    if "C" in results:
        summary["C"] = {
            "best_K":      int(results["C"]["best_K"]),
            "best_K_gap":  int(results["C"].get("best_K_gap", -1)),
            "sil_scores":  {int(k): float(v) for k, v in results["C"]["sil_scores"].items()},
            "gap_scores":  {int(k): float(v) for k, v in results["C"].get("gap_scores", {}).items()},
            "n_hdbscan_clusters": int(results["C"].get("n_hdbscan_clusters", -1)),
            "n_hdbscan_noise":    int(results["C"].get("n_noise", -1)),
            "basin_stats": results["C"]["basin_stats"],
        }
    if "E" in results:
        summary["E"] = {
            "PR_img":     [float(x) for x in results["E"]["PR_img"]],
            "PR_txt":     [float(x) for x in results["E"]["PR_txt"]],
            "rank95":     int(results["E"]["rank95"]),
            "MI_img_per_k": [float(x) for x in results["E"]["MI_img_per_k"]],
            "MI_txt_per_k": [float(x) for x in results["E"]["MI_txt_per_k"]],
        }
    if "F" in results and results["F"] is not None:
        F = results["F"]
        f_summary = {
            "perm_full_T":         float(F["perm_full"]["T_observed"]),
            "perm_full_p":         float(F["perm_full"]["p_value"]),
            "perm_full_d":         float(F["perm_full"]["cohen_d"]),
            "mean_d_same_full":    float(F["perm_full"]["mean_d_same"]),
            "mean_d_diff_full":    float(F["perm_full"]["mean_d_diff"]),
            "n_stable":            int(F.get("n_stable", 0)),
            "n_with_positives":    int(F.get("n_with_positives", 0)),
            "n_dropped_all_neg":   int(F.get("n_dropped_all_neg", 0)),
            "r2_faithful":         float(F["r2_faithful"]),
            "labels":              F["labels"],
        }
        if F.get("perm_stable"):
            f_summary.update({
                "perm_stable_T":      float(F["perm_stable"]["T_observed"]),
                "perm_stable_p":      float(F["perm_stable"]["p_value"]),
                "perm_stable_d":      float(F["perm_stable"]["cohen_d"]),
                "n_stable_with_pos":  int(F["n_stable_with_pos"]),
            })
        summary["F"] = f_summary

    if "C" in results:
        c_summary = {
            "best_K_canonical": int(results["C"]["best_K"]),
            "best_K_gap":       int(results["C"]["best_K_gap"]),
            "n_hdbscan_clusters": int(results["C"]["n_hdbscan_clusters"]),
            "n_noise":          int(results["C"]["n_noise"]),
            "configs": {},
        }
        for cfg_name, cfg in results["C"].get("configs", {}).items():
            c_summary["configs"][cfg_name] = {
                "best_K_sil":     int(cfg["best_K"]),
                "best_sil_score": float(cfg["sil_scores"].get(cfg["best_K"], np.nan)),
                "best_K_gap":     int(cfg["best_K_gap"]),
                "hdb_n_clusters": int(cfg["n_hdb_clusters"]),
                "hdb_pct_noise":  float(100 * cfg["n_noise"] / len(trajs)),
            }
        summary["C"] = c_summary

    out_json = os.path.join(args.out_dir, "analysis_results.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"\nResults summary → {out_json}")
    logger.info(f"All outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()