"""
all_supp_figs.py — Supplementary figure generator for the AIM2 NeurIPS 2026 paper.

Sibling to all_figs.py. Each panel is one PNG. Multi-panel layouts are
assembled manually. Legends are saved separately when possible.

Panel registry
--------------
S1 — Trajectory geometry per K (Block A, long_horizon_results.json)
    figS1a_anchor_distance       d_K vs K, image+text
    figS1b_step_size             ‖z_K − z_{K-1}‖ vs K
    figS1c_modal_coupling        image↔text coupling ρ_K vs K

S2 — Per-pathology survival hierarchy (block_K_results.json single_label)
    figS2a_pathology_heatmap     log-fold change vs GT, all 14 labels × probe-K
    figS2b_pathology_curves      selected pathology positive-rate trajectories

S3 — Extended persistence (Block J)
    figS3a_autocorr_extended     full + late-window autocorr vs lag, both modalities
    figS3b_knn_jaccard_extended  kNN-Jaccard vs lag, both modalities

S4 — Per-anchor Lyapunov (Block B)
    figS4_anchor_lambda          per-anchor λ_a final values, sorted scatter

S5 — Cluster structure robustness (Block C)
    figS5_cluster_dim_robust     silhouette/gap/HDBSCAN at PCA-20/50/256-d

S6 — Basin × pathology null (Block F)
    figS6_basin_pathology_null   permutation T/d/p, full vs stable subset

S7 — Modality asymmetry (cohort summary, JSON-only)
    figS7_asymmetry_quant        kNN-to-training, image-IN vs text-OUT bars

S_oov_inflation — OOV inflation factor table (Block K, GT-OOV baseline)
    figS_oov_inflation           per-category GT vs K=100 cohort prevalence
                                 with inflation factors. Shows the loop
                                 SELECTIVELY amplifies diffuse parenchymal
                                 templates (COPD ~3.5x, plaques ~3.6x,
                                 fibrosis ~2.3x) and SUPPRESSES focal
                                 device findings (catheters, pacemakers,
                                 sternotomy, all to ~0%).

S_mi_calibration — MI estimator validation
    figS_mi_calibration          synthetic Gaussian calibration at exact
                                 (N=1008, d=256, n_pca=20, kNN=5) plus
                                 kNN sensitivity sweep at iter-1. Shows
                                 the iter-1 image MI of 0.17 nats sits
                                 ~23x above the estimator floor (~0.007
                                 nats) and is robust across kNN choices.

S_lyap_saturation — Estimator saturation under bounded support
    figS_lyap_saturation         finite-separation divergence-rate
                                 estimator on coupled logistic maps
                                 (known classical λ = ln 2) alongside
                                 paper Block B curves. Both decay
                                 monotonically with horizon, illustrating
                                 the bounded-support behavior described
                                 in §3.2: absolute magnitudes are
                                 finite-separation estimates, not
                                 classical Lyapunov exponents.

S_perm_gap — Permutation null gap statistic
    figS_perm_gap                Bernoulli vs column-permutation null
                                 gap statistic at K=10 and K=100. Shows
                                 the K=100 K_c=2 collapse is robust to
                                 null choice; K=10 cluster count is
                                 mildly null-sensitive (3 vs 4).

S_jaccard_autocorr — Two-timescale memory decomposition (App. G, Fig 11)
    figS_jaccard_autocorr        Late-window position autocorrelation
                                 (left) and kNN-Jaccard persistence
                                 (right) vs lag, both modalities, with
                                 their respective random baselines. The
                                 separation in decay rate between the
                                 two metrics — autocorr stays >2x
                                 baseline at lag-50 while Jaccard
                                 collapses to ~2x by lag-40 — is the
                                 two-timescale signature of motion
                                 within an attractor basin.

S_sampler_sensitivity — CFG/step sweep (App. B.3, Fig 12)
    figS_sampler_sensitivity     Image and text cosine to anchor across
                                 sampler-setting sweeps: CFG ∈ {2, 7}
                                 (left, T=100) and DiT denoising steps
                                 T ∈ {25, 50} (right, CFG=4), 20-anchor
                                 subset over 5 iterations each. The
                                 main run (CFG=4, T=100) sits between
                                 these settings. Trajectories are
                                 qualitatively identical across all
                                 sampler settings.

S8 — Modality asymmetry exemplar (UMAP) — REQUIRES HPC DATA
    figS_asym_umap_exemplar      one image traj drifting IN, one text traj drifting OUT
                                 plus cohort 256-d cosine kNN histograms
                                 (needs --main_dir + --ref_dir; skips otherwise)

Usage
-----
    python all_supp_figs.py --panels all
    python all_supp_figs.py --panels figS2a,figS2b
    python all_supp_figs.py --panels figS_asym_umap_exemplar \\
        --main_dir /n/.../chexgen_long \\
        --ref_dir  /n/.../reference_embeddings
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import pickle
from glob import glob
from pathlib import Path

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42  # TrueType for NeurIPS compliance
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import numpy as np

from style import (
    apply_style, new_panel,
    PANEL_FULL_W, PANEL_HALF_W, PANEL_DEFAULT_H,
    C_GT, C_ITER0, C_ITERK,
    C_IMG, C_TXT,
    C_LSYS, C_LANC, C_LANC_LIGHT,
    C_OOV_DOMINANT, C_OOV_FIBROSIS, C_OOV_SCOLIOSIS, C_OOV_OTHER,
    C_NORMAL, C_RESIDUAL,
    C_ELEVATOR, C_AMPLIFIED, C_PRESERVED, C_DIFFUSE, C_NOVEL,
    C_ANNOT, C_PRED, C_ZERO,
    shorten_label,
)


# ════════════════════════════════════════════════════════════════════════════
#  Path defaults
# ════════════════════════════════════════════════════════════════════════════

DEFAULTS = {
    "block_k":        "/n/groups/training/bmif203/AIM2/Experiments/attractor_loop/analysis_long/block_K_results.json",
    "analysis_json":  "/n/groups/training/bmif203/AIM2/Experiments/attractor_loop/analysis_long/analysis_results.json",
    "long_horizon":   "/n/groups/training/bmif203/AIM2/Experiments/attractor_loop/analysis_long/long_horizon_results.json",
    "geom_summary":   "/n/groups/training/bmif203/AIM2/Experiments/attractor_loop/analysis_long/I_geometry_summary.json",
    "mi_calibration": "/n/groups/training/bmif203/AIM2/Experiments/attractor_loop/analysis_long/mi_calibration/mi_calibration.json",
    "lyap_synthetic": "/n/groups/training/bmif203/AIM2/Experiments/attractor_loop/analysis_long/lyap_synthetic/lyapunov_synthetic.json",
    "sweep_cfg":      "/n/groups/training/bmif203/AIM2/Experiments/attractor_loop/results/chexgen_cfg_sweep/sweep_summary.json",
    "sweep_step":     "/n/groups/training/bmif203/AIM2/Experiments/attractor_loop/results/chexgen_step_sweep/sweep_summary.json",
    "out_dir":        "/n/groups/training/bmif203/AIM2/Experiments/attractor_loop/figures/figures_supp",
    "main_dir":       "/n/groups/training/bmif203/AIM2/Experiments/attractor_loop/results/chexgen_long",   # HPC trajectory dir (only for UMAP panel)
    "ref_dir":        "/n/groups/training/bmif203/AIM2/Experiments/attractor_loop/reference_embeddings",   # HPC reference embeddings (only for UMAP panel)
}


# ════════════════════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════════════════════

def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _save(fig, out_dir, name, also_pdf=True):
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{name}.png"
    fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.1)
    if also_pdf:
        fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"  → {png.name}")


def _bootstrap_mean_ci(values, n_boot=5000, ci=0.95, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    arr = np.asarray(values, dtype=float)
    means = np.array([rng.choice(arr, size=len(arr), replace=True).mean()
                      for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    return float(arr.mean()), float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


# ════════════════════════════════════════════════════════════════════════════
#  S1 — Trajectory geometry per K (Block A)
# ════════════════════════════════════════════════════════════════════════════

def _block_a_per_k(lh, modality, field):
    img = lh["A"][modality]
    ks = np.array(sorted(int(k) for k in img.keys()))
    vals = np.array([img[str(k)][field] for k in ks])
    return ks, vals


def figS1a_anchor_distance(cfg, out_dir):
    """Anchor distance d_K vs K. Plateaus around k≈5–10 in both modalities."""
    lh = _load_json(cfg["long_horizon"])
    if "A" not in lh:
        raise RuntimeError("Block A missing in long_horizon_results.json")

    ks, d_img_mean = _block_a_per_k(lh, "image", "d_K_mean")
    _,  d_img_std  = _block_a_per_k(lh, "image", "d_K_std")
    _,  d_txt_mean = _block_a_per_k(lh, "text",  "d_K_mean")
    _,  d_txt_std  = _block_a_per_k(lh, "text",  "d_K_std")

    fig, ax = new_panel(width=PANEL_FULL_W, height=PANEL_DEFAULT_H + 0.4)
    ax.fill_between(ks, d_img_mean - d_img_std, d_img_mean + d_img_std,
                    color=C_IMG, alpha=0.15, lw=0)
    ax.fill_between(ks, d_txt_mean - d_txt_std, d_txt_mean + d_txt_std,
                    color=C_TXT, alpha=0.15, lw=0)
    ax.plot(ks, d_img_mean, marker="o", ms=3.5, lw=1.5, color=C_IMG, label="Image")
    ax.plot(ks, d_txt_mean, marker="s", ms=3.5, lw=1.5, color=C_TXT, label="Text")
    ax.set_xlabel("Iteration $K$")
    ax.set_ylabel(r"Anchor distance  $\|z_K - z_0\|_2$")
    ax.set_title("Anchor distance saturates by $K\\!\\sim\\!10$, then stays bounded")
    ax.set_xticks([k for k in [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100] if k <= ks.max()])
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(False)
    fig.tight_layout()
    _save(fig, out_dir, "figS1a_anchor_distance", cfg["pdf"])


def figS1b_step_size(cfg, out_dir):
    """Per-iteration step size ‖z_K − z_{K-1}‖. Tiny, bounded, signals random
    walk on the attractor not free diffusion."""
    lh = _load_json(cfg["long_horizon"])
    if "A" not in lh:
        raise RuntimeError("Block A missing in long_horizon_results.json")

    ks, s_img_mean = _block_a_per_k(lh, "image", "step_K_mean")
    _,  s_img_std  = _block_a_per_k(lh, "image", "step_K_std")
    _,  s_txt_mean = _block_a_per_k(lh, "text",  "step_K_mean")
    _,  s_txt_std  = _block_a_per_k(lh, "text",  "step_K_std")

    fig, ax = new_panel(width=PANEL_FULL_W, height=PANEL_DEFAULT_H + 0.4)
    ax.fill_between(ks, s_img_mean - s_img_std, s_img_mean + s_img_std,
                    color=C_IMG, alpha=0.15, lw=0)
    ax.fill_between(ks, s_txt_mean - s_txt_std, s_txt_mean + s_txt_std,
                    color=C_TXT, alpha=0.15, lw=0)
    ax.plot(ks, s_img_mean, marker="o", ms=3.5, lw=1.5, color=C_IMG, label="Image")
    ax.plot(ks, s_txt_mean, marker="s", ms=3.5, lw=1.5, color=C_TXT, label="Text")
    ax.set_xlabel("Iteration $K$")
    ax.set_ylabel(r"Step size  $\|z_K - z_{K-1}\|_2$")
    ax.set_title("Per-iteration step size: small and bounded across the horizon")
    ax.set_xticks([k for k in [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100] if k <= ks.max()])
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(False)
    fig.tight_layout()
    _save(fig, out_dir, "figS1b_step_size", cfg["pdf"])


def figS1c_modal_coupling(cfg, out_dir):
    """Image↔text coupling ρ_K = corr(image-side d_k, text-side d_k) over k≤K.
    Shows the two modalities move in concert across the trajectory."""
    lh = _load_json(cfg["long_horizon"])
    if "A" not in lh or "modal_coupling" not in lh["A"]:
        raise RuntimeError("Block A modal_coupling missing in long_horizon_results.json")
    mc = lh["A"]["modal_coupling"]
    ks = np.array(sorted(int(k) for k in mc.keys()))
    rho = np.array([mc[str(k)] for k in ks])

    fig, ax = new_panel(width=PANEL_FULL_W, height=PANEL_DEFAULT_H + 0.4)
    ax.plot(ks, rho, marker="o", ms=3.5, lw=1.5, color=C_ANNOT)
    ax.axhline(0, color="#666", lw=0.5, ls="--", alpha=0.5)
    ax.axhline(1, color="#666", lw=0.5, ls=":", alpha=0.4)
    ax.set_xlabel("Iteration $K$")
    ax.set_ylabel(r"Modal coupling $\rho_K = \mathrm{corr}(d^{\mathrm{img}}_k, d^{\mathrm{txt}}_k)_{k\leq K}$")
    ax.set_title("Image and text trajectories move together: persistent positive coupling")
    ax.set_xticks([k for k in [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100] if k <= ks.max()])
    ax.set_ylim(-0.1, 1.0)
    ax.grid(False)
    fig.tight_layout()
    _save(fig, out_dir, "figS1c_modal_coupling", cfg["pdf"])


# ════════════════════════════════════════════════════════════════════════════
#  S2 — Per-pathology survival hierarchy (block_K_results)
# ════════════════════════════════════════════════════════════════════════════

def _build_pathology_matrix(bk):
    """Returns labels (M,), probe_ks (T,), and matrix M of shape (M, T) where
    M[i,k] = positive_rate of label i at probe-K probe_ks[k]. Plus gt rates."""
    labels = bk["label_names"]
    probe_ks = sorted(int(k) for k in bk["probe_iters"])
    mat = np.zeros((len(labels), len(probe_ks)))
    for j, k in enumerate(probe_ks):
        sl = bk[f"iter_{k}"]["single_label"]
        for i, lbl in enumerate(labels):
            mat[i, j] = sl.get(lbl, {}).get("positive_rate", 0.0)
    gt_rates = np.array([bk["gt"]["single_label"].get(lbl, {}).get("positive_rate", 0.0)
                         for lbl in labels])
    return np.array(labels), np.array(probe_ks), mat, gt_rates


def figS2a_pathology_heatmap(cfg, out_dir):
    """log2(positive_rate / GT_rate) per pathology per probe-K. Red ⇒ inflated;
    blue ⇒ depleted. Sorted by inflation at the largest K."""
    bk = _load_json(cfg["block_k"])
    labels, ks, mat, gt = _build_pathology_matrix(bk)
    eps = 1e-3
    log_fold = np.log2((mat + eps) / (gt[:, None] + eps))   # (M, T)

    # Sort by max log-fold across iterations (most-inflated first)
    order = np.argsort(-log_fold[:, -1])
    labels_s = labels[order]
    log_fold_s = log_fold[order]
    gt_s = gt[order]

    vmax = float(np.nanmax(np.abs(log_fold_s)))
    vmax = min(vmax, 4.0)   # cap for legibility

    fig, ax = new_panel(width=PANEL_FULL_W + 1.0, height=PANEL_DEFAULT_H + 1.5)
    im = ax.imshow(log_fold_s, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax,
                   extent=[ks[0] - 0.5, ks[-1] + 0.5,
                           len(labels_s) - 0.5, -0.5])
    ax.set_yticks(range(len(labels_s)))
    ax.set_yticklabels([f"{l}  (GT={100*gt_s[i]:.0f}%)" for i, l in enumerate(labels_s)],
                       fontsize=8)
    ax.set_xlabel("Iteration $K$")
    ax.set_xticks([k for k in [0, 5, 10, 20, 30, 50, 70, 100] if k <= ks.max()])
    ax.set_title(r"Per-pathology survival vs ground truth   $\log_2(\mathrm{rate}_K\,/\,\mathrm{rate}_{\mathrm{GT}})$")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(r"$\log_2$ fold change", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    _save(fig, out_dir, "figS2a_pathology_heatmap", cfg["pdf"])


def figS2b_pathology_curves(cfg, out_dir):
    """Selected pathology positive-rate trajectories vs K, with GT references.
    Legend saved separately as figS2b_legend.png to keep the panel clean."""
    bk = _load_json(cfg["block_k"])
    labels, ks, mat, gt = _build_pathology_matrix(bk)

    final_fold = np.log2((mat[:, -1] + 1e-3) / (gt + 1e-3))
    order = np.argsort(-final_fold)
    keep = [int(order[0]), int(order[1]), int(order[-1]), int(order[-2])]
    canon = ["Cardiomegaly", "Support Devices", "Lung Opacity"]
    for c in canon:
        if c in labels.tolist():
            keep.append(labels.tolist().index(c))
    keep = list(dict.fromkeys(keep))[:7]

    palette = ["#C73E1D", "#E69138", "#2E7D87", "#7E4196", "#5D8C4F", "#21295C", "#888888"]

    fig, ax = new_panel(width=PANEL_FULL_W, height=PANEL_DEFAULT_H + 0.6)
    for j, idx in enumerate(keep):
        rates = mat[idx]
        ax.plot(ks, rates, marker="o", ms=3, lw=1.4,
                color=palette[j % len(palette)])
        ax.axhline(gt[idx], color=palette[j % len(palette)],
                   ls="--", lw=1.0, alpha=0.75)
    ax.set_xlabel("Iteration $K$")
    ax.set_ylabel("Cohort positive rate")
    ax.set_title("Per-pathology survival trajectories: amplification, decay, novelty")
    ax.set_xticks([k for k in [0, 5, 10, 20, 30, 50, 70, 100] if k <= ks.max()])
    ax.grid(False)
    fig.tight_layout()
    _save(fig, out_dir, "figS2b_pathology_curves", cfg["pdf"])

    _save_pathology_curves_legend(out_dir, cfg, labels, gt, keep, palette)


def _save_pathology_curves_legend(out_dir, cfg, labels, gt, keep, palette):
    """Standalone legend for figS2b — pathology name + GT prevalence."""
    fig = plt.figure(figsize=(3.8, 2.4))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    y = 0.94
    y -= 0.10
    for j, idx in enumerate(keep):
        c = palette[j % len(palette)]
        ax.plot([0.06, 0.18], [y, y], color=c, lw=1.6)
        ax.scatter([0.12], [y], s=18, color=c, marker="o", zorder=3)
        ax.text(0.22, y, f"{labels[idx]}",
                fontsize=8.5, va="center")
        y -= 0.085
    y -= 0.02
    ax.plot([0.06, 0.18], [y, y], color="#666", ls="--", lw=1.0, alpha=0.75)
    ax.text(0.22, y, "GT prevalence (dashed)", fontsize=8.5, va="center", color="#444")

    fig.savefig(out_dir / "figS2b_legend.png", dpi=600,
                bbox_inches="tight", pad_inches=0.15)
    if cfg["pdf"]:
        fig.savefig(out_dir / "figS2b_legend.pdf",
                    bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"  → figS2b_legend.png")


# ════════════════════════════════════════════════════════════════════════════
#  S3 — Extended persistence (Block J)
# ════════════════════════════════════════════════════════════════════════════

def figS3a_autocorr_extended(cfg, out_dir):
    """Full + late-window autocorrelation vs lag, both modalities, with random
    baselines and τ_mix annotation."""
    lh = _load_json(cfg["long_horizon"])
    if "J" not in lh:
        raise RuntimeError("Block J missing in long_horizon_results.json")
    J_img = lh["J"]["image"]; J_txt = lh["J"]["text"]
    lags = np.asarray(J_img["lags"], dtype=int)
    af_img = np.asarray(J_img["autocorr_full_mean"], dtype=float)
    al_img = np.asarray(J_img["autocorr_late_mean"], dtype=float)
    af_txt = np.asarray(J_txt["autocorr_full_mean"], dtype=float)
    al_txt = np.asarray(J_txt["autocorr_late_mean"], dtype=float)
    rb_img = float(J_img["random_pair_autocorr"])
    rb_txt = float(J_txt["random_pair_autocorr"])
    tau_img = J_img.get("tau_mix_iters")
    tau_txt = J_txt.get("tau_mix_iters")

    fig, ax = new_panel(width=PANEL_FULL_W, height=PANEL_DEFAULT_H + 0.5)
    ax.plot(lags, af_img, marker="o", ms=3, lw=1.0, color=C_IMG, alpha=0.45,
            ls="--", label="Image full window")
    ax.plot(lags, al_img, marker="o", ms=3.5, lw=1.6, color=C_IMG,
            label="Image late window")
    ax.plot(lags, af_txt, marker="s", ms=3, lw=1.0, color=C_TXT, alpha=0.45,
            ls="--", label="Text full window")
    ax.plot(lags, al_txt, marker="s", ms=3.5, lw=1.6, color=C_TXT,
            label="Text late window")
    ax.axhline(rb_img, color=C_IMG, lw=0.8, ls=":", alpha=0.65,
               label=f"Image random baseline ({rb_img:.2f})")
    ax.axhline(rb_txt, color=C_TXT, lw=0.8, ls=":", alpha=0.65,
               label=f"Text random baseline ({rb_txt:.2f})")

    ax.set_xlabel("Lag $\\ell$ (iterations)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Position autocorrelation vs lag, full and late windows")
    ax.set_xlim(0, lags.max() + 2)
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.92)
    ax.grid(False)
    fig.tight_layout()
    _save(fig, out_dir, "figS3a_autocorr_extended", cfg["pdf"])


def figS3b_knn_jaccard_extended(cfg, out_dir):
    """kNN-Jaccard persistence vs lag for both modalities, full and late
    windows, with analytic random baseline."""
    lh = _load_json(cfg["long_horizon"])
    if "J" not in lh:
        raise RuntimeError("Block J missing in long_horizon_results.json")
    J_img = lh["J"]["image"]; J_txt = lh["J"]["text"]
    lags = np.asarray(J_img["lags"], dtype=int)
    jf_img = np.asarray(J_img["knn_jaccard_full_mean"], dtype=float)
    jl_img = np.asarray(J_img["knn_jaccard_late_mean"], dtype=float)
    jf_txt = np.asarray(J_txt["knn_jaccard_full_mean"], dtype=float)
    jl_txt = np.asarray(J_txt["knn_jaccard_late_mean"], dtype=float)
    rb_img = float(J_img["random_pair_jaccard"])
    rb_txt = float(J_txt["random_pair_jaccard"])

    fig, ax = new_panel(width=PANEL_FULL_W, height=PANEL_DEFAULT_H + 0.5)
    ax.plot(lags, jf_img, marker="o", ms=3, lw=1.0, color=C_IMG, alpha=0.45,
            ls="--", label="Image full window")
    ax.plot(lags, jl_img, marker="o", ms=3.5, lw=1.6, color=C_IMG,
            label="Image late window")
    ax.plot(lags, jf_txt, marker="s", ms=3, lw=1.0, color=C_TXT, alpha=0.45,
            ls="--", label="Text full window")
    ax.plot(lags, jl_txt, marker="s", ms=3.5, lw=1.6, color=C_TXT,
            label="Text late window")
    ax.axhline(rb_img, color="#666", lw=0.8, ls=":", alpha=0.65,
               label=f"Random (analytic) ≈ {rb_img:.3f}")

    ax.set_xlabel("Lag $\\ell$ (iterations)")
    ax.set_ylabel("kNN-Jaccard overlap")
    ax.set_title("Local neighborhood persistence: micro-mixing vs macro-position memory")
    ax.set_xlim(0, lags.max() + 2)
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9)
    ax.grid(False)
    fig.tight_layout()
    _save(fig, out_dir, "figS3b_knn_jaccard_extended", cfg["pdf"])


# ════════════════════════════════════════════════════════════════════════════
#  S4 — Per-anchor Lyapunov scatter (Block B)
# ════════════════════════════════════════════════════════════════════════════

def figS4_anchor_lambda(cfg, out_dir):
    """Per-anchor finite-time Lyapunov exponent λ_a at K_max, sorted, with
    bootstrap mean and 95% CI. Shows that all anchors are individually positive."""
    aj = _load_json(cfg["analysis_json"])
    if "B" not in aj or "lyapunov_per_anchor" not in aj["B"]:
        raise RuntimeError("Block B missing in analysis_results.json")
    per_anchor = aj["B"]["lyapunov_per_anchor"]
    sids = sorted(per_anchor.keys())
    arrs = np.stack([np.asarray(per_anchor[s], dtype=float) for s in sids])
    K = arrs.shape[1] - 1
    finals = arrs[:, -1]
    mean, lo, hi = _bootstrap_mean_ci(finals)
    n_pos = int(np.sum(finals > 0))

    order = np.argsort(finals)
    finals_s = finals[order]

    fig, ax = new_panel(width=PANEL_FULL_W, height=PANEL_DEFAULT_H + 0.5)
    x = np.arange(len(finals_s))
    colors = [C_LANC if v > 0 else C_LSYS for v in finals_s]
    ax.scatter(x, finals_s, s=42, c=colors, edgecolors="white",
               linewidths=0.5, zorder=4)
    ax.axhline(mean, color=C_LANC, lw=1.0, zorder=2,
               label=f"Mean $= {mean:+.4f}$")
    ax.axhspan(lo, hi, color=C_LANC, alpha=0.15, zorder=1,
               label=f"95% CI $[{lo:+.4f},\\,{hi:+.4f}]$")
    ax.axhline(0, color=C_ZERO, lw=0.6, ls="--", alpha=0.5)

    ax.text(0.02, 0.96,
            f"{n_pos}/{len(finals)} anchors with $\\bar\\lambda_a > 0$ at $K{{=}}{K}$",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=8, color=C_ANNOT,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#CCC", lw=0.5))

    ax.set_xlabel(f"Anchor (sorted, {len(finals)} total)")
    ax.set_ylabel(rf"$\bar\lambda_a$ at $K{{=}}{K}$")
    ax.set_title("Per-anchor Lyapunov exponent: within-anchor divergence is universal")
    ax.set_xticks([])
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(False)
    fig.tight_layout()
    _save(fig, out_dir, "figS4_anchor_lambda", cfg["pdf"])


# ════════════════════════════════════════════════════════════════════════════
#  S5 — Cluster structure robustness (Block C)
# ════════════════════════════════════════════════════════════════════════════

def figS5_cluster_dim_robust(cfg, out_dir):
    """Best-K, silhouette score, and HDBSCAN noise across three dimensionality
    settings. Shows that no dimensionality recovers strong discrete clusters
    in embedding space — basins are content-defined, not geometric.
    Legend saved separately as figS5_legend.png."""
    aj = _load_json(cfg["analysis_json"])
    if "C" not in aj or "configs" not in aj["C"]:
        raise RuntimeError("Block C configs missing in analysis_results.json")
    configs = aj["C"]["configs"]

    config_order = ["raw_256d_cos", "pca20_euc", "pca50_euc"]
    config_labels = ["Raw 256-d\n(cosine)", "PCA-20\n(euclidean)", "PCA-50\n(euclidean)"]

    best_K_sil = [configs[c]["best_K_sil"] for c in config_order]
    best_sil   = [configs[c]["best_sil_score"] for c in config_order]
    best_K_gap = [configs[c]["best_K_gap"] for c in config_order]
    hdb_n      = [configs[c]["hdb_n_clusters"] for c in config_order]
    hdb_noise  = [configs[c]["hdb_pct_noise"] for c in config_order]

    fig, axes = plt.subplots(1, 3, figsize=(PANEL_FULL_W + 3.0, PANEL_DEFAULT_H + 1.0))

    x = np.arange(len(config_order))
    width = 0.4

    # Panel 1 — best K (silhouette and gap)
    ax = axes[0]
    ax.bar(x - width/2, best_K_sil, width, color=C_IMG, alpha=0.75,
           edgecolor="white")
    ax.bar(x + width/2, best_K_gap, width, color=C_TXT, alpha=0.75,
           edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(config_labels, fontsize=8)
    ax.set_ylabel("Best cluster count")
    ax.set_title("Best $K$ (silhouette / gap)", fontsize=9.5)
    ax.grid(False)

    # Panel 2 — silhouette score (low = weak cluster structure)
    ax = axes[1]
    bars = ax.bar(x, best_sil, width=0.55, color=C_LANC, alpha=0.78,
                  edgecolor="white")
    for b, v in zip(bars, best_sil):
        ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.2f}",
                ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(config_labels, fontsize=8)
    ax.set_ylabel("Best silhouette score")
    ax.set_ylim(0, max(0.5, max(best_sil) * 1.25))
    ax.set_title("Silhouette: weak partitioning everywhere", fontsize=9.5)
    ax.axhline(0.5, color="#888", lw=0.6, ls="--", alpha=0.4)
    ax.text(0.98, 0.95, "0.5 = strong clusters",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7, color="#888", style="italic")
    ax.grid(False)

    # Panel 3 — HDBSCAN noise
    ax = axes[2]
    ax.bar(x - width/2, hdb_n, width, color=C_LSYS, alpha=0.75,
           edgecolor="white")
    ax2 = ax.twinx()
    ax2.bar(x + width/2, hdb_noise, width, color=C_RESIDUAL, alpha=0.75,
            edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(config_labels, fontsize=8)
    ax.set_ylabel("HDBSCAN $n_{\\mathrm{clusters}}$", fontsize=9)
    ax2.set_ylabel("HDBSCAN noise %", fontsize=9)
    ax2.set_ylim(0, max(60, max(hdb_noise) * 1.2))
    ax.set_title("HDBSCAN: high noise fraction", fontsize=9.5)
    ax.grid(False); ax2.grid(False)

    fig.suptitle("Embedding-space cluster structure is weak across dimensionality choices",
                 fontsize=10.5, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, out_dir, "figS5_cluster_dim_robust", cfg["pdf"])

    _save_cluster_dim_legend(out_dir, cfg)


def _save_cluster_dim_legend(out_dir, cfg):
    """Standalone legend covering the three figS5 panels."""
    fig = plt.figure(figsize=(3.6, 2.6))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    y = 0.95
    ax.text(0.04, y, "Bar colors", fontweight="bold", fontsize=10.5, va="top")

    y -= 0.10
    ax.add_patch(plt.Rectangle((0.06, y - 0.03), 0.14, 0.06,
                               color=C_IMG, alpha=0.78))
    ax.text(0.23, y, "Best $K$ by silhouette", fontsize=9, va="center")

    y -= 0.10
    ax.add_patch(plt.Rectangle((0.06, y - 0.03), 0.14, 0.06,
                               color=C_TXT, alpha=0.78))
    ax.text(0.23, y, "Best $K$ by gap statistic", fontsize=9, va="center")

    y -= 0.10
    ax.add_patch(plt.Rectangle((0.06, y - 0.03), 0.14, 0.06,
                               color=C_LANC, alpha=0.78))
    ax.text(0.23, y, "Best silhouette score", fontsize=9, va="center")

    y -= 0.10
    ax.add_patch(plt.Rectangle((0.06, y - 0.03), 0.14, 0.06,
                               color=C_LSYS, alpha=0.78))
    ax.text(0.23, y, "HDBSCAN $n_{\\mathrm{clusters}}$", fontsize=9, va="center")

    y -= 0.10
    ax.add_patch(plt.Rectangle((0.06, y - 0.03), 0.14, 0.06,
                               color=C_RESIDUAL, alpha=0.78))
    ax.text(0.23, y, "HDBSCAN noise %", fontsize=9, va="center")

    y -= 0.12
    ax.plot([0.06, 0.20], [y, y], color="#888", ls="--", lw=0.8)
    ax.text(0.23, y, "Silhouette 0.5 = strong", fontsize=8.5,
            va="center", color="#666", style="italic")

    fig.savefig(out_dir / "figS5_legend.png", dpi=600,
                bbox_inches="tight", pad_inches=0.15)
    if cfg["pdf"]:
        fig.savefig(out_dir / "figS5_legend.pdf",
                    bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"  → figS5_legend.png")


# ════════════════════════════════════════════════════════════════════════════
#  S6 — Basin × pathology null result (Block F)
# ════════════════════════════════════════════════════════════════════════════

def figS6_basin_pathology_null(cfg, out_dir):
    """Permutation test (Block F) on whether embedding-space basins preserve
    GT pathology profile. Full cohort vs stable subset. Negligible effect
    sizes corroborate §5.2: large modes are loop-induced, not patient-derived.
    Legend saved separately as figS6_legend.png."""
    aj = _load_json(cfg["analysis_json"])
    if "F" not in aj:
        raise RuntimeError("Block F missing in analysis_results.json")
    F = aj["F"]
    T_full   = F.get("perm_full_T")
    p_full   = F.get("perm_full_p")
    d_full   = F.get("perm_full_d")
    T_stab   = F.get("perm_stable_T")
    p_stab   = F.get("perm_stable_p")
    d_stab   = F.get("perm_stable_d", None)
    n_full   = F.get("perm_full_T") is not None and (F.get("n_with_positives") or 1054)
    n_stable = F.get("n_stable")
    r2       = F.get("r2_faithful")

    fig, axes = plt.subplots(1, 2, figsize=(PANEL_FULL_W, PANEL_DEFAULT_H + 0.6))

    # ─── Panel 1 — bar chart of effect sizes ────────────────────────────────
    ax = axes[0]
    metrics = []
    full_vals = []
    stab_vals = []
    if T_full is not None and T_stab is not None:
        metrics.append(r"perm $T$")
        full_vals.append(T_full); stab_vals.append(T_stab)
    if d_full is not None:
        metrics.append(r"Cohen's $d$")
        full_vals.append(d_full); stab_vals.append(d_stab if d_stab is not None else 0.0)

    x = np.arange(len(metrics))
    width = 0.38
    ax.bar(x - width/2, full_vals, width, color=C_LSYS, alpha=0.78,
           edgecolor="white")
    ax.bar(x + width/2, stab_vals, width, color=C_LANC, alpha=0.78,
           edgecolor="white")

    # Value labels: above positive bars, below negative bars (no overlap)
    label_offset = 0.0006
    for i, v in enumerate(full_vals):
        offset = label_offset if v >= 0 else -label_offset
        va = "bottom" if v >= 0 else "top"
        ax.text(x[i] - width/2, v + offset, f"{v:.4f}",
                ha="center", va=va, fontsize=7.5)
    for i, v in enumerate(stab_vals):
        offset = label_offset if v >= 0 else -label_offset
        va = "bottom" if v >= 0 else "top"
        ax.text(x[i] + width/2, v + offset, f"{v:.4f}",
                ha="center", va=va, fontsize=7.5)

    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.axhline(0, color="#666", lw=0.5)
    ax.set_ylabel(r"Effect size  ($\bar{d}_{\mathrm{same}} - \bar{d}_{\mathrm{diff}}$)")
    ax.set_title("Permutation test: basins do not preserve GT profile",
                 fontsize=9.5)
    # Symmetric y-limit padding so labels fit cleanly above/below bars
    all_vals = full_vals + stab_vals
    if all_vals:
        ymin = min(all_vals); ymax = max(all_vals)
        pad = max(abs(ymin), abs(ymax)) * 0.45
        ax.set_ylim(ymin - pad, ymax + pad if ymax > 0 else pad)
    ax.grid(False)

    # ─── Panel 2 — p-values + faithfulness R² ──────────────────────────────
    ax = axes[1]
    items = []
    vals  = []
    if p_full is not None:
        items.append(r"perm $p$" + "\n(full)"); vals.append(p_full)
    if p_stab is not None:
        items.append(r"perm $p$" + "\n(stable)"); vals.append(p_stab)
    if r2 is not None:
        items.append(r"$R^2$ faithfulness"); vals.append(r2)
    bars = ax.bar(range(len(items)), vals, width=0.55,
                  color=[C_LSYS, C_LANC, C_AMPLIFIED][:len(items)],
                  alpha=0.78, edgecolor="white")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.012, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(items))); ax.set_xticklabels(items, fontsize=8)
    ax.axhline(0.05, color="#888", ls="--", lw=0.6, alpha=0.5)
    ax.text(0.98, 0.96, r"$\alpha = 0.05$",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7, color="#888", style="italic")
    ax.set_ylabel("Value")
    ax.set_title("Significance and faithfulness", fontsize=9.5)
    ax.set_ylim(0, max(1.05, max(vals) * 1.2) if vals else 1.0)
    ax.grid(False)

    fig.suptitle(
        "Embedding-space basins are not patient-specific (null finding)",
        fontsize=10.5, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, out_dir, "figS6_basin_pathology_null", cfg["pdf"])

    _save_basin_pathology_legend(out_dir, cfg, n_stable)


def _save_basin_pathology_legend(out_dir, cfg, n_stable):
    """Standalone legend for figS6 — bar colors and effect-size convention."""
    fig = plt.figure(figsize=(3.6, 2.4))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    y = 0.92
    ax.text(0.04, y, "Bar colors", fontweight="bold", fontsize=10.5, va="top")

    y -= 0.13
    ax.add_patch(plt.Rectangle((0.06, y - 0.035), 0.14, 0.07,
                               color=C_LSYS, alpha=0.78))
    ax.text(0.23, y, "Full cohort (n = 1054 with positives)",
            fontsize=9, va="center")

    y -= 0.13
    ax.add_patch(plt.Rectangle((0.06, y - 0.035), 0.14, 0.07,
                               color=C_LANC, alpha=0.78))
    ax.text(0.23, y,
            (f"Stable trajectory subset (n = {n_stable})"
             if n_stable is not None else "Stable trajectory subset"),
            fontsize=9, va="center")

    y -= 0.13
    ax.add_patch(plt.Rectangle((0.06, y - 0.035), 0.14, 0.07,
                               color=C_AMPLIFIED, alpha=0.78))
    ax.text(0.23, y, r"$R^{2}$ faithfulness (right panel)",
            fontsize=9, va="center")

    y -= 0.16
    ax.text(0.04, y,
            (r"Sign convention: $T = \bar{d}_{\mathrm{same}} - "
             r"\bar{d}_{\mathrm{diff}}$" + "\n"
             r"Negative $T$ $\Rightarrow$ within-basin distances are not "
             r"smaller than between-basin distances $\Rightarrow$ basins "
             r"carry no GT-pathology signal."),
            fontsize=8, va="top", color="#444", style="italic")

    fig.savefig(out_dir / "figS6_legend.png", dpi=600,
                bbox_inches="tight", pad_inches=0.15)
    if cfg["pdf"]:
        fig.savefig(out_dir / "figS6_legend.pdf",
                    bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"  → figS6_legend.png")


# ════════════════════════════════════════════════════════════════════════════
#  S7 — Modality asymmetry quantitative (cohort summary, JSON-only)
# ════════════════════════════════════════════════════════════════════════════

def figS7_asymmetry_quant(cfg, out_dir):
    """Cohort-level asymmetric drift summary from I_geometry_summary.json:
    image kNN-to-training shrinks (drift IN), text kNN-to-training grows
    (drift OUT). Plus cohort MIPD contraction. Same iter-0 vs iter-K data
    used for the UMAP exemplar panel below."""
    geom_path = cfg.get("geom_summary")
    if not geom_path or not os.path.exists(geom_path):
        raise RuntimeError(
            f"I_geometry_summary.json not found at {geom_path}.\n"
            "  → run analysis_knn_alignment.py to produce it.")
    G = _load_json(geom_path)
    img = G["image"]; txt = G["text"]

    fig, axes = plt.subplots(1, 2, figsize=(PANEL_FULL_W + 0.5, PANEL_DEFAULT_H + 1.0))

    # Panel 1 — kNN-to-training: drift IN vs OUT
    ax = axes[0]
    labels = ["Image", "Text"]
    iter0 = [img["knn_iter_0"], txt["knn_iter_0"]]
    iterK = [img["knn_iter_K"], txt["knn_iter_K"]]
    x = np.arange(len(labels))
    width = 0.38
    ax.bar(x - width/2, iter0, width, color="#A0A0A0", alpha=0.65,
           edgecolor="white")
    ax.bar(x + width/2, iterK, width,
           color=[C_IMG, C_TXT], alpha=0.85,
           edgecolor="white")
    for xi, d0, dK in zip(x, iter0, iterK):
        ax.annotate("", xy=(xi + width/2, dK), xytext=(xi - width/2, d0),
                    arrowprops=dict(arrowstyle="->", color="#444", lw=1.0))
        delta = dK - d0
        direction = "OUT" if delta > 0 else "IN"
        ax.text(xi, max(d0, dK) * 1.10,
                f"Δ={delta:+.3f}\n({direction})",
                ha="center", fontsize=8.5, fontweight="bold",
                color=(C_TXT if delta > 0 else C_IMG))
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Mean cosine distance to 10-NN in training")
    ax.set_title("kNN-to-training: image IN, text OUT", fontsize=9.5)
    ax.set_ylim(0, max(iter0 + iterK) * 1.45)
    ax.grid(False)

    # Panel 2 — cohort contraction (MIPD shrinks for both modalities)
    ax = axes[1]
    iter0_m = [img["mipd_iter_0"], txt["mipd_iter_0"]]
    iterK_m = [img["mipd_iter_K"], txt["mipd_iter_K"]]
    ax.bar(x - width/2, iter0_m, width, color="#A0A0A0", alpha=0.65,
           edgecolor="white")
    ax.bar(x + width/2, iterK_m, width,
           color=[C_IMG, C_TXT], alpha=0.85,
           edgecolor="white")
    for xi, d0, dK in zip(x, iter0_m, iterK_m):
        ax.annotate("", xy=(xi + width/2, dK), xytext=(xi - width/2, d0),
                    arrowprops=dict(arrowstyle="->", color="#444", lw=1.0))
        delta = dK - d0
        ax.text(xi, max(d0, dK) * 1.07,
                f"Δ={delta:+.3f}",
                ha="center", fontsize=8.5, fontweight="bold",
                color="#444")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Mean intra-cohort pairwise distance (MIPD)")
    ax.set_title("Cohort contraction: both modalities shrink", fontsize=9.5)
    ax.set_ylim(0, max(iter0_m + iterK_m) * 1.30)
    ax.grid(False)

    fig.suptitle(
        "Asymmetric drift: ChexGen pulls images IN, MAIRA-2 pushes text OUT",
        fontsize=10.5, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, out_dir, "figS7_asymmetry_quant", cfg["pdf"])


# ════════════════════════════════════════════════════════════════════════════
#  S8 — Modality asymmetry exemplar UMAP (REQUIRES HPC TRAJECTORY DATA)
# ════════════════════════════════════════════════════════════════════════════

def _load_hpc_trajectories(main_dir, modality, K_max=10):
    """Load per-study (anchor, trajectory) embeddings from HPC main_dir.
    Returns sids list, A (N, D), Z (N, K+1, D)."""
    main_dir = Path(main_dir)
    suffix_anchor = f"anchor_{modality}_embed.npy"
    suffix_iter   = f"{modality}_embed_iter_"

    candidate_dirs = sorted([d for d in main_dir.iterdir() if d.is_dir()])
    sids, A_list, Z_list = [], [], []
    for d in candidate_dirs:
        a_path = d / suffix_anchor
        if not a_path.exists():
            continue
        iter_paths = [d / f"{suffix_iter}{k:03d}.npy" for k in range(K_max + 1)]
        if not all(p.exists() for p in iter_paths):
            continue
        try:
            A_list.append(np.load(a_path))
            Z_list.append(np.stack([np.load(p) for p in iter_paths], axis=0))
            sids.append(d.name)
        except Exception:
            continue
    if not sids:
        raise RuntimeError(f"No complete studies found under {main_dir} for modality={modality}")
    return sids, np.stack(A_list), np.stack(Z_list)


def _knn_distance_to_training(query_2d, ref_2d, k=10):
    """For each query point, distance to its k-th nearest training neighbor in
    UMAP space. Smaller = closer to dense training region. Returns (Q,) array."""
    from sklearn.neighbors import NearestNeighbors
    knn = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(ref_2d)
    dists, _ = knn.kneighbors(query_2d)
    return dists[:, -1]


def _knn_cosine_distance_to_training_256d(query, ref_256, k=10):
    """For each query point in 256-d, MEAN cosine distance to its k nearest
    training neighbors. This is the same metric Block I uses to detect
    modality-asymmetric drift; it survives in 256-d but collapses to noise
    in the UMAP-2D projection. Smaller = closer to training distribution.
    Returns (Q,) array."""
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import normalize
    q_n   = normalize(query)
    ref_n = normalize(ref_256)
    knn = NearestNeighbors(n_neighbors=k, metric="cosine",
                           algorithm="brute", n_jobs=-1).fit(ref_n)
    dists, _ = knn.kneighbors(q_n)
    return dists.mean(axis=1)


def _pick_modality_asymmetry_traj_256d(Z_traj, A_emb, ref_256, role):
    """Pick exemplar trajectory using 256-d cosine kNN distance to the
    training distribution. The 2D UMAP picker (legacy) can fail because
    UMAP collapses the modality-asymmetric drift signal; this picker uses
    the same metric as Block I.

    role='image_in':  iter-0 starts far from training core (high d), iter-K close (low d)
    role='text_out':  iter-0 starts close (low d), iter-K far (high d)
    """
    d0 = _knn_cosine_distance_to_training_256d(A_emb,             ref_256, k=10)
    dK = _knn_cosine_distance_to_training_256d(Z_traj[:, -1, :],  ref_256, k=10)
    if role == "image_in":
        score = d0 - dK
    elif role == "text_out":
        score = dK - d0
    else:
        raise ValueError(f"unknown role {role}")

    # Monotone-drift filter using ALL intermediate iterations.
    K_iters = Z_traj.shape[1]
    d_full = np.zeros((len(Z_traj), K_iters))
    for k_ in range(K_iters):
        d_full[:, k_] = _knn_cosine_distance_to_training_256d(
            Z_traj[:, k_, :], ref_256, k=10)
    monotone = np.zeros(len(Z_traj))
    for i in range(len(Z_traj)):
        if role == "image_in":
            ok = (d_full[i, 1:] <= d_full[i, 0]).sum() / (K_iters - 1)
        else:
            ok = (d_full[i, 1:] >= d_full[i, 0]).sum() / (K_iters - 1)
        monotone[i] = ok
    combined = score * (monotone > 0.6).astype(float)
    rank = np.argsort(combined)[::-1]
    best = int(rank[0])
    return best, float(d0[best]), float(dK[best]), d_full[best]


def _pick_modality_asymmetry_traj(Z_traj, A_emb, reducer, ref_2d, role):
    """UMAP-2D fallback picker for when 256-d reference embeddings are missing.
    Should not normally be needed in our setup but kept for robustness."""
    A_2d = reducer.transform(A_emb)
    K_2d = reducer.transform(Z_traj[:, -1, :])
    d0 = _knn_distance_to_training(A_2d,  ref_2d, k=10)
    dK = _knn_distance_to_training(K_2d,  ref_2d, k=10)
    if role == "image_in":
        score = d0 - dK
    elif role == "text_out":
        score = dK - d0
    else:
        raise ValueError(f"unknown role {role}")

    K_iters = Z_traj.shape[1]
    inter_2d = np.stack([reducer.transform(Z_traj[:, k, :]) for k in range(K_iters)],
                        axis=1)
    monotone_score = np.zeros(len(Z_traj))
    for i in range(len(Z_traj)):
        d_per_iter = _knn_distance_to_training(inter_2d[i], ref_2d, k=10)
        if role == "image_in":
            ok = (d_per_iter[1:] <= d_per_iter[0] + 0.5).sum() / (K_iters - 1)
        else:
            ok = (d_per_iter[1:] >= d_per_iter[0] - 0.5).sum() / (K_iters - 1)
        monotone_score[i] = ok
    combined = score * (monotone_score > 0.6).astype(float)
    rank = np.argsort(combined)[::-1]
    best = int(rank[0])
    return best, float(d0[best]), float(dK[best])


def figS_asym_umap_exemplar(cfg, out_dir):
    """Image and text exemplar trajectories + cohort 256-d cosine kNN
    histograms — drifting INTO and OUT of the MedCLIP training distribution.

    Splits into 5 PNGs (matches the main-figure convention of separable
    sub-panels): 4 panels + 1 unified legend.

        figS_asym_image_traj.png    image exemplar UMAP trajectory
        figS_asym_image_hist.png    image cohort 256-d kNN histogram
        figS_asym_text_traj.png     text exemplar UMAP trajectory
        figS_asym_text_hist.png     text cohort 256-d kNN histogram
        figS_asym_legend.png        unified legend covering all 4 panels

    Story: ChexGen pulls images toward its training prior (image kNN-cos
    shrinks); MAIRA-2 pushes text outside MedCLIP's text training manifold
    (text kNN-cos grows). The asymmetry signal lives in 256-d cosine
    geometry — the UMAP-2D projection compresses it away, so we use
    256-d cosine kNN for the cohort histograms and UMAP-2D only for the
    visual exemplar trajectory.

    Requires HPC trajectory data:
        --main_dir /n/.../chexgen_long              per-study .npy files
        --ref_dir  /n/.../reference_embeddings       img/txt embeddings + UMAP reducers
    """
    main_dir = cfg.get("main_dir")
    ref_dir = cfg.get("ref_dir")
    if not (main_dir and ref_dir and os.path.isdir(main_dir) and os.path.isdir(ref_dir)):
        raise RuntimeError(
            "figS_asym_umap_exemplar requires --main_dir and --ref_dir "
            "(per-study trajectory .npy files + reference embeddings + UMAP reducers).\n"
            "  Skipping: this panel must be generated on the HPC where the data lives.")
    ref_dir = Path(ref_dir)

    print("  loading reference embeddings + UMAP reducers ...")
    ref_img_256 = np.load(ref_dir / "img_embeds.npy")
    ref_txt_256 = np.load(ref_dir / "txt_embeds.npy")
    ref_img_2d  = np.load(ref_dir / "umap_img_2d.npy")
    ref_txt_2d  = np.load(ref_dir / "umap_txt_2d.npy")
    with open(ref_dir / "umap_img.pkl", "rb") as f: reducer_img = pickle.load(f)
    with open(ref_dir / "umap_txt.pkl", "rb") as f: reducer_txt = pickle.load(f)

    print("  loading per-study trajectory embeddings ...")
    K_max_load = int(cfg.get("asym_K_load", 10))
    sids_img, A_img, Z_img = _load_hpc_trajectories(main_dir, "img", K_max=K_max_load)
    sids_txt, A_txt, Z_txt = _load_hpc_trajectories(main_dir, "text", K_max=K_max_load)
    print(f"  loaded {len(sids_img)} image trajectories, {len(sids_txt)} text trajectories"
          f" (K_max={K_max_load})")

    have_256d = (ref_img_256 is not None) and (ref_txt_256 is not None)
    if have_256d:
        i_img, d0_img, dK_img, _ = _pick_modality_asymmetry_traj_256d(
            Z_img, A_img, ref_img_256, role="image_in")
        i_txt, d0_txt, dK_txt, _ = _pick_modality_asymmetry_traj_256d(
            Z_txt, A_txt, ref_txt_256, role="text_out")
        metric_label = "256-d cosine"
    else:
        print("  WARNING: falling back to UMAP-2D picker (256-d ref embeddings missing)")
        i_img, d0_img, dK_img = _pick_modality_asymmetry_traj(
            Z_img, A_img, reducer_img, ref_img_2d, role="image_in")
        i_txt, d0_txt, dK_txt = _pick_modality_asymmetry_traj(
            Z_txt, A_txt, reducer_txt, ref_txt_2d, role="text_out")
        metric_label = "UMAP-2D euclidean (FALLBACK)"

    print(f"  image-IN  exemplar: idx={i_img} ({sids_img[i_img][:8]})  "
          f"d_kNN(iter 0)={d0_img:.3f} → d_kNN(iter K)={dK_img:.3f} ({metric_label})")
    print(f"  text-OUT  exemplar: idx={i_txt} ({sids_txt[i_txt][:8]})  "
          f"d_kNN(iter 0)={d0_txt:.3f} → d_kNN(iter K)={dK_txt:.3f} ({metric_label})")

    # UMAP-project the chosen exemplar trajectories
    A_img_2d    = reducer_img.transform(A_img[i_img][None, :])[0]
    traj_img_2d = reducer_img.transform(Z_img[i_img])
    A_txt_2d    = reducer_txt.transform(A_txt[i_txt][None, :])[0]
    traj_txt_2d = reducer_txt.transform(Z_txt[i_txt])

    # Cohort-level kNN distances. Use 256-d cosine if available (real signal).
    if have_256d:
        cohort_d0_img = _knn_cosine_distance_to_training_256d(A_img,            ref_img_256, k=10)
        cohort_dK_img = _knn_cosine_distance_to_training_256d(Z_img[:, -1, :], ref_img_256, k=10)
        cohort_d0_txt = _knn_cosine_distance_to_training_256d(A_txt,            ref_txt_256, k=10)
        cohort_dK_txt = _knn_cosine_distance_to_training_256d(Z_txt[:, -1, :], ref_txt_256, k=10)
        x_axis_label = "Mean cosine distance to 10-NN in training distribution"
    else:
        A_img_all_2d = reducer_img.transform(A_img)
        K_img_all_2d = reducer_img.transform(Z_img[:, -1, :])
        cohort_d0_img = _knn_distance_to_training(A_img_all_2d, ref_img_2d, k=10)
        cohort_dK_img = _knn_distance_to_training(K_img_all_2d, ref_img_2d, k=10)
        A_txt_all_2d = reducer_txt.transform(A_txt)
        K_txt_all_2d = reducer_txt.transform(Z_txt[:, -1, :])
        cohort_d0_txt = _knn_distance_to_training(A_txt_all_2d, ref_txt_2d, k=10)
        cohort_dK_txt = _knn_distance_to_training(K_txt_all_2d, ref_txt_2d, k=10)
        x_axis_label = "Distance to 10-NN in UMAP-2D space (FALLBACK)"

    med_d0_img, med_dK_img = float(np.median(cohort_d0_img)), float(np.median(cohort_dK_img))
    med_d0_txt, med_dK_txt = float(np.median(cohort_d0_txt)), float(np.median(cohort_dK_txt))
    delta_img = med_dK_img - med_d0_img
    delta_txt = med_dK_txt - med_d0_txt
    print(f"  cohort image: median {med_d0_img:.3f} → {med_dK_img:.3f}  "
          f"(Δ={delta_img:+.3f}; {'IN' if delta_img < 0 else 'OUT'})")
    print(f"  cohort text : median {med_d0_txt:.3f} → {med_dK_txt:.3f}  "
          f"(Δ={delta_txt:+.3f}; {'OUT' if delta_txt > 0 else 'IN'})")

    # Color palette — match the main MI cliff figure conventions
    C_IMG_MI = "#21295C"   # accent paper blue (image)
    C_TXT_MI = "#C73E1D"   # warning soft red (text)
    C_TRAIN  = "#B8C5D6"
    C_ANCHOR = "#FFD166"
    C_MUTED  = "#A0A0A0"
    K_n = traj_img_2d.shape[0]

    # ─── Trajectory panels (clean, no inline legend) ─────────────────────────
    def _render_traj(traj_2d, anchor_2d, ref_2d, color, title, fname, rng_seed):
        fig, ax = plt.subplots(figsize=(7.0, 5.5))
        sub = np.random.default_rng(rng_seed).choice(
            len(ref_2d), size=min(20000, len(ref_2d)), replace=False)
        ax.scatter(ref_2d[sub, 0], ref_2d[sub, 1], s=3,
                   c=C_TRAIN, alpha=0.35, rasterized=True)
        for k in range(K_n):
            alpha = 0.30 + 0.70 * (k / max(K_n - 1, 1))
            size  = 30 + 90 * (k / max(K_n - 1, 1))
            ax.scatter(traj_2d[k, 0], traj_2d[k, 1],
                       s=size, c=color, alpha=alpha,
                       edgecolors="white", linewidths=0.7, zorder=4)
        ax.plot(traj_2d[:, 0], traj_2d[:, 1], "-",
                color=color, lw=2.2, alpha=0.7, zorder=3)
        ax.scatter(anchor_2d[0], anchor_2d[1], s=380, marker="*",
                   c=C_ANCHOR, edgecolors=color, linewidths=2.4, zorder=5)
        ax.scatter(traj_2d[-1, 0], traj_2d[-1, 1], s=180, marker="s",
                   c=color, edgecolors="white", linewidths=2.0, zorder=5)
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        ax.set_title(title, fontweight="bold", fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        _save(fig, out_dir, fname, cfg["pdf"])

    _render_traj(traj_img_2d, A_img_2d, ref_img_2d, C_IMG_MI,
                 "Image trajectory drifts INTO MedCLIP training distribution",
                 "figS_asym_image_traj", rng_seed=0)
    _render_traj(traj_txt_2d, A_txt_2d, ref_txt_2d, C_TXT_MI,
                 "Text trajectory drifts OUT of MedCLIP training distribution",
                 "figS_asym_text_traj", rng_seed=1)

    # ─── Histogram panels (clean, no inline legend) ──────────────────────────
    def _render_hist(d0, dK, color, title_prefix, n_total, delta, fname):
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        bins = np.linspace(0, max(d0.max(), dK.max()) * 1.05, 35)
        med0, medK = float(np.median(d0)), float(np.median(dK))
        ax.hist(d0, bins=bins, color=C_MUTED, alpha=0.55, edgecolor="white")
        ax.hist(dK, bins=bins, color=color,   alpha=0.75, edgecolor="white")
        ax.axvline(med0, color=C_MUTED, ls="--", lw=1.5)
        ax.axvline(medK, color=color,   ls="--", lw=1.5)
        ax.set_xlabel(x_axis_label, fontsize=10)
        ax.set_ylabel("Trajectories")
        direction = "OUT" if delta > 0 else "IN"
        ax.set_title(
                     f"Δmedian = {delta:+.3f}, drifts {direction}",
                     fontsize=12)
        ax.grid(False)
        fig.tight_layout()
        _save(fig, out_dir, fname, cfg["pdf"])

    _render_hist(cohort_d0_img, cohort_dK_img, C_IMG_MI,
                 "Image", len(cohort_d0_img), delta_img, "figS_asym_image_hist")
    _render_hist(cohort_d0_txt, cohort_dK_txt, C_TXT_MI,
                 "Text",  len(cohort_d0_txt), delta_txt, "figS_asym_text_hist")

    # ─── Unified legend (covers all 4 panels) ────────────────────────────────
    _save_asym_legend(out_dir, cfg, C_IMG_MI, C_TXT_MI,
                      C_TRAIN, C_ANCHOR, C_MUTED, K_n)


def _save_asym_legend(out_dir, cfg, C_IMG_MI, C_TXT_MI,
                      C_TRAIN, C_ANCHOR, C_MUTED, K_n):
    """Standalone legend covering both trajectory and histogram panels."""
    fig = plt.figure(figsize=(4.5, 4.8))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    y = 0.97
    ax.text(0.04, y, "Trajectory panels", fontweight="bold", fontsize=10.5, va="top")

    y -= 0.075
    ax.scatter([0.13], [y], s=22, c=C_TRAIN, alpha=0.6)
    ax.text(0.23, y, "Training distribution", fontsize=9, va="center")

    y -= 0.075
    ax.scatter([0.13], [y], s=200, marker="*", c=C_ANCHOR,
               edgecolors="#444", linewidths=1.2)
    ax.text(0.23, y, "Anchor (iter 0)", fontsize=9, va="center")

    y -= 0.075
    for i, k in enumerate([0.0, 0.33, 0.66, 1.0]):
        ax.scatter([0.07 + i*0.035], [y], s=24 + 80*k,
                   c=C_IMG_MI, alpha=0.30 + 0.70*k,
                   edgecolors="white", linewidths=0.6)
    ax.text(0.27, y, "Image iter $1{\\ldots}K{-}1$", fontsize=9, va="center", color=C_IMG_MI)

    y -= 0.075
    for i, k in enumerate([0.0, 0.33, 0.66, 1.0]):
        ax.scatter([0.07 + i*0.035], [y], s=24 + 80*k,
                   c=C_TXT_MI, alpha=0.30 + 0.70*k,
                   edgecolors="white", linewidths=0.6)
    ax.text(0.27, y, "Text iter $1{\\ldots}K{-}1$", fontsize=9, va="center", color=C_TXT_MI)

    y -= 0.075
    ax.scatter([0.13], [y], s=80, marker="s", c=C_IMG_MI,
               edgecolors="white", linewidths=1.4)
    ax.text(0.23, y, f"Image endpoint (iter {K_n-1})", fontsize=9, va="center", color=C_IMG_MI)

    y -= 0.075
    ax.scatter([0.13], [y], s=80, marker="s", c=C_TXT_MI,
               edgecolors="white", linewidths=1.4)
    ax.text(0.23, y, f"Text endpoint (iter {K_n-1})", fontsize=9, va="center", color=C_TXT_MI)

    # Histogram conventions
    y -= 0.10
    ax.text(0.04, y, "Histogram panels", fontweight="bold", fontsize=10.5, va="top")

    y -= 0.075
    ax.add_patch(plt.Rectangle((0.06, y - 0.022), 0.14, 0.044,
                               color=C_MUTED, alpha=0.55))
    ax.text(0.23, y, "Iter 0 cohort", fontsize=9, va="center")

    y -= 0.075
    ax.add_patch(plt.Rectangle((0.06, y - 0.022), 0.14, 0.044,
                               color=C_IMG_MI, alpha=0.75))
    ax.text(0.23, y, f"Image iter {K_n-1} cohort", fontsize=9, va="center", color=C_IMG_MI)

    y -= 0.075
    ax.add_patch(plt.Rectangle((0.06, y - 0.022), 0.14, 0.044,
                               color=C_TXT_MI, alpha=0.75))
    ax.text(0.23, y, f"Text iter {K_n-1} cohort", fontsize=9, va="center", color=C_TXT_MI)

    y -= 0.075
    ax.plot([0.06, 0.20], [y, y], color=C_MUTED, ls="--", lw=1.5)
    ax.text(0.23, y, "Iter 0 median", fontsize=9, va="center")

    y -= 0.07
    ax.plot([0.06, 0.20], [y, y], color="#444", ls="--", lw=1.5)
    ax.text(0.23, y, "Iter K median (in modality color)", fontsize=9, va="center")

    fig.savefig(out_dir / "figS_asym_legend.png", dpi=600,
                bbox_inches="tight", pad_inches=0.15)
    if cfg["pdf"]:
        fig.savefig(out_dir / "figS_asym_legend.pdf",
                    bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"  → figS_asym_legend.png")


# ════════════════════════════════════════════════════════════════════════════
#  S_oov_inflation — OOV inflation factor table (Block K)
# ════════════════════════════════════════════════════════════════════════════
#
# Per-OOV-category GT prevalence vs K=100 cohort prevalence with inflation
# factor (= K100_frac / GT_frac). Directly addresses the reviewer concern
# "is 12.5% COPD really inflation?" by extracting GT prevalence of every
# OOV category from the FINDINGS column of the cohort and comparing to
# the loop's iter-100 prevalence.
#
# The story: the loop SELECTIVELY amplifies diffuse parenchymal templates
# (COPD ~3.5x, pleural plaques ~3.6x, pulmonary fibrosis ~2.3x) and
# EXTINGUISHES focal/anatomically-anchored findings (catheters, pacemakers,
# sternotomy wires, all going to ~0%). This asymmetry is the empirical
# fingerprint of the image-side mode-seeking mechanism: ChexGen's prior
# over generated images is biased toward diffuse parenchymal patterns,
# and MAIRA-2 then templates these into the corresponding OOV phrasings.

def figS_oov_inflation(cfg, out_dir):
    """Per-OOV-category GT prevalence vs K=100 prevalence, with inflation
    factors. Shows the loop's selective amplification of diffuse parenchymal
    OOV templates and suppression of focal/device-related findings.
    Legend saved separately as figS_oov_inflation_legend.png."""
    bk = _load_json(cfg["block_k"])
    infl_table = bk.get("oov_inflation_table")
    if not infl_table:
        raise RuntimeError("oov_inflation_table missing in block_K_results.json")
    K_max = max(bk["probe_iters"])
    K_max_str = str(K_max)
    n_cohort = bk["n_studies"]

    # Collect per-category numbers, drop categories that are 0 at both
    # GT and K=100 (truly empty taxonomy slots that contribute nothing).
    rows = []
    for cat, row in infl_table.items():
        it = row["iters"][K_max_str]
        if row["gt_count"] == 0 and it["iter_count"] == 0:
            continue
        rows.append({
            "cat":          cat,
            "gt_count":     row["gt_count"],
            "gt_frac":      row["gt_fraction"],
            "iter_count":   it["iter_count"],
            "iter_frac":    it["iter_fraction"],
            "inflation":    it["inflation"],
        })
    # Sort by K=100 fraction descending so amplified categories appear at top
    rows.sort(key=lambda r: -r["iter_frac"])
    # Cap to top-12 for readability
    rows = rows[:12]

    cats = [shorten_label(r["cat"], 30) for r in rows]
    gt_pct = np.array([100 * r["gt_frac"] for r in rows])
    it_pct = np.array([100 * r["iter_frac"] for r in rows])
    infl_v = np.array([r["inflation"] for r in rows])

    # Color the iter-K bar by regime: amplified / preserved / suppressed.
    def _color(infl):
        if not np.isfinite(infl):
            return C_AMPLIFIED
        if infl >= 1.5:
            return C_AMPLIFIED
        if infl <= 0.5:
            return C_ELEVATOR
        return C_PRESERVED
    bar_colors = [_color(v) for v in infl_v]

    fig, ax = new_panel(width=PANEL_FULL_W, height=PANEL_DEFAULT_H + 1.6)
    y = np.arange(len(cats))
    ax.barh(y - 0.2, gt_pct, height=0.36, color=C_GT, alpha=0.7,
            edgecolor="white", lw=0.4)
    ax.barh(y + 0.2, it_pct, height=0.36, color=bar_colors, alpha=0.85,
            edgecolor="white", lw=0.4)

    # Inflation factor annotations to the right
    x_max = max(gt_pct.max(), it_pct.max())
    for i, infl in enumerate(infl_v):
        if not np.isfinite(infl):
            txt = r"$\infty$"
        elif infl == 0:
            txt = r"0.00$\times$"
        else:
            txt = rf"{infl:.2f}$\times$"
        ax.text(x_max * 1.04, y[i], txt, fontsize=7.5,
                va="center", color=C_ANNOT)

    ax.set_yticks(y)
    ax.set_yticklabels(cats, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel(rf"Cohort prevalence (\% of $N{{=}}{n_cohort}$)")
    ax.set_xlim(0, x_max * 1.20)
    ax.set_title(
        rf"Out-of-vocabulary inflation: GT vs $K{{=}}{K_max}$ "
        rf"(inflation factor at right of each pair)"
    )

    ax.grid(axis="x", alpha=0.3, lw=0.4)
    fig.tight_layout()
    _save(fig, out_dir, "figS_oov_inflation", cfg["pdf"])
    _save_oov_inflation_legend(out_dir, cfg)


def _save_oov_inflation_legend(out_dir, cfg):
    """Standalone legend for figS_oov_inflation."""
    fig = plt.figure(figsize=(3.4, 1.9))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    y = 0.92
    ax.text(0.04, y, "Bars (per category)", fontweight="bold",
            fontsize=10.5, va="top")

    y -= 0.16
    ax.add_patch(plt.Rectangle((0.06, y - 0.04), 0.14, 0.08,
                                color=C_GT, alpha=0.7))
    ax.text(0.23, y, "GT prevalence", fontsize=9, va="center")

    y -= 0.16
    ax.add_patch(plt.Rectangle((0.06, y - 0.04), 0.14, 0.08,
                                color=C_AMPLIFIED, alpha=0.85))
    ax.text(0.23, y, r"Amplified at $K{=}100$ ($> 1.5\times$ GT)",
            fontsize=9, va="center")

    y -= 0.16
    ax.add_patch(plt.Rectangle((0.06, y - 0.04), 0.14, 0.08,
                                color=C_PRESERVED, alpha=0.85))
    ax.text(0.23, y, r"Preserved at $K{=}100$ ($0.5$--$1.5\times$ GT)",
            fontsize=9, va="center")

    y -= 0.16
    ax.add_patch(plt.Rectangle((0.06, y - 0.04), 0.14, 0.08,
                                color=C_ELEVATOR, alpha=0.85))
    ax.text(0.23, y, r"Suppressed at $K{=}100$ ($< 0.5\times$ GT)",
            fontsize=9, va="center")

    fig.savefig(out_dir / "figS_oov_inflation_legend.png", dpi=600,
                bbox_inches="tight", pad_inches=0.15)
    if cfg["pdf"]:
        fig.savefig(out_dir / "figS_oov_inflation_legend.pdf",
                    bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"  → figS_oov_inflation_legend.png")


# ════════════════════════════════════════════════════════════════════════════
#  S_mi_calibration — MI estimator validation (synthetic + kNN sweep)
# ════════════════════════════════════════════════════════════════════════════
#
# Two-panel:
#   (left)  Synthetic Gaussian calibration. Run the SAME MI estimator (KSG
#           with shared-PCA-20, kNN=5) at the SAME (N=1008, d=256) regime
#           on synthetic data with known I(X;Y), reading off:
#             - the estimator FLOOR at I=0 (~0.007 nats)
#             - the bias curve relative to identity
#           The paper iter-0 (5.64) and iter-1 (0.17) values are overlaid as
#           horizontal reference lines.  Conclusion: the iter-1 0.17-nat
#           value sits ~23x above the estimator floor and is therefore not
#           an artifact; the >97% drop is real.
#   (right) kNN sensitivity at iter-1. MI(k=1) for image and text under
#           kNN in {3,5,10,20}. Spread is <5%, confirming the headline
#           collapse is robust to estimator hyperparameter choice.

def figS_mi_calibration(cfg, out_dir):
    """MI estimator calibration on synthetic Gaussian + kNN sensitivity.
    Legend saved separately as figS_mi_calibration_legend.png."""
    if not os.path.exists(cfg["mi_calibration"]):
        raise RuntimeError(f"mi_calibration.json not found at {cfg['mi_calibration']}")
    mc = _load_json(cfg["mi_calibration"])
    aj = _load_json(cfg["analysis_json"])

    rows = mc["rows"]
    true_mi = np.array([r["target_per_dim_MI_nats"] for r in rows])
    est_mean = np.array([r["estimated_MI_mean"] for r in rows])
    est_std = np.array([r["estimated_MI_std"] for r in rows])

    paper_iter0_img = float(aj["E"]["MI_img_per_k"][0])
    paper_iter1_img = float(aj["E"]["MI_img_per_k"][1])

    fig, axes = plt.subplots(
        1, 2, figsize=(PANEL_FULL_W, PANEL_DEFAULT_H + 0.4),
        gridspec_kw={"width_ratios": [1.45, 1.0]},
    )

    # ── Left: calibration curve ─────────────────────────────────────────────
    ax = axes[0]
    x_ref = np.linspace(0, max(true_mi) * 1.05, 100)
    ax.plot(x_ref, x_ref, color="#888", ls="--", lw=0.8, zorder=1)
    ax.errorbar(true_mi, est_mean, yerr=est_std, fmt="o", ms=5,
                color=C_IMG, ecolor=C_IMG, elinewidth=0.8, capsize=2.5,
                zorder=4)
    ax.axhline(paper_iter0_img, color=C_ITER0, ls=":", lw=0.8,
               alpha=0.7, zorder=2)
    ax.axhline(paper_iter1_img, color=C_ITERK, ls=":", lw=0.8,
               alpha=0.7, zorder=2)

    ax.set_xlabel("True per-dim MI (nats)")
    ax.set_ylabel("Estimated per-dim MI (nats)")
    ax.set_title(
        rf"Synthetic Gaussian calibration"
        rf" ($N{{=}}{mc['N']}$, $d{{=}}{mc['d']}$, kNN$={mc['knn']}$)"
    )
    ax.grid(True, alpha=0.3, lw=0.4)
    ax.set_xlim(-0.2, max(true_mi) * 1.05)

    # ── Right: kNN sensitivity at iter-1 ────────────────────────────────────
    ax = axes[1]
    sweep = aj.get("E", {}).get("MI_iter1_knn_sweep", {})
    floor_val = est_mean[0]  # I_true=0 row
    if sweep:
        kns = sorted(sweep["image"].keys(), key=int)
        kn_int = [int(k) for k in kns]
        mi_img_v = [sweep["image"][k] for k in kns]
        mi_txt_v = [sweep["text"][k] for k in kns]
        ax.plot(kn_int, mi_img_v, "o-", color=C_IMG, ms=6, lw=1.4)
        ax.plot(kn_int, mi_txt_v, "s-", color=C_TXT, ms=6, lw=1.4)
        ax.axhline(floor_val, color="#888", ls=":", lw=0.7, alpha=0.7)
        ax.set_xlabel("KSG kNN parameter")
        ax.set_ylabel(r"MI($z^{(0)};z^{(1)}$) (nats)")
        ax.set_title("kNN sensitivity at iter-1")
        ax.set_xticks(kn_int)
        ax.grid(True, alpha=0.3, lw=0.4)
        ax.set_ylim(0, max(max(mi_img_v), max(mi_txt_v)) * 1.4)
    else:
        ax.text(0.5, 0.5, "MI_iter1_knn_sweep not available",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=8, color="#888")
        ax.set_axis_off()

    fig.tight_layout()
    _save(fig, out_dir, "figS_mi_calibration", cfg["pdf"])
    _save_mi_calibration_legend(out_dir, cfg, paper_iter0_img,
                                  paper_iter1_img, floor_val)


def _save_mi_calibration_legend(out_dir, cfg, paper_iter0_img,
                                  paper_iter1_img, floor_val):
    """Standalone legend covering both panels of figS_mi_calibration."""
    fig = plt.figure(figsize=(3.8, 2.4))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    y = 0.96
    ax.text(0.04, y, "Left panel (calibration)",
            fontweight="bold", fontsize=10.5, va="top")

    y -= 0.13
    ax.plot([0.06, 0.20], [y, y], color="#888", ls="--", lw=0.9)
    ax.text(0.23, y, r"Identity ($I_{\rm est} = I_{\rm true}$)",
            fontsize=8.5, va="center")

    y -= 0.10
    ax.errorbar([0.13], [y], yerr=[0.012], fmt="o", ms=5,
                color=C_IMG, ecolor=C_IMG, elinewidth=0.7, capsize=2)
    ax.text(0.23, y, "KSG estimator (mean $\\pm$ SD, 10 reps)",
            fontsize=8.5, va="center")

    y -= 0.10
    ax.plot([0.06, 0.20], [y, y], color=C_ITER0, ls=":", lw=0.9)
    ax.text(0.23, y, f"Paper iter-0 image MI = {paper_iter0_img:.2f} nats",
            fontsize=8.5, va="center")

    y -= 0.10
    ax.plot([0.06, 0.20], [y, y], color=C_ITERK, ls=":", lw=0.9)
    ax.text(0.23, y, f"Paper iter-1 image MI = {paper_iter1_img:.2f} nats",
            fontsize=8.5, va="center")

    y -= 0.16
    ax.text(0.04, y, "Right panel (kNN sweep)",
            fontweight="bold", fontsize=10.5, va="top")

    y -= 0.10
    ax.plot([0.06, 0.20], [y, y], color=C_IMG, lw=1.4)
    ax.scatter([0.13], [y], color=C_IMG, s=22, marker="o", zorder=3)
    ax.text(0.23, y, r"Image MI($k{=}1$)", fontsize=8.5, va="center")

    y -= 0.10
    ax.plot([0.06, 0.20], [y, y], color=C_TXT, lw=1.4)
    ax.scatter([0.13], [y], color=C_TXT, s=22, marker="s", zorder=3)
    ax.text(0.23, y, r"Text MI($k{=}1$)", fontsize=8.5, va="center")

    y -= 0.10
    ax.plot([0.06, 0.20], [y, y], color="#888", ls=":", lw=0.9)
    ax.text(0.23, y,
            f"Estimator floor at $I{{=}}0$ ({floor_val:.4f} nats)",
            fontsize=8.5, va="center")

    fig.savefig(out_dir / "figS_mi_calibration_legend.png", dpi=600,
                bbox_inches="tight", pad_inches=0.15)
    if cfg["pdf"]:
        fig.savefig(out_dir / "figS_mi_calibration_legend.pdf",
                    bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"  → figS_mi_calibration_legend.png")


# ════════════════════════════════════════════════════════════════════════════
#  S_lyap_saturation — Estimator saturation under bounded support
# ════════════════════════════════════════════════════════════════════════════
#
# This panel CHARACTERIZES the saturation behavior of the finite-separation
# divergence-rate estimator under bounded support, on a synthetic system
# (coupled logistic, r=4, c=0.1) where the analytical Lyapunov exponent is
# known to be λ_true = ln(2) ≈ 0.6931. It is NOT a validation that our
# estimator equals the classical exponent; it shows that the estimator
# DECAYS monotonically with horizon as the trajectories occupy the basin,
# which is exactly the bounded-support behavior described in §3.2.
#
# The paper's per-anchor curve from Block B (chest X-ray loop) is plotted
# on the right with the same axes, showing the same monotonic decay
# pattern. This makes the bounded-support framing of the paper's absolute
# numbers self-evident.
#
# The framing is: same estimator, two systems with different physics,
# both produce the same monotonic decay because both have bounded support.
# We do not extract a "correct" Lyapunov exponent from either; we use
# the SIGNS and STABILITY across k as the inferentially meaningful
# quantities (per the rephrasing in §3.2 and §5.3).

def figS_lyap_saturation(cfg, out_dir):
    """Finite-separation divergence-rate estimator: saturation under
    bounded support, on synthetic and paper data side by side.
    Legend saved separately as figS_lyap_saturation_legend.png."""
    if not os.path.exists(cfg["lyap_synthetic"]):
        raise RuntimeError(f"lyapunov_synthetic.json not found at {cfg['lyap_synthetic']}")
    ls = _load_json(cfg["lyap_synthetic"])
    aj = _load_json(cfg["analysis_json"])

    fig, axes = plt.subplots(
        1, 2, figsize=(PANEL_FULL_W, PANEL_DEFAULT_H + 0.4),
        sharex=False,
    )

    # ── Left: synthetic system (coupled logistic) ─────────────────────────
    ax = axes[0]
    lam_true = ls["lambda_true_analytical"]
    lam_a_per_k = np.array(ls["lambda_a_cohort_mean_per_k"])
    lam_sys_per_k = np.array(ls["lambda_sys_per_k"])
    K_synth = len(lam_a_per_k) - 1
    ks = np.arange(K_synth + 1)

    ax.axhline(lam_true, color="#222", ls="--", lw=1.1, alpha=0.75, zorder=3)
    ax.plot(ks[1:], lam_a_per_k[1:], "-", color=C_LANC, lw=1.6, zorder=4)
    ax.plot(ks[1:], lam_sys_per_k[1:], "-", color=C_LSYS, lw=1.4, zorder=4)
    ax.axhline(0, color=C_ZERO, lw=0.6, ls="-", alpha=0.4, zorder=1)

    ax.set_xlabel("Iteration $k$")
    ax.set_ylabel("Finite-time divergence rate")
    ax.set_title(
        rf"Synthetic: coupled logistic ($r{{=}}{ls['args']['r']}$, "
        rf"$c{{=}}{ls['args']['c']}$)"
    )
    ax.grid(True, alpha=0.3, lw=0.4)
    ax.set_xlim(0, K_synth)

    # ── Right: paper data (chest X-ray loop) ──────────────────────────────
    ax = axes[1]
    paper_lsys = np.array(aj["B"]["lambda_sys_per_k"])
    K_paper = len(paper_lsys) - 1
    ks_p = np.arange(K_paper + 1)

    per_anchor = aj["B"]["lyapunov_per_anchor"]
    sids = sorted(per_anchor.keys())
    arrs = np.stack([np.asarray(per_anchor[s], dtype=float) for s in sids])
    per_anchor_mean = arrs.mean(axis=0)

    ax.plot(ks_p[1:], per_anchor_mean[1:], "-", color=C_LANC, lw=1.6, zorder=4)
    ax.plot(ks_p[1:], paper_lsys[1:], "-", color=C_LSYS, lw=1.4, zorder=4)
    ax.axhline(0, color=C_ZERO, lw=0.6, ls="-", alpha=0.4, zorder=1)

    ax.set_xlabel("Iteration $k$")
    ax.set_ylabel("Finite-time divergence rate")
    ax.set_title("Chest X-ray loop (paper data)")
    ax.grid(True, alpha=0.3, lw=0.4)
    ax.set_xlim(0, K_paper)

    fig.suptitle(
        "Same estimator, two bounded-support systems: monotonic decay of "
        r"$\bar\lambda_a$ with horizon",
        fontsize=8.5, y=1.02,
    )
    fig.tight_layout()
    _save(fig, out_dir, "figS_lyap_saturation", cfg["pdf"])
    _save_lyap_saturation_legend(out_dir, cfg, lam_true)


def _save_lyap_saturation_legend(out_dir, cfg, lam_true):
    """Standalone legend covering both panels of figS_lyap_saturation."""
    fig = plt.figure(figsize=(3.6, 1.9))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    y = 0.92
    ax.text(0.04, y, "Curves (both panels)", fontweight="bold",
            fontsize=10.5, va="top")

    y -= 0.16
    ax.plot([0.06, 0.20], [y, y], color=C_LANC, lw=1.8)
    ax.text(0.23, y,
            r"Estimator $\bar\lambda_a$ (cohort mean, within-anchor)",
            fontsize=8.5, va="center")

    y -= 0.16
    ax.plot([0.06, 0.20], [y, y], color=C_LSYS, lw=1.6)
    ax.text(0.23, y,
            r"Estimator $\lambda_{\rm sys}$ (cross-anchor pairwise)",
            fontsize=8.5, va="center")

    y -= 0.16
    ax.plot([0.06, 0.20], [y, y], color="#222", ls="--", lw=1.0)
    ax.text(0.23, y,
            rf"Classical $\lambda = \ln 2 \approx {lam_true:.3f}$ "
            r"(left panel only)",
            fontsize=8.5, va="center")

    y -= 0.16
    ax.plot([0.06, 0.20], [y, y], color=C_ZERO, lw=0.8)
    ax.text(0.23, y, "zero", fontsize=8.5, va="center", color="#666")

    fig.savefig(out_dir / "figS_lyap_saturation_legend.png", dpi=600,
                bbox_inches="tight", pad_inches=0.15)
    if cfg["pdf"]:
        fig.savefig(out_dir / "figS_lyap_saturation_legend.pdf",
                    bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"  → figS_lyap_saturation_legend.png")


# ════════════════════════════════════════════════════════════════════════════
#  S_perm_gap — Permutation null gap statistic
# ════════════════════════════════════════════════════════════════════════════
#
# Compares Bernoulli null (existing, marginal-matched, ignores label
# correlations) and column-independent permutation null (new, preserves
# marginals, also destroys correlations) for the soft-clustering gap
# statistic at K=10 and K=100. Reviewers asked whether the K_c=4 -> K_c=2
# consolidation is driven by null choice. The permutation null at K=100
# returns K_c=2 (matches Bernoulli), while at K=10 it returns K_c=3
# (Bernoulli returned K_c=4). The K=100 collapse is therefore robust to
# null choice; the intermediate K=10 cluster count is mildly null-sensitive.

def figS_perm_gap(cfg, out_dir):
    """Bernoulli vs permutation null gap statistic at K=10 and K=100.

    Note: in the original analysis the Bernoulli soft-cluster was computed
    only at K=0 and K=100 (the most informative iters). The permutation null
    is the new analysis added in revision and is computed at K=10 and K=100.
    The K=10 panel therefore shows the permutation curve alone; the K=100
    panel shows both curves and is the direct robustness check.
    Legend saved separately as figS_perm_gap_legend.png.
    """
    bk = _load_json(cfg["block_k"])
    perm = bk.get("gap_perm_null")
    if not perm:
        raise RuntimeError("gap_perm_null missing in block_K_results.json")

    fig, axes = plt.subplots(
        1, 2, figsize=(PANEL_FULL_W, PANEL_DEFAULT_H + 0.3),
        sharey=False,
    )

    best_perm_per_iter = {}
    best_bern_per_iter = {}

    for ax, k_probe_str in zip(axes, ["iter_10", "iter_100"]):
        k_probe = int(k_probe_str.split("_")[1])
        soft = bk.get(f"iter_{k_probe}", {}).get("soft")
        permres = perm.get(k_probe_str)
        if not permres:
            ax.text(0.5, 0.5, f"perm null missing at {k_probe_str}",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue

        ks_perm = sorted([int(k) for k in permres["per_k"].keys()])
        gap_perm_v = [permres["per_k"][str(k)]["gap"] for k in ks_perm]
        gap_perm_se = [permres["per_k"][str(k)]["gap_se"] for k in ks_perm]
        best_perm = permres["best_k_gap"]
        best_perm_per_iter[k_probe] = best_perm

        ax.errorbar(ks_perm, gap_perm_v, yerr=gap_perm_se, fmt="s-",
                    color=C_OOV_DOMINANT, ecolor=C_OOV_DOMINANT,
                    elinewidth=0.7, capsize=2, ms=4, lw=1.2, zorder=4)

        if soft is not None and "per_k" in soft:
            per_k_soft = soft["per_k"]
            ks_b = sorted([int(k) for k in per_k_soft.keys()])
            gap_b = [per_k_soft[str(k)]["gap"] for k in ks_b]
            gap_b_se = [per_k_soft[str(k)]["gap_se"] for k in ks_b]
            best_b = soft.get("best_k_gap")
            best_bern_per_iter[k_probe] = best_b
            ax.errorbar(ks_b, gap_b, yerr=gap_b_se, fmt="o-",
                        color=C_LSYS, ecolor=C_LSYS,
                        elinewidth=0.7, capsize=2, ms=4, lw=1.2, zorder=4)
        else:
            ax.text(
                0.98, 0.04,
                "Bernoulli null not computed at this iter",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=6.5, color="#888", style="italic",
            )

        # Annotate best-K values directly on the panel (varies per panel,
        # so does not belong in a global legend)
        anno_lines = [rf"Permutation null: best $K_c{{=}}{best_perm}$"]
        if k_probe in best_bern_per_iter:
            anno_lines.append(
                rf"Bernoulli null: best $K_c{{=}}{best_bern_per_iter[k_probe]}$"
            )
        ax.text(
            0.98, 0.98, "\n".join(anno_lines),
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor="#CCC", lw=0.5),
        )

        ax.axhline(0, color=C_ZERO, lw=0.5, ls="--", alpha=0.4, zorder=1)
        ax.set_xlabel(r"Cluster count $K_c$")
        ax.set_ylabel("Gap statistic")
        ax.set_title(rf"Probe iteration $K{{=}}{k_probe}$")
        ax.grid(True, alpha=0.3, lw=0.4)

    fig.suptitle("Soft-cluster gap statistic: Bernoulli vs permutation null",
                  fontsize=9, y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, "figS_perm_gap", cfg["pdf"])
    _save_perm_gap_legend(out_dir, cfg)


def _save_perm_gap_legend(out_dir, cfg):
    """Standalone legend for figS_perm_gap."""
    fig = plt.figure(figsize=(3.6, 1.6))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    y = 0.90
    ax.text(0.04, y, "Curves", fontweight="bold", fontsize=10.5, va="top")

    y -= 0.18
    ax.errorbar([0.13], [y], yerr=[0.018], fmt="s-", color=C_OOV_DOMINANT,
                ecolor=C_OOV_DOMINANT, elinewidth=0.7, capsize=2, ms=4, lw=1.4)
    ax.text(0.23, y,
            "Permutation null (column-shuffled, correlation-aware)",
            fontsize=8.5, va="center")

    y -= 0.18
    ax.errorbar([0.13], [y], yerr=[0.018], fmt="o-", color=C_LSYS,
                ecolor=C_LSYS, elinewidth=0.7, capsize=2, ms=4, lw=1.4)
    ax.text(0.23, y,
            "Bernoulli null (marginal-matched, original)",
            fontsize=8.5, va="center")

    y -= 0.18
    ax.plot([0.06, 0.20], [y, y], color=C_ZERO, lw=0.7, ls="--")
    ax.text(0.23, y, "zero", fontsize=8.5, va="center", color="#666")

    fig.savefig(out_dir / "figS_perm_gap_legend.png", dpi=600,
                bbox_inches="tight", pad_inches=0.15)
    if cfg["pdf"]:
        fig.savefig(out_dir / "figS_perm_gap_legend.pdf",
                    bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"  → figS_perm_gap_legend.png")


# ════════════════════════════════════════════════════════════════════════════
#  S_jaccard_autocorr — Two-timescale memory decomposition (Fig 11, App. G)
# ════════════════════════════════════════════════════════════════════════════
#
# Companion to Figure 6D in the main text, providing the full multi-lag
# curves from which the Appendix G Table 1 numbers are read. Two-panel
# figure: position autocorrelation vs lag (left) and kNN-Jaccard
# persistence vs lag (right), both for image and text channels in the
# late window k ∈ [50, 100], with their respective random baselines as
# dashed references. The separation in decay rate between autocorr and
# Jaccard is the two-timescale signature of motion within a basin
# (τ_macro ≫ τ_micro).

def figS_jaccard_autocorr(cfg, out_dir):
    """Late-window position autocorrelation and kNN-Jaccard persistence
    vs lag, image + text channels, with random baselines.

    Two-panel layout matching figS_lyap_saturation. Data source:
    long_horizon_results.json -> J -> {image,text} ->
    {autocorr_late_mean, knn_jaccard_late_mean, lags, random_pair_*}.
    Legend saved separately as figS_jaccard_autocorr_legend.png.
    """
    lh = _load_json(cfg["long_horizon"])
    if "J" not in lh:
        raise RuntimeError("Block J missing in long_horizon_results.json")

    J_img = lh["J"]["image"]
    J_txt = lh["J"]["text"]
    lags = np.asarray(J_img["lags"], dtype=int)

    # Late-window curves (k ≥ late_iter_start = 50)
    ac_img = np.asarray(J_img["autocorr_late_mean"], dtype=float)
    ac_txt = np.asarray(J_txt["autocorr_late_mean"], dtype=float)
    ac_img_se = np.asarray(J_img["autocorr_late_sem"], dtype=float)
    ac_txt_se = np.asarray(J_txt["autocorr_late_sem"], dtype=float)
    jc_img = np.asarray(J_img["knn_jaccard_late_mean"], dtype=float)
    jc_txt = np.asarray(J_txt["knn_jaccard_late_mean"], dtype=float)
    jc_img_se = np.asarray(J_img["knn_jaccard_late_sem"], dtype=float)
    jc_txt_se = np.asarray(J_txt["knn_jaccard_late_sem"], dtype=float)

    rb_ac_img = float(J_img["random_pair_autocorr"])
    rb_ac_txt = float(J_txt["random_pair_autocorr"])
    rb_jc = float(J_img["random_pair_jaccard"])  # same analytic baseline both channels
    k_nn_used = int(J_img.get("k_nn_used", 10))
    late_start = int(J_img.get("late_iter_start", 50))

    fig, axes = plt.subplots(
        1, 2, figsize=(PANEL_FULL_W, PANEL_DEFAULT_H + 0.4),
        sharex=True,
    )

    # ── Left: position autocorrelation vs lag ─────────────────────────────
    ax = axes[0]
    ax.fill_between(lags, ac_img - ac_img_se, ac_img + ac_img_se,
                    color=C_IMG, alpha=0.18, lw=0)
    ax.fill_between(lags, ac_txt - ac_txt_se, ac_txt + ac_txt_se,
                    color=C_TXT, alpha=0.18, lw=0)
    ax.plot(lags, ac_img, marker="o", ms=3.5, lw=1.5, color=C_IMG, zorder=4)
    ax.plot(lags, ac_txt, marker="s", ms=3.5, lw=1.5, color=C_TXT, zorder=4)
    ax.axhline(rb_ac_img, color=C_IMG, lw=0.8, ls=":", alpha=0.7, zorder=2)
    ax.axhline(rb_ac_txt, color=C_TXT, lw=0.8, ls=":", alpha=0.7, zorder=2)

    ax.set_xlabel(r"Lag $\ell$ (iterations)")
    ax.set_ylabel("Position autocorrelation")
    ax.set_title("Macroscopic position memory")
    ax.set_xlim(0, lags.max() + 2)
    ax.grid(True, alpha=0.3, lw=0.4)

    # ── Right: kNN-Jaccard persistence vs lag ─────────────────────────────
    ax = axes[1]
    ax.fill_between(lags, jc_img - jc_img_se, jc_img + jc_img_se,
                    color=C_IMG, alpha=0.18, lw=0)
    ax.fill_between(lags, jc_txt - jc_txt_se, jc_txt + jc_txt_se,
                    color=C_TXT, alpha=0.18, lw=0)
    ax.plot(lags, jc_img, marker="o", ms=3.5, lw=1.5, color=C_IMG, zorder=4)
    ax.plot(lags, jc_txt, marker="s", ms=3.5, lw=1.5, color=C_TXT, zorder=4)
    ax.axhline(rb_jc, color="#666", lw=0.8, ls=":", alpha=0.75, zorder=2)

    ax.set_xlabel(r"Lag $\ell$ (iterations)")
    ax.set_ylabel(rf"$k$NN-Jaccard overlap ($k{{=}}{k_nn_used}$)")
    ax.set_title("Local neighbourhood persistence")
    ax.set_xlim(0, lags.max() + 2)
    ax.grid(True, alpha=0.3, lw=0.4)

    fig.suptitle(
        rf"Two-timescale memory decomposition (late window $k \geq {late_start}$): "
        r"$\tau_{\rm macro} \gg \tau_{\rm micro}$",
        fontsize=8.5, y=1.02,
    )
    fig.tight_layout()
    _save(fig, out_dir, "figS_jaccard_autocorr", cfg["pdf"])
    _save_jaccard_autocorr_legend(out_dir, cfg, rb_ac_img, rb_ac_txt, rb_jc)


def _save_jaccard_autocorr_legend(out_dir, cfg, rb_ac_img, rb_ac_txt, rb_jc):
    """Standalone legend covering both panels of figS_jaccard_autocorr."""
    fig = plt.figure(figsize=(4.0, 2.4))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    y = 0.94
    ax.text(0.04, y, "Curves (both panels)", fontweight="bold",
            fontsize=10.5, va="top")

    y -= 0.13
    ax.plot([0.06, 0.20], [y, y], color=C_IMG, lw=1.6, marker="o", ms=4)
    ax.text(0.23, y, "Image channel (late window)", fontsize=8.5, va="center")

    y -= 0.13
    ax.plot([0.06, 0.20], [y, y], color=C_TXT, lw=1.6, marker="s", ms=4)
    ax.text(0.23, y, "Text channel (late window)", fontsize=8.5, va="center")

    y -= 0.16
    ax.text(0.04, y, "Random baselines", fontweight="bold",
            fontsize=10.5, va="top")

    y -= 0.13
    ax.plot([0.06, 0.20], [y, y], color=C_IMG, lw=0.9, ls=":")
    ax.text(0.23, y,
            rf"Image autocorr (cohort, ${rb_ac_img:.3f}$) — left panel",
            fontsize=8.5, va="center")

    y -= 0.13
    ax.plot([0.06, 0.20], [y, y], color=C_TXT, lw=0.9, ls=":")
    ax.text(0.23, y,
            rf"Text autocorr (cohort, ${rb_ac_txt:.3f}$) — left panel",
            fontsize=8.5, va="center")

    y -= 0.13
    ax.plot([0.06, 0.20], [y, y], color="#666", lw=0.9, ls=":")
    ax.text(0.23, y,
            rf"$k$NN-Jaccard (analytic, ${rb_jc:.4f}$) — right panel",
            fontsize=8.5, va="center")

    fig.savefig(out_dir / "figS_jaccard_autocorr_legend.png", dpi=600,
                bbox_inches="tight", pad_inches=0.15)
    if cfg["pdf"]:
        fig.savefig(out_dir / "figS_jaccard_autocorr_legend.pdf",
                    bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"  → figS_jaccard_autocorr_legend.png")


# ════════════════════════════════════════════════════════════════════════════
#  S_sampler_sensitivity — CFG / DiT-step sweep (Fig 12, App. B.3)
# ════════════════════════════════════════════════════════════════════════════
#
# Two-panel sensitivity check on the loop's sampler hyperparameters.
# CFG sweep (left, T=100): CFG ∈ {2, 7}, with main run at CFG=4.
# Step sweep (right, CFG=4): T ∈ {25, 50}, with main run at T=100.
# Each setting: 20 anchors × 5 iterations. Quantity is mean cosine
# similarity between iter-k embedding and the GT anchor (image and
# text channels), with shaded ±SEM bands. The qualitative trajectory
# shape (rapid drop iter-0 → iter-3, plateau by iter-5) is identical
# across all four off-default settings, supporting the claim in §B.3
# that the information-loss and attractor findings are not artifacts
# of the sampler hyperparameters.

def _sweep_extract(summary, group_dict_key, group_value_label):
    """Extract per-iter mean ± SEM cosine arrays from a sweep summary.

    summary is the loaded JSON; group_dict_key is the per-group dict
    key ("per_cfg" or "per_steps"); group_value_label is the inner
    label that we want ("2.0", "7.0", "025", "050").

    Returns (iters, img_mean, img_sem, txt_mean, txt_sem, n).
    """
    grp = summary[group_dict_key][group_value_label]
    n = int(grp["n_trajectories"])
    iter_keys = sorted(
        [k for k in grp if k.startswith("iter_")],
        key=lambda k: int(k.split("_")[1]),
    )
    iters = np.array([int(k.split("_")[1]) for k in iter_keys])
    img_mu = np.array([grp[k]["image_cos_mean"] for k in iter_keys])
    img_sd = np.array([grp[k]["image_cos_std"]  for k in iter_keys])
    txt_mu = np.array([grp[k]["text_cos_mean"]  for k in iter_keys])
    txt_sd = np.array([grp[k]["text_cos_std"]   for k in iter_keys])
    img_se = img_sd / np.sqrt(max(n, 1))
    txt_se = txt_sd / np.sqrt(max(n, 1))
    return iters, img_mu, img_se, txt_mu, txt_se, n


def figS_sampler_sensitivity(cfg, out_dir):
    """Sampler-setting sensitivity: cosine-to-anchor across CFG and
    DiT-step sweeps. Two-panel layout, image + text channels.

    Data sources:
        cfg["sweep_cfg"]  : per_cfg with CFG ∈ {2.0, 7.0}, T fixed at 100
        cfg["sweep_step"] : per_steps with T ∈ {25, 50}, CFG fixed at 4

    Legend saved separately as figS_sampler_sensitivity_legend.png.
    """
    if not os.path.exists(cfg["sweep_cfg"]):
        raise RuntimeError(f"sweep_cfg not found at {cfg['sweep_cfg']}")
    if not os.path.exists(cfg["sweep_step"]):
        raise RuntimeError(f"sweep_step not found at {cfg['sweep_step']}")
    sc = _load_json(cfg["sweep_cfg"])
    ss = _load_json(cfg["sweep_step"])

    # Settings to plot (lighter → darker within each sweep)
    cfg_settings = [
        ("2.0", "$s{=}2$", "#9ecae1"),
        ("7.0", "$s{=}7$", "#08519c"),
    ]
    step_settings = [
        ("025", "$T{=}25$", "#fdae6b"),
        ("050", "$T{=}50$", "#a63603"),
    ]

    fig, axes = plt.subplots(
        1, 2, figsize=(PANEL_FULL_W, PANEL_DEFAULT_H + 0.4),
        sharey=True,
    )

    # ── Left: CFG sweep (T=100 fixed) ─────────────────────────────────────
    ax = axes[0]
    n_cfg = None
    for label_key, _label_disp, color in cfg_settings:
        iters, img_mu, img_se, txt_mu, txt_se, n = _sweep_extract(
            sc, "per_cfg", label_key
        )
        n_cfg = n
        # Image: solid line + filled SEM band
        ax.fill_between(iters, img_mu - img_se, img_mu + img_se,
                        color=color, alpha=0.20, lw=0)
        ax.plot(iters, img_mu, "-o", color=color, lw=1.5, ms=4, zorder=4)
        # Text: dashed line, same color, no band (to keep panel readable)
        ax.plot(iters, txt_mu, "--s", color=color, lw=1.2, ms=3.5,
                alpha=0.85, zorder=4)

    ax.axhline(0, color=C_ZERO, lw=0.6, ls="-", alpha=0.4, zorder=1)
    ax.set_xlabel("Iteration $k$")
    ax.set_ylabel("Cosine to anchor")
    ax.set_title(rf"CFG sweep ($T{{=}}100$, $n{{=}}{n_cfg}$ anchors)")
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.grid(True, alpha=0.3, lw=0.4)

    # ── Right: DiT-step sweep (CFG=4 fixed) ───────────────────────────────
    ax = axes[1]
    n_step = None
    for label_key, _label_disp, color in step_settings:
        iters, img_mu, img_se, txt_mu, txt_se, n = _sweep_extract(
            ss, "per_steps", label_key
        )
        n_step = n
        ax.fill_between(iters, img_mu - img_se, img_mu + img_se,
                        color=color, alpha=0.20, lw=0)
        ax.plot(iters, img_mu, "-o", color=color, lw=1.5, ms=4, zorder=4)
        ax.plot(iters, txt_mu, "--s", color=color, lw=1.2, ms=3.5,
                alpha=0.85, zorder=4)

    ax.axhline(0, color=C_ZERO, lw=0.6, ls="-", alpha=0.4, zorder=1)
    ax.set_xlabel("Iteration $k$")
    ax.set_title(rf"DiT-step sweep ($s{{=}}4$, $n{{=}}{n_step}$ anchors)")
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.grid(True, alpha=0.3, lw=0.4)

    fig.suptitle(
        "Sampler-setting sensitivity: cosine-to-anchor across CFG and "
        "denoising-step sweeps",
        fontsize=8.5, y=1.02,
    )
    fig.tight_layout()
    _save(fig, out_dir, "figS_sampler_sensitivity", cfg["pdf"])
    _save_sampler_sensitivity_legend(
        out_dir, cfg,
        cfg_colors=[c[2] for c in cfg_settings],
        cfg_labels=[c[1] for c in cfg_settings],
        step_colors=[c[2] for c in step_settings],
        step_labels=[c[1] for c in step_settings],
    )


def _save_sampler_sensitivity_legend(out_dir, cfg,
                                     cfg_colors, cfg_labels,
                                     step_colors, step_labels):
    """Standalone legend covering both panels of figS_sampler_sensitivity."""
    fig = plt.figure(figsize=(4.0, 2.6))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    y = 0.95
    ax.text(0.04, y, "Modality (line style)", fontweight="bold",
            fontsize=10.0, va="top")

    y -= 0.11
    ax.plot([0.06, 0.20], [y, y], color="#444", lw=1.6, marker="o", ms=4)
    ax.text(0.23, y, "Image cosine to anchor (with SEM band)",
            fontsize=8.5, va="center")

    y -= 0.11
    ax.plot([0.06, 0.20], [y, y], color="#444", lw=1.2, ls="--",
            marker="s", ms=3.5)
    ax.text(0.23, y, "Text cosine to anchor", fontsize=8.5, va="center")

    y -= 0.13
    ax.text(0.04, y, "CFG sweep (left panel)", fontweight="bold",
            fontsize=10.0, va="top")
    for color, label in zip(cfg_colors, cfg_labels):
        y -= 0.10
        ax.plot([0.06, 0.20], [y, y], color=color, lw=1.6)
        ax.text(0.23, y, rf"{label} (CFG scale)", fontsize=8.5, va="center")

    y -= 0.13
    ax.text(0.04, y, "DiT-step sweep (right panel)", fontweight="bold",
            fontsize=10.0, va="top")
    for color, label in zip(step_colors, step_labels):
        y -= 0.10
        ax.plot([0.06, 0.20], [y, y], color=color, lw=1.6)
        ax.text(0.23, y, rf"{label} (denoising steps)",
                fontsize=8.5, va="center")

    fig.savefig(out_dir / "figS_sampler_sensitivity_legend.png", dpi=600,
                bbox_inches="tight", pad_inches=0.15)
    if cfg["pdf"]:
        fig.savefig(out_dir / "figS_sampler_sensitivity_legend.pdf",
                    bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"  → figS_sampler_sensitivity_legend.png")


# ════════════════════════════════════════════════════════════════════════════
#  Panel registry + main
# ════════════════════════════════════════════════════════════════════════════

PANELS = {
    "figS1a":  figS1a_anchor_distance,
    "figS1b":  figS1b_step_size,
    "figS1c":  figS1c_modal_coupling,
    "figS2a":  figS2a_pathology_heatmap,
    "figS2b":  figS2b_pathology_curves,
    "figS3a":  figS3a_autocorr_extended,
    "figS3b":  figS3b_knn_jaccard_extended,
    "figS4":   figS4_anchor_lambda,
    "figS5":   figS5_cluster_dim_robust,
    "figS6":   figS6_basin_pathology_null,
    "figS7":   figS7_asymmetry_quant,
    "figS_oov_inflation":  figS_oov_inflation,
    "figS_mi_calibration": figS_mi_calibration,
    "figS_lyap_saturation": figS_lyap_saturation,
    "figS_perm_gap":       figS_perm_gap,
    "figS_jaccard_autocorr":    figS_jaccard_autocorr,
    "figS_sampler_sensitivity": figS_sampler_sensitivity,
    "figS_asym_umap_exemplar": figS_asym_umap_exemplar,
}


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--panels", default="all",
                   help='"all" or comma-separated panel keys')
    p.add_argument("--out_dir",       default=DEFAULTS["out_dir"])
    p.add_argument("--block_k",       default=DEFAULTS["block_k"])
    p.add_argument("--analysis_json", default=DEFAULTS["analysis_json"])
    p.add_argument("--long_horizon",  default=DEFAULTS["long_horizon"])
    p.add_argument("--geom_summary",  default=DEFAULTS["geom_summary"])
    p.add_argument("--mi_calibration", default=DEFAULTS["mi_calibration"],
                   help="Path to mi_calibration.json (for figS_mi_calibration)")
    p.add_argument("--lyap_synthetic", default=DEFAULTS["lyap_synthetic"],
                   help="Path to lyapunov_synthetic.json (for figS_lyap_saturation)")
    p.add_argument("--sweep_cfg", default=DEFAULTS["sweep_cfg"],
                   help="Path to chexgen_cfg_sweep/sweep_summary.json "
                        "(for figS_sampler_sensitivity)")
    p.add_argument("--sweep_step", default=DEFAULTS["sweep_step"],
                   help="Path to chexgen_step_sweep/sweep_summary.json "
                        "(for figS_sampler_sensitivity)")
    p.add_argument("--main_dir",      default=DEFAULTS["main_dir"],
                   help="HPC trajectory dir (only for figS_asym_umap_exemplar)")
    p.add_argument("--ref_dir",       default=DEFAULTS["ref_dir"],
                   help="HPC reference embeddings dir (only for figS_asym_umap_exemplar)")
    p.add_argument("--asym_K_load", type=int, default=10,
                   help="Iterations to load for the UMAP asymmetry panel (default 10)")
    p.add_argument("--pdf", action="store_true",
                   help="Also save PDF copies alongside PNGs")
    p.add_argument("--list", action="store_true", help="List panels and exit")
    args = p.parse_args()

    if args.list:
        for k in PANELS:
            print(f"  {k}")
        return

    apply_style()

    cfg = {
        "block_k":        args.block_k,
        "analysis_json":  args.analysis_json,
        "long_horizon":   args.long_horizon,
        "geom_summary":   args.geom_summary,
        "mi_calibration": args.mi_calibration,
        "lyap_synthetic": args.lyap_synthetic,
        "sweep_cfg":      args.sweep_cfg,
        "sweep_step":     args.sweep_step,
        "main_dir":       args.main_dir,
        "ref_dir":        args.ref_dir,
        "asym_K_load":    args.asym_K_load,
        "pdf":            args.pdf,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.panels == "all":
        keys = list(PANELS.keys())
    else:
        keys = [k.strip() for k in args.panels.split(",") if k.strip()]
        unknown = [k for k in keys if k not in PANELS]
        if unknown:
            print(f"ERROR: unknown panel(s): {unknown}")
            print(f"Available: {list(PANELS.keys())}")
            sys.exit(1)

    print(f"[all_supp_figs] Generating {len(keys)} panel(s) → {out_dir}")
    for k in keys:
        print(f"[{k}]")
        try:
            PANELS[k](cfg, out_dir)
        except Exception as e:
            print(f"  ✗ FAILED: {type(e).__name__}: {e}")
    print("[all_supp_figs] Done.")


if __name__ == "__main__":
    main()