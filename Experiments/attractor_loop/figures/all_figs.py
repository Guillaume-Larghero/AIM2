"""
all_figs.py — Unified figure generator for the AIM2 NeurIPS 2026 paper.

Each panel is an independent function that produces ONE PNG. Multi-panel
layouts are assembled manually.

Data flow
---------
All numeric values are pulled from the JSON outputs of attractor_analysis
and analysis_long_horizon. Nothing is hardcoded except path defaults.

Required JSON contents
----------------------
analysis_results.json:
    "B".lambda_sys_final              (scalar)
    "B".lyapunov_per_anchor           (dict: sid -> list of length K+1)
    "B".monotone_V_frac               (scalar)
    "B".basin_radius_p95              (scalar)
    "E".MI_img_per_k, "E".MI_txt_per_k (lists of length K+1)
    "n_trajectories"                   (scalar)

block_K_results.json:
    "n_studies", "probe_iters"
    "iter_<k>".hard.{entropy_bits, distinct_profiles, top_n_coverage, top_profiles}
    "iter_<k>".mode_purity, "iter_<k>".empty_breakdown

long_horizon_results.json:
    "A".image / "A".text / "<K>".tortuosity_mean
    "J".image / "J".text / autocorr_late_mean, lags, random_pair_autocorr,
        tau_mix_iters

Usage
-----
    python all_figs.py --panels all
    python all_figs.py --panels fig5a,fig6a
    python all_figs.py --panels all \\
        --block_k        ...block_K_results.json \\
        --analysis_json  ...analysis_results.json \\
        --long_horizon   ...long_horizon_results.json

Panel registry
--------------
Fig 1 (profile distributions)
    fig1a / fig1b / fig1c   GT / iter-10 / iter-100 distributions
Fig 2 (mode purity)
    fig2a / fig2b           K=10 / K=100 mode purity scatter
    fig2a_labeled /
    fig2b_labeled           Diagnostic versions with profile labels next to each
                            marker (used for internal review; not for camera-ready)
    fig2c                   top-4 coverage + GT match across K
Fig 3
    fig3                    MI(z_0; z_k) full curve
Fig 4 (OOV/COPD)
    fig4a                   empty-class composition stacked bar
    fig4b                   per-OOV-category cohort fraction across K
Fig 5 (Lyapunov)
    fig5a                   per-anchor λ_a(k) traces + cohort mean + λ_sys(k)
                            (combined; legend saved separately as fig5_legend.png)
    fig5c                   tortuosity vs K
Fig 6 (long-horizon dynamics, no Lyapunov-prediction overlays)
    fig6a                   H(K) two-regime entropy curve
    fig6b                   Block-J autocorrelation curve
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42  # TrueType for NeurIPS compliance
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import numpy as np

from style import (
    apply_style, new_panel, save_panel,
    PANEL_FULL_W, PANEL_HALF_W, PANEL_DEFAULT_H, PANEL_TALL_H, PANEL_SHORT_H,
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
#  Path defaults only — everything numeric comes from the JSONs
# ════════════════════════════════════════════════════════════════════════════

DEFAULTS = {
    "block_k":        "/n/groups/training/bmif203/AIM2/Experiments/attractor_loop/analysis_long/block_K_results.json",
    "analysis_json":  "/n/groups/training/bmif203/AIM2/Experiments/attractor_loop/analysis_long/analysis_results.json",
    "long_horizon":   "/n/groups/training/bmif203/AIM2/Experiments/attractor_loop/analysis_long/long_horizon_results.json",
    "out_dir":        "/n/groups/training/bmif203/AIM2/Experiments/attractor_loop/figures",
}


# ════════════════════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════════════════════

def _load_json(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Required data file missing: {p}")
    with open(p) as f:
        return json.load(f)


def _save(fig, out_dir, name, also_pdf=True):
    png = out_dir / f"{name}.png"
    fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.02)
    if also_pdf:
        fig.savefig(out_dir / f"{name}.pdf", format="pdf",
                    bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  → {png.name}")


def _bootstrap_mean_ci(values, n_boot=5000, ci=0.95, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    n = len(values)
    boot = np.array([float(np.mean(values[rng.integers(0, n, size=n)]))
                     for _ in range(n_boot)])
    return (float(np.mean(values)),
            float(np.percentile(boot, 100 * (1 - ci) / 2)),
            float(np.percentile(boot, 100 * (1 + ci) / 2)))


def _entropy_curve(block_k):
    ks = np.asarray(block_k["probe_iters"], dtype=int)
    H = np.array([block_k[f"iter_{k}"]["hard"]["entropy_bits"] for k in ks])
    distinct = np.array([block_k[f"iter_{k}"]["hard"]["distinct_profiles"] for k in ks])
    return ks, H, distinct


def _topn_coverage_and_match(block_k, n=4):
    ks = np.asarray(block_k["probe_iters"], dtype=int)
    cov, match = [], []
    n_total = block_k["n_studies"]
    for k in ks:
        mp = block_k[f"iter_{k}"]["mode_purity"]
        topn = mp[:n]
        sizes = np.array([m["size_K"] for m in topn], dtype=float)
        gtm = np.array([m["gt_match_rate"] for m in topn], dtype=float)
        cov.append(float(sizes.sum() / n_total))
        if sizes.sum() > 0:
            match.append(float((gtm * sizes).sum() / sizes.sum()))
        else:
            match.append(float("nan"))
    return ks, np.array(cov), np.array(match)


def _detect_k_contract(ks, H, floor_window=(50, 100), floor_tolerance=1.05):
    mask = (ks >= floor_window[0]) & (ks <= floor_window[1])
    if not mask.any():
        return int(ks[int(np.argmin(H))])
    H_floor = float(np.median(H[mask]))
    threshold = H_floor * floor_tolerance
    candidates = ks[H <= threshold]
    if len(candidates) == 0:
        return int(ks[int(np.argmin(H))])
    return int(candidates.min())


def _load_lyapunov(cfg):
    """Pull all Lyapunov data from analysis_results.json. Fail loudly if
    Block B is missing — no placeholder fallback.

    Returns a dict with:
        per_anchor              : dict sid -> np.array of length K+1
        traj_array              : np.array of shape (n_anchors, K+1)
        K_max                   : int
        n_anchors               : int
        lambda_sys_per_k        : np.array of length K+1, cross-anchor cohort
                                  Lyapunov trajectory (None if not in JSON)
        lambda_sys_final        : float, asymptotic cohort FTLE at K_max
        lambda_a_final_per_anchor : np.array length n_anchors
        lambda_a_final_mean / _lo / _hi : bootstrap mean and 95% CI
        monotone_frac, basin_radius_p95 : auxiliary scalars
    """
    aj = _load_json(cfg["analysis_json"])
    if "B" not in aj or "lyapunov_per_anchor" not in aj["B"]:
        raise RuntimeError(
            f"Block B not found in {cfg['analysis_json']}.\n"
            "  → re-run attractor_analysis.py with --blocks B."
        )
    B = aj["B"]
    per_anchor = {sid: np.asarray(v, dtype=float)
                  for sid, v in B["lyapunov_per_anchor"].items()}
    sids = sorted(per_anchor.keys())
    arrs = np.stack([per_anchor[s] for s in sids])  # (n_anchors, K+1)
    K_max = arrs.shape[1] - 1
    n_anchors = arrs.shape[0]

    # λ_sys trajectory — only present if attractor_analysis.py was run with
    # the patched serializer that saves lambda_sys_per_k.
    lambda_sys_per_k = None
    if "lambda_sys_per_k" in B:
        lambda_sys_per_k = np.asarray(B["lambda_sys_per_k"], dtype=float)

    lam_a_final = arrs[:, K_max]
    lam_final_mean, lam_final_lo, lam_final_hi = _bootstrap_mean_ci(lam_a_final)

    return {
        "sids":                       sids,
        "per_anchor":                 per_anchor,
        "traj_array":                 arrs,
        "K_max":                      K_max,
        "n_anchors":                  n_anchors,
        "lambda_sys_per_k":           lambda_sys_per_k,
        "lambda_sys_final":           float(B["lambda_sys_final"]),
        "lambda_a_final_per_anchor":  lam_a_final,
        "lambda_a_final_mean":        lam_final_mean,
        "lambda_a_final_lo":          lam_final_lo,
        "lambda_a_final_hi":          lam_final_hi,
        "monotone_frac":              float(B.get("monotone_V_frac", 0.0)),
        "basin_radius_p95":           float(B.get("basin_radius_p95", float("nan"))),
        "n_traj":                     int(aj.get("n_trajectories", -1)),
    }


# ════════════════════════════════════════════════════════════════════════════
#  Fig 1 — Profile distributions
# ════════════════════════════════════════════════════════════════════════════
 
EMPTY_LABEL = "OCV"
OCV_COLOR   = "#E8B83C"
HATCH_STYLE = "////"
HATCH_LW    = 0.6
EMPTY_KEYS  = {"", "{}"}   # raw labels treated as the empty CheXpert profile
 
 
def _profile_panel(ax, level_label, hard, default_color, *,
                   top_n=10, xmax=None,
                   future_top4_labels=None, mark_future=False):
    import matplotlib as mpl
    profiles = hard["top_profiles"][:top_n]
    raw_labels = [tp["label"] for tp in profiles]
    fracs = np.array([tp["fraction"] for tp in profiles])
    display_labels = [EMPTY_LABEL if l in EMPTY_KEYS else l for l in raw_labels]
    bar_colors = [OCV_COLOR if l in EMPTY_KEYS else default_color for l in raw_labels]
    other = max(0.0, 1.0 - fracs.sum())
    display_labels.append("other")
    fracs = np.append(fracs, other)
    bar_colors.append("#BBBBBB")
    raw_with_other = raw_labels + ["__other__"]
    y_pos = np.arange(len(fracs))[::-1]
    bars = ax.barh(y_pos, fracs, height=0.72, color=bar_colors, edgecolor="none")
 
    # Mark bars whose profile is in the K=100 top-4 with red striations.
    # Bump the hatch line width so striations are visible on small bars.
    if mark_future and future_top4_labels:
        future_set = set(future_top4_labels)
        old_hlw = mpl.rcParams.get("hatch.linewidth", 1.0)
        mpl.rcParams["hatch.linewidth"] = 1.2
        try:
            for bar, raw in zip(bars, raw_with_other):
                if raw in future_set:
                    bar.set_hatch(HATCH_STYLE)
                    bar.set_edgecolor(C_ITERK)
                    bar.set_linewidth(HATCH_LW)
        finally:
            mpl.rcParams["hatch.linewidth"] = old_hlw
    ax.set_yticks(y_pos)
    ax.set_yticklabels([shorten_label(l, 26) for l in display_labels])
    ax.tick_params(axis="y", which="both", length=0)
    if xmax is None:
        xmax = max(0.30, fracs.max() * 1.18)
    ax.set_xlim(0, xmax)
    ax.set_xlabel("Fraction of cohort")
    for bar, f in zip(bars, fracs):
        if f > 0.001:
            ax.text(bar.get_width() + xmax * 0.012,
                    bar.get_y() + bar.get_height() / 2,
                    f"{100*f:.1f}%", va="center", ha="left", fontsize=7)
    ax.set_title(
        f"{level_label}\n"
        f"$N_{{\\rm distinct}}{{=}}{hard['distinct_profiles']}$, "
        f"$H{{=}}{hard['entropy_bits']:.2f}$ bits, "
        f"top-1$\\,$=$\\,${100*hard['top_n_coverage']['top1']:.1f}%",
        fontsize=9,
    )
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", alpha=0.25, linewidth=0.4)
 
 
def _fig1_one(level_key, level_title, color, cfg, out_dir, fname):
    bk = _load_json(cfg["block_k"])
    future_top4 = [tp["label"] for tp in bk["iter_100"]["hard"]["top_profiles"][:4]]
    fig, ax = new_panel(width=PANEL_HALF_W + 0.6, height=PANEL_DEFAULT_H + 0.4)
    _profile_panel(ax, level_title, bk[level_key]["hard"], color,
                   future_top4_labels=future_top4,
                   mark_future=(level_key != "iter_100"))
    fig.tight_layout()
    _save(fig, out_dir, fname, cfg["pdf"])
 
 
def fig1a_profile_gt(cfg, out_dir):
    _fig1_one("gt", "Ground truth", C_GT, cfg, out_dir, "fig1a_profile_gt")
 
def fig1b_profile_iter1(cfg, out_dir):
    _fig1_one("iter_1", "Iteration 1", C_ITER0, cfg, out_dir, "fig1b_profile_iter1")
 
def fig1c_profile_iter10(cfg, out_dir):
    _fig1_one("iter_10", "Iteration 10", C_ITER0, cfg, out_dir, "fig1c_profile_iter10")
 
def fig1d_profile_iter100(cfg, out_dir):
    _fig1_one("iter_100", "Iteration 100", C_ITERK, cfg, out_dir, "fig1d_profile_iter100")



# ════════════════════════════════════════════════════════════════════════════
#  Fig 2 — Mode purity
# ════════════════════════════════════════════════════════════════════════════

CLASS_STYLE = {
    "elevator-music (inflated, anchor-independent)":
        dict(label="Attractor",  color=C_ELEVATOR,  marker="o"),
    "amplified-but-faithful":
        dict(label="Amplified",  color=C_AMPLIFIED, marker="s"),
    "preserved (faithful to GT)":
        dict(label="Preserved",  color=C_PRESERVED, marker="D"),
    "novel_mode (no GT prevalence; entirely loop-induced)":
        dict(label="Novel",      color=C_NOVEL,     marker="^"),
}
CLASS_ORDER = list(CLASS_STYLE.keys())

LIFT_FLOOR = 0.10   # log-axis floor for modes with match_rate = 0
NOVEL_Y    = 0.04   # y-position for novel-mode band


def _mode_purity_panel(ax, mode_purity, n_total, probe_k, add_labels=False):
    main  = [e for e in mode_purity if e.get("prev_GT", 0) > 0]
    novel = [e for e in mode_purity if e.get("prev_GT", 0) == 0]

    by_class = {c: [] for c in CLASS_ORDER}
    for e in main:
        if e["classification"] in by_class:
            by_class[e["classification"]].append(e)

    max_size = max((e["size_K"] for e in mode_purity), default=10)
    rng = np.random.default_rng(42)

    # Track plotted positions and labels for optional annotation pass.
    label_records = []  # list of (x, y, label_str)

    # Main scatter — log-jitter the lift=0 (floor-clipped) points so that
    # different categories at the floor don't stack perfectly on top of each other.
    for cls in CLASS_ORDER:
        es = by_class[cls]
        if not es:
            continue
        sizes = np.array([e["size_K"] for e in es], dtype=float)
        raw_lifts = np.array([e["lift"] for e in es], dtype=float)
        is_clipped = raw_lifts <= 0
        plotted = np.where(
            is_clipped,
            LIFT_FLOOR * np.exp(rng.uniform(-0.20, 0.20, size=len(raw_lifts))),
            np.maximum(raw_lifts, LIFT_FLOOR),
        )
        s_marker = 30 + 1.2 * sizes
        sty = CLASS_STYLE[cls]
        ax.scatter(sizes, plotted, s=s_marker, c=sty["color"], marker=sty["marker"],
                   alpha=0.78, edgecolor="white", linewidth=0.7, zorder=4)
        if add_labels:
            for x_pt, y_pt, e in zip(sizes, plotted, es):
                label_str = e.get("label", "?")
                # Truncate very long composite labels to keep the diagnostic plot legible.
                if len(label_str) > 28:
                    label_str = label_str[:25] + "..."
                label_records.append((float(x_pt), float(y_pt), label_str))

    if novel:
        sizes_n = np.array([e["size_K"] for e in novel], dtype=float)
        s_n = 30 + 1.2 * sizes_n
        y_n = NOVEL_Y * np.exp(rng.uniform(-0.20, 0.20, size=len(sizes_n)))
        sty = CLASS_STYLE["novel_mode (no GT prevalence; entirely loop-induced)"]
        ax.scatter(sizes_n, y_n, s=s_n, c=sty["color"], marker=sty["marker"],
                   alpha=0.78, edgecolor="white", linewidth=0.7, zorder=4)
        ax.axhspan(NOVEL_Y * 0.5, NOVEL_Y * 2.0, color="#888", alpha=0.05, zorder=1)
        if add_labels:
            for x_pt, y_pt, e in zip(sizes_n, y_n, novel):
                label_str = e.get("label", "?")
                if len(label_str) > 28:
                    label_str = label_str[:25] + "..."
                label_records.append((float(x_pt), float(y_pt), label_str))

    # Chance line — geometric only, no text
    ax.axhline(1.0, color="#444", ls="--", lw=0.8, alpha=0.6, zorder=2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.7, max_size * 1.6)
    all_lifts = [max(e.get("lift", LIFT_FLOOR), LIFT_FLOOR) for e in main]
    ymax = max(all_lifts + [10]) * 2.5
    ax.set_ylim(NOVEL_Y * 0.3, ymax)
    ax.set_xlabel(f"Number of patients in mode at iter-{probe_k}  ($N{{=}}{n_total}$, log scale)")
    ax.set_ylabel(r"Enrichment $= \mathrm{match}/\mathrm{prev}_{\rm GT}$  (log)")
    ax.set_title(f"K={probe_k}: large attractors recruit patients at chance — patient-independent")

    # Optional annotation pass — drawn last so labels sit above all markers.
    if add_labels:
        for x_pt, y_pt, label_str in label_records:
            ax.annotate(
                label_str,
                xy=(x_pt, y_pt),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=6,
                color="#222",
                alpha=0.95,
                zorder=6,
            )


def _save_mode_purity_legend(out_dir, cfg):
    """Combined legend for fig2a/fig2b — class markers, mode-size scale,
    and reference markers (chance line, novel-modes band)."""
    fig = plt.figure(figsize=(3.4, 3.8))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Section 1 — Classes
    y = 0.97
    ax.text(0.04, y, "Class", fontweight="bold", fontsize=10.5, va="top")
    for cls, sty in CLASS_STYLE.items():
        y -= 0.07
        ax.scatter(0.12, y, s=85, c=sty["color"], marker=sty["marker"],
                   alpha=0.78, edgecolor="white", linewidth=0.7)
        ax.text(0.22, y, sty["label"], fontsize=9.5, va="center")

    # Section 2 — Mode size scale
    y -= 0.10
    ax.text(0.04, y, "Mode size  (marker area)", fontweight="bold",
            fontsize=10.5, va="top")
    y -= 0.10
    for i, n_mode in enumerate([5, 30, 100]):
        s_marker = 30 + 1.2 * n_mode
        x = 0.14 + 0.22 * i
        ax.scatter(x, y, s=s_marker, c="#888", marker="o",
                   alpha=0.5, edgecolor="white", linewidth=0.7)
        ax.text(x, y - 0.07, f"n = {n_mode}", fontsize=9, ha="center", va="top")

    # Section 3 — Reference markers
    y -= 0.18
    ax.text(0.04, y, "Reference", fontweight="bold", fontsize=10.5, va="top")
    y -= 0.07
    ax.plot([0.08, 0.20], [y, y], color="#444", ls="--", lw=0.9)
    ax.text(0.22, y, "chance level  (lift = 1)", fontsize=9.5, va="center")
    y -= 0.07
    ax.add_patch(plt.Rectangle((0.08, y - 0.020), 0.12, 0.040,
                               color="#888", alpha=0.20))
    ax.text(0.22, y, r"novel modes  (prev$_{\rm GT}\!=\!0$)",
            fontsize=9.5, va="center")

    fig.savefig(out_dir / "fig2_legend.png", dpi=600,
                bbox_inches="tight", pad_inches=0.15)
    if cfg["pdf"]:
        fig.savefig(out_dir / "fig2_legend.pdf",
                    bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"  → fig2_legend.png")


def fig2a_mode_purity_k10(cfg, out_dir):
    bk = _load_json(cfg["block_k"])
    fig, ax = new_panel(width=PANEL_FULL_W, height=PANEL_DEFAULT_H + 0.7)
    _mode_purity_panel(ax, bk["iter_10"]["mode_purity"], bk["n_studies"], probe_k=10)
    fig.tight_layout()
    _save(fig, out_dir, "fig2a_mode_purity_k10", cfg["pdf"])
    _save_mode_purity_legend(out_dir, cfg)


def fig2b_mode_purity_k100(cfg, out_dir):
    bk = _load_json(cfg["block_k"])
    fig, ax = new_panel(width=PANEL_FULL_W, height=PANEL_DEFAULT_H + 0.7)
    _mode_purity_panel(ax, bk["iter_100"]["mode_purity"], bk["n_studies"], probe_k=100)
    fig.tight_layout()
    _save(fig, out_dir, "fig2b_mode_purity_k100", cfg["pdf"])


def fig2a_mode_purity_k10_labeled(cfg, out_dir):
    """Diagnostic variant of fig2a with a CheXpert-profile label next to each
    marker. Used to identify which dot corresponds to which mode when reviewing
    or writing about the figure. Not intended for the camera-ready paper."""
    bk = _load_json(cfg["block_k"])
    fig, ax = new_panel(width=PANEL_FULL_W * 1.4, height=PANEL_DEFAULT_H + 1.6)
    _mode_purity_panel(ax, bk["iter_10"]["mode_purity"], bk["n_studies"],
                       probe_k=10, add_labels=True)
    fig.tight_layout()
    _save(fig, out_dir, "fig2a_mode_purity_k10_labeled", cfg["pdf"])


def fig2b_mode_purity_k100_labeled(cfg, out_dir):
    """Diagnostic variant of fig2b with a CheXpert-profile label next to each
    marker. Used to identify which dot corresponds to which mode when reviewing
    or writing about the figure. Not intended for the camera-ready paper."""
    bk = _load_json(cfg["block_k"])
    fig, ax = new_panel(width=PANEL_FULL_W * 1.4, height=PANEL_DEFAULT_H + 1.6)
    _mode_purity_panel(ax, bk["iter_100"]["mode_purity"], bk["n_studies"],
                       probe_k=100, add_labels=True)
    fig.tight_layout()
    _save(fig, out_dir, "fig2b_mode_purity_k100_labeled", cfg["pdf"])


def fig2c_failure_tradeoff(cfg, out_dir):
    bk = _load_json(cfg["block_k"])
    ks, cov, match = _topn_coverage_and_match(bk, n=4)

    fig, ax = new_panel(width=PANEL_FULL_W, height=PANEL_DEFAULT_H + 0.4)
    ax2 = ax.twinx()
    l1 = ax.plot(ks, cov * 100, marker="o", ms=5, lw=1.5, color=C_ELEVATOR,
                 label="Top-4 mode coverage")
    l2 = ax2.plot(ks, match * 100, marker="s", ms=5, lw=1.5, color=C_PRESERVED,
                  label="Mean within-mode GT match", linestyle="--")
    ax.set_xlabel("Iteration $K$")
    ax.set_ylabel("Top-4 mode coverage (% of cohort)", color=C_ELEVATOR)
    ax2.set_ylabel("Mean within-mode GT match (%)", color=C_PRESERVED)
    ax.tick_params(axis="y", labelcolor=C_ELEVATOR)
    ax2.tick_params(axis="y", labelcolor=C_PRESERVED)
    ax.set_ylim(0, 100)
    ax2.set_ylim(0, max(8, float(np.nanmax(match) * 100) * 1.4))
    ax.grid(axis="x", visible=False)
    lines = l1 + l2
    ax.legend(lines, [l.get_label() for l in lines],
              loc="center right", fontsize=8, framealpha=0.95)
    ax.set_title("Long horizon makes the loop more, not less, patient-independent")
    fig.tight_layout()
    _save(fig, out_dir, "fig2c_failure_tradeoff", cfg["pdf"])


# ════════════════════════════════════════════════════════════════════════════
#  Fig 3 — MI cliff
# ════════════════════════════════════════════════════════════════════════════

# Modality colors
C_MI_IMG = "#21295C"   # accent  — soft paper blue (image)
C_MI_TXT = "#C73E1D"   # warning — soft red       (text)


def _mi_modality_panel(out_dir, ks, mi_vals, cfg, *,
                       color, modality_label, fname,
                       k_left_max=10, k_right_min=90):
    """Single panel with broken x-axis: shows k ∈ [0, k_left_max] then a
    visual gap, then k ∈ [k_right_min, K_max]. The flat middle is hidden."""
    K = int(ks.max())
    fig = plt.figure(figsize=(PANEL_FULL_W, PANEL_DEFAULT_H + 0.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1.0], wspace=0.06)
    ax_l = fig.add_subplot(gs[0])
    ax_r = fig.add_subplot(gs[1], sharey=ax_l)

    m_l = ks <= k_left_max
    m_r = ks >= k_right_min
    ax_l.plot(ks[m_l], mi_vals[m_l], marker="o", ms=4, lw=1.6, color=color)
    ax_r.plot(ks[m_r], mi_vals[m_r], marker="o", ms=4, lw=1.6, color=color)
    for ax in (ax_l, ax_r):
        ax.axhline(0, color="k", lw=0.5, alpha=0.4)

    drop = (mi_vals[0] - mi_vals[1]) / mi_vals[0] * 100
    ax_l.annotate(
        f"$-{drop:.0f}\\%$\nin one iteration",
        xy=(0.5, (mi_vals[0] + mi_vals[1]) / 2),
        xytext=(2.3, mi_vals[0] * 0.55),
        fontsize=8.5, color=C_ANNOT, ha="left",
        arrowprops=dict(arrowstyle="->", color=C_ANNOT, lw=0.7,
                        connectionstyle="arc3,rad=0.2"),
    )

    ax_l.set_xlim(-0.4, k_left_max + 0.4)
    ax_r.set_xlim(k_right_min - 0.4, K + 0.4)
    ax_l.spines["right"].set_visible(False)
    ax_r.spines["left"].set_visible(False)
    ax_r.tick_params(axis="y", which="both", left=False, labelleft=False)

    d = 0.012
    kw = dict(transform=ax_l.transAxes, color="k", clip_on=False, lw=0.8)
    ax_l.plot([1 - d, 1 + d], [-d, +d], **kw)
    ax_l.plot([1 - d, 1 + d], [1 - d, 1 + d], **kw)
    kw = dict(transform=ax_r.transAxes, color="k", clip_on=False, lw=0.8)
    ax_r.plot([-d, +d], [-d, +d], **kw)
    ax_r.plot([-d, +d], [1 - d, 1 + d], **kw)

    ax_l.set_xticks(range(0, k_left_max + 1, 2))
    ax_r.set_xticks(list(range(k_right_min, K + 1, 2)))
    ax_l.set_ylabel(rf"$I(z_0;z_k)$ [nats]")
    ymax = float(mi_vals.max()) * 1.15
    ax_l.set_ylim(-0.15, ymax)

    fig.text(0.5, -0.02, "Iteration $k$", ha="center", va="top", fontsize=9)
    fig.suptitle(f"Patient information collapse — {modality_label} modality",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, fname, cfg["pdf"])


def fig3_mi_cliff(cfg, out_dir):
    aj = _load_json(cfg["analysis_json"])
    if "E" not in aj or "MI_img_per_k" not in aj["E"]:
        raise RuntimeError(
            "Block E (MI per k) not found in analysis_results.json.\n"
            "  → re-run attractor_analysis.py with --blocks E."
        )
    mi_img = np.asarray(aj["E"]["MI_img_per_k"], dtype=float)
    mi_txt = np.asarray(aj["E"]["MI_txt_per_k"], dtype=float)
    ks = np.arange(len(mi_img))

    _mi_modality_panel(out_dir, ks, mi_img, cfg,
                       color=C_MI_IMG, modality_label="image",
                       fname="fig3a_mi_cliff_image")
    _mi_modality_panel(out_dir, ks, mi_txt, cfg,
                       color=C_MI_TXT, modality_label="text",
                       fname="fig3b_mi_cliff_text")


# ════════════════════════════════════════════════════════════════════════════
#  Fig 4 — OOV / COPD attractor
# ════════════════════════════════════════════════════════════════════════════

EMPTY_CATS = [
    ("COPD/emphysema",                  "COPD / emphysema",            C_OOV_DOMINANT),
    ("pulmonary fibrosis/ILD",          "Pulmonary fibrosis / ILD",    C_OOV_FIBROSIS),
    ("scoliosis/kyphosis",              "Scoliosis / kyphosis",        C_OOV_SCOLIOSIS),
    ("__OOV_OTHER__",                   "Other OOV findings",          C_OOV_OTHER),
    ("__VOCAB_MISS__",                  "CheXpert-vocab miss",         "#A48EB6"),
    ("explicit-normal (CheXpert miss)", "Explicit-normal miss",        C_NORMAL),
    ("other-unclassified",              "Residual unclassified",       C_RESIDUAL),
]
OOV_OTHER_MEMBERS = [
    "aortic elongation/tortuosity", "metastatic disease/multiple nodules",
    "bullae", "hiatal hernia", "pleural thickening/plaques",
    "calcified granuloma (old)", "volume loss",
]
VOCAB_MISS_MEMBERS = [
    "mass/nodule (likely Lung Lesion miss)",
    "sternotomy/post-surgical (likely SD miss)",
]


def _empty_counts(eb):
    raw = eb["primary_category_counts"]
    out = {
        "COPD/emphysema":                  raw.get("COPD/emphysema", 0),
        "pulmonary fibrosis/ILD":          raw.get("pulmonary fibrosis/ILD", 0),
        "scoliosis/kyphosis":              raw.get("scoliosis/kyphosis", 0),
        "__OOV_OTHER__":   sum(raw.get(k, 0) for k in OOV_OTHER_MEMBERS),
        "__VOCAB_MISS__":  sum(raw.get(k, 0) for k in VOCAB_MISS_MEMBERS),
        "explicit-normal (CheXpert miss)": raw.get("explicit-normal (CheXpert miss)", 0),
        "other-unclassified":              raw.get("other-unclassified", 0),
    }
    accounted = {"COPD/emphysema", "pulmonary fibrosis/ILD", "scoliosis/kyphosis",
                 "explicit-normal (CheXpert miss)", "other-unclassified"} \
                | set(OOV_OTHER_MEMBERS) | set(VOCAB_MISS_MEMBERS)
    leftover = set(raw.keys()) - accounted
    out["other-unclassified"] += sum(raw[k] for k in leftover)
    return out


def fig4a_empty_stacked(cfg, out_dir):
    bk = _load_json(cfg["block_k"])
    iters_wanted = [0, 1, 5, 10, 30, 50, 70, 100]
    iters = [k for k in bk["probe_iters"] if k in iters_wanted]
    n_total = bk["n_studies"]

    cat_keys, cat_labels, cat_colors = zip(*EMPTY_CATS)
    counts = np.zeros((len(iters), len(cat_keys)), dtype=int)
    for i, k in enumerate(iters):
        ext = _empty_counts(bk[f"iter_{k}"]["empty_breakdown"])
        for j, ck in enumerate(cat_keys):
            counts[i, j] = ext[ck]
    fracs = counts / float(n_total)

    fig, ax = new_panel(width=PANEL_FULL_W, height=PANEL_TALL_H + 0.3)
    x = np.arange(len(iters))
    bottom = np.zeros(len(iters))
    for j, (lbl, col) in enumerate(zip(cat_labels, cat_colors)):
        ax.bar(x, fracs[:, j], bottom=bottom, width=0.62, color=col,
               edgecolor="white", linewidth=0.5, label=lbl)
        bottom += fracs[:, j]

    for xi, top in zip(x, bottom):
        ax.text(xi, top + 0.004, f"{100*top:.1f}%",
                ha="center", va="bottom", fontsize=8, color="#444")

    ax.set_xticks(x)
    ax.set_xticklabels([f"$k{{=}}{i}$" for i in iters])
    ax.set_xlabel("Iteration $k$")
    ax.set_ylabel(f"Fraction of cohort ($N{{=}}{n_total}$)")
    ax.set_title("Composition of the empty CheXpert-profile class")
    ax.set_ylim(0, bottom.max() * 1.18)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100*v:.0f}%"))
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5),
              fontsize=7.5, frameon=False, handlelength=1.0,
              handletextpad=0.5, borderaxespad=0.5)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, out_dir, "fig4a_empty_stacked", cfg["pdf"])


def fig4b_oov_trajectories(cfg, out_dir):
    bk = _load_json(cfg["block_k"])
    iters = bk["probe_iters"]
    n_total = bk["n_studies"]
    cat_keys, cat_labels, cat_colors = zip(*EMPTY_CATS)
    counts = np.zeros((len(iters), len(cat_keys)), dtype=int)
    for i, k in enumerate(iters):
        ext = _empty_counts(bk[f"iter_{k}"]["empty_breakdown"])
        for j, ck in enumerate(cat_keys):
            counts[i, j] = ext[ck]
    fracs = counts / float(n_total)

    fig, ax = new_panel(width=PANEL_FULL_W, height=PANEL_DEFAULT_H + 0.5)
    for j, (lbl, col) in enumerate(zip(cat_labels, cat_colors)):
        lw = 1.7 if cat_keys[j] == "COPD/emphysema" else 1.0
        ms = 5 if cat_keys[j] == "COPD/emphysema" else 3.5
        ax.plot(iters, fracs[:, j], marker="o", ms=ms, lw=lw, color=col, label=lbl)
    ax.set_xlabel("Iteration $k$")
    ax.set_ylabel("Fraction of cohort")
    ax.set_title("Per-OOV-category cohort fraction across $K{=}100$")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100*v:.0f}%"))
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5),
              fontsize=7.5, frameon=False, handlelength=1.5,
              handletextpad=0.5, borderaxespad=0.5)
    fig.tight_layout()
    _save(fig, out_dir, "fig4b_oov_trajectories", cfg["pdf"])


# ════════════════════════════════════════════════════════════════════════════
#  Fig 5 — Two-timescale Lyapunov
#
#  Headline scalar: saturated log-dispersion C = mean over k in [lo, hi] of
#  k * λ_a(k). Physically: log of asymptotic seed-spread ratio. C > 0 ⇔
#  bounded attractor with finite local volume. k-independent (averaged).
# ════════════════════════════════════════════════════════════════════════════

def fig5a_lyapunov_combined(cfg, out_dir):
    """Combined two-timescale Lyapunov panel:
      - 20 per-anchor λ_a(k) traces (light red, transparent)
      - Cohort mean λ_a(k) (bold red)  — within-anchor (same anchor, diff seeds)
      - λ_sys(k) (bold blue)           — cross-anchor pairwise divergence
    All on the same axes. y-axis is symmetric-log (linthresh=0.005) so the
    near-zero λ_sys trajectory is visible alongside the larger per-anchor
    transient. No legend in the panel; saved separately as fig5_legend.png.
    """
    L = cfg["lyapunov"]
    arrs = L["traj_array"]
    K = L["K_max"]
    ks = np.arange(K + 1)
    mean_a = arrs.mean(axis=0)
    sys_t = L["lambda_sys_per_k"]

    fig, ax = new_panel(width=PANEL_FULL_W, height=PANEL_DEFAULT_H + 0.7)

    for row in arrs:
        ax.plot(ks[1:], row[1:], color=C_LANC_LIGHT, lw=0.7, alpha=0.55, zorder=2)

    ax.plot(ks[1:], mean_a[1:], color=C_LANC, lw=2.0,
            marker="o", ms=2.8, zorder=4)

    if sys_t is not None:
        ax.plot(ks[1:], sys_t[1:], color=C_LSYS, lw=2.0,
                marker="s", ms=2.8, zorder=4)
    else:
        print("  WARNING: lambda_sys_per_k missing — only asymptotic point shown.")
        ax.scatter([K], [L["lambda_sys_final"]], color=C_LSYS, s=45,
                   marker="s", zorder=4)

    ax.axhline(0, color=C_ZERO, lw=0.7, ls="--", alpha=0.6)

    from matplotlib.ticker import FuncFormatter, FixedLocator
    ax.set_yscale("symlog", linthresh=0.01, linscale=0.6)
    arr_max = float(np.nanmax(arrs[:, 1:]))
    sys_min = float(np.nanmin(sys_t[1:])) if sys_t is not None else L["lambda_sys_final"]
    ax.set_ylim(min(sys_min * 1.6, -0.005), arr_max * 1.15)
    # Major ticks at 0.5, 0.1, 0.01, 0, and the lambda_sys final value
    yticks = [0.5, 0.1, 0.01, 0.0]
    if sys_min < -0.005:
        yticks.append(round(sys_min, 4))
    ax.yaxis.set_major_locator(FixedLocator(sorted(set(yticks))))
    ax.yaxis.set_minor_locator(FixedLocator([]))   # suppress minor-tick clutter
    def _fmt_dec(v, _):
        if abs(v) < 1e-9:  return "0"
        if abs(v) < 0.01:  return f"{v:.4f}".rstrip("0").rstrip(".")
        if abs(v) < 1:     return f"{v:.2f}".rstrip("0").rstrip(".")
        return f"{v:.1f}"
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_dec))

    ax.set_xlabel("Iteration $k$")
    ax.set_ylabel(r"Finite-time Lyapunov exponent  (symlog)")
    ax.set_title("Two-timescale Lyapunov: within-anchor divergence above, cohort contraction below")
    ax.set_xlim(0, K)
    fig.tight_layout()
    _save(fig, out_dir, "fig5a_lyapunov_combined", cfg["pdf"])
    _save_lyapunov_legend(out_dir, cfg)


def _save_lyapunov_legend(out_dir, cfg):
    """Standalone legend + numerical summary for fig5a."""
    L = cfg["lyapunov"]
    K = L["K_max"]
    n_anchors = L["n_anchors"]
    n_pos = int(np.sum(L["lambda_a_final_per_anchor"] > 0))

    fig = plt.figure(figsize=(3.8, 3.4))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    y = 0.97
    ax.text(0.04, y, "Curves", fontweight="bold", fontsize=10.5, va="top")

    y -= 0.09
    ax.plot([0.06, 0.20], [y, y], color=C_LANC_LIGHT, lw=0.9, alpha=0.7)
    ax.text(0.23, y, f"Per-anchor $\\bar\\lambda_a(k)$  ({n_anchors} traces)",
            fontsize=9.5, va="center")

    y -= 0.09
    ax.plot([0.06, 0.20], [y, y], color=C_LANC, lw=2.2)
    ax.scatter([0.13], [y], color=C_LANC, s=22, marker="o", zorder=3)
    ax.text(0.23, y, r"Cohort mean $\langle\bar\lambda_a\rangle(k)$  (within-anchor)",
            fontsize=9.5, va="center")

    y -= 0.09
    if L["lambda_sys_per_k"] is not None:
        ax.plot([0.06, 0.20], [y, y], color=C_LSYS, lw=2.2)
        ax.scatter([0.13], [y], color=C_LSYS, s=22, marker="s", zorder=3)
    else:
        ax.scatter([0.13], [y], color=C_LSYS, s=40, marker="s")
    ax.text(0.23, y, r"$\lambda_{\rm sys}(k)$  (cross-anchor pairwise)",
            fontsize=9.5, va="center")

    y -= 0.09
    ax.plot([0.06, 0.20], [y, y], color=C_ZERO, lw=0.8, ls="--")
    ax.text(0.23, y, "zero", fontsize=9, va="center", color="#666")

    y -= 0.13
    ax.text(0.04, y, f"At $K{{=}}{K}$", fontweight="bold", fontsize=10.5, va="top")
    y -= 0.08
    ax.text(0.06, y,
            rf"$\langle\bar\lambda_a\rangle = {L['lambda_a_final_mean']:+.4f}$",
            fontsize=9.5, va="center")
    y -= 0.07
    ax.text(0.06, y,
            rf"95% CI $[{L['lambda_a_final_lo']:+.4f},\,{L['lambda_a_final_hi']:+.4f}]$",
            fontsize=8.5, va="center", color="#555")
    y -= 0.07
    ax.text(0.06, y,
            f"{n_pos}/{n_anchors} anchors with $\\bar\\lambda_a > 0$",
            fontsize=8.5, va="center", color="#555")
    y -= 0.09
    ax.text(0.06, y,
            rf"$\lambda_{{\rm sys}} = {L['lambda_sys_final']:+.4f}$",
            fontsize=9.5, va="center")
    y -= 0.07
    ax.text(0.06, y,
            rf"$V(z)$ monotone-decreasing fraction: ${L['monotone_frac']:.2f}$",
            fontsize=8.5, va="center", color="#555")

    fig.savefig(out_dir / "fig5_legend.png", dpi=600,
                bbox_inches="tight", pad_inches=0.15)
    if cfg["pdf"]:
        fig.savefig(out_dir / "fig5_legend.pdf",
                    bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"  → fig5_legend.png")


def fig5c_tortuosity(cfg, out_dir):
    """Path tortuosity T = path length / endpoint distance, vs probe K.
    Plotted as median ± IQR (P25–P75 band) for both modalities. Median is
    robust to anchor-level L/D blow-ups that crush the mean (when a
    trajectory accidentally lands near its origin, individual anchor T
    can spike by orders of magnitude — see tortuosity_n_outliers_5x).

    Linear growth in K = bounded random-walk fingerprint.
    """
    lh = _load_json(cfg["long_horizon"])
    if "A" not in lh:
        raise RuntimeError("Block A not in long_horizon_results.json.")
    img = lh["A"]["image"]; txt = lh["A"]["text"]
    ks = np.array(sorted(int(k) for k in img.keys()))

    sample = img[str(ks[0])]
    if "tortuosity_median" not in sample:
        raise RuntimeError(
            "tortuosity_median missing from long_horizon_results.json.\n"
            "  → re-run analysis_long_horizon.py with the patched Block A "
            "(adds median, p25, p75, p10, p90)."
        )

    def _pull(d, key):
        return np.array([d[str(k)][key] for k in ks])

    T_img_med, T_img_p25, T_img_p75 = (_pull(img, k) for k in
                                       ["tortuosity_median", "tortuosity_p25", "tortuosity_p75"])
    T_txt_med, T_txt_p25, T_txt_p75 = (_pull(txt, k) for k in
                                       ["tortuosity_median", "tortuosity_p25", "tortuosity_p75"])

    fig, ax = new_panel(width=PANEL_FULL_W, height=PANEL_DEFAULT_H + 0.4)

    ax.fill_between(ks, T_img_p25, T_img_p75, color=C_IMG, alpha=0.15, lw=0,
                    label="Image IQR (P25–P75)")
    ax.fill_between(ks, T_txt_p25, T_txt_p75, color=C_TXT, alpha=0.15, lw=0,
                    label="Text IQR (P25–P75)")
    ax.plot(ks, T_img_med, marker="o", ms=4, lw=1.6, color=C_IMG, label="Image median")
    ax.plot(ks, T_txt_med, marker="s", ms=4, lw=1.6, color=C_TXT, label="Text median")

    ax.axhline(1, color="#666", lw=0.7, ls="--", alpha=0.6)


    ymax = float(max(T_img_p75.max(), T_txt_p75.max())) * 1.18
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Iteration $K$ (probe endpoint)")
    ax.set_ylabel(r"Tortuosity $T = $ path length $/$ endpoint distance")
    ax.set_title("Tortuosity grows linearly with horizon: bounded random-walk fingerprint")
    ax.set_xticks([k for k in [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100] if k <= ks.max()])
    ax.legend(loc="lower right", fontsize=7.5, ncol=2, columnspacing=1.0)
    ax.grid(False)
    fig.tight_layout()
    _save(fig, out_dir, "fig5c_tortuosity", cfg["pdf"])


# ════════════════════════════════════════════════════════════════════════════
#  Fig 6 — Long-horizon dynamics  (no Lyapunov-prediction overlays)
# ════════════════════════════════════════════════════════════════════════════

def fig6a_entropy_kmin(cfg, out_dir):
    """H(K) two-regime entropy: fast contraction phase, then stochastic
    mixing floor. Panel only — legend (regime shading + reference markers)
    saved separately as fig6a_legend.png.
    """
    bk = _load_json(cfg["block_k"])
    ks, H, distinct = _entropy_curve(bk)
    N = bk["n_studies"]

    K_contract = _detect_k_contract(ks, H, floor_window=(50, 100), floor_tolerance=1.05)
    mask_floor = (ks >= 50) & (ks <= ks.max())
    H_floor_med = float(np.median(H[mask_floor])) if mask_floor.any() else float(H.min())
    H_floor_lo  = float(H[mask_floor].min())     if mask_floor.any() else float(H.min())
    H_floor_hi  = float(H[mask_floor].max())     if mask_floor.any() else float(H.min())

    fig, ax = new_panel(width=PANEL_FULL_W, height=PANEL_DEFAULT_H + 0.7)
    H_lo = float(min(H.min() - 0.3, H_floor_lo - 0.2))
    H_hi = float(H[0] + 0.3)

    ax.axvspan(0, K_contract, color=C_LSYS, alpha=0.07, zorder=0)
    ax.axvspan(K_contract, ks.max(), color=C_LANC, alpha=0.07, zorder=0)
    ax.axhspan(H_floor_lo, H_floor_hi, color="#888", alpha=0.10, zorder=0,
               xmin=K_contract / ks.max(), xmax=1.0)

    ax.axvline(K_contract, color=C_LANC, lw=1.4, ls="-", alpha=0.7, zorder=2)
    ax.plot(ks, H, marker="o", ms=4, lw=1.4, color="#222", zorder=4)
    ax.plot([K_contract, ks.max()], [H_floor_med, H_floor_med],
            color="#666", lw=0.8, ls="-.", zorder=3, alpha=0.7)

    ax.set_xlabel("Iteration $K$")
    ax.set_ylabel("Profile entropy $H(K)$ [bits]")
    ax.set_title("Two regimes: fast contraction, then stochastic mixing floor")
    ax.set_xlim(-2, ks.max() + 2)
    ax.set_ylim(H_lo, H_hi)
    fig.tight_layout()
    _save(fig, out_dir, "fig6a_entropy_kcontract", cfg["pdf"])

    _save_entropy_legend(out_dir, cfg, K_contract, H_floor_med,
                         H_floor_lo, H_floor_hi, N, ks.max())


def _save_entropy_legend(out_dir, cfg, K_contract, H_floor_med,
                         H_floor_lo, H_floor_hi, N, K_max):
    fig = plt.figure(figsize=(3.6, 2.6))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    y = 0.95
    ax.text(0.04, y, "Curve & regimes", fontweight="bold", fontsize=10.5, va="top")

    y -= 0.11
    ax.plot([0.06, 0.20], [y, y], color="#222", lw=1.6, marker="o", ms=4)
    ax.text(0.23, y, "Profile entropy $H(K)$", fontsize=9.5, va="center")

    y -= 0.11
    ax.add_patch(plt.Rectangle((0.06, y - 0.025), 0.14, 0.05,
                               color=C_LSYS, alpha=0.20))
    ax.text(0.23, y, "Fast contraction phase", fontsize=9.5, va="center")

    y -= 0.11
    ax.add_patch(plt.Rectangle((0.06, y - 0.025), 0.14, 0.05,
                               color=C_LANC, alpha=0.20))
    ax.text(0.23, y, "Stochastic mixing floor", fontsize=9.5, va="center")

    y -= 0.11
    ax.plot([0.06, 0.20], [y, y], color=C_LANC, lw=1.6)
    ax.text(0.23, y, fr"$K_{{\rm contract}} = {K_contract}$", fontsize=9.5, va="center")

    y -= 0.11
    ax.plot([0.06, 0.20], [y, y], color="#666", lw=0.9, ls="-.")
    ax.text(0.23, y, f"floor median $= {H_floor_med:.2f}$ bits",
            fontsize=9.5, va="center")

    y -= 0.13
    ax.text(0.04, y, f"$N{{=}}{N}$, $K_{{\\max}}{{=}}{K_max}$",
            fontsize=8.5, va="top", color="#555")
    y -= 0.08
    ax.text(0.04, y, f"floor range: $[{H_floor_lo:.2f},\\,{H_floor_hi:.2f}]$ bits",
            fontsize=8.5, va="top", color="#555")

    fig.savefig(out_dir / "fig6a_legend.png", dpi=600,
                bbox_inches="tight", pad_inches=0.15)
    if cfg["pdf"]:
        fig.savefig(out_dir / "fig6a_legend.pdf",
                    bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"  → fig6a_legend.png")


def fig6b_autocorr(cfg, out_dir):
    """Block-J late-window autocorrelation vs lag, image and text modalities.
    Random baseline = autocorrelation between two unrelated trajectory steps
    in the same embedding space (cosine-similarity floor in high-dim).
    τ_mix = smallest lag where autocorrelation reaches the baseline.
    Panel only — legend saved separately as fig6b_legend.png.
    """
    lh = _load_json(cfg["long_horizon"])
    if "J" not in lh:
        raise RuntimeError("Block J not in long_horizon_results.json.")
    J_img = lh["J"]["image"]; J_txt = lh["J"]["text"]
    lags = np.asarray(J_img["lags"], dtype=int)
    ac_img = np.asarray(J_img["autocorr_late_mean"], dtype=float)
    ac_txt = np.asarray(J_txt["autocorr_late_mean"], dtype=float)
    rb_img = float(J_img["random_pair_autocorr"])
    rb_txt = float(J_txt["random_pair_autocorr"])
    tau_img = J_img.get("tau_mix_iters")
    tau_txt = J_txt.get("tau_mix_iters")

    fig, ax = new_panel(width=PANEL_FULL_W, height=PANEL_DEFAULT_H + 0.5)
    ax.plot(lags, ac_img, marker="o", ms=3.5, lw=1.4, color=C_IMG)
    ax.plot(lags, ac_txt, marker="s", ms=3.5, lw=1.4, color=C_TXT)
    ax.axhline(rb_img, color=C_IMG, lw=0.8, ls=":", alpha=0.65)
    ax.axhline(rb_txt, color=C_TXT, lw=0.8, ls=":", alpha=0.65)
    if tau_img is not None:
        ax.axvline(int(tau_img), color=C_IMG, lw=0.9, ls="-.", alpha=0.7)
    if tau_txt is not None:
        ax.axvline(int(tau_txt), color=C_TXT, lw=0.9, ls="-.", alpha=0.7)

    ax.set_xlabel("Lag $\\ell$ (iterations)")
    ax.set_ylabel("Late-window autocorrelation")
    ax.set_title("Step-to-step memory persists well above random baseline")
    ax.set_xlim(0, lags.max() + 2)
    ax.set_ylim(0, max(float(ac_img.max()), float(ac_txt.max())) * 1.1)
    fig.tight_layout()
    _save(fig, out_dir, "fig6b_autocorr", cfg["pdf"])

    _save_autocorr_legend(out_dir, cfg, rb_img, rb_txt, tau_img, tau_txt,
                          lags.max())


def _save_autocorr_legend(out_dir, cfg, rb_img, rb_txt, tau_img, tau_txt, max_lag):
    fig = plt.figure(figsize=(3.6, 2.8))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    y = 0.95
    ax.text(0.04, y, "Curves", fontweight="bold", fontsize=10.5, va="top")

    y -= 0.11
    ax.plot([0.06, 0.20], [y, y], color=C_IMG, lw=1.6)
    ax.scatter([0.13], [y], color=C_IMG, s=20, marker="o", zorder=3)
    ax.text(0.23, y, "Image autocorrelation", fontsize=9.5, va="center")

    y -= 0.11
    ax.plot([0.06, 0.20], [y, y], color=C_TXT, lw=1.6)
    ax.scatter([0.13], [y], color=C_TXT, s=20, marker="s", zorder=3)
    ax.text(0.23, y, "Text autocorrelation",  fontsize=9.5, va="center")

    y -= 0.11
    ax.plot([0.06, 0.20], [y, y], color=C_IMG, lw=0.9, ls=":")
    ax.text(0.23, y, f"Image random baseline ({rb_img:.2f})",
            fontsize=9, va="center")

    y -= 0.10
    ax.plot([0.06, 0.20], [y, y], color=C_TXT, lw=0.9, ls=":")
    ax.text(0.23, y, f"Text random baseline ({rb_txt:.2f})",
            fontsize=9, va="center")

    y -= 0.11
    ax.plot([0.13, 0.13], [y - 0.02, y + 0.02], color="#666", lw=0.9, ls="-.")
    ax.text(0.23, y, r"Mixing time $\tau_{\rm mix}$", fontsize=9.5, va="center")

    y -= 0.13
    img_str = (f"$\\tau_{{\\rm mix}}^{{\\rm img}} = {int(tau_img)}$"
               if tau_img is not None else
               f"$\\tau_{{\\rm mix}}^{{\\rm img}} > {max_lag}$")
    txt_str = (f"$\\tau_{{\\rm mix}}^{{\\rm txt}} = {int(tau_txt)}$"
               if tau_txt is not None else
               f"$\\tau_{{\\rm mix}}^{{\\rm txt}} > {max_lag}$")
    ax.text(0.04, y, img_str, fontsize=9.5, va="top", color=C_IMG)
    y -= 0.08
    ax.text(0.04, y, txt_str, fontsize=9.5, va="top", color=C_TXT)

    fig.savefig(out_dir / "fig6b_legend.png", dpi=600,
                bbox_inches="tight", pad_inches=0.15)
    if cfg["pdf"]:
        fig.savefig(out_dir / "fig6b_legend.pdf",
                    bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"  → fig6b_legend.png")


# ════════════════════════════════════════════════════════════════════════════
#  Registry & dispatcher
# ════════════════════════════════════════════════════════════════════════════

PANELS = {
    "fig1a": fig1a_profile_gt,
    "fig1b": fig1b_profile_iter1,
    "fig1c": fig1c_profile_iter10,
    "fig1d": fig1d_profile_iter100,
    "fig2a": fig2a_mode_purity_k10,
    "fig2b": fig2b_mode_purity_k100,
    "fig2a_labeled": fig2a_mode_purity_k10_labeled,
    "fig2b_labeled": fig2b_mode_purity_k100_labeled,
    "fig2c": fig2c_failure_tradeoff,
    "fig3":  fig3_mi_cliff,
    "fig4a": fig4a_empty_stacked,
    "fig4b": fig4b_oov_trajectories,
    "fig5a": fig5a_lyapunov_combined,
    "fig5c": fig5c_tortuosity,
    "fig6a": fig6a_entropy_kmin,
    "fig6b": fig6b_autocorr,
}

# Panels that need Lyapunov data preloaded
LYAP_PANELS = {"fig5a"}


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
        "block_k":       args.block_k,
        "analysis_json": args.analysis_json,
        "long_horizon":  args.long_horizon,
        "pdf":           args.pdf,
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

    # Pre-load Lyapunov data once if any panel needs it
    if any(k in LYAP_PANELS for k in keys):
        cfg["lyapunov"] = _load_lyapunov(cfg)
        L = cfg["lyapunov"]
        K = L["K_max"]
        print(f"[all_figs] Lyapunov summary (from analysis_results.json Block B):")
        print(f"  K_max                                  = {K}")
        print(f"  n_anchors                              = {L['n_anchors']}")
        print(f"  λ_sys final  (cross-anchor, K={K})   = {L['lambda_sys_final']:+.4f}")
        if L["lambda_sys_per_k"] is not None:
            sys_t = L["lambda_sys_per_k"]
            print(f"  λ_sys trajectory length              = {len(sys_t)}  "
                  f"(min {sys_t[1:].min():+.4f}, max {sys_t[1:].max():+.4f})")
        else:
            print(f"  λ_sys trajectory                     = MISSING "
                  f"(re-run attractor_analysis.py with patched serializer)")
        print(f"  <λ_a> at K={K}                       = {L['lambda_a_final_mean']:+.4f}  "
              f"95% CI [{L['lambda_a_final_lo']:+.4f}, {L['lambda_a_final_hi']:+.4f}]  "
              f"({int(np.sum(L['lambda_a_final_per_anchor']>0))}/{L['n_anchors']} positive)")
        print(f"  monotone V(z) fraction                 = {L['monotone_frac']:.4f}")
        print(f"  basin radius (95th pct)                = {L['basin_radius_p95']:.3f}")

    print(f"[all_figs] Generating {len(keys)} panel(s) → {out_dir}")
    for k in keys:
        print(f"[{k}]")
        try:
            PANELS[k](cfg, out_dir)
        except Exception as e:
            print(f"  ✗ FAILED: {type(e).__name__}: {e}")
    print(f"[all_figs] Done.")


if __name__ == "__main__":
    main()