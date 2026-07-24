"""
Shared matplotlib style for AIM2 NeurIPS 2027 figures.

Each figure script imports `apply_style()` from this module and the color/size
constants below to ensure consistent typography, palette, and dimensions
across all main-text panels.

NeurIPS 2027 single-column text width is ~5.50in. Each figure here is sized
to be a self-contained panel: assemble manually via TeX subfigure or LaTeX
\\includegraphics side-by-side as desired.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────────────
#  Typography & rcParams
# ──────────────────────────────────────────────────────────────────────────────

def apply_style() -> None:
    """Apply NeurIPS-friendly matplotlib defaults. Call once at script start."""
    mpl.rcParams.update({
        # Fonts
        "font.family":        "serif",
        "font.serif":         ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset":   "cm",
        "font.size":          9.0,
        "axes.titlesize":     9.5,
        "axes.labelsize":     9.0,
        "xtick.labelsize":    8.0,
        "ytick.labelsize":    8.0,
        "legend.fontsize":    7.5,
        "figure.titlesize":   10.0,

        # Lines
        "lines.linewidth":    1.2,
        "lines.markersize":   4.0,
        "axes.linewidth":     0.7,
        "xtick.major.width":  0.7,
        "ytick.major.width":  0.7,
        "xtick.major.size":   3.0,
        "ytick.major.size":   3.0,

        # Spines
        "axes.spines.top":    False,
        "axes.spines.right":  False,

        # Layout
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype":       42,   # editable TrueType in PDF
        "ps.fonttype":        42,

        # Grid
        "axes.grid":          True,
        "grid.alpha":         0.25,
        "grid.linestyle":     "-",
        "grid.linewidth":     0.4,

        # Legend
        "legend.frameon":     False,
        "legend.handlelength": 1.5,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.0,
        "legend.borderaxespad": 0.3,
    })


# ──────────────────────────────────────────────────────────────────────────────
#  Color palette
# ──────────────────────────────────────────────────────────────────────────────
# Designed to be print-friendly, colorblind-safe, and consistent across panels.
# - GT / iter-0 / iter-K share a sequential palette (light → dark = "older to
#   later" in iteration time). Iter-K is the "loaded" color so it draws the eye.
# - Image (C_IMG) and text (C_TXT) are reserved for modality-paired plots.
# - Categorical attractor colors are used for empty-class breakdown.

# Iteration palette (GT → iter-0 → iter-K)
C_GT     = "#4A6FA5"   # muted blue
C_ITER0  = "#8FA8C7"   # lighter blue
C_ITERK  = "#C44E52"   # warm red — the "elevator-music" iter-K
C_INT    = "#7E96B6"   # intermediate blue for in-between iters

# Modality palette
C_IMG    = "#3B6FBF"   # image — blue
C_TXT    = "#D08540"   # text  — orange

# Lyapunov / dynamics palette
C_LSYS   = "#3B6FBF"   # system-level (negative, contractive)
C_LANC   = "#C44E52"   # per-anchor (positive, divergent)
C_LANC_LIGHT = "#E8B4B6"  # per-anchor light (for individual anchor traces)

# Empty-class category palette (warm-OOV vs cool-explicit-normal)
C_OOV_DOMINANT = "#A6373F"   # COPD — dominant OOV
C_OOV_FIBROSIS = "#D17B49"   # fibrosis
C_OOV_SCOLIOSIS = "#E0B26B"  # scoliosis
C_OOV_OTHER    = "#C9A36F"   # other OOV
C_NORMAL       = "#6D9DC5"   # explicit-normal
C_RESIDUAL     = "#A8A8A8"   # residual unclassified

# Mode-purity classification colors (for K3)
C_ELEVATOR = "#C44E52"   # elevator-music
C_AMPLIFIED = "#D9A441"  # amplified-but-faithful
C_PRESERVED = "#5D8C4F"  # preserved
C_DIFFUSE   = "#7E96B6"  # diffuse
C_NOVEL     = "#9670A5"  # novel mode

# Annotation accents
C_ANNOT    = "#444444"   # neutral dark grey for callouts
C_PRED     = "#666666"   # prediction line dashed grey
C_ZERO     = "#000000"   # zero line


# ──────────────────────────────────────────────────────────────────────────────
#  Standard panel dimensions (inches)
# ──────────────────────────────────────────────────────────────────────────────
# Each panel is intended as a stand-alone PDF for manual assembly.
# Widths assume eventual placement at full single-column or 1/2 column width.

PANEL_FULL_W   = 5.50    # full single-column NeurIPS width
PANEL_HALF_W   = 2.70    # half single-column
PANEL_THIRD_W  = 1.80    # third single-column

PANEL_DEFAULT_H = 2.20   # standard panel height
PANEL_TALL_H    = 2.80   # taller panel (for stacked legends or multi-row data)
PANEL_SHORT_H   = 1.80   # short panel (for entropy curves with annotation)


def new_panel(width=PANEL_FULL_W, height=PANEL_DEFAULT_H):
    """Create a single-axes figure at the requested panel size."""
    fig, ax = plt.subplots(1, 1, figsize=(width, height))
    return fig, ax


def save_panel(fig, path: str) -> None:
    """Save a panel PDF; never call plt.show()."""
    fig.savefig(path, format="pdf")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
#  Misc helpers
# ──────────────────────────────────────────────────────────────────────────────

def shorten_label(label: str, max_len: int = 24) -> str:
    """Trim long CheXpert profile labels for axis ticks."""
    if not label:
        return r"$\varnothing$"   # empty profile glyph
    if len(label) > max_len:
        return label[:max_len - 1] + "…"
    return label


def add_panel_letter(ax, letter: str, *, x=-0.13, y=1.04):
    """Add (a)/(b)/... letter at top-left of axes (use sparingly)."""
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top", ha="left")