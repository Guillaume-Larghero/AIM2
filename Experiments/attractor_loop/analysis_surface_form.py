#!/usr/bin/env python3
"""
AIM2 — Surface-Form Fidelity Analysis (Block G).

Posthoc analysis quantifying how well GENERATED REPORTS preserve the
content of the GROUND-TRUTH report across iterations, using metrics that
are INDEPENDENT of the MedCLIP encoder.

WHY THIS MATTERS FOR THE PAPER:
  Our main results (Blocks A–F, H, I) all observe trajectories through
  MedCLIP embeddings. A reviewer might object: "your encoder is biased —
  what if MedCLIP just doesn't capture clinical info, but the reports
  themselves preserve anchor identity?" This analysis tests that directly
  using surface-form metrics on the generated text.

METRICS:
  1. BLEU-4 between generated FINDINGS at iter k and GT FINDINGS.
     Surface-form lexical similarity. Drops to near-zero quickly if
     MAIRA-2 produces templated reports for many anchors.
  2. CheXpert label preservation:
       recall_k    = |labels(report_k) ∩ labels(GT)| / |labels(GT)|
       precision_k = |labels(report_k) ∩ labels(GT)| / |labels(report_k)|
       jaccard_k   = |∩| / |∪|
     These quantify how many of the original pathologies are still
     mentioned in the iter-k report.
  3. Per-pathology preservation: which pathologies survive iteration?

OUTPUTS:
  analysis/G_surface_form/
  ├── G_per_iter_summary.csv         — mean ± std of all metrics per iter
  ├── G_per_label_preservation.csv   — per-pathology survival rate
  ├── G_per_trajectory.csv           — long-format full data
  ├── G_surface_form.npz             — raw arrays
  └── G_surface_form.pdf             — 4-panel figure

USAGE:
  python analysis_surface_form.py \\
      --main_dir   .../results/chexgen_main \\
      --out_dir    .../analysis \\
      --data_csv   .../processed_data/processed_data.csv \\
      --use_chexpert {auto|gen|none}     # how to label generated reports

CHEXPERT LABELING:
  We use the user's existing GENERATION.chexpert.extractor.CheXpertLabelExtractor
  if importable; otherwise we fall back to a rule-based labeler that pattern-
  matches the 14 standard CheXpert categories. Rule-based is a baseline only —
  for the paper we recommend running with CheXpertLabelExtractor.
"""

import argparse
import json
import logging
import os
import re
import sys
from glob import glob
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  CheXpert-14 labels (standard order)
# ══════════════════════════════════════════════════════════════════════════════

CHEXPERT_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
    "Lung Opacity", "No Finding", "Pleural Effusion", "Pleural Other",
    "Pneumonia", "Pneumothorax", "Support Devices",
]


# ══════════════════════════════════════════════════════════════════════════════
#  BLEU implementation (no NLTK dependency)
# ══════════════════════════════════════════════════════════════════════════════

def _ngrams(tokens, n):
    """Generator of n-gram tuples from a token list."""
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def bleu4_smoothed(reference: str, hypothesis: str) -> float:
    """BLEU-4 with add-1 smoothing on numerator and denominator.

    Both reference and hypothesis are tokenized with simple whitespace
    splitting (sufficient for our usage on short clinical reports).
    Returns 0.0 if either is empty.
    """
    import math

    ref = (reference or "").split()
    hyp = (hypothesis or "").split()
    if not ref or not hyp:
        return 0.0

    log_precs = []
    for n in range(1, 5):
        ref_grams = {}
        for g in _ngrams(ref, n):
            ref_grams[g] = ref_grams.get(g, 0) + 1
        hyp_grams = {}
        for g in _ngrams(hyp, n):
            hyp_grams[g] = hyp_grams.get(g, 0) + 1
        clipped = sum(min(c, ref_grams.get(g, 0)) for g, c in hyp_grams.items())
        total = max(sum(hyp_grams.values()), 1)
        # Add-1 smoothing
        prec = (clipped + 1) / (total + 1)
        log_precs.append(math.log(max(prec, 1e-12)))

    # Brevity penalty
    bp = min(1.0, math.exp(1 - len(ref) / max(len(hyp), 1)))
    return float(bp * math.exp(sum(log_precs) / 4))


# ══════════════════════════════════════════════════════════════════════════════
#  Rule-based CheXpert fallback (only used if the real extractor isn't importable)
# ══════════════════════════════════════════════════════════════════════════════

# Rough-and-ready patterns. For the actual paper run we recommend the real
# CheXpert NLP labeler (Stanford). These patterns capture obvious mentions
# but miss negations, uncertainty, and many synonyms.
RULEBASED_PATTERNS = {
    "Atelectasis":               r"atelecta",
    "Cardiomegaly":              r"cardiomegal|enlarged?\s+(cardiac|heart)",
    "Consolidation":             r"consolidat",
    "Edema":                     r"\bedema\b|pulmonary\s+edema",
    "Enlarged Cardiomediastinum": r"enlarged?\s+cardiomediastinum|widened?\s+mediastinum",
    "Fracture":                  r"fracture",
    "Lung Lesion":               r"lung\s+(lesion|mass|nodule)|pulmonary\s+(mass|nodule|lesion)",
    "Lung Opacity":              r"lung\s+opacit|pulmonary\s+opacit|opacities",
    "No Finding":                r"no\s+(acute\s+)?(cardiopulmonary|finding|abnormal)",
    "Pleural Effusion":          r"pleural\s+effusion|effusion",
    "Pleural Other":             r"pleural\s+(thickening|scarring|plaque)",
    "Pneumonia":                 r"pneumoni",
    "Pneumothorax":              r"pneumothora",
    "Support Devices":           r"\b(et\s*tube|endotrach|ng\s*tube|nasogastric|"
                                 r"central\s*line|swan[\-\s]ganz|pacemaker|"
                                 r"icd\b|aicd\b|chest\s+tube|catheter|stent|"
                                 r"\bpicc\b|port|line\s+placement|tube\b)",
}

NEGATION_PATTERN = re.compile(
    r"\b(no|without|negative\s+for|denies|free\s+of|absen[ct])\s+\w*",
    re.IGNORECASE,
)


def rulebased_chexpert_labels(text: str) -> List[str]:
    """Return list of POSITIVE CheXpert labels detected in text via regex.

    Honors a coarse negation check: if the keyword sits within ~60 chars
    after a negation cue, treat as negative. This is approximate; for the
    paper we recommend using the real CheXpert labeler.
    """
    text = (text or "").lower()
    if not text:
        return []
    # Strip negated spans (~60 char window after negation cue)
    spans_to_skip = []
    for m in NEGATION_PATTERN.finditer(text):
        spans_to_skip.append((m.start(), min(len(text), m.end() + 60)))

    positive = []
    for label, pat in RULEBASED_PATTERNS.items():
        for m in re.finditer(pat, text, re.IGNORECASE):
            ms = m.start()
            if any(s <= ms < e for s, e in spans_to_skip):
                continue
            positive.append(label)
            break
    return positive


def get_chexpert_extractor():
    """Try to import the project's CheXpertLabelExtractor.

    The real API (from GENERATION/chexpert/extractor.py):
      ex = CheXpertLabelExtractor()
      result = ex.extract_labels(text)
      # result is a LabelExtractionResult dataclass with:
      #   result.labels       — List[float] aligned with result.label_names
      #                          (1.0=positive, 0.0=negative, -1.0=uncertain)
      #   result.label_names  — List[str]
      #   result.evidence     — Dict[str, List[str]]
      # We treat ONLY 1.0 as positive (uncertain → not preserved, conservative).

    Returns (callable, name) where callable: str -> List[str] (positive labels).
    Falls back to rule-based regex on import failure.

    Adds candidate AIM2 base paths to sys.path before import so this works
    regardless of which directory the script is invoked from. The script's
    own location (Experiments/attractor_loop/) is a sub-folder of the AIM2
    base, so we walk up from __file__ to find the GENERATION/ directory.
    """
    # Find AIM2 base by walking up from this script's location until we
    # see a sibling GENERATION/ directory. This makes the import work no
    # matter where the script is invoked from.
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    walk = here
    for _ in range(6):
        if os.path.isdir(os.path.join(walk, "GENERATION", "chexpert")):
            candidates.append(walk)
            break
        walk = os.path.dirname(walk)
    # Also add common environment hint as a fallback
    if "AIM2_BASE" in os.environ:
        candidates.append(os.environ["AIM2_BASE"])
    candidates.append("/n/groups/training/bmif203/AIM2")  # known O2 location

    added_paths = []
    for c in candidates:
        if c and os.path.isdir(c) and c not in sys.path:
            sys.path.insert(0, c)
            added_paths.append(c)

    if added_paths:
        logger.info(f"  Added to sys.path for CheXpert import: {added_paths}")

    try:
        from GENERATION.chexpert.extractor import CheXpertLabelExtractor  # type: ignore
        ex = CheXpertLabelExtractor()

        def fn(text: str) -> List[str]:
            if not text or not text.strip():
                return []
            result = ex.extract_labels(text)
            return [name for name, val in zip(result.label_names, result.labels)
                    if val == 1.0]

        # Smoke probe — confirm the API works as expected
        probe = fn("Cardiomegaly is present. No acute pneumothorax.")
        if "Cardiomegaly" not in probe:
            logger.warning(f"  CheXpert smoke probe unexpected: {probe}")
        else:
            logger.info(f"  CheXpert smoke probe OK: {probe}")
        return fn, "CheXpertLabelExtractor (project)"
    except Exception as e:
        logger.warning(f"  Could not load project CheXpert extractor: {e}")
        logger.warning(f"  sys.path = {sys.path[:8]} ...")
        return rulebased_chexpert_labels, "rule-based (fallback)"


# ══════════════════════════════════════════════════════════════════════════════
#  GT label loader — uses the existing CheXpert columns in processed_data.csv
# ══════════════════════════════════════════════════════════════════════════════

def load_gt_labels(data_csv: str):
    """Return a study_id -> set of positive CheXpert labels mapping.

    Uses CheXpert columns directly from the master CSV (these come from
    the original CheXpert NLP labeler applied during dataset construction,
    so they're our gold-standard for GT labels).
    """
    df = pd.read_csv(data_csv, low_memory=False)
    df["study_id"] = df["study_id"].astype(str)
    df = df.groupby("study_id", as_index=False).first()
    df = df.set_index("study_id")

    available = [c for c in CHEXPERT_LABELS if c in df.columns]
    if len(available) < 14:
        logger.warning(f"  Only {len(available)}/14 CheXpert columns in CSV. "
                       f"Missing: {[c for c in CHEXPERT_LABELS if c not in df.columns]}")
    if not available:
        return {}, []

    gt_map = {}
    for sid, row in df.iterrows():
        pos = []
        for lbl in available:
            v = row[lbl]
            if pd.notna(v) and float(v) == 1.0:
                pos.append(lbl)
        gt_map[sid] = set(pos)
    return gt_map, available


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--main_dir", required=True)
    p.add_argument("--out_dir",  required=True)
    p.add_argument("--data_csv", required=True,
                   help="Master processed_data.csv with CheXpert columns "
                        "(used for GT pathology labels).")
    p.add_argument("--use_chexpert", choices=["auto", "rulebased", "none"],
                   default="auto",
                   help="Labeling backend for GENERATED reports. "
                        "'auto' tries project extractor, falls back to rules. "
                        "'rulebased' forces the rule-based fallback. "
                        "'none' skips CheXpert preservation entirely.")
    p.add_argument("--max_studies", type=int, default=-1,
                   help="Cap on number of studies (debugging). -1 = all.")
    return p.parse_args()


def main():
    args = parse_args()
    out_subdir = os.path.join(args.out_dir, "G_surface_form")
    cache_dir  = os.path.join(args.out_dir, "cache")
    fig_dir    = os.path.join(args.out_dir, "figures")
    table_dir  = os.path.join(args.out_dir, "tables")
    for d in (out_subdir, cache_dir, fig_dir, table_dir):
        os.makedirs(d, exist_ok=True)

    logger.info("=" * 60)
    logger.info("AIM2 Surface-Form Fidelity Analysis (Block G)")
    logger.info("=" * 60)
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")

    # ── Load all per-study metrics.json (this contains findings + GT) ─────────
    metric_files = sorted(glob(os.path.join(args.main_dir, "*", "metrics.json")))
    logger.info(f"\nFound {len(metric_files)} metrics.json files")
    if args.max_studies > 0:
        metric_files = metric_files[: args.max_studies]
        logger.info(f"  truncated to {len(metric_files)} for debug")

    # ── GT labels from CSV ────────────────────────────────────────────────────
    gt_map, available_labels = load_gt_labels(args.data_csv)
    logger.info(f"\nLoaded GT labels for {len(gt_map)} studies, "
                f"{len(available_labels)} CheXpert columns")

    # ── CheXpert labeler for generated reports ────────────────────────────────
    if args.use_chexpert == "none":
        chx_fn, chx_name = None, "disabled"
    elif args.use_chexpert == "rulebased":
        chx_fn, chx_name = rulebased_chexpert_labels, "rule-based (forced)"
    else:
        chx_fn, chx_name = get_chexpert_extractor()
    logger.info(f"\nGenerated-report labeler: {chx_name}")
    if chx_fn is rulebased_chexpert_labels:
        logger.warning("  Using rule-based fallback. For paper, re-run with "
                       "the project CheXpert NLP labeler if available.")

    # ── Iterate trajectories ──────────────────────────────────────────────────
    rows = []
    K_canonical = None

    for i, mf in enumerate(metric_files):
        sid = os.path.basename(os.path.dirname(mf))
        with open(mf) as f:
            m = json.load(f)
        findings = m.get("findings", [])
        gt_text  = m.get("gt_findings", "")
        if not findings or not gt_text:
            continue
        if K_canonical is None:
            K_canonical = len(findings)
        elif len(findings) != K_canonical:
            continue  # skip incomplete

        gt_labels = gt_map.get(sid, set())

        for k, gen_text in enumerate(findings):
            bleu_k = bleu4_smoothed(gt_text, gen_text or "")

            row = {
                "study_id": sid,
                "iter":     k,
                "bleu":     bleu_k,
                "gen_len":  len((gen_text or "").split()),
                "gt_len":   len(gt_text.split()),
            }

            if chx_fn is not None and gt_labels:
                gen_labels = set(chx_fn(gen_text or ""))
                inter = gt_labels & gen_labels
                row["recall"]    = len(inter) / max(len(gt_labels), 1)
                row["precision"] = (len(inter) / len(gen_labels)
                                     if gen_labels else 0.0)
                union = gt_labels | gen_labels
                row["jaccard"]   = len(inter) / len(union) if union else 0.0
                row["n_gt_labels"]  = len(gt_labels)
                row["n_gen_labels"] = len(gen_labels)
                # Per-label hit columns
                for lbl in available_labels:
                    row[f"hit::{lbl}"] = int(lbl in gen_labels and lbl in gt_labels)
                    row[f"gt::{lbl}"]  = int(lbl in gt_labels)

            rows.append(row)

        if (i + 1) % 100 == 0:
            logger.info(f"  processed {i+1}/{len(metric_files)} studies")

    if not rows:
        logger.error("No rows produced — empty findings or no matching studies.")
        sys.exit(1)

    df_long = pd.DataFrame(rows)
    df_long.to_csv(os.path.join(out_subdir, "G_per_trajectory.csv"), index=False)
    logger.info(f"\nWrote {out_subdir}/G_per_trajectory.csv  ({len(df_long)} rows)")

    # ── Per-iter aggregate ────────────────────────────────────────────────────
    K = K_canonical
    iters = np.arange(K)
    summary_per_iter = []
    for k in iters:
        sub = df_long[df_long["iter"] == k]
        s = {"iter": int(k), "n": int(len(sub))}
        for col in ("bleu", "recall", "precision", "jaccard"):
            if col in sub.columns:
                s[f"{col}_mean"] = float(sub[col].mean())
                s[f"{col}_std"]  = float(sub[col].std())
                s[f"{col}_median"] = float(sub[col].median())
        summary_per_iter.append(s)
    df_summary = pd.DataFrame(summary_per_iter)
    df_summary.to_csv(os.path.join(out_subdir, "G_per_iter_summary.csv"), index=False)
    logger.info(f"Wrote {out_subdir}/G_per_iter_summary.csv")

    # ── Per-label preservation table (only studies where the label was GT-positive) ───
    per_label_rows = []
    if "n_gt_labels" in df_long.columns:
        for lbl in available_labels:
            hit_col = f"hit::{lbl}"
            gt_col  = f"gt::{lbl}"
            if hit_col not in df_long.columns:
                continue
            for k in iters:
                sub = df_long[(df_long["iter"] == k) & (df_long[gt_col] == 1)]
                n = len(sub)
                if n == 0:
                    rate = float("nan")
                else:
                    rate = float(sub[hit_col].mean())
                per_label_rows.append({
                    "label":      lbl,
                    "iter":       int(k),
                    "n_gt_pos":   n,
                    "preserved_rate": rate,
                })
    df_per_label = pd.DataFrame(per_label_rows)
    if len(df_per_label):
        df_per_label.to_csv(os.path.join(out_subdir, "G_per_label_preservation.csv"),
                              index=False)
        logger.info(f"Wrote {out_subdir}/G_per_label_preservation.csv")

    # ── Save raw npz ──────────────────────────────────────────────────────────
    np.savez(os.path.join(cache_dir, "G_surface_form.npz"),
             K=K, n_studies=len(df_long["study_id"].unique()),
             chexpert_backend=chx_name)

    # ── Figure: 4-panel summary ───────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    has_chx = "recall_mean" in df_summary.columns

    # Panel 1: BLEU
    ax = axes[0, 0]
    ax.plot(df_summary["iter"], df_summary["bleu_mean"], "o-", color="C0",
            lw=2, ms=5)
    ax.fill_between(df_summary["iter"],
                    df_summary["bleu_mean"] - df_summary["bleu_std"],
                    df_summary["bleu_mean"] + df_summary["bleu_std"],
                    color="C0", alpha=0.18)
    ax.set_xlabel("Iteration $k$"); ax.set_ylabel("BLEU-4 vs GT")
    ax.set_title("Lexical fidelity"); ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(0.4, df_summary["bleu_mean"].max() * 1.2))

    # Panel 2: recall + precision + jaccard
    ax = axes[0, 1]
    if has_chx:
        for col, color, label in [("recall_mean",    "C2", "Recall"),
                                   ("precision_mean", "C3", "Precision"),
                                   ("jaccard_mean",   "C4", "Jaccard")]:
            ax.plot(df_summary["iter"], df_summary[col], "o-",
                    color=color, lw=2, ms=4, label=label)
        ax.set_xlabel("Iteration $k$")
        ax.set_ylabel("CheXpert preservation")
        ax.set_title(f"Pathology preservation\n(labeler: {chx_name})", fontsize=9)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1)
    else:
        ax.text(0.5, 0.5, "CheXpert disabled", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_axis_off()

    # Panel 3: per-label preservation heatmap
    ax = axes[1, 0]
    if len(df_per_label):
        wide = df_per_label.pivot(index="label", columns="iter",
                                   values="preserved_rate")
        # Order labels by iter-K preservation, descending
        if K - 1 in wide.columns:
            wide = wide.sort_values(K - 1, ascending=False)
        im = ax.imshow(wide.values, aspect="auto", cmap="RdYlGn",
                        vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks(np.arange(len(wide.columns)))
        ax.set_xticklabels(wide.columns)
        ax.set_yticks(np.arange(len(wide.index)))
        ax.set_yticklabels(wide.index, fontsize=8)
        ax.set_xlabel("Iteration $k$")
        ax.set_title("Per-label preservation rate", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, label="hit rate")
    else:
        ax.set_axis_off()

    # Panel 4: report length over iterations
    ax = axes[1, 1]
    grouped = df_long.groupby("iter")["gen_len"].agg(["mean", "std"]).reset_index()
    ax.plot(grouped["iter"], grouped["mean"], "o-", color="C5", lw=2, ms=5)
    ax.fill_between(grouped["iter"],
                    grouped["mean"] - grouped["std"],
                    grouped["mean"] + grouped["std"],
                    color="C5", alpha=0.18)
    gt_avg = float(df_long["gt_len"].mean())
    ax.axhline(gt_avg, color="k", ls="--", lw=0.8, alpha=0.7,
               label=f"GT avg = {gt_avg:.0f}")
    ax.set_xlabel("Iteration $k$"); ax.set_ylabel("Generated report length (words)")
    ax.set_title("Report-length stationarity"); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(fig_dir, "G_surface_form.pdf")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"\nFigure → {fig_path}")

    # Top-line summary JSON
    summary = {
        "args": vars(args),
        "n_studies":  int(len(df_long["study_id"].unique())),
        "n_iterations": int(K),
        "chexpert_backend": chx_name,
        "bleu_iter_0": float(df_summary["bleu_mean"].iloc[0]),
        "bleu_iter_K": float(df_summary["bleu_mean"].iloc[-1]),
    }
    if has_chx:
        summary.update({
            "recall_iter_0":    float(df_summary["recall_mean"].iloc[0]),
            "recall_iter_K":    float(df_summary["recall_mean"].iloc[-1]),
            "jaccard_iter_0":   float(df_summary["jaccard_mean"].iloc[0]),
            "jaccard_iter_K":   float(df_summary["jaccard_mean"].iloc[-1]),
            "precision_iter_0": float(df_summary["precision_mean"].iloc[0]),
            "precision_iter_K": float(df_summary["precision_mean"].iloc[-1]),
        })
    with open(os.path.join(args.out_dir, "G_surface_form_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Summary → {args.out_dir}/G_surface_form_summary.json")
    logger.info("Done.")


if __name__ == "__main__":
    main()