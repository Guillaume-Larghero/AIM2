#!/usr/bin/env python3
"""
AIM2 — Block K: Generated-pathology mode structure analysis
                (the "elevator-music attractor" test).

============================================================================
QUESTION
============================================================================
At iter K, do the loop's generated CheXpert profiles cluster into a small
number of canonical modes ("Cardiomegaly mode", "No Finding mode", etc.),
or remain diffusely distributed across patient-specific profiles?

The "elevator-music attractor" hypothesis (Hintze et al. 2026, Cell Patterns)
predicts that long-iterated coupled generative loops collapse onto a
small set of generic motifs. If true for our medical loop, we should
see: at iter K, a small number of canonical CheXpert profiles cover
most of the cohort, AND those profiles are generated regardless of
input patient (i.e., mode size at iter K >> GT prevalence of that
profile, and within-mode GT match rate ≈ chance).

============================================================================
THREE DEFINITIONS OF "MODE" (reported side-by-side; honest)
============================================================================
1. HARD-PROFILE MODE — every unique 14-bit CheXpert profile is its own
   mode. Direct test for elevator-music: does the iter-K cohort collapse
   onto a small number of *identical* profiles? No clustering choices.
   Reports: distinct profile count, top-N coverage, Shannon entropy,
   perplexity (effective number of profiles).

2. SOFT-CLUSTER MODE — k-means on the 14-bit profile vectors (Hamming
   ≡ squared-Euclidean for binary). Captures family structure (e.g.,
   "Cardiomegaly + any subset" as one family). Reports: best K by
   silhouette, modal profile per cluster, per-cluster size and purity.

3. SINGLE-LABEL MARGINAL — for each of 14 labels, fraction of cohort
   with that label positive at iter K. Compared to iter-0 and GT.
   Loses co-occurrence info but maximally interpretable.

============================================================================
MODE PURITY (the crucial second test)
============================================================================
For each top profile m at iter K with prevalence p_K(m):
  • size_K(m)         = N · p_K(m)
  • GT_match_rate(m)  = P(study's GT profile == m | study's iter-K profile == m)
  • GT_prevalence(m)  = P(study's GT profile == m)
  • Lift(m)           = GT_match_rate(m) / GT_prevalence(m)
  • Inflation(m)      = p_K(m) / GT_prevalence(m)

Interpretation:
  Lift ≈ 1, Inflation >> 1   → ELEVATOR MUSIC: mode generated regardless of input
  Lift >> 1, Inflation ≈ 1   → PRESERVATION:   mode reflects faithful copy of GT
  Lift ≈ 1, Inflation ≈ 1    → DIFFUSE:        no mode collapse at all

============================================================================
INPUT
============================================================================
  Trajectory directory (chexgen_main K=11 OR chexgen_long K=101, same schema)
  CheXpert columns from processed_data.csv (GT)
  Project's CheXpertLabelExtractor (sys.path walk-up; rule-based fallback)

============================================================================
OUTPUT
============================================================================
  block_K_results.json     — all numbers per probe iter
  figures/K_*.pdf          — profile distributions, mode purity, marginals,
                              entropy/distinct-count trajectories
  tables/K_top_profiles_iter_*.tsv — top 30 profiles per iter (for appendix)
"""

import argparse
import json
import logging
import os
import re
import sys
from collections import Counter
from glob import glob

import numpy as np
import pandas as pd

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

CHEXPERT_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
    "Lung Opacity", "No Finding", "Pleural Effusion", "Pleural Other",
    "Pneumonia", "Pneumothorax", "Support Devices",
]


# ══════════════════════════════════════════════════════════════════════════════
#  Empty-profile (no-CheXpert-label) characterization
# ══════════════════════════════════════════════════════════════════════════════
#
# A report whose CheXpert profile is {} falls into 3 heterogeneous categories
# that matter for paper interpretation:
#   1. EXPLICIT-NORMAL — phrasings like "without alterations", "within normal
#      limits" that are clinically "No Finding" but missed by the rule-based
#      extractor (CheXpert 14-label "No Finding" misses).
#   2. OOV-DISEASE — real pathologies outside the 14-label CheXpert vocabulary
#      (COPD, scoliosis, aortic elongation, hiatal hernia, ...). These are
#      genuine attractor modes the loop produces — invisible to CheXpert
#      but clinically meaningful.
#   3. SHORT/OTHER — very short or unclassifiable reports.
#
# The "primary category" assignment is ordered by clinical primacy: active
# pulmonary disease > anatomical variants/structural > old/healed findings >
# devices/post-surgical (likely Support Devices misses) > short/other.

EMPTY_DISPLAY_LABEL = "No CheXpert label"   # Replaces "{}" in figures/tables

# Explicit-normal phrasings (should clinically map to "No Finding"
# but were missed by the rule-based extractor).
EXPLICIT_NORMAL_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bwithout (?:radiological |significant )?alterations? of significance\b",
        r"\bwithout radiological alterations?\b",
        r"\bwithout significant alterations?\b",
        r"\bwithout (?:relevant )?pathological findings?\b",
        r"\bwithin normal limits\b",
        r"\bwithin normality\b",
        r"\bno relevant pathological findings?\b",
        r"\bno significant (?:radiological )?(?:alterations|findings|abnormalit)\w*\b",
        r"\bno other significant findings?\b",
        r"\bno further (?:significant )?findings?\b",
        r"\bno (?:acute |relevant )?(?:cardiopulmonary )?(?:abnormalit|finding)\w+\b",
        r"\bno acute skeletal findings?\b",
        r"\bunremarkable\b",
        r"\bclear lungs?\b",
        r"\blungs? (?:are |is )?(?:well[- ]expanded(?: and)? )?clear\b",
        r"\blungs? and pleural surfaces? (?:are )?clear\b",
        r"\blungs? (?:are )?well[- ]expanded\b",
        r"\b(?:heart size|cardiac silhouette|mediastinal contours?|hilar contours?)"
        r" (?:are |is )?(?:within )?normal\b",
        r"\bchanges? consistent with the patient.?s? age\b",
        r"\bpreoperative\..*within normality\b",
    ]
]

# OOV pattern groups — ordered by clinical primacy (first match wins for primary)
OOV_PATTERN_GROUPS = {
    # Active pulmonary disease (highest primacy)
    "COPD/emphysema": [
        r"\bCOPD\b",
        r"\bchronic obstructive pulmonary\b",
        r"\bemphysema(?!\s+(?:subcut|of\s+the\s+chest))\b",
        r"\bair trapping\b",
        r"\bhyperinflat\w+\b",
    ],
    "bullae": [
        r"\bbullae?\b",
        r"\bblebs?\b",
    ],
    "pulmonary fibrosis/ILD": [
        r"\bpulmonary fibrosis\b",
        r"\binterstitial lung disease\b",
        r"\binterstitial (?:abnormalit\w+|pattern|changes?|opacities)\b",
        r"\bhoneycombing\b",
        r"\breticular[- ]?nodular\b",
        r"\breticulonodular\b",
        r"\bsubpleural (?:reticulation|changes?|honeycomb)\b",
        r"\bdiffuse interstitial\b",
        r"\busual interstitial pneumon\w*\b",
    ],
    "subcutaneous emphysema/pneumomediastinum": [
        r"\bsubcutaneous emphysema\b",
        r"\bpneumomediastinum\b",
    ],
    "metastatic disease/multiple nodules": [
        r"\bmetasta\w+\b",
        r"\bmultiple (?:bilateral )?(?:pulmonary |lung )?(?:nodules?|masses?)\b",
        r"\bmiliary (?:pattern|nodules)\b",
    ],
    "mass/nodule (likely Lung Lesion miss)": [
        r"\bmediastinal mass\b",
        r"\bpulmonary mass\b",
        r"\bnodular (?:density|image|lesion|opacit\w+)\b",
        r"\bnodule (?:in|of) the\b",
        r"\bdominant mass\b",
        r"\b\d+(?:\.\d+)?\s*cm\s+(?:nodule|mass|lesion)\b",
    ],
    "pleural thickening/plaques": [
        r"\bcalcified pleural plaques?\b",
        r"\bcalcified pleurisy\b",
        r"\bpleural thickening\b",
        r"\basbestos\b",
        r"\bpleuroparenchymal (?:tracts?|changes?|scar)\b",
    ],
    # Anatomical variants / structural (mid primacy)
    "aortic elongation/tortuosity": [
        r"\baortic elongation\b",
        r"\belongation of the (?:thoracic )?aorta\b",
        r"\btortuous aorta\b",
        r"\baortic tortuosity\b",
        r"\baorta is tortuous\b",
        r"\bdiffusely tortuous\b",
        r"\bunfolding of the aorta\b",
    ],
    "scoliosis/kyphosis": [
        r"\bscoliosis\b",
        r"\bkyphosis\b",
        r"\bdorsolumbar (?:scoliosis|curve)\b",
        r"\bdextroscoliosis\b",
        r"\bdorsal scoliosis\b",
    ],
    "hiatal hernia": [
        r"\bhiatal hernia\b",
        r"\bhiatus hernia\b",
    ],
    "volume loss": [
        r"\bvolume loss\b",
        r"\bloss of volume\b",
    ],
    # Old/healed findings (lower primacy)
    "calcified granuloma (old)": [
        r"\bcalcified granuloma\w*\b",
        r"\bgranulomas? (?:in|of) (?:both )?(?:upper|lower|inferior|right|left)\b",
    ],
    "fracture callus (healed)": [
        r"\bfracture callus(?:es)?\b",
        r"\bhealed (?:rib )?fractures?\b",
        r"\bcalluses\b",
    ],
    # Devices / post-surgical (likely Support Devices extractor misses)
    "sternotomy/post-surgical (likely SD miss)": [
        r"\bsternotomy\b",
        r"\bcerclage\b",
        r"\bsuture wires?\b",
        r"\bsurgical clips\b",
        r"\bpost-?surgical\b",
    ],
    "pacemaker/ICD (likely SD miss)": [
        r"\bpacemaker\b",
        r"\bICD\b",
        r"\bdefibrillator\b",
    ],
    "valve prosthesis (likely SD miss)": [
        r"\bvalve prosthesis\b",
        r"\bmetallic (?:mitral |aortic |heart )?valve\b",
        r"\bprosthetic valve\b",
    ],
    "catheter/line (likely SD miss)": [
        r"\bPICC\b",
        r"\bcentral (?:venous )?(?:catheter|line)\b",
        r"\bport-a-cath\b",
        r"\bsubclavian (?:vascular )?(?:catheter|stent|approach|Port)\b",
        r"\bjugular catheter\b",
        r"\bhemodialysis catheter\b",
    ],
    # Other anatomical descriptors
    "other-anatomic": [
        r"\bgynecomastia\b",
        r"\bprominent hilum\b",
        r"\bcalcified mediastinal\b",
    ],
}

OOV_REGEXES = {
    cat: [re.compile(p, re.IGNORECASE) for p in pats]
    for cat, pats in OOV_PATTERN_GROUPS.items()
}
OOV_DISPLAY_ORDER = list(OOV_PATTERN_GROUPS.keys())  # Clinical-primacy ordering


def categorize_empty_report(text):
    """Classify a {} CheXpert profile report into one primary category.

    Returns dict with:
        primary_category:    str (highest-primacy match)
        all_oov_categories:  list[str] (every OOV group matched, can be multi)
        is_explicit_normal:  bool (any explicit-normal pattern matched)
        n_chars:             int
    """
    text = (text or "").strip()
    n_chars = len(text)

    if n_chars == 0:
        return {
            "primary_category":   "truly-empty",
            "all_oov_categories": [],
            "is_explicit_normal": False,
            "n_chars":            0,
        }

    is_normal = any(p.search(text) for p in EXPLICIT_NORMAL_PATTERNS)
    oov_matches = [cat for cat in OOV_DISPLAY_ORDER
                    if any(r.search(text) for r in OOV_REGEXES[cat])]

    # Primary category: first OOV match in primacy order, else explicit-normal,
    # else short, else other.
    if oov_matches:
        primary = oov_matches[0]
    elif is_normal:
        primary = "explicit-normal (CheXpert miss)"
    elif n_chars < 25:
        primary = "short/other"
    else:
        primary = "other-unclassified"

    return {
        "primary_category":   primary,
        "all_oov_categories": oov_matches,
        "is_explicit_normal": is_normal,
        "n_chars":            n_chars,
    }


def analyze_empty_profiles_at_iter(profiles_at_iter, findings_at_iter, sids,
                                     n_examples_per_category=5, seed=42):
    """For each {} profile at this iter, categorize the report.

    Args:
        profiles_at_iter:  list of frozensets aligned with sids
        findings_at_iter:  dict[sid -> text]
        sids:              list of study ids
    """
    empty_reports = []
    for sid, prof in zip(sids, profiles_at_iter):
        if len(prof) == 0:
            text = findings_at_iter.get(sid, "")
            cat_info = categorize_empty_report(text)
            empty_reports.append({"sid": sid, "text": text, **cat_info})

    n_empty = len(empty_reports)
    n_total = len(sids)

    if n_empty == 0:
        return {
            "n_total_empty":              0,
            "n_total_studies":            n_total,
            "fraction_empty":             0.0,
            "primary_category_counts":    {},
            "primary_category_fractions": {},
            "oov_category_counts":        {},
            "length_stats":               None,
            "examples":                   {},
        }

    primary_counts = Counter(r["primary_category"] for r in empty_reports)
    primary_fractions = {k: v / n_empty for k, v in primary_counts.items()}

    # OOV any-match counts (one report can contribute to multiple categories)
    oov_counts = Counter()
    for r in empty_reports:
        for cat in r["all_oov_categories"]:
            oov_counts[cat] += 1

    lengths = np.array([r["n_chars"] for r in empty_reports])
    length_stats = {
        "mean":   float(lengths.mean()),
        "median": float(np.median(lengths)),
        "p25":    float(np.percentile(lengths, 25)),
        "p75":    float(np.percentile(lengths, 75)),
        "min":    int(lengths.min()),
        "max":    int(lengths.max()),
    }

    rng = np.random.RandomState(seed)
    examples = {}
    for cat in primary_counts:
        cat_reports = [r for r in empty_reports if r["primary_category"] == cat]
        n_take = min(n_examples_per_category, len(cat_reports))
        if n_take == 0:
            continue
        if len(cat_reports) > n_take:
            idx = rng.choice(len(cat_reports), size=n_take, replace=False)
        else:
            idx = list(range(len(cat_reports)))
        examples[cat] = [
            {"sid": cat_reports[i]["sid"], "text": cat_reports[i]["text"][:300]}
            for i in idx
        ]

    return {
        "n_total_empty":              n_empty,
        "n_total_studies":            n_total,
        "fraction_empty":             n_empty / n_total,
        "primary_category_counts":    dict(primary_counts),
        "primary_category_fractions": primary_fractions,
        "oov_category_counts":        dict(oov_counts),
        "length_stats":               length_stats,
        "examples":                   examples,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Three-regime entropy dynamics fit (the Lyapunov bounce-back signature)
# ══════════════════════════════════════════════════════════════════════════════
#
# The two-timescale Lyapunov picture (Block B) predicts a non-monotonic
# entropy trajectory:
#   • Cohort-level contraction (λ_sys < 0): pulls all trajectories onto the
#     deepest attractor modes → entropy DROPS in early iterations, fast.
#   • Per-anchor divergence (λ̄_a > 0): kicks individual trajectories between
#     attractor modes → entropy RECOVERS toward H_∞ at long horizon, slow.
#
# Empirical signature:
#   - Fast contraction: K = 0..K_min     (timescale ~ 1/λ̄_a ≈ 32 iters)
#   - Minimum:          K = K_min
#   - Macro-mixing:     K > K_min        (timescale = τ_macro >> τ_micro)
#
# Fit:  H(K) = H_∞ + (H_min - H_∞) · exp(-(K - K_min) / τ_macro)  for K ≥ K_min
#
# τ_macro is the macro-mixing timescale of the bounded attractor, providing
# a paper-grade point estimate (Block J only gives a lower bound, τ_macro > 50).

def fit_lyapunov_three_regime(probe_iters, entropies):
    """Fit entropy bounce-back curve and extract τ_macro point estimate.

    Returns dict with:
        K_min, H_min:                     location of entropy minimum
        H_K0, H_Kmax:                     anchor values
        initial_entropy_drop:             H_K0 - H_min  (contraction strength)
        recovery_amplitude:               H_Kmax - H_min (recovery strength)
        n_post_min_points:                n probe iters at or after K_min
        tau_macro_fit, H_inf_fit:         fitted parameters (None if no fit)
        tau_macro_ci_lower/upper:         95% CI from Hessian
        fit_succeeded:                    bool
        fit_error:                        str (if failed)
        classification:                   "monotone-collapse",
                                          "bouncing-attractor", "flat-attractor"
    """
    iters = np.array(sorted(probe_iters), dtype=float)
    H = np.array(entropies, dtype=float)

    K_min_idx = int(np.argmin(H))
    K_min = float(iters[K_min_idx])
    H_min = float(H[K_min_idx])
    H_K0 = float(H[0])
    H_Kmax = float(H[-1])

    initial_drop = H_K0 - H_min
    recovery_amp = H_Kmax - H_min

    if K_min_idx == len(iters) - 1:
        classification = "monotone-collapse"
    elif recovery_amp < 0.05:
        classification = "flat-attractor"
    else:
        classification = "bouncing-attractor"

    base = {
        "K_min":                K_min,
        "H_min":                H_min,
        "H_K0":                 H_K0,
        "H_Kmax":               H_Kmax,
        "initial_entropy_drop": initial_drop,
        "recovery_amplitude":   recovery_amp,
        "classification":       classification,
        "n_post_min_points":    int(len(iters) - K_min_idx),
        "tau_macro_fit":        None,
        "H_inf_fit":            None,
        "tau_macro_ci_lower":   None,
        "tau_macro_ci_upper":   None,
        "fit_succeeded":        False,
        "is_well_identified":   False,
        "fit_error":            None,
    }

    # Need ≥3 post-min points to fit 2 params; skip non-bouncing
    if classification != "bouncing-attractor" or (len(iters) - K_min_idx) < 3:
        base["fit_error"] = ("τ_macro fit requires bouncing-attractor "
                              "classification and ≥3 post-min probe iters; "
                              f"got {classification} with "
                              f"{len(iters) - K_min_idx} post-min points")
        return base

    try:
        from scipy.optimize import curve_fit
        K_post = iters[K_min_idx:] - K_min  # K=0 at minimum
        H_post = H[K_min_idx:]

        def model(K, H_inf, tau):
            return H_inf + (H_min - H_inf) * np.exp(-K / tau)

        H_inf_guess = float(H_post[-1])
        tau_guess = max(float(K_post[-1]) / 2.0, 5.0)

        popt, pcov = curve_fit(
            model, K_post, H_post,
            p0=[H_inf_guess, tau_guess],
            bounds=([H_min, 1.0], [H_K0 + 1.0, 1e4]),
            maxfev=10000,
        )
        H_inf, tau_macro = popt
        sigma_tau = float(np.sqrt(np.diag(pcov))[1]) if pcov is not None else None

        ci_lower = (max(1.0, tau_macro - 1.96 * sigma_tau)
                    if sigma_tau is not None else None)
        ci_upper = (tau_macro + 1.96 * sigma_tau
                    if sigma_tau is not None else None)

        # τ_macro is well-identified when:
        #   (i)  CI width is reasonable (upper/lower < 10x), and
        #   (ii) point estimate is within ~3x of observation horizon
        K_horizon = float(max(iters) - K_min)
        well_id = False
        if ci_lower and ci_upper and ci_lower > 0:
            ci_ratio = ci_upper / ci_lower
            well_id = (ci_ratio < 10.0) and (tau_macro < 3.0 * K_horizon)

        base.update({
            "tau_macro_fit":      float(tau_macro),
            "H_inf_fit":          float(H_inf),
            "tau_macro_ci_lower": float(ci_lower) if ci_lower is not None else None,
            "tau_macro_ci_upper": float(ci_upper) if ci_upper is not None else None,
            "fit_succeeded":      True,
            "is_well_identified": bool(well_id),
        })
    except Exception as e:
        base["fit_error"] = f"{type(e).__name__}: {e}"

    return base


# ══════════════════════════════════════════════════════════════════════════════
#  CheXpert extractor (mirroring analysis_surface_form.get_chexpert_extractor)
# ══════════════════════════════════════════════════════════════════════════════

def get_chexpert_extractor(use_chexpert="auto"):
    """sys.path walk-up for the project CheXpertLabelExtractor."""
    if use_chexpert == "none":
        return None, "disabled"
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    walk = here
    for _ in range(6):
        if os.path.isdir(os.path.join(walk, "GENERATION", "chexpert")):
            candidates.append(walk)
            break
        walk = os.path.dirname(walk)
    if "AIM2_BASE" in os.environ:
        candidates.append(os.environ["AIM2_BASE"])
    candidates.append("/n/groups/training/bmif203/AIM2")
    for c in candidates:
        if c and os.path.isdir(c) and c not in sys.path:
            sys.path.insert(0, c)

    if use_chexpert in ("auto", "extractor"):
        try:
            from GENERATION.chexpert.extractor import CheXpertLabelExtractor  # type: ignore
            ex = CheXpertLabelExtractor()
            def fn(text):
                if not text or not text.strip():
                    return frozenset()
                r = ex.extract_labels(text)
                return frozenset(name for name, val in zip(r.label_names, r.labels)
                                  if val == 1.0)
            probe = fn("Cardiomegaly is present. No acute pneumothorax.")
            assert "Cardiomegaly" in probe, f"smoke probe failed: {probe}"
            logger.info(f"  CheXpert smoke probe OK: {sorted(probe)}")
            return fn, "CheXpertLabelExtractor (project)"
        except Exception as e:
            logger.warning(f"  Could not load project CheXpertLabelExtractor: {e}")
    # Minimal regex fallback (sufficient for sanity, but use project for paper)
    PATTERNS = {
        "Cardiomegaly":     re.compile(r"\bcardiomegal\w*\b", re.I),
        "Pleural Effusion": re.compile(r"\bpleural\s+effusion|\beffusion\w*\b", re.I),
        "Pneumothorax":     re.compile(r"\bpneumothorax\b|\bptx\b", re.I),
        "Atelectasis":      re.compile(r"\batelecta\w*\b", re.I),
        "Consolidation":    re.compile(r"\bconsolidat\w*\b", re.I),
        "Edema":            re.compile(r"\bedema\b", re.I),
        "Pneumonia":        re.compile(r"\bpneumonia\b|\bpneumonitis\b", re.I),
        "Support Devices":  re.compile(r"\b(et\s+tube|ett|ng\s+tube|pacemaker|catheter|chest\s+tube)\b", re.I),
        "Lung Opacity":     re.compile(r"\bopacit\w+\b", re.I),
        "Fracture":         re.compile(r"\bfracture\w*\b", re.I),
        "Lung Lesion":      re.compile(r"\b(lung|pulmonary)\s+(mass|lesion|nodule)\b", re.I),
        "No Finding":       re.compile(r"\bno\s+(acute\s+)?(cardiopulmonary\s+)?(abnormalit|finding)\w*\b|\bunremarkable\b|\bclear\s+lungs?\b", re.I),
    }
    NEG = re.compile(r"\b(no\s+(evidence\s+of\s+)?|without|absent|negative\s+for|not\s+(seen|identified))\b", re.I)
    def rule_fn(text):
        if not text or not text.strip():
            return frozenset()
        labels = []
        for sentence in re.split(r"[.!?]\s+", text):
            for lbl, pat in PATTERNS.items():
                m = pat.search(sentence)
                if m and not NEG.search(sentence[:m.start()]):
                    labels.append(lbl)
        return frozenset(labels)
    logger.warning("  Falling back to rule-based regex labeler")
    return rule_fn, "rule-based fallback"


def load_gt_labels(data_csv):
    """study_id → frozenset of GT positive CheXpert labels (val == 1.0)."""
    df = pd.read_csv(data_csv, low_memory=False)
    df["study_id"] = df["study_id"].astype(str)
    df = df.groupby("study_id", as_index=False).first().set_index("study_id")
    available = [c for c in CHEXPERT_LABELS if c in df.columns]
    gt_map = {}
    for sid, row in df.iterrows():
        pos = frozenset(lbl for lbl in available
                          if pd.notna(row[lbl]) and float(row[lbl]) == 1.0)
        gt_map[sid] = pos
    return gt_map, available


# ══════════════════════════════════════════════════════════════════════════════
#  Trajectory loader — handles main_dir (K=11) or long_dir (K=101)
# ══════════════════════════════════════════════════════════════════════════════

def load_trajectory_findings(traj_dir, probe_iters, max_studies=-1):
    """For each study, load findings text at the requested probe iters.

    Returns:
        sids:       list[str]
        findings:   dict[iter -> dict[sid -> text]]  (only sids present at all iters)
        K_max_seen: highest iter index found in any study (informational)
    """
    sids_all = sorted([d for d in os.listdir(traj_dir)
                       if os.path.isdir(os.path.join(traj_dir, d)) and d.isdigit()])
    if max_studies > 0:
        sids_all = sids_all[:max_studies]
    logger.info(f"  Found {len(sids_all)} candidate study directories")

    findings = {k: {} for k in probe_iters}
    K_max_seen = -1
    valid_sids = set(sids_all)
    for sid in sids_all:
        sd = os.path.join(traj_dir, sid)
        # Determine K_max for this study
        f_files = sorted(glob(os.path.join(sd, "findings_iter_*.txt")))
        if not f_files:
            valid_sids.discard(sid); continue
        max_k_this = max(int(os.path.basename(f).split("_")[-1].split(".")[0])
                          for f in f_files)
        K_max_seen = max(K_max_seen, max_k_this)
        # Verify we have all requested probe iters for this study
        for k in probe_iters:
            f = os.path.join(sd, f"findings_iter_{k:03d}.txt")
            if not os.path.exists(f):
                valid_sids.discard(sid); break
            with open(f) as fp:
                findings[k][sid] = fp.read()
    sids = [s for s in sids_all if s in valid_sids]
    # Drop incomplete sids from findings
    for k in findings:
        findings[k] = {s: findings[k][s] for s in sids}
    logger.info(f"  Loaded {len(sids)} valid trajectories with all probe iters present")
    logger.info(f"  K_max in directory: {K_max_seen}")
    return sids, findings, K_max_seen


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def profile_to_label(p, max_chars=40):
    """Render a frozenset profile as a sortable, readable label string."""
    if len(p) == 0:
        return "{}"
    s = "+".join(sorted(p))
    return s if len(s) <= max_chars else s[:max_chars - 3] + "..."


def profile_to_vector(p, label_names):
    """frozenset → 14-dim binary vector."""
    return np.array([1 if lbl in p else 0 for lbl in label_names], dtype=np.int8)


# ══════════════════════════════════════════════════════════════════════════════
#  Hard-profile analysis (Definition 1)
# ══════════════════════════════════════════════════════════════════════════════

def hard_profile_stats(profiles, top_n=30):
    """Stats over the empirical distribution of distinct profiles.
       profiles: list of frozensets, length N.
    """
    N = len(profiles)
    counts = Counter(profiles)
    distinct = len(counts)
    sorted_items = counts.most_common()  # [(profile, count), ...]
    counts_arr = np.array([c for _, c in sorted_items])
    p = counts_arr / N

    # Shannon entropy in nats AND bits
    entropy_nats = float(-(p * np.log(p)).sum())
    entropy_bits = float(-(p * np.log2(p)).sum())
    perplexity   = float(np.exp(entropy_nats))           # effective # of profiles
    entropy_norm = float(entropy_nats / np.log(N))       # 0=fully concentrated, 1=uniform
    H_max_bits   = float(np.log2(N))                     # uniform over N studies

    # Top-N coverage
    coverage = {}
    for n in [1, 3, 5, 10, 20, 30, 50]:
        coverage[f"top{n}"] = float(counts_arr[:n].sum() / N)

    # Build top profiles table
    top_profiles = []
    for i, (prof, cnt) in enumerate(sorted_items[:top_n]):
        top_profiles.append({
            "rank":         i + 1,
            "profile":      sorted(list(prof)),
            "label":        profile_to_label(prof),
            "size":         int(cnt),
            "fraction":     float(cnt / N),
            "n_pathologies": len(prof),
        })

    # Empty-profile vs single-label vs multi-label breakdown
    n_empty       = sum(1 for prof in profiles if len(prof) == 0)
    n_no_finding  = sum(1 for prof in profiles if prof == frozenset({"No Finding"}))
    n_single_lbl  = sum(1 for prof in profiles if len(prof) == 1)
    n_multi       = sum(1 for prof in profiles if len(prof) >= 2)

    return {
        "N":                     int(N),
        "distinct_profiles":     int(distinct),
        "entropy_nats":          entropy_nats,
        "entropy_bits":          entropy_bits,
        "entropy_normalized":    entropy_norm,
        "perplexity":            perplexity,
        "H_max_bits_uniform":    H_max_bits,
        "top_n_coverage":        coverage,
        "top_profiles":          top_profiles,
        "n_empty_profile":       n_empty,
        "n_only_no_finding":     n_no_finding,
        "n_single_label":        n_single_lbl,
        "n_multi_label":         n_multi,
        "fraction_empty":        float(n_empty / N),
        "fraction_no_finding":   float(n_no_finding / N),
        "fraction_single_label": float(n_single_lbl / N),
        "fraction_multi_label":  float(n_multi / N),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Soft-cluster analysis (Definition 2)
# ══════════════════════════════════════════════════════════════════════════════

def soft_cluster_stats(profiles, label_names, K_range=(2, 25), n_gap_refs=10,
                         rng_seed=42):
    """k-means on 14-bit binary vectors. For binary {0,1}, ||a-b||² == Hamming(a,b).
    Reports per-K silhouette + gap statistic + best K by both criteria, plus
    per-cluster modal profile.

    NOTE on cluster-count selection:
      Silhouette is monotonically increasing for binary data with mode collapse
      and a long tail of rare profiles, because k-means at high K splits the
      long tail into clusters of near-identical points (silhouette ≈ 1 per
      cluster). It is NOT a reliable cluster-count test for this data.

      Gap statistic (Tibshirani et al. 2001) compares observed within-cluster
      inertia W_k to the expected W_k under a marginal-matched null
      (per-bit Bernoulli with the same positive rate as the observed cohort).
      The natural maximum of Gap_k indicates the cluster count above which
      additional clusters provide no more separation than the null.

      Best practice for the paper: report hard-profile statistics (top-N
      coverage, entropy, perplexity) as primary evidence; use gap-statistic
      best-K from this function as a corroborating soft-cluster count.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    X = np.stack([profile_to_vector(p, label_names) for p in profiles]).astype(np.float64)
    N, D = X.shape
    rng = np.random.default_rng(rng_seed)

    # Per-bit positive rates (used for marginal-matched null reference clouds)
    p_per_bit = X.mean(axis=0)

    out_per_k = {}
    log_W_obs = []
    log_W_ref_mean = []
    log_W_ref_sd = []
    sils = []
    inertias = []
    best_assigns = {}
    best_centers = {}

    for k in range(K_range[0], K_range[1] + 1):
        km = KMeans(n_clusters=k, random_state=rng_seed, n_init=10).fit(X)
        inertias.append(float(km.inertia_))
        # Silhouette
        try:
            sil = float(silhouette_score(X, km.labels_, metric="hamming"))
        except Exception:
            sil = float("nan")
        sils.append(sil)
        log_W_obs.append(float(np.log(max(km.inertia_, 1e-12))))
        # Gap reference: per-bit Bernoulli with matched marginal
        ref_log_Ws = []
        for _ in range(n_gap_refs):
            Xref = (rng.random((N, D)) < p_per_bit).astype(np.float64)
            km_ref = KMeans(n_clusters=k, random_state=rng_seed,
                              n_init=5).fit(Xref)
            ref_log_Ws.append(np.log(max(km_ref.inertia_, 1e-12)))
        log_W_ref_mean.append(float(np.mean(ref_log_Ws)))
        log_W_ref_sd.append(float(np.std(ref_log_Ws, ddof=1)))
        # Cache
        best_assigns[k] = km.labels_
        best_centers[k] = km.cluster_centers_

    # Gap_k = E[log W_ref] - log W_obs; s_k = sqrt(1+1/B) * SD[log W_ref]
    gap_arr = np.array(log_W_ref_mean) - np.array(log_W_obs)
    s_arr = np.array(log_W_ref_sd) * np.sqrt(1.0 + 1.0 / max(n_gap_refs, 1))
    ks = list(range(K_range[0], K_range[1] + 1))

    for i, k in enumerate(ks):
        out_per_k[k] = {
            "silhouette":   sils[i],
            "inertia":      inertias[i],
            "log_W_obs":    log_W_obs[i],
            "log_W_ref":    log_W_ref_mean[i],
            "gap":          float(gap_arr[i]),
            "gap_se":       float(s_arr[i]),
        }

    # Best K by silhouette (max). Likely monotone-increasing for this data.
    best_sil_k = ks[int(np.argmax(sils))]
    best_sil_value = float(np.max(sils))
    # Best K by Tibshirani's gap rule:
    #   smallest k such that Gap(k) >= Gap(k+1) - s(k+1)
    # If no k satisfies, pick argmax(gap).
    best_gap_k = ks[int(np.argmax(gap_arr))]  # default fallback
    for i in range(len(ks) - 1):
        if gap_arr[i] >= gap_arr[i + 1] - s_arr[i + 1]:
            best_gap_k = ks[i]
            break
    best_gap_value = float(gap_arr[ks.index(best_gap_k)])

    # Per-cluster modal profile, using GAP-best K (more reliable than silhouette)
    chosen_k = best_gap_k
    chosen_assign = best_assigns[chosen_k]
    chosen_centers = best_centers[chosen_k]
    clusters = []
    for c in range(chosen_k):
        mask = chosen_assign == c
        n_c = int(mask.sum())
        if n_c == 0:
            continue
        modal_vec = (chosen_centers[c] >= 0.5).astype(int)
        modal_profile = frozenset(label_names[i] for i, b in enumerate(modal_vec) if b)
        cluster_profiles = [profiles[i] for i in np.where(mask)[0]]
        n_match_modal = sum(1 for p in cluster_profiles if p == modal_profile)
        marg = {label_names[i]: float(((np.stack([profile_to_vector(p, label_names)
                                                      for p in cluster_profiles])[:, i]).mean()))
                for i in range(len(label_names))} if cluster_profiles else {}
        clusters.append({
            "cluster":           int(c),
            "size":              n_c,
            "fraction":          float(n_c / N),
            "modal_profile":     sorted(list(modal_profile)),
            "modal_label":       profile_to_label(modal_profile),
            "modal_purity":      float(n_match_modal / max(n_c, 1)),
            "per_pathology_pos_rate": marg,
        })
    clusters.sort(key=lambda d: -d["size"])

    return {
        "best_k_silhouette":   int(best_sil_k),
        "best_silhouette":     best_sil_value,
        "silhouette_caveat":   ("Silhouette is monotonically increasing on binary "
                                  "long-tail data and is NOT a reliable cluster-count "
                                  "test. Use gap-statistic best-K instead."),
        "best_k_gap":          int(best_gap_k),
        "best_gap":            best_gap_value,
        "k_used_for_clusters": int(chosen_k),
        "per_k":               {str(k): v for k, v in out_per_k.items()},
        "clusters":            clusters,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Single-label marginal analysis (Definition 3)
# ══════════════════════════════════════════════════════════════════════════════

def single_label_marginals(profiles, label_names):
    """Per-label positive rate + Shannon entropy of the binary marginal."""
    out = {}
    N = len(profiles)
    for lbl in label_names:
        n_pos = sum(1 for p in profiles if lbl in p)
        rate = n_pos / N
        # Shannon entropy of Bernoulli marginal (bits) — 0 if rate∈{0,1}, 1 if rate=0.5
        if rate <= 0 or rate >= 1:
            ent = 0.0
        else:
            ent = float(-(rate * np.log2(rate) + (1 - rate) * np.log2(1 - rate)))
        out[lbl] = {
            "positive_rate":          float(rate),
            "n_positive":             int(n_pos),
            "marginal_entropy_bits":  ent,
        }
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Mode purity vs GT
# ══════════════════════════════════════════════════════════════════════════════

def mode_purity_analysis(profiles_iter_K, gt_profiles, top_n=30):
    """For each top profile m at iter K, compute size / GT-prevalence / purity.

    Args:
        profiles_iter_K: list of frozensets, length N (iter-K profiles per study)
        gt_profiles:     list of frozensets, length N (GT profiles per study, same order)
        top_n:           number of top iter-K profiles to characterize

    Lift = GT_match_rate / GT_prevalence:
        ≈ 1   → mode is anchor-independent (chance-level GT match)
        >> 1  → mode preserves patient identity
        Combined with Inflation = (size_K / size_GT):
            high inflation + low lift → ELEVATOR MUSIC
            low inflation + high lift → PRESERVATION
    """
    N = len(profiles_iter_K)
    counter_K = Counter(profiles_iter_K)
    counter_GT = Counter(gt_profiles)
    top_K = counter_K.most_common(top_n)

    rows = []
    for prof, n_K in top_K:
        # size at iter K
        size_K = n_K
        size_GT = counter_GT.get(prof, 0)
        prev_K  = size_K / N
        prev_GT = size_GT / N
        # Within-mode GT match: of patients whose iter-K profile = prof, how many had GT == prof?
        n_match = sum(1 for i, p in enumerate(profiles_iter_K) if p == prof and gt_profiles[i] == prof)
        gt_match_rate = n_match / size_K  # P(GT==prof | iter-K==prof)
        # Lift relative to chance (GT_prevalence)
        lift = (gt_match_rate / prev_GT) if prev_GT > 0 else float("inf")
        # Inflation: how much over-represented at iter-K vs GT
        inflation = (prev_K / prev_GT) if prev_GT > 0 else float("inf")
        # Classify the mode
        if prev_GT < 1e-6:
            classification = "novel_mode (no GT prevalence; entirely loop-induced)"
        elif inflation >= 2.0 and lift < 2.0:
            classification = "elevator-music (inflated, anchor-independent)"
        elif lift >= 2.0 and inflation < 2.0:
            classification = "preserved (faithful to GT)"
        elif inflation >= 2.0 and lift >= 2.0:
            classification = "amplified-but-faithful"
        else:
            classification = "diffuse"
        rows.append({
            "profile":            sorted(list(prof)),
            "label":              profile_to_label(prof),
            "size_K":             int(size_K),
            "size_GT":            int(size_GT),
            "prev_K":             float(prev_K),
            "prev_GT":            float(prev_GT),
            "gt_match_rate":      float(gt_match_rate),
            "lift":               float(lift) if lift != float("inf") else None,
            "inflation":          float(inflation) if inflation != float("inf") else None,
            "classification":     classification,
        })
    return rows


# ══════════════════════════════════════════════════════════════════════════════
#  Distribution-level comparison (TV distance, KL div) between iter K and GT
# ══════════════════════════════════════════════════════════════════════════════

def distribution_distance(profiles_a, profiles_b, eps=1e-12):
    """TV and KL between profile distributions (over the union of observed profiles)."""
    N_a, N_b = len(profiles_a), len(profiles_b)
    c_a = Counter(profiles_a); c_b = Counter(profiles_b)
    keys = set(c_a) | set(c_b)
    p_a = np.array([c_a.get(k, 0) / N_a for k in keys])
    p_b = np.array([c_b.get(k, 0) / N_b for k in keys])
    tv = float(0.5 * np.abs(p_a - p_b).sum())
    # Symmetric KL (for stability)
    p_a_s = p_a + eps; p_a_s /= p_a_s.sum()
    p_b_s = p_b + eps; p_b_s /= p_b_s.sum()
    kl_ab = float((p_a_s * np.log(p_a_s / p_b_s)).sum())
    kl_ba = float((p_b_s * np.log(p_b_s / p_a_s)).sum())
    return {"tv_distance": tv, "kl_a_to_b": kl_ab, "kl_b_to_a": kl_ba,
            "symm_kl": (kl_ab + kl_ba) / 2.0}


# ══════════════════════════════════════════════════════════════════════════════
#  Figures
# ══════════════════════════════════════════════════════════════════════════════

def make_figures(results, out_dir, label_names, probe_iters):
    import matplotlib.pyplot as plt
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # ── Figure K1: Profile distribution at GT vs iter-0 vs iter-K (top 10 + other) ──
    K_max = max(probe_iters)
    # Three panels: GT, iter-0, iter-K_max
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, key, title in [
        (axes[0], "gt", "Ground truth (CSV labels)"),
        (axes[1], f"iter_{probe_iters[0]}", f"Iter {probe_iters[0]} (best generation)"),
        (axes[2], f"iter_{K_max}", f"Iter {K_max} (asymptotic)"),
    ]:
        if key not in results: continue
        top = results[key]["hard"]["top_profiles"][:10]
        labels  = [(EMPTY_DISPLAY_LABEL if t["label"] == "{}"
                     else (t["label"] if t["label"] else EMPTY_DISPLAY_LABEL))
                    for t in top]
        fracs   = [t["fraction"] for t in top]
        other = 1.0 - sum(fracs)
        labels.append(f"other ({results[key]['hard']['distinct_profiles'] - len(top)})")
        fracs.append(other)
        bars = ax.barh(range(len(labels))[::-1], fracs)
        ax.set_yticks(range(len(labels))[::-1])
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Fraction of cohort")
        ax.set_xlim(0, max(fracs) * 1.1)
        ent = results[key]["hard"]["entropy_bits"]
        dist = results[key]["hard"]["distinct_profiles"]
        plex = results[key]["hard"]["perplexity"]
        ax.set_title(f"{title}\n"
                       f"distinct={dist}, "
                       f"H={ent:.2f} bits, "
                       f"perplexity={plex:.1f}")
        for bar, f in zip(bars, fracs):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                     f"{f*100:.1f}%", va="center", fontsize=8)
    fig.suptitle("Block K — Profile distribution: GT vs iter-0 vs iter-K",
                  fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "K1_profile_distributions.pdf")); plt.close(fig)
    logger.info(f"  → {fig_dir}/K1_profile_distributions.pdf")

    # ── Figure K2: Mode-collapse trajectory across iters ──
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    iters = sorted(probe_iters)
    distinct_traj = [results[f"iter_{k}"]["hard"]["distinct_profiles"] for k in iters]
    entropy_traj  = [results[f"iter_{k}"]["hard"]["entropy_bits"] for k in iters]
    perplex_traj  = [results[f"iter_{k}"]["hard"]["perplexity"] for k in iters]
    top1_traj     = [results[f"iter_{k}"]["hard"]["top_n_coverage"]["top1"] for k in iters]
    top5_traj     = [results[f"iter_{k}"]["hard"]["top_n_coverage"]["top5"] for k in iters]
    top10_traj    = [results[f"iter_{k}"]["hard"]["top_n_coverage"]["top10"] for k in iters]

    axes[0].plot(iters, distinct_traj, "o-", color="#2962FF", lw=2)
    axes[0].axhline(results["gt"]["hard"]["distinct_profiles"], color="gray", ls="--",
                      lw=1, label="GT")
    axes[0].set_xlabel("Iteration $K$"); axes[0].set_ylabel("# distinct profiles")
    axes[0].set_title("Distinct profile count\n(↓ = mode collapse)")
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

    axes[1].plot(iters, entropy_traj, "o-", color="#388E3C", lw=2)
    axes[1].axhline(results["gt"]["hard"]["entropy_bits"], color="gray", ls="--",
                      lw=1, label="GT")
    axes[1].set_xlabel("Iteration $K$"); axes[1].set_ylabel("Profile entropy (bits)")
    axes[1].set_title("Profile entropy\n(↓ = mode collapse)")
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

    axes[2].plot(iters, perplex_traj, "o-", color="#E65100", lw=2)
    axes[2].axhline(results["gt"]["hard"]["perplexity"], color="gray", ls="--",
                      lw=1, label="GT")
    axes[2].set_xlabel("Iteration $K$"); axes[2].set_ylabel("Perplexity")
    axes[2].set_title("Effective # profiles\n(↓ = mode collapse)")
    axes[2].legend(fontsize=9); axes[2].grid(True, alpha=0.3)

    axes[3].plot(iters, top1_traj, "o-", color="#C62828", lw=2, label="top-1")
    axes[3].plot(iters, top5_traj, "o-", color="#6A1B9A", lw=2, label="top-5")
    axes[3].plot(iters, top10_traj, "o-", color="#00838F", lw=2, label="top-10")
    axes[3].set_xlabel("Iteration $K$"); axes[3].set_ylabel("Cumulative coverage")
    axes[3].set_title("Mode dominance\n(↑ = mode collapse)")
    axes[3].legend(fontsize=9); axes[3].grid(True, alpha=0.3); axes[3].set_ylim(0, 1.05)

    fig.suptitle("Block K — Mode-collapse trajectory across iterations",
                  fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "K2_mode_collapse_trajectory.pdf")); plt.close(fig)
    logger.info(f"  → {fig_dir}/K2_mode_collapse_trajectory.pdf")

    # ── Figure K3: Mode purity scatter at iter-K_max ──
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    purity_rows = results[f"iter_{K_max}"]["mode_purity"]
    for r in purity_rows:
        sz = r["size_K"]
        gtm = r["gt_match_rate"]
        infl = r.get("inflation")
        lift = r.get("lift")
        # Color/marker by classification
        if r["classification"].startswith("elevator-music"):
            color, marker = "#C62828", "o"
        elif r["classification"].startswith("preserved"):
            color, marker = "#2E7D32", "s"
        elif r["classification"].startswith("amplified-but-faithful"):
            color, marker = "#F57C00", "^"
        elif r["classification"].startswith("novel"):
            color, marker = "#6A1B9A", "D"
        else:
            color, marker = "#90A4AE", "x"
        if marker == "x":
            ax.scatter(sz, gtm, s=80, c=color, marker=marker, alpha=0.7,
                         linewidths=1.5)
        else:
            ax.scatter(sz, gtm, s=80, c=color, marker=marker, alpha=0.7,
                         edgecolors="black", linewidths=0.5)
        _annot_label = EMPTY_DISPLAY_LABEL if r["label"] == "{}" else r["label"]
        ax.annotate(_annot_label, xy=(sz, gtm), xytext=(5, 4),
                      textcoords="offset points", fontsize=7)
    ax.set_xscale("log"); ax.set_xlabel("Mode size at iter K (log scale)")
    ax.set_ylabel("Within-mode GT match rate")
    ax.set_title(f"Block K — Mode purity scatter at iter {K_max}")
    ax.grid(True, alpha=0.3)
    # Legend
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#C62828",
                  markeredgecolor="black", markersize=10, label="elevator-music"),
        Line2D([0],[0], marker="s", color="w", markerfacecolor="#2E7D32",
                  markeredgecolor="black", markersize=10, label="preserved"),
        Line2D([0],[0], marker="^", color="w", markerfacecolor="#F57C00",
                  markeredgecolor="black", markersize=10, label="amplified-but-faithful"),
        Line2D([0],[0], marker="D", color="w", markerfacecolor="#6A1B9A",
                  markeredgecolor="black", markersize=10, label="novel mode"),
        Line2D([0],[0], marker="x", color="#90A4AE", markeredgecolor="#90A4AE",
                  markersize=10, label="diffuse"),
    ], fontsize=9, loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "K3_mode_purity_scatter.pdf")); plt.close(fig)
    logger.info(f"  → {fig_dir}/K3_mode_purity_scatter.pdf")

    # ── Figure K4: Per-pathology marginal trajectory ──
    fig, ax = plt.subplots(1, 1, figsize=(13, 7))
    cmap = plt.get_cmap("tab20")
    for i, lbl in enumerate(label_names):
        rates = [results[f"iter_{k}"]["single_label"][lbl]["positive_rate"]
                  for k in iters]
        ax.plot(iters, rates, "o-", color=cmap(i), lw=1.7,
                  label=lbl, markersize=5)
        gt_rate = results["gt"]["single_label"][lbl]["positive_rate"]
        ax.axhline(gt_rate, color=cmap(i), ls=":", lw=1, alpha=0.5)
    ax.set_xlabel("Iteration $K$"); ax.set_ylabel("Positive rate at iter $K$")
    ax.set_title("Block K — Per-pathology marginal trajectory\n"
                  "(solid: iter-K rate; dotted: GT rate for same color)")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "K4_pathology_marginals.pdf")); plt.close(fig)
    logger.info(f"  → {fig_dir}/K4_pathology_marginals.pdf")

    # ── Figure K5: Soft-cluster cluster-count selection at iter-K_max ──
    if results[f"iter_{K_max}"].get("soft"):
        soft_K_max = results[f"iter_{K_max}"]["soft"]
        if soft_K_max.get("per_k"):
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            ks = sorted(int(k) for k in soft_K_max["per_k"])
            sils = [soft_K_max["per_k"][str(k)]["silhouette"] for k in ks]
            gaps = [soft_K_max["per_k"][str(k)]["gap"] for k in ks]
            gap_se = [soft_K_max["per_k"][str(k)]["gap_se"] for k in ks]

            # Silhouette panel — shown but with caveat
            axes[0].plot(ks, sils, "o-", color="#90A4AE", lw=2)
            best_sil_k = soft_K_max.get("best_k_silhouette")
            if best_sil_k:
                axes[0].axvline(best_sil_k, color="#90A4AE", ls="--", lw=1.2)
            axes[0].set_xlabel("# soft clusters $K_c$")
            axes[0].set_ylabel("Silhouette (Hamming)")
            axes[0].set_title("Silhouette — UNRELIABLE for binary mode-collapse data\n"
                                "(monotonically rises as long tail is split)")
            axes[0].grid(True, alpha=0.3)

            # Gap statistic panel — the rigorous metric
            gaps_arr = np.array(gaps); ses_arr = np.array(gap_se)
            axes[1].errorbar(ks, gaps, yerr=ses_arr, fmt="o-", color="#2962FF",
                              lw=2, capsize=4)
            best_gap_k = soft_K_max.get("best_k_gap")
            if best_gap_k:
                axes[1].axvline(best_gap_k, color="red", ls="--", lw=1.4,
                                  label=f"best K (Tibshirani rule) = {best_gap_k}")
            axes[1].set_xlabel("# soft clusters $K_c$")
            axes[1].set_ylabel("Gap statistic = $E[\\log W_{ref}] - \\log W_{obs}$")
            axes[1].set_title("Gap statistic — rigorous unsupervised cluster count\n"
                                "(maximum or first 'flattening' indicates true K)")
            axes[1].legend(); axes[1].grid(True, alpha=0.3)

            fig.suptitle(f"Block K — Soft-cluster cluster count at iter {K_max}",
                          fontsize=13, fontweight="bold")
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, "K5_soft_cluster_selection.pdf"))
            plt.close(fig)
            logger.info(f"  → {fig_dir}/K5_soft_cluster_selection.pdf")

    # ── Figure K6: 'No CheXpert label' class breakdown across iterations ──
    # Stacked bar: at each probe iter, what fraction of cohort is in {} class,
    # decomposed by primary category (explicit-normal / OOV-disease / ...).
    iters_with_empty = [k for k in probe_iters
                          if results[f"iter_{k}"].get("empty_breakdown")
                          and results[f"iter_{k}"]["empty_breakdown"]["n_total_empty"] > 0]
    if iters_with_empty:
        # Union of all primary categories observed across iters, ordered:
        #   explicit-normal first, OOV groups next (clinical-primacy order),
        #   short/other and unclassified last.
        seen_cats = set()
        for k in iters_with_empty:
            seen_cats.update(
                results[f"iter_{k}"]["empty_breakdown"]["primary_category_counts"].keys())
        ordered_cats = []
        if "explicit-normal (CheXpert miss)" in seen_cats:
            ordered_cats.append("explicit-normal (CheXpert miss)")
            seen_cats.discard("explicit-normal (CheXpert miss)")
        for c in OOV_DISPLAY_ORDER:
            if c in seen_cats:
                ordered_cats.append(c); seen_cats.discard(c)
        for tail in ["short/other", "other-unclassified", "truly-empty"]:
            if tail in seen_cats:
                ordered_cats.append(tail); seen_cats.discard(tail)
        ordered_cats.extend(sorted(seen_cats))  # any leftovers

        # Build matrix: rows = categories, cols = probe iters, values = fraction
        # of total cohort (NOT fraction of empty)
        data = np.zeros((len(ordered_cats), len(probe_iters)))
        for j, k in enumerate(probe_iters):
            br = results[f"iter_{k}"].get("empty_breakdown")
            if not br or br["n_total_empty"] == 0:
                continue
            n_total = br["n_total_studies"]
            for i, cat in enumerate(ordered_cats):
                count = br["primary_category_counts"].get(cat, 0)
                data[i, j] = count / n_total

        # Color palette: explicit-normal in green, OOV in oranges/reds, other in gray
        n_oov = sum(1 for c in ordered_cats if c in OOV_PATTERN_GROUPS)
        cmap_oov = plt.get_cmap("YlOrRd")
        colors = []
        oov_idx = 0
        for c in ordered_cats:
            if c == "explicit-normal (CheXpert miss)":
                colors.append("#2E7D32")
            elif c in OOV_PATTERN_GROUPS:
                colors.append(cmap_oov(0.25 + 0.65 * (oov_idx / max(1, n_oov - 1))))
                oov_idx += 1
            else:
                colors.append("#9E9E9E")

        fig, ax = plt.subplots(1, 1, figsize=(13, 6.5))
        bottoms = np.zeros(len(probe_iters))
        for i, cat in enumerate(ordered_cats):
            ax.bar(probe_iters, data[i], bottom=bottoms, color=colors[i],
                    edgecolor="white", linewidth=0.5, label=cat, width=0.7
                    if max(probe_iters) <= 11 else 4.0)
            bottoms += data[i]
        ax.set_xlabel("Iteration $K$")
        ax.set_ylabel(r"Fraction of cohort with profile = '" + EMPTY_DISPLAY_LABEL + "'")
        ax.set_title("Block K (supplementary) — '" + EMPTY_DISPLAY_LABEL +
                     "' class decomposition across iterations\n"
                     "(stacked: green=explicit-normal extractor misses, "
                     "warm=out-of-vocabulary clinical entities, gray=other)")
        ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "K6_empty_class_breakdown.pdf"))
        plt.close(fig)
        logger.info(f"  → {fig_dir}/K6_empty_class_breakdown.pdf")

    # ── Figure K7: Three-regime entropy dynamics with Lyapunov fit ──
    if results.get("lyapunov_three_regime"):
        lyap = results["lyapunov_three_regime"]
        iters = sorted(probe_iters)
        H_traj = [results[f"iter_{k}"]["hard"]["entropy_bits"] for k in iters]

        fig, ax = plt.subplots(1, 1, figsize=(11, 6))

        # Background shading: three regimes
        K_min = lyap["K_min"]
        if lyap["classification"] == "bouncing-attractor":
            ax.axvspan(0, K_min, color="#FFE0B2", alpha=0.4,
                        label="Regime 1: fast contraction (λ_sys < 0)")
            ax.axvspan(K_min, max(iters), color="#C8E6C9", alpha=0.4,
                        label=r"Regime 2: macro-mixing recovery ($\bar\lambda_a > 0$)")

        # Data points
        ax.plot(iters, H_traj, "o-", color="#388E3C", lw=2.2, markersize=8,
                  label=f"Observed entropy H(K)", zorder=5)

        # GT reference
        gt_H = results["gt"]["hard"]["entropy_bits"]
        ax.axhline(gt_H, color="gray", ls="--", lw=1, alpha=0.7,
                     label=f"GT entropy = {gt_H:.2f} bits")

        # Mark K_min
        ax.axvline(K_min, color="red", ls=":", lw=1.5, alpha=0.7)
        ax.annotate(f"K_min = {int(K_min)}\nH_min = {lyap['H_min']:.2f} bits",
                      xy=(K_min, lyap["H_min"]),
                      xytext=(K_min + 0.05 * max(iters), lyap["H_min"] + 0.15),
                      fontsize=10, color="red",
                      arrowprops=dict(arrowstyle="->", color="red", lw=1))

        # Fit overlay (only if well-identified — otherwise CI swamps the figure)
        if lyap["fit_succeeded"] and lyap.get("is_well_identified", False):
            K_dense = np.linspace(K_min, max(iters), 200)
            H_inf = lyap["H_inf_fit"]; tau = lyap["tau_macro_fit"]
            H_min_fit = lyap["H_min"]
            H_fit = H_inf + (H_min_fit - H_inf) * np.exp(-(K_dense - K_min) / tau)
            ax.plot(K_dense, H_fit, "--", color="#1565C0", lw=2,
                      label=f"Fit: H(K) = {H_inf:.2f} + ({H_min_fit:.2f}−{H_inf:.2f})·"
                            r"$\exp(-(K-K_{min})/\tau_{macro})$")
            ax.axhline(H_inf, color="#1565C0", ls=":", lw=1, alpha=0.5)
            # τ_macro annotation
            ci_str = ""
            if lyap.get("tau_macro_ci_lower") is not None:
                ci_str = (f" (95% CI [{lyap['tau_macro_ci_lower']:.1f}, "
                          f"{lyap['tau_macro_ci_upper']:.1f}])")
            ax.text(0.98, 0.97,
                     f"τ_macro = {tau:.1f} iters{ci_str}\n"
                     f"H_∞ = {H_inf:.2f} bits\n"
                     f"Recovery amplitude = {lyap['recovery_amplitude']:.2f} bits",
                     transform=ax.transAxes, ha="right", va="top",
                     fontsize=10, family="monospace",
                     bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                               edgecolor="#1565C0", linewidth=1.2))
        elif lyap["fit_succeeded"]:
            # Fit converged but is poorly identified — note this on the figure
            ax.text(0.98, 0.97,
                     "τ_macro fit poorly identified at this N\n"
                     f"(point estimate {lyap['tau_macro_fit']:.1f}, "
                     f"CI ratio {lyap['tau_macro_ci_upper']/max(0.01, lyap['tau_macro_ci_lower']):.1f}×)\n"
                     f"Recovery amplitude = {lyap['recovery_amplitude']:.2f} bits\n"
                     "Lock τ_macro at full N=1081.",
                     transform=ax.transAxes, ha="right", va="top",
                     fontsize=9.5, family="monospace",
                     bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF9C4",
                               edgecolor="#F57F17", linewidth=1.2))

        ax.set_xlabel("Iteration $K$")
        ax.set_ylabel("Profile entropy H(K) [bits]")
        ax.set_title(f"Block K — Three-regime entropy dynamics: {lyap['classification']}\n"
                     f"(Lyapunov framework predicts non-monotonic H(K) when "
                     r"$\lambda_{sys} < 0 < \bar\lambda_a$)")
        ax.legend(fontsize=9, loc="lower right" if lyap["fit_succeeded"] else "best")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "K7_lyapunov_three_regime.pdf"))
        plt.close(fig)
        logger.info(f"  → {fig_dir}/K7_lyapunov_three_regime.pdf")
        if not lyap["fit_succeeded"]:
            logger.info(f"      (no τ_macro fit overlay: {lyap.get('fit_error')})")


def write_top_profiles_tables(results, out_dir, probe_iters):
    """Per-iter TSV of top profiles (for paper appendix)."""
    table_dir = os.path.join(out_dir, "tables")
    os.makedirs(table_dir, exist_ok=True)
    for key, label in [("gt", "GT")] + [(f"iter_{k}", f"iter_{k}") for k in probe_iters]:
        if key not in results: continue
        top = results[key]["hard"]["top_profiles"]
        rows = []
        for t in top:
            disp = (EMPTY_DISPLAY_LABEL if t["label"] == "{}"
                     else profile_to_label(frozenset(t["profile"]), max_chars=120))
            rows.append({
                "rank":          t["rank"],
                "size":          t["size"],
                "fraction":      f"{t['fraction']:.4f}",
                "n_pathologies": t["n_pathologies"],
                "profile":       disp,
            })
        df = pd.DataFrame(rows)
        path = os.path.join(table_dir, f"K_top_profiles_{label}.tsv")
        df.to_csv(path, sep="\t", index=False)
    logger.info(f"  Top-profile tables → {table_dir}/")


def write_empty_breakdown_tables(results, out_dir, probe_iters):
    """Per-iter TSV of {} class breakdown (counts, fractions, OOV any-match,
    plus a separate examples TSV for paper appendix)."""
    table_dir = os.path.join(out_dir, "tables")
    os.makedirs(table_dir, exist_ok=True)

    # Summary table: rows = (iter, category), columns = count, frac_of_empty,
    # frac_of_cohort
    summary_rows = []
    for k in probe_iters:
        br = results[f"iter_{k}"].get("empty_breakdown")
        if not br or br["n_total_empty"] == 0:
            continue
        n_emp = br["n_total_empty"]; n_tot = br["n_total_studies"]
        for cat, cnt in sorted(br["primary_category_counts"].items(),
                                  key=lambda kv: -kv[1]):
            summary_rows.append({
                "iter":             k,
                "category":         cat,
                "count":            cnt,
                "frac_of_empty":    f"{cnt/n_emp:.4f}",
                "frac_of_cohort":   f"{cnt/n_tot:.4f}",
            })
    if summary_rows:
        df_sum = pd.DataFrame(summary_rows)
        df_sum.to_csv(os.path.join(table_dir, "K_empty_class_breakdown.tsv"),
                       sep="\t", index=False)

    # OOV any-match counts (one report can be in multiple categories)
    oov_rows = []
    for k in probe_iters:
        br = results[f"iter_{k}"].get("empty_breakdown")
        if not br or br["n_total_empty"] == 0:
            continue
        for cat, cnt in sorted(br["oov_category_counts"].items(),
                                  key=lambda kv: -kv[1]):
            oov_rows.append({
                "iter":         k,
                "oov_category": cat,
                "any_match_count": cnt,
                "frac_of_empty":   f"{cnt/br['n_total_empty']:.4f}",
                "frac_of_cohort":  f"{cnt/br['n_total_studies']:.4f}",
            })
    if oov_rows:
        pd.DataFrame(oov_rows).to_csv(
            os.path.join(table_dir, "K_empty_oov_any_match.tsv"),
            sep="\t", index=False)

    # Example reports (for appendix illustration)
    ex_rows = []
    for k in probe_iters:
        br = results[f"iter_{k}"].get("empty_breakdown")
        if not br: continue
        for cat, examples in br.get("examples", {}).items():
            for ex in examples:
                ex_rows.append({
                    "iter":     k,
                    "category": cat,
                    "study_id": ex["sid"],
                    "n_chars":  len(ex["text"]),
                    "text":     ex["text"].replace("\t", " ").replace("\n", " "),
                })
    if ex_rows:
        pd.DataFrame(ex_rows).to_csv(
            os.path.join(table_dir, "K_empty_examples.tsv"),
            sep="\t", index=False)
        logger.info(f"  Empty-class breakdown tables → {table_dir}/"
                    "K_empty_*.tsv")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                  description=__doc__)
    p.add_argument("--trajectory_dir", required=True,
                    help="chexgen_main (K=11) or chexgen_long (K=101).")
    p.add_argument("--data_csv",
                    default="/n/groups/training/bmif203/AIM2/processed_data/processed_data.csv")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--probe_iters", default=None,
                    help="Comma-separated list of iters. Default: auto by K_max.")
    p.add_argument("--use_chexpert", choices=["auto", "extractor", "rulebased", "none"],
                    default="auto")
    p.add_argument("--max_studies", type=int, default=-1)
    p.add_argument("--soft_K_range", default="2,25",
                    help="Comma-separated k_min,k_max for soft k-means (default 2,25). "
                         "Wider range than v1 to detect saturation; gap statistic is "
                         "the primary cluster-count metric (silhouette is unreliable "
                         "for binary mode-collapse data).")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("AIM2 Block K — Generated-pathology mode structure")
    logger.info("=" * 60)
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")

    # ── Determine probe iters ────────────────────────────────────────────────
    if args.probe_iters:
        probe_iters = [int(x) for x in args.probe_iters.split(",")]
    else:
        # Quick peek at K_max in dir
        any_sd = next((d for d in os.listdir(args.trajectory_dir)
                        if os.path.isdir(os.path.join(args.trajectory_dir, d))
                        and d.isdigit()), None)
        K_peek = -1
        if any_sd:
            f_files = sorted(glob(os.path.join(args.trajectory_dir, any_sd,
                                                  "findings_iter_*.txt")))
            if f_files:
                K_peek = max(int(os.path.basename(f).split("_")[-1].split(".")[0])
                              for f in f_files)
        if K_peek <= 11:
            probe_iters = [0, 1, 2, 5, 10]
        else:
            probe_iters = [0, 1, 5, 10, 20, 30, 50, 70, 100]
        logger.info(f"  Auto-selected probe_iters by K_max={K_peek}: {probe_iters}")

    # ── Load CheXpert extractor + GT labels ──────────────────────────────────
    chx_fn, chx_name = get_chexpert_extractor(args.use_chexpert)
    logger.info(f"  CheXpert extractor: {chx_name}")
    if chx_fn is None:
        raise RuntimeError("CheXpert extractor not available; cannot run Block K")
    gt_map, available_labels = load_gt_labels(args.data_csv)
    logger.info(f"  GT labels: {len(gt_map)} studies, "
                f"{len(available_labels)} CheXpert columns")

    # ── Load trajectories ────────────────────────────────────────────────────
    sids, findings, K_max_seen = load_trajectory_findings(
        args.trajectory_dir, probe_iters, max_studies=args.max_studies)
    if not sids:
        raise RuntimeError("No valid trajectories found")
    # Filter to studies with GT labels too
    sids = [s for s in sids if s in gt_map]
    for k in findings:
        findings[k] = {s: findings[k][s] for s in sids}
    logger.info(f"  N = {len(sids)} studies with GT + all probe iters")

    # GT profiles aligned to sids
    gt_profiles = [gt_map[s] for s in sids]
    # Per-iter generated profiles aligned to sids
    iter_profiles = {}
    for k in probe_iters:
        logger.info(f"  Extracting profiles at iter {k}...")
        iter_profiles[k] = [chx_fn(findings[k][s]) for s in sids]

    # ── Run analyses ─────────────────────────────────────────────────────────
    results = {
        "args":              vars(args),
        "n_studies":         len(sids),
        "probe_iters":       probe_iters,
        "label_names":       available_labels,
        "chx_extractor":     chx_name,
    }

    # GT analysis
    logger.info("\n  Block K analysis: GT distribution")
    results["gt"] = {
        "hard":         hard_profile_stats(gt_profiles),
        "single_label": single_label_marginals(gt_profiles, available_labels),
    }
    g = results["gt"]["hard"]
    logger.info(f"    GT: distinct={g['distinct_profiles']}, "
                f"top1={g['top_n_coverage']['top1']:.3f}, "
                f"top5={g['top_n_coverage']['top5']:.3f}, "
                f"H={g['entropy_bits']:.2f} bits, perplexity={g['perplexity']:.1f}")

    K_min, K_max = [int(x) for x in args.soft_K_range.split(",")]
    for k in probe_iters:
        logger.info(f"\n  Block K analysis: iter {k}")
        prof_k = iter_profiles[k]
        rk = {
            "hard":         hard_profile_stats(prof_k),
            "single_label": single_label_marginals(prof_k, available_labels),
            "mode_purity":  mode_purity_analysis(prof_k, gt_profiles, top_n=30),
            "vs_gt":        distribution_distance(prof_k, gt_profiles),
        }
        # Soft clustering only for the most informative iters (slow at 1081)
        if k in [0, probe_iters[-1]] or len(probe_iters) <= 5:
            rk["soft"] = soft_cluster_stats(prof_k, available_labels,
                                              K_range=(K_min, K_max))
        else:
            rk["soft"] = None
        results[f"iter_{k}"] = rk

        h = rk["hard"]
        logger.info(f"    iter {k}: distinct={h['distinct_profiles']}, "
                    f"top1={h['top_n_coverage']['top1']:.3f}, "
                    f"top5={h['top_n_coverage']['top5']:.3f}, "
                    f"top10={h['top_n_coverage']['top10']:.3f}")
        logger.info(f"    iter {k}: H={h['entropy_bits']:.2f} bits, "
                    f"perplexity={h['perplexity']:.1f}, "
                    f"empty={h['fraction_empty']:.2%}, "
                    f"single-label={h['fraction_single_label']:.2%}, "
                    f"multi-label={h['fraction_multi_label']:.2%}")
        logger.info(f"    iter {k}: TV(iter,GT)={rk['vs_gt']['tv_distance']:.3f}, "
                    f"symm-KL={rk['vs_gt']['symm_kl']:.3f}")

        # Quick mode-purity summary
        em_modes = [r for r in rk["mode_purity"] if r["classification"].startswith("elevator-music")]
        pres_modes = [r for r in rk["mode_purity"] if r["classification"].startswith("preserved")]
        novel_modes = [r for r in rk["mode_purity"] if r["classification"].startswith("novel")]
        logger.info(f"    iter {k}: elevator-music modes: {len(em_modes)} "
                    f"(top: {em_modes[0]['label'] if em_modes else 'none'})")
        logger.info(f"    iter {k}: preserved modes:      {len(pres_modes)}")
        logger.info(f"    iter {k}: novel modes:          {len(novel_modes)}")

        # ── '{}' / 'No CheXpert label' class breakdown ──
        # Categorize each empty-profile report into explicit-normal vs OOV
        # vs short/other. Reports the heterogeneity of the {} class
        # (essential for honest headline interpretation).
        rk["empty_breakdown"] = analyze_empty_profiles_at_iter(
            prof_k, findings[k], sids, n_examples_per_category=5, seed=42)
        eb = rk["empty_breakdown"]
        if eb["n_total_empty"] > 0:
            top_cats = sorted(eb["primary_category_counts"].items(),
                                key=lambda kv: -kv[1])[:3]
            top_cats_str = ", ".join(f"{c}={n}" for c, n in top_cats)
            logger.info(f"    iter {k}: {{}} class N={eb['n_total_empty']} "
                          f"({eb['fraction_empty']:.1%}); top categories: {top_cats_str}")

    # Iter-0 vs iter-K_max comparison
    K_max_iter = max(probe_iters)
    if 0 in probe_iters:
        results["iter0_vs_iterK"] = distribution_distance(
            iter_profiles[0], iter_profiles[K_max_iter])
        logger.info(f"\n  iter-0 vs iter-{K_max_iter}: "
                    f"TV={results['iter0_vs_iterK']['tv_distance']:.3f}, "
                    f"symm-KL={results['iter0_vs_iterK']['symm_kl']:.3f}")

    # ── Lyapunov three-regime entropy fit ────────────────────────────────────
    # Predicted by Block B: contraction (λ_sys < 0) → entropy drop, then
    # divergence (λ̄_a > 0) → entropy recovery. Provides τ_macro point estimate.
    if len(probe_iters) >= 4:
        H_traj = [results[f"iter_{k}"]["hard"]["entropy_bits"]
                   for k in sorted(probe_iters)]
        results["lyapunov_three_regime"] = fit_lyapunov_three_regime(
            sorted(probe_iters), H_traj)
        ly = results["lyapunov_three_regime"]
        logger.info(f"\n  Lyapunov three-regime fit: classification = "
                    f"{ly['classification']}")
        logger.info(f"    K_min = {int(ly['K_min'])}, "
                    f"H_min = {ly['H_min']:.3f} bits, "
                    f"H_K0 = {ly['H_K0']:.3f}, "
                    f"H_Kmax = {ly['H_Kmax']:.3f}")
        logger.info(f"    Initial entropy drop = {ly['initial_entropy_drop']:.3f} bits, "
                    f"recovery amplitude = {ly['recovery_amplitude']:.3f} bits")
        if ly["fit_succeeded"]:
            ci = (f" [{ly['tau_macro_ci_lower']:.1f}, {ly['tau_macro_ci_upper']:.1f}]"
                   if ly['tau_macro_ci_lower'] is not None else "")
            logger.info(f"    τ_macro = {ly['tau_macro_fit']:.2f} iters{ci}, "
                        f"H_∞ = {ly['H_inf_fit']:.3f} bits")
        else:
            logger.info(f"    (no fit: {ly['fit_error']})")
    else:
        logger.info(f"\n  Lyapunov three-regime fit skipped: need ≥4 probe iters, "
                    f"got {len(probe_iters)}")

    # ── Save results JSON ────────────────────────────────────────────────────
    out_json = os.path.join(args.out_dir, "block_K_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2,
                   default=lambda o: float(o) if isinstance(o, np.floating) else
                                      list(o) if isinstance(o, frozenset) else
                                      str(o))
    logger.info(f"\nResults → {out_json}")

    # ── Figures + tables ─────────────────────────────────────────────────────
    make_figures(results, args.out_dir, available_labels, probe_iters)
    write_top_profiles_tables(results, args.out_dir, probe_iters)
    write_empty_breakdown_tables(results, args.out_dir, probe_iters)

    # ── Final headline summary ───────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("BLOCK K HEADLINE SUMMARY")
    logger.info("=" * 60)
    g = results["gt"]["hard"]
    h0 = results[f"iter_{probe_iters[0]}"]["hard"]
    hK = results[f"iter_{K_max_iter}"]["hard"]
    logger.info(f"  Distinct profiles:  GT={g['distinct_profiles']}, "
                f"iter-{probe_iters[0]}={h0['distinct_profiles']}, "
                f"iter-{K_max_iter}={hK['distinct_profiles']}")
    logger.info(f"  Top-1 coverage:     GT={g['top_n_coverage']['top1']:.3f}, "
                f"iter-{probe_iters[0]}={h0['top_n_coverage']['top1']:.3f}, "
                f"iter-{K_max_iter}={hK['top_n_coverage']['top1']:.3f}")
    logger.info(f"  Top-5 coverage:     GT={g['top_n_coverage']['top5']:.3f}, "
                f"iter-{probe_iters[0]}={h0['top_n_coverage']['top5']:.3f}, "
                f"iter-{K_max_iter}={hK['top_n_coverage']['top5']:.3f}")
    logger.info(f"  Profile entropy:    GT={g['entropy_bits']:.2f} bits, "
                f"iter-{probe_iters[0]}={h0['entropy_bits']:.2f}, "
                f"iter-{K_max_iter}={hK['entropy_bits']:.2f}")
    logger.info(f"  Perplexity:         GT={g['perplexity']:.1f}, "
                f"iter-{probe_iters[0]}={h0['perplexity']:.1f}, "
                f"iter-{K_max_iter}={hK['perplexity']:.1f}")

    # {} class headline at the asymptotic iter
    eb_K = results[f"iter_{K_max_iter}"].get("empty_breakdown")
    if eb_K and eb_K["n_total_empty"] > 0:
        logger.info(f"  '{EMPTY_DISPLAY_LABEL}' class at iter {K_max_iter}: "
                      f"N={eb_K['n_total_empty']} ({eb_K['fraction_empty']:.1%})")
        for cat, cnt in sorted(eb_K["primary_category_counts"].items(),
                                  key=lambda kv: -kv[1])[:5]:
            logger.info(f"    {cat:50s}  n={cnt:4d}  "
                          f"({cnt/eb_K['n_total_empty']:.1%} of empty, "
                          f"{cnt/eb_K['n_total_studies']:.1%} of cohort)")

    # Lyapunov three-regime headline
    ly = results.get("lyapunov_three_regime")
    if ly:
        logger.info(f"  Lyapunov three-regime: {ly['classification']}, "
                      f"K_min={int(ly['K_min'])}, H_min={ly['H_min']:.2f} bits, "
                      f"recovery={ly['recovery_amplitude']:.2f} bits")
        if ly["fit_succeeded"]:
            ci = (f" [{ly['tau_macro_ci_lower']:.1f}, {ly['tau_macro_ci_upper']:.1f}]"
                   if ly['tau_macro_ci_lower'] is not None else "")
            logger.info(f"    τ_macro = {ly['tau_macro_fit']:.1f} iters{ci}, "
                          f"H_∞ = {ly['H_inf_fit']:.2f} bits")

    logger.info(f"\nAll outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()