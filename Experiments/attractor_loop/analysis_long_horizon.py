#!/usr/bin/env python3
"""
AIM2 — Long-horizon analysis (K=100 N=48).

Runs a subset of the standard analysis blocks against the chexgen_long
trajectories, parameterized by K (iteration depth at which to "stop").
Asks: which findings from the K=11 main run hold asymptotically, which
saturate between K=11 and K=100, and which (if any) qualitatively
change?

Probes K ∈ {1..10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 45, 50, 60, 70,
80, 90, 100}.

Blocks run:
  A  — trajectory geometry (anchor dist, step size, tortuosity, modal coupling)
  C  — cluster structure (silhouette + gap stats; HDBSCAN skipped at N=48)
  E  — high-D structure (PR, rank-95; MI skipped, KSG broken at N=48)
  G  — surface-form fidelity (BLEU, CheXpert recall, report length, per-pathology survival)
  H  — training-cluster dynamics (entropy, dominant-fraction, top-mass)
  I  — geometric mixing (kNN-to-training, MIPD, displacement alignment)
  J  — autocorrelation + kNN persistence vs LAG (the headline result)

Blocks NOT run:
  B  — Lyapunov (handled separately by 20-anchor × 10-seed dedicated run)
  D  — phase portraits (visualization only)
  F  — basin × pathology test (no meaningful basins at N=48)

Outputs to /n/groups/training/bmif203/AIM2/Experiments/attractor_loop/analysis_long/
  long_horizon_results.json  — all numerical results per block per K
  figures/*.pdf               — per-block summary figures
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from glob import glob

import numpy as np
from PIL import Image

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ─── K-probing schedule ───────────────────────────────────────────────────────
PROBE_KS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            12, 14, 16, 18, 20,
            25, 30, 35, 40, 45, 50,
            60, 70, 80, 90, 100]

# ─── Lag schedule for Block J (autocorrelation, persistence) ──────────────────
LAGS_FOR_BLOCK_J = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                     12, 14, 16, 18, 20, 25, 30, 35, 40, 45, 50]


# ══════════════════════════════════════════════════════════════════════════════
#  Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_long_trajectories(long_dir, K_max=100, min_iters_required=100):
    """Load all completed long-horizon trajectories.

    Returns:
        Z_img:    (N, K_max+1, 256) float64
        Z_txt:    (N, K_max+1, 256) float64
        A_img:    (N, 256)          — iter-0 anchor image embeddings
        A_txt:    (N, 256)          — iter-0 anchor text embeddings
        sids:     list[str]
        findings: list[list[str]]   — outer over study, inner K_max+1 strings
        gt_findings: list[str]      — ground-truth findings text per study
                                      (used as BLEU reference in Block G)
    """
    sids_all = sorted([d for d in os.listdir(long_dir)
                       if os.path.isdir(os.path.join(long_dir, d)) and d.isdigit()])
    logger.info(f"  Found {len(sids_all)} candidate study directories")

    valid = []
    Z_img_list, Z_txt_list, findings_list, gt_findings_list = [], [], [], []
    A_img_list, A_txt_list = [], []
    for sid in sids_all:
        sd = os.path.join(long_dir, sid)
        anchor_img_p = os.path.join(sd, "anchor_img_embed.npy")
        anchor_txt_p = os.path.join(sd, "anchor_text_embed.npy")
        gt_findings_p = os.path.join(sd, "gt_findings.txt")
        if not (os.path.exists(anchor_img_p) and os.path.exists(anchor_txt_p)):
            logger.warning(f"  [{sid}] missing anchor; skipping")
            continue
        img_files = sorted(glob(os.path.join(sd, "img_embed_iter_*.npy")))
        txt_files = sorted(glob(os.path.join(sd, "text_embed_iter_*.npy")))
        f_files   = sorted(glob(os.path.join(sd, "findings_iter_*.txt")))
        if len(img_files) < min_iters_required + 1:
            logger.warning(f"  [{sid}] only {len(img_files)} img iters; skipping")
            continue
        if len(txt_files) < min_iters_required + 1:
            logger.warning(f"  [{sid}] only {len(txt_files)} txt iters; skipping")
            continue
        if len(f_files) < min_iters_required + 1:
            logger.warning(f"  [{sid}] only {len(f_files)} findings; skipping")
            continue

        img_seq = np.stack([np.load(os.path.join(sd, f"img_embed_iter_{k:03d}.npy"))
                            for k in range(K_max + 1)], axis=0)
        txt_seq = np.stack([np.load(os.path.join(sd, f"text_embed_iter_{k:03d}.npy"))
                            for k in range(K_max + 1)], axis=0)
        findings_seq = []
        for k in range(K_max + 1):
            with open(os.path.join(sd, f"findings_iter_{k:03d}.txt")) as f:
                findings_seq.append(f.read())
        if os.path.exists(gt_findings_p):
            with open(gt_findings_p) as f:
                gt_text = f.read()
        else:
            logger.warning(f"  [{sid}] missing gt_findings.txt; using iter-0 findings as fallback")
            gt_text = findings_seq[0]

        Z_img_list.append(img_seq)
        Z_txt_list.append(txt_seq)
        findings_list.append(findings_seq)
        gt_findings_list.append(gt_text)
        A_img_list.append(np.load(anchor_img_p))
        A_txt_list.append(np.load(anchor_txt_p))
        valid.append(sid)

    if not valid:
        raise RuntimeError(f"No valid trajectories found in {long_dir}")

    Z_img = np.stack(Z_img_list, axis=0).astype(np.float64)
    Z_txt = np.stack(Z_txt_list, axis=0).astype(np.float64)
    A_img = np.stack(A_img_list, axis=0).astype(np.float64)
    A_txt = np.stack(A_txt_list, axis=0).astype(np.float64)
    logger.info(f"  Loaded N={len(valid)}, K_max+1={Z_img.shape[1]}, "
                f"D={Z_img.shape[2]}")
    return Z_img, Z_txt, A_img, A_txt, valid, findings_list, gt_findings_list


def load_reference(ref_dir):
    """Load 256-d training-distribution embeddings + UMAP for blocks H and I."""
    img_p = os.path.join(ref_dir, "img_embeds.npy")
    txt_p = os.path.join(ref_dir, "txt_embeds.npy")
    ref_img = np.load(img_p) if os.path.exists(img_p) else None
    ref_txt = np.load(txt_p) if os.path.exists(txt_p) else None
    return ref_img, ref_txt


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def normalize_rows(X):
    n = np.linalg.norm(X, axis=-1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def cohort_centroid(Z_iter):
    """Cohort mean of L2-normalized embeddings, then re-normalized."""
    c = Z_iter.mean(axis=0)
    return c / np.maximum(np.linalg.norm(c), 1e-12)


def participation_ratio(X):
    """PR = (Σλ)² / Σλ² for cov eigenvalues. Effective dimensionality."""
    Xc = X - X.mean(axis=0, keepdims=True)
    cov = Xc.T @ Xc / max(len(X) - 1, 1)
    evals = np.linalg.eigvalsh(cov)
    evals = np.maximum(evals, 0.0)
    s = evals.sum()
    s2 = (evals ** 2).sum()
    return float(s ** 2 / max(s2, 1e-12))


def rank_95(Z_traj_centered):
    """Number of SVs needed to capture 95% of variance in the trajectory tensor.
    Z_traj_centered: (N, K, D) per-iter-centered → flatten to (N, K*D)."""
    N, K, D = Z_traj_centered.shape
    flat = Z_traj_centered.reshape(N, K * D)
    sv = np.linalg.svd(flat, compute_uv=False)
    var = sv ** 2
    cumvar = np.cumsum(var) / var.sum()
    return int(np.searchsorted(cumvar, 0.95) + 1)


# ══════════════════════════════════════════════════════════════════════════════
#  Block A — trajectory geometry
# ══════════════════════════════════════════════════════════════════════════════

def block_A_long(Z_img, Z_txt, A_img, A_txt, probe_ks):
    """At each probe K, compute per-trajectory geometric statistics treating
    iter K as the endpoint:
      • anchor distance d_K = ||z_K - z_0||  (cohort mean ± SD)
      • step size Δ_K = ||z_K - z_{K-1}||    (cohort mean)
      • path length L_K = Σ_{k<K} ||z_{k+1}-z_k||
      • net displacement D_K = ||z_K - z_0||
      • tortuosity T_K = L_K / D_K
      • modal coupling ρ_K = corr(image-side d_k, text-side d_k) over k≤K
    """
    logger.info("Block A — trajectory geometry across K")
    N = Z_img.shape[0]
    out = {}

    for modality, Z in [("image", Z_img), ("text", Z_txt)]:
        per_K = {}
        for K in probe_ks:
            # Anchor distance
            d_K = np.linalg.norm(Z[:, K, :] - Z[:, 0, :], axis=1)   # (N,)
            # Step size at iter K (i.e., from K-1 to K)
            step_K = np.linalg.norm(Z[:, K, :] - Z[:, K-1, :], axis=1)
            # Path length 0..K, net 0..K, tortuosity
            steps_all = np.linalg.norm(Z[:, 1:K+1, :] - Z[:, :K, :], axis=2)  # (N, K)
            path_len = steps_all.sum(axis=1)
            net_disp = d_K
            tortuosity = path_len / np.maximum(net_disp, 1e-9)

            per_K[str(K)] = {
                "d_K_mean": float(d_K.mean()),
                "d_K_std":  float(d_K.std(ddof=1)),
                "step_K_mean": float(step_K.mean()),
                "step_K_std":  float(step_K.std(ddof=1)),
                # Tortuosity: mean is sensitive to anchor-level L/D blow-ups
                # (when net_disp ≈ 0). Median + IQR + p10/p90 are robust to
                # these outliers and recover the true linear-growth signal.
                "tortuosity_mean":   float(tortuosity.mean()),
                "tortuosity_std":    float(tortuosity.std(ddof=1)),
                "tortuosity_median": float(np.median(tortuosity)),
                "tortuosity_p25":    float(np.percentile(tortuosity, 25)),
                "tortuosity_p75":    float(np.percentile(tortuosity, 75)),
                "tortuosity_p10":    float(np.percentile(tortuosity, 10)),
                "tortuosity_p90":    float(np.percentile(tortuosity, 90)),
                "tortuosity_n_outliers_5x": int(
                    np.sum(tortuosity > 5 * np.median(tortuosity))
                ),
            }
        out[modality] = per_K
        logger.info(f"  {modality}: d_K(K=10) = {out[modality]['10']['d_K_mean']:.3f},  "
                    f"d_K(K=100) = {out[modality]['100']['d_K_mean']:.3f},  "
                    f"step(K=100) = {out[modality]['100']['step_K_mean']:.3f}")

    # Modal coupling: correlation of d_k between image and text over each
    # trajectory's k=1..K_max, summarized per K
    per_K_coupling = {}
    K_max = Z_img.shape[1] - 1
    d_img_all = np.linalg.norm(Z_img - Z_img[:, :1, :], axis=2)   # (N, K_max+1)
    d_txt_all = np.linalg.norm(Z_txt - Z_txt[:, :1, :], axis=2)
    for K in probe_ks:
        if K < 2:
            per_K_coupling[str(K)] = float("nan")
            continue
        # Per-trajectory Pearson correlation of d_img(k) vs d_txt(k) for k in 1..K
        rs = []
        for i in range(N):
            x = d_img_all[i, 1:K+1]
            y = d_txt_all[i, 1:K+1]
            if x.std() < 1e-9 or y.std() < 1e-9:
                continue
            rs.append(float(np.corrcoef(x, y)[0, 1]))
        per_K_coupling[str(K)] = float(np.mean(rs)) if rs else float("nan")
    out["modal_coupling"] = per_K_coupling
    logger.info(f"  modal coupling ρ at K=100: {out['modal_coupling']['100']:.3f}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Block C — cluster structure (silhouette + gap; skip HDBSCAN)
# ══════════════════════════════════════════════════════════════════════════════

def block_C_long(Z_img, Z_txt, probe_ks, K_range=(2, 8), n_gap_refs=20, rng_seed=0):
    """At each probe K, compute silhouette and gap statistic on raw 256-d
    cosine for K_clust in K_range. Reports the silhouette-best K_clust.
    HDBSCAN omitted (unreliable at N=48)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from scipy.spatial.distance import pdist
    logger.info("Block C — cluster structure across K (raw 256-d cosine)")

    out = {}
    rng = np.random.default_rng(rng_seed)
    for modality, Z in [("image", Z_img), ("text", Z_txt)]:
        per_K = {}
        for K in probe_ks:
            Xn = normalize_rows(Z[:, K, :])
            sil_per_kc = {}
            best_sil = -2.0
            best_kc = None
            for kc in range(K_range[0], K_range[1] + 1):
                km = KMeans(n_clusters=kc, random_state=rng_seed,
                              n_init=10).fit(Xn)
                # silhouette in cosine ≡ silhouette in euclidean on L2-normed data
                sil = float(silhouette_score(Xn, km.labels_,
                                                metric="cosine"))
                sil_per_kc[str(kc)] = sil
                if sil > best_sil:
                    best_sil, best_kc = sil, kc

            # Gap statistic: log Wk_obs vs E[log Wk_uniform-on-PCA-bbox]
            # We compare against null clouds matched in mean & cov for simplicity.
            xc = Xn - Xn.mean(axis=0, keepdims=True)
            # within-cluster dispersion at k_best
            km_best = KMeans(n_clusters=best_kc, random_state=rng_seed,
                              n_init=10).fit(Xn)
            W_obs = 0.0
            for c in range(best_kc):
                mask = km_best.labels_ == c
                if mask.sum() < 2:
                    continue
                W_obs += pdist(Xn[mask]).sum() / (2 * mask.sum())
            # Reference: shuffle each dim independently
            W_ref = []
            for _ in range(n_gap_refs):
                Xref = np.empty_like(Xn)
                for d in range(Xn.shape[1]):
                    Xref[:, d] = rng.permutation(Xn[:, d])
                Xref = normalize_rows(Xref)
                km_r = KMeans(n_clusters=best_kc, random_state=rng_seed,
                                n_init=10).fit(Xref)
                W_r = 0.0
                for c in range(best_kc):
                    mask = km_r.labels_ == c
                    if mask.sum() < 2:
                        continue
                    W_r += pdist(Xref[mask]).sum() / (2 * mask.sum())
                W_ref.append(W_r)
            gap = float(np.log(np.mean(W_ref) + 1e-9) - np.log(W_obs + 1e-9))

            per_K[str(K)] = {
                "best_kc": int(best_kc),
                "silhouette_best": best_sil,
                "silhouette_per_kc": sil_per_kc,
                "gap_at_best_kc": gap,
            }
        out[modality] = per_K
        logger.info(f"  {modality}: silhouette(K=10) = {out[modality]['10']['silhouette_best']:.3f},  "
                    f"silhouette(K=100) = {out[modality]['100']['silhouette_best']:.3f}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Block E — PR + rank-95 (skip MI; KSG broken at N=48)
# ══════════════════════════════════════════════════════════════════════════════

def block_E_long(Z_img, Z_txt, probe_ks):
    logger.info("Block E — PR + rank-95 across K (MI skipped)")
    out = {}

    for modality, Z in [("image", Z_img), ("text", Z_txt)]:
        per_K = {}
        for K in probe_ks:
            X = Z[:, K, :]
            pr = participation_ratio(X)
            per_K[str(K)] = {"PR": pr}
        # Rank-95 over the trajectory tensor truncated at iter K, per-iter centered
        rank_per_K = {}
        for K in probe_ks:
            Z_trunc = Z[:, :K + 1, :]
            Z_centered = Z_trunc - Z_trunc.mean(axis=0, keepdims=True)
            r95 = rank_95(Z_centered)
            rank_per_K[str(K)] = r95
            per_K[str(K)]["rank95"] = r95
        out[modality] = per_K
        logger.info(f"  {modality}: PR(K=10) = {out[modality]['10']['PR']:.2f},  "
                    f"PR(K=100) = {out[modality]['100']['PR']:.2f},  "
                    f"rank-95(K=100) = {out[modality]['100']['rank95']}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Block G — surface-form fidelity
# ══════════════════════════════════════════════════════════════════════════════

def _bleu4_safe(reference, candidate):
    """Cumulative BLEU-4 with NLTK; returns 0 on degenerate inputs."""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        ref_tokens  = reference.lower().split()
        cand_tokens = candidate.lower().split()
        if not ref_tokens or len(cand_tokens) < 4:
            return 0.0
        return float(sentence_bleu(
            [ref_tokens], cand_tokens,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=SmoothingFunction().method1,
        ))
    except Exception:
        return 0.0


# CheXpert label list — must match the project extractor's ordering
CHEXPERT_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
    "Lung Opacity", "No Finding", "Pleural Effusion", "Pleural Other",
    "Pneumonia", "Pneumothorax", "Support Devices",
]


def _get_chexpert_extractor(use_chexpert="auto"):
    """Mirror of analysis_surface_form.get_chexpert_extractor.

    Walks up from this script's location to find an AIM2 base with a
    sibling GENERATION/chexpert/ directory, adds it to sys.path, then
    imports CheXpertLabelExtractor. On failure falls back to a minimal
    rule-based regex labeler.

    Returns (callable, name) where callable: str -> List[str] of positive
    labels.
    """
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

    added = []
    for c in candidates:
        if c and os.path.isdir(c) and c not in sys.path:
            sys.path.insert(0, c)
            added.append(c)
    if added:
        logger.info(f"  Added to sys.path for CheXpert import: {added}")

    if use_chexpert in ("auto", "extractor"):
        try:
            from GENERATION.chexpert.extractor import CheXpertLabelExtractor  # type: ignore
            ex = CheXpertLabelExtractor()

            def fn(text: str):
                if not text or not text.strip():
                    return []
                result = ex.extract_labels(text)
                return [name for name, val in zip(result.label_names, result.labels)
                        if val == 1.0]

            probe = fn("Cardiomegaly is present. No acute pneumothorax.")
            if "Cardiomegaly" not in probe:
                logger.warning(f"  CheXpert smoke probe unexpected: {probe}")
            else:
                logger.info(f"  CheXpert smoke probe OK: {probe}")
            return fn, "CheXpertLabelExtractor (project)"
        except Exception as e:
            logger.warning(f"  Could not load project CheXpertLabelExtractor: {e}")
            if use_chexpert == "extractor":
                return None, "failed"

    # Fallback: minimal rule-based regex labeler
    import re
    PATTERNS = {
        "Cardiomegaly":  re.compile(r"\bcardiomegal\w*\b", re.I),
        "Pleural Effusion": re.compile(r"\bpleural\s+effusion|\beffusion\w*\b", re.I),
        "Pneumothorax":  re.compile(r"\bpneumothorax\b|\bptx\b", re.I),
        "Atelectasis":   re.compile(r"\batelecta\w*\b", re.I),
        "Consolidation": re.compile(r"\bconsolidat\w*\b", re.I),
        "Edema":         re.compile(r"\bedema\b|\bvascular\s+congestion\b", re.I),
        "Pneumonia":     re.compile(r"\bpneumonia\b|\bpneumonitis\b", re.I),
        "Support Devices": re.compile(r"\b(et\s+tube|ett|ng\s+tube|pacemaker|catheter|chest\s+tube)\b", re.I),
        "Lung Opacity":  re.compile(r"\bopacit\w+\b|\bhaziness\b", re.I),
        "Fracture":      re.compile(r"\bfracture\w*\b", re.I),
        "Lung Lesion":   re.compile(r"\b(lung|pulmonary)\s+(mass|lesion|nodule)\b", re.I),
        "No Finding":    re.compile(r"\bno\s+(acute\s+)?(cardiopulmonary\s+)?(abnormalit|finding)\w*\b|\bunremarkable\b|\bclear\s+lungs?\b", re.I),
    }
    NEG = re.compile(r"\b(no\s+(evidence\s+of\s+)?|without|absent|negative\s+for|not\s+(seen|identified))\b", re.I)
    def rule_fn(text: str):
        if not text or not text.strip():
            return []
        labels = []
        for sentence in re.split(r"[.!?]\s+", text):
            for lbl, pat in PATTERNS.items():
                m = pat.search(sentence)
                if m and not NEG.search(sentence[:m.start()]):
                    labels.append(lbl)
        return list(set(labels))
    logger.warning("  Falling back to rule-based regex labeler")
    return rule_fn, "rule-based fallback"


def load_gt_labels(data_csv):
    """Return (study_id -> set of positive CheXpert labels), and the list of
    available label columns. Mirrors analysis_surface_form.load_gt_labels.
    """
    import pandas as pd
    df = pd.read_csv(data_csv, low_memory=False)
    df["study_id"] = df["study_id"].astype(str)
    df = df.groupby("study_id", as_index=False).first()
    df = df.set_index("study_id")
    available = [c for c in CHEXPERT_LABELS if c in df.columns]
    if len(available) < 14:
        logger.warning(f"  Only {len(available)}/14 CheXpert columns in CSV")
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


def block_G_long(findings, gt_findings, gt_labels_map, sids, probe_ks,
                   chx_fn, chx_name, available_labels):
    """At each probe K, compute (mirroring canonical analysis_surface_form.py):
      • BLEU-4 of generated findings_K vs GT findings (from gt_findings.txt)
      • CheXpert recall = |GT_pos ∩ gen_pos| / |GT_pos|
      • CheXpert precision = |GT_pos ∩ gen_pos| / |gen_pos|
      • CheXpert Jaccard = |GT_pos ∩ gen_pos| / |GT_pos ∪ gen_pos|
      • Mean report length (words)
      • Vocabulary diversity (unique tokens / total tokens at iter K)

    Reference texts are GT (gt_findings.txt), reference labels are GT
    (from processed_data.csv CheXpert columns). Candidate texts are the
    iter-K generated findings, candidate labels are extracted on the fly.
    Conservative: only val == 1.0 counts as positive (uncertain → negative).
    """
    logger.info("Block G — surface-form fidelity across K (canonical method)")
    N = len(findings)
    out = {}
    per_K = {}

    # Per-pathology survival arrays (count of "still-present at iter K"
    # studies among those where GT-pos)
    n_with_gt = sum(1 for sid in sids if gt_labels_map.get(sid))
    logger.info(f"  N={N} trajectories, {n_with_gt} with GT CheXpert labels")

    for K in probe_ks:
        bleus, word_lens, char_lens = [], [], []
        recalls, precisions, jaccards = [], [], []
        unique_tokens, total_tokens = 0, 0
        per_label_hits = {lbl: [0, 0] for lbl in available_labels}  # [hits, gt_pos_count]

        for i, sid in enumerate(sids):
            cand = findings[i][K]
            ref  = gt_findings[i]
            bleus.append(_bleu4_safe(ref, cand))
            words = cand.split()
            word_lens.append(len(words))
            char_lens.append(len(cand))
            tokens = [w.lower() for w in words]
            unique_tokens += len(set(tokens))
            total_tokens  += len(tokens)

            gt_pos = gt_labels_map.get(sid, set())
            if chx_fn is not None and gt_pos:
                gen_pos = set(chx_fn(cand))
                inter = gt_pos & gen_pos
                union = gt_pos | gen_pos
                recalls.append(len(inter) / max(len(gt_pos), 1))
                if gen_pos:
                    precisions.append(len(inter) / len(gen_pos))
                jaccards.append(len(inter) / max(len(union), 1))
                # Per-pathology bookkeeping
                for lbl in gt_pos:
                    per_label_hits[lbl][1] += 1
                    if lbl in gen_pos:
                        per_label_hits[lbl][0] += 1

        # Per-pathology preservation rates
        per_label_preservation = {}
        for lbl, (hits, n_gt) in per_label_hits.items():
            if n_gt > 0:
                per_label_preservation[lbl] = {
                    "preservation_rate": hits / n_gt,
                    "n_gt_positive": n_gt,
                }

        per_K[str(K)] = {
            "bleu4_mean":   float(np.mean(bleus)),
            "bleu4_median": float(np.median(bleus)),
            "report_length_words_mean": float(np.mean(word_lens)),
            "report_length_words_std":  float(np.std(word_lens, ddof=1)),
            "report_length_chars_mean": float(np.mean(char_lens)),
            "vocab_diversity": float(unique_tokens / max(total_tokens, 1)),
            "chexpert_recall_mean":    float(np.mean(recalls)) if recalls else None,
            "chexpert_precision_mean": float(np.mean(precisions)) if precisions else None,
            "chexpert_jaccard_mean":   float(np.mean(jaccards)) if jaccards else None,
            "n_with_chexpert_eval":    len(recalls),
            "per_label_preservation":  per_label_preservation,
        }

    out["per_K"] = per_K
    out["chx_extractor_name"] = chx_name

    # Logging summary
    logger.info(f"  BLEU(K=1)   = {per_K['1']['bleu4_mean']:.4f}")
    logger.info(f"  BLEU(K=10)  = {per_K['10']['bleu4_mean']:.4f}")
    logger.info(f"  BLEU(K=100) = {per_K['100']['bleu4_mean']:.4f}")
    if per_K['10']['chexpert_recall_mean'] is not None:
        logger.info(f"  CheXpert recall(K=1)   = {per_K['1']['chexpert_recall_mean']:.3f}")
        logger.info(f"  CheXpert recall(K=10)  = {per_K['10']['chexpert_recall_mean']:.3f}")
        logger.info(f"  CheXpert recall(K=100) = {per_K['100']['chexpert_recall_mean']:.3f}")
    logger.info(f"  Report length (words): K=1 {per_K['1']['report_length_words_mean']:.0f}, "
                f"K=10 {per_K['10']['report_length_words_mean']:.0f}, "
                f"K=100 {per_K['100']['report_length_words_mean']:.0f}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Block H — training-cluster trajectory dynamics
# ══════════════════════════════════════════════════════════════════════════════

def block_H_long(Z_img, Z_txt, ref_img, ref_txt, probe_ks,
                  K_clust=15, rng_seed=0):
    """At each probe K, assign trajectory points to nearest training-distribution
    cluster (k-means on training distribution at K_clust=15). Reports
    per-iter Shannon entropy of cluster assignments and dominant-cluster
    fraction.
    """
    from sklearn.cluster import KMeans
    logger.info("Block H — training-cluster trajectory dynamics across K")
    out = {}

    for modality, (Z, ref) in [("image", (Z_img, ref_img)),
                                  ("text",  (Z_txt, ref_txt))]:
        if ref is None:
            logger.warning(f"  {modality}: ref_256 missing; skipping")
            continue
        # Cast to float64 to match Z dtype (KMeans is dtype-strict on predict)
        ref_n = normalize_rows(ref.astype(np.float64))
        # Fit k-means on training distribution
        km = KMeans(n_clusters=K_clust, random_state=rng_seed,
                     n_init=10).fit(ref_n)

        per_K = {}
        for K in probe_ks:
            Xn = normalize_rows(Z[:, K, :])
            assignments = km.predict(Xn)
            # Shannon entropy in bits
            counts = np.bincount(assignments, minlength=K_clust)
            p = counts / max(counts.sum(), 1)
            p = p[p > 0]
            ent = float(-(p * np.log2(p)).sum())
            ent_norm = ent / np.log2(K_clust)
            dom_frac = float(counts.max() / max(counts.sum(), 1))
            top3 = float(np.sort(counts)[-3:].sum() / max(counts.sum(), 1))

            per_K[str(K)] = {
                "entropy_bits": ent,
                "entropy_normalized": ent_norm,
                "dominant_cluster_fraction": dom_frac,
                "top3_cluster_fraction": top3,
            }
        out[modality] = per_K
        logger.info(f"  {modality}: entropy(K=10) = {out[modality]['10']['entropy_bits']:.2f},  "
                    f"entropy(K=100) = {out[modality]['100']['entropy_bits']:.2f}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Block I — geometric mixing (kNN-to-training, MIPD, displacement alignment)
# ══════════════════════════════════════════════════════════════════════════════

def block_I_long(Z_img, Z_txt, ref_img, ref_txt, probe_ks, k_nn=10,
                  rng_seed=42):
    """Per probe K (mirroring canonical analysis_knn_alignment.py):
      • mean cosine kNN-to-training distance (cohort-mean to k_nn nearest training points)
      • mean inter-point distance (MIPD) within cohort (cosine)
      • displacement alignment: mean pairwise cosine between step vectors Δ_i
        across the cohort. → 1.0 = coherent collapse direction; → 0 = isotropic.
    """
    from sklearn.neighbors import NearestNeighbors
    logger.info("Block I — geometric mixing across K (canonical method)")
    out = {}
    rng = np.random.default_rng(rng_seed)

    for modality, (Z, ref) in [("image", (Z_img, ref_img)),
                                  ("text",  (Z_txt, ref_txt))]:
        per_K = {}
        ref_n = normalize_rows(ref.astype(np.float64)) if ref is not None else None
        if ref_n is not None:
            nn = NearestNeighbors(n_neighbors=k_nn, metric="cosine",
                                    algorithm="brute", n_jobs=-1).fit(ref_n)

        for K in probe_ks:
            Xn = normalize_rows(Z[:, K, :])
            # kNN-to-training (mean over k nearest training neighbors)
            if ref_n is not None:
                d, _ = nn.kneighbors(Xn)
                knn_to_train = float(d.mean())
            else:
                knn_to_train = float("nan")

            # MIPD: mean pairwise cosine distance among cohort
            sims = Xn @ Xn.T
            mask = np.triu(np.ones_like(sims, dtype=bool), k=1)
            mipd = float((1.0 - sims[mask]).mean())

            # Canonical displacement alignment: mean pairwise cosine of step
            # vectors (Δ_i = z^K_i − z^{K-1}_i). Measures cohort coherence:
            # are all trajectories drifting in the same direction?
            if K >= 1:
                disp = Z[:, K, :] - Z[:, K-1, :]                # (N, D)
                norms = np.linalg.norm(disp, axis=1, keepdims=True) + 1e-12
                disp_n = disp / norms                            # (N, D)
                pair_sim = disp_n @ disp_n.T
                iu = np.triu_indices(len(disp_n), k=1)
                disp_align = float(pair_sim[iu].mean())
                step_size_mean = float(norms.mean())
            else:
                disp_align = float("nan")
                step_size_mean = float("nan")

            per_K[str(K)] = {
                "knn_to_training_mean": knn_to_train,
                "mipd": mipd,
                "displacement_alignment": disp_align,  # cohort-coherence ∈ [-1, 1]
                "step_size_mean": step_size_mean,
            }
        out[modality] = per_K
        logger.info(f"  {modality}: kNN-to-train(K=10) = {out[modality]['10']['knn_to_training_mean']:.3f},  "
                    f"kNN-to-train(K=100) = {out[modality]['100']['knn_to_training_mean']:.3f}")
        logger.info(f"  {modality}: disp_align(K=10) = {out[modality]['10']['displacement_alignment']:+.3f},  "
                    f"disp_align(K=100) = {out[modality]['100']['displacement_alignment']:+.3f}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Block J — autocorrelation + kNN persistence vs LAG (HEADLINE)
# ══════════════════════════════════════════════════════════════════════════════

def block_J_long(Z_img, Z_txt, K_max=100, lags=None, k_nn=10,
                   late_iter_start=None, rng_seed=42):
    """Compute autocorrelation and kNN-Jaccard persistence as functions of lag.
    Mirrors canonical analysis_local_persistence.py.

    Autocorrelation at lag ℓ:
      For each k_start in [0, K-ℓ-1], cohort-mean cos(z_k, z_{k+ℓ}).
      We report two summaries:
        autocorr_full_mean  — average over ALL k_start (early + late)
        autocorr_late_mean  — average over k_start ≥ late_iter_start (asymptotic)
      Random-pair baseline: cohort-mean cosine between random different-trajectory
      pairs at iter 0 (the "prior" similarity of unrelated trajectories).

    kNN Jaccard persistence at lag ℓ:
      At iter k, find each trajectory's k_nn nearest neighbors (cosine).
      At iter k+ℓ, find them again. Jaccard overlap. Average over k and traj.
      Random baseline (analytic, uniform sampling without replacement):
        E[Jaccard] ≈ (k_nn²/(N-1)) / (2·k_nn − k_nn²/(N-1))

    Mixing time τ_mix: lag at which late-window autocorr first crosses
    the random-pair baseline (within 1 SE).
    """
    from sklearn.neighbors import NearestNeighbors
    if lags is None:
        lags = LAGS_FOR_BLOCK_J
    if late_iter_start is None:
        late_iter_start = K_max // 2

    logger.info(f"Block J — autocorr + persistence vs lag (k_nn={k_nn}, "
                f"late_start={late_iter_start})")
    rng = np.random.default_rng(rng_seed)
    out = {}

    for modality, Z in [("image", Z_img), ("text", Z_txt)]:
        N, K_total, D = Z.shape
        Zn = normalize_rows(Z.reshape(-1, D)).reshape(N, K_total, D)

        # Random-pair baseline at iter 0 (canonical convention)
        pair_idx = rng.choice(N, size=(2000, 2), replace=True)
        pair_idx = pair_idx[pair_idx[:, 0] != pair_idx[:, 1]]
        rand_sim = float((Zn[pair_idx[:, 0], 0, :] *
                            Zn[pair_idx[:, 1], 0, :]).sum(axis=1).mean())

        # Autocorrelation at each lag — both full and late windows
        autocorr_full_mean,  autocorr_full_sem  = [], []
        autocorr_late_mean,  autocorr_late_sem  = [], []
        for ell in lags:
            # All valid k_start: 0..K_total-1-ell
            k_hi = K_total - 1 - ell
            if k_hi < 0:
                for arr in (autocorr_full_mean, autocorr_full_sem,
                             autocorr_late_mean, autocorr_late_sem):
                    arr.append(float("nan"))
                continue
            # Full window
            sims_full = []
            for k in range(0, k_hi + 1):
                sims_full.append(
                    float((Zn[:, k, :] * Zn[:, k + ell, :]).sum(axis=1).mean())
                )
            autocorr_full_mean.append(float(np.mean(sims_full)))
            autocorr_full_sem.append(float(np.std(sims_full, ddof=1) / np.sqrt(max(len(sims_full), 1)))
                                       if len(sims_full) > 1 else 0.0)
            # Late window (k_start ≥ late_iter_start)
            k_lo = max(late_iter_start, 0)
            if k_hi < k_lo:
                autocorr_late_mean.append(float("nan"))
                autocorr_late_sem.append(float("nan"))
            else:
                sims_late = sims_full[k_lo:]
                autocorr_late_mean.append(float(np.mean(sims_late)))
                autocorr_late_sem.append(float(np.std(sims_late, ddof=1) / np.sqrt(max(len(sims_late), 1)))
                                          if len(sims_late) > 1 else 0.0)

        # kNN Jaccard persistence at each lag
        # Pre-fit kNN at every iteration once (memoize) for efficiency
        neighbor_sets_per_iter = {}
        def get_neighbors(k):
            if k in neighbor_sets_per_iter: return neighbor_sets_per_iter[k]
            kn = NearestNeighbors(n_neighbors=k_nn + 1, metric="cosine",
                                    algorithm="brute", n_jobs=-1).fit(Zn[:, k, :])
            _, idx = kn.kneighbors(Zn[:, k, :])
            sets_k = [set(idx[i, 1:]) for i in range(N)]
            neighbor_sets_per_iter[k] = sets_k
            return sets_k

        knn_full_mean, knn_full_sem = [], []
        knn_late_mean, knn_late_sem = [], []
        for ell in lags:
            k_hi = K_total - 1 - ell
            if k_hi < 0:
                for arr in (knn_full_mean, knn_full_sem,
                             knn_late_mean, knn_late_sem):
                    arr.append(float("nan"))
                continue
            jacs_full = []
            for k in range(0, k_hi + 1):
                a_sets = get_neighbors(k)
                b_sets = get_neighbors(k + ell)
                ja_per_traj = [len(a_sets[i] & b_sets[i]) / max(len(a_sets[i] | b_sets[i]), 1)
                                for i in range(N)]
                jacs_full.append(float(np.mean(ja_per_traj)))
            knn_full_mean.append(float(np.mean(jacs_full)))
            knn_full_sem.append(float(np.std(jacs_full, ddof=1) / np.sqrt(max(len(jacs_full), 1)))
                                  if len(jacs_full) > 1 else 0.0)
            k_lo = max(late_iter_start, 0)
            if k_hi < k_lo:
                knn_late_mean.append(float("nan"))
                knn_late_sem.append(float("nan"))
            else:
                jacs_late = jacs_full[k_lo:]
                knn_late_mean.append(float(np.mean(jacs_late)))
                knn_late_sem.append(float(np.std(jacs_late, ddof=1) / np.sqrt(max(len(jacs_late), 1)))
                                      if len(jacs_late) > 1 else 0.0)

        # Analytic kNN-Jaccard random baseline
        expected_inter = (k_nn * k_nn) / (N - 1)
        expected_union = 2 * k_nn - expected_inter
        knn_random_baseline = float(expected_inter / max(expected_union, 1))

        # Mixing time: first lag where late autocorr drops below random baseline + 1 SE
        tau_mix = None
        for j, m in enumerate(autocorr_late_mean):
            if not np.isnan(m) and m < rand_sim + autocorr_late_sem[j]:
                tau_mix = lags[j]
                break

        out[modality] = {
            "lags": list(lags),
            "autocorr_full_mean": autocorr_full_mean,
            "autocorr_full_sem":  autocorr_full_sem,
            "autocorr_late_mean": autocorr_late_mean,
            "autocorr_late_sem":  autocorr_late_sem,
            "knn_jaccard_full_mean": knn_full_mean,
            "knn_jaccard_full_sem":  knn_full_sem,
            "knn_jaccard_late_mean": knn_late_mean,
            "knn_jaccard_late_sem":  knn_late_sem,
            "random_pair_autocorr":   rand_sim,
            "random_pair_jaccard":    knn_random_baseline,
            "tau_mix_iters":          tau_mix,
            "k_nn_used":              k_nn,
            "late_iter_start":        late_iter_start,
        }
        logger.info(f"  {modality}: autocorr_late(lag=1) = {autocorr_late_mean[0]:.3f}, "
                    f"autocorr_late(lag=50) = {autocorr_late_mean[-1] if len(lags)>0 else float('nan'):.3f}, "
                    f"random_pair = {rand_sim:.3f}, τ_mix = {tau_mix}")
        logger.info(f"  {modality}: kNN-Jacc_late(lag=1) = {knn_late_mean[0]:.3f}, "
                    f"kNN-Jacc_late(lag=50) = {knn_late_mean[-1]:.3f}, "
                    f"random = {knn_random_baseline:.3f}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Figures
# ══════════════════════════════════════════════════════════════════════════════

def make_figures(results, out_dir, probe_ks, lags):
    """Per-block summary figures showing K-evolution + lag-evolution."""
    import matplotlib.pyplot as plt
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Figure A — anchor distance, step size, tortuosity vs K
    if "A" in results:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        for modality, color in [("image", "#2962FF"), ("text", "#E65100")]:
            data = results["A"][modality]
            Ks = np.array([int(k) for k in probe_ks])
            d_K = np.array([data[str(K)]["d_K_mean"] for K in probe_ks])
            d_sd = np.array([data[str(K)]["d_K_std"] for K in probe_ks])
            step = np.array([data[str(K)]["step_K_mean"] for K in probe_ks])
            tort = np.array([data[str(K)]["tortuosity_mean"] for K in probe_ks])

            axes[0].plot(Ks, d_K, "o-", color=color, label=modality, lw=2)
            axes[0].fill_between(Ks, d_K - d_sd, d_K + d_sd, color=color, alpha=0.18)
            axes[1].plot(Ks, step, "o-", color=color, label=modality, lw=2)
            axes[2].plot(Ks, tort, "o-", color=color, label=modality, lw=2)
        for ax, ylabel in zip(axes,
                               ["Anchor distance $d_K$",
                                "Step size $\\Delta_K$",
                                "Tortuosity $L_K / D_K$"]):
            ax.set_xlabel("Iteration $K$"); ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3); ax.legend(fontsize=10)
        fig.suptitle("Block A — trajectory geometry across K", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "A_trajectory_geometry.pdf")); plt.close(fig)
        logger.info(f"  → {fig_dir}/A_trajectory_geometry.pdf")

    # Figure C — silhouette + gap vs K
    if "C" in results:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for modality, color in [("image", "#2962FF"), ("text", "#E65100")]:
            data = results["C"][modality]
            Ks = np.array([int(k) for k in probe_ks])
            sil = np.array([data[str(K)]["silhouette_best"] for K in probe_ks])
            gap = np.array([data[str(K)]["gap_at_best_kc"] for K in probe_ks])
            axes[0].plot(Ks, sil, "o-", color=color, label=modality, lw=2)
            axes[1].plot(Ks, gap, "o-", color=color, label=modality, lw=2)
        axes[0].axhline(0.5, color="gray", ls=":", lw=1, label="strong cluster")
        axes[0].axhline(0.25, color="gray", ls="--", lw=1, label="weak/artifact")
        axes[0].set_xlabel("Iteration $K$"); axes[0].set_ylabel("Best silhouette")
        axes[1].set_xlabel("Iteration $K$"); axes[1].set_ylabel("Gap statistic at best $K_c$")
        for ax in axes:
            ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
        fig.suptitle("Block C — cluster structure across K", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "C_cluster_structure.pdf")); plt.close(fig)
        logger.info(f"  → {fig_dir}/C_cluster_structure.pdf")

    # Figure E — PR + rank-95 vs K
    if "E" in results:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for modality, color in [("image", "#2962FF"), ("text", "#E65100")]:
            data = results["E"][modality]
            Ks = np.array([int(k) for k in probe_ks])
            pr = np.array([data[str(K)]["PR"] for K in probe_ks])
            r95 = np.array([data[str(K)]["rank95"] for K in probe_ks])
            axes[0].plot(Ks, pr, "o-", color=color, label=modality, lw=2)
            axes[1].plot(Ks, r95, "o-", color=color, label=modality, lw=2)
        axes[0].set_xlabel("Iteration $K$"); axes[0].set_ylabel("Participation ratio")
        axes[1].set_xlabel("Iteration $K$"); axes[1].set_ylabel("Rank-95 (trajectory tensor 0..K)")
        for ax in axes:
            ax.grid(True, alpha=0.3); ax.legend(fontsize=10)
        fig.suptitle("Block E — high-D structure across K", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "E_high_d_structure.pdf")); plt.close(fig)
        logger.info(f"  → {fig_dir}/E_high_d_structure.pdf")

    # Figure G — surface-form vs K
    if "G" in results:
        fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
        data = results["G"]["per_K"]
        Ks = np.array([int(k) for k in probe_ks])
        bleu = np.array([data[str(K)]["bleu4_mean"] for K in probe_ks])
        rlen = np.array([data[str(K)]["report_length_words_mean"] for K in probe_ks])
        rlen_sd = np.array([data[str(K)]["report_length_words_std"] for K in probe_ks])
        vocab = np.array([data[str(K)]["vocab_diversity"] for K in probe_ks])
        recall_vals = [data[str(K)].get("chexpert_recall_mean") for K in probe_ks]
        recall = np.array([float("nan") if v is None else v for v in recall_vals])

        axes[0].plot(Ks, bleu, "o-", color="#2962FF", lw=2)
        axes[0].set_xlabel("Iteration $K$"); axes[0].set_ylabel("BLEU-4 vs GT")
        axes[0].set_title("Surface-form fidelity")

        axes[1].plot(Ks, recall, "o-", color="#C62828", lw=2)
        axes[1].set_xlabel("Iteration $K$"); axes[1].set_ylabel("CheXpert recall vs GT")
        axes[1].set_title("Pathology preservation")

        axes[2].plot(Ks, rlen, "o-", color="#388E3C", lw=2)
        axes[2].fill_between(Ks, rlen - rlen_sd, rlen + rlen_sd, color="#388E3C", alpha=0.2)
        axes[2].set_xlabel("Iteration $K$"); axes[2].set_ylabel("Report length (words)")
        axes[2].set_title("Report length collapse")

        axes[3].plot(Ks, vocab, "o-", color="#E65100", lw=2)
        axes[3].set_xlabel("Iteration $K$"); axes[3].set_ylabel("Vocabulary diversity")
        axes[3].set_title("Vocabulary collapse")

        for ax in axes:
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"Block G — surface-form fidelity across K "
                       f"(extractor: {results['G'].get('chx_extractor_name', 'unknown')})",
                       fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "G_surface_form.pdf")); plt.close(fig)
        logger.info(f"  → {fig_dir}/G_surface_form.pdf")

    # Figure H — entropy + dominant fraction vs K
    if "H" in results:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for modality, color in [("image", "#2962FF"), ("text", "#E65100")]:
            if modality not in results["H"]:
                continue
            data = results["H"][modality]
            Ks = np.array([int(k) for k in probe_ks])
            ent = np.array([data[str(K)]["entropy_bits"] for K in probe_ks])
            dom = np.array([data[str(K)]["dominant_cluster_fraction"] for K in probe_ks])
            axes[0].plot(Ks, ent, "o-", color=color, label=modality, lw=2)
            axes[1].plot(Ks, dom, "o-", color=color, label=modality, lw=2)
        axes[0].set_xlabel("Iteration $K$"); axes[0].set_ylabel("Cluster entropy (bits)")
        axes[1].set_xlabel("Iteration $K$"); axes[1].set_ylabel("Dominant cluster fraction")
        for ax in axes:
            ax.grid(True, alpha=0.3); ax.legend(fontsize=10)
        fig.suptitle("Block H — training-cluster dynamics across K", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "H_training_clusters.pdf")); plt.close(fig)
        logger.info(f"  → {fig_dir}/H_training_clusters.pdf")

    # Figure I — geometric mixing vs K
    if "I" in results:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        for modality, color in [("image", "#2962FF"), ("text", "#E65100")]:
            data = results["I"][modality]
            Ks = np.array([int(k) for k in probe_ks])
            knn_t = np.array([data[str(K)]["knn_to_training_mean"] for K in probe_ks])
            mipd = np.array([data[str(K)]["mipd"] for K in probe_ks])
            disp = np.array([data[str(K)]["displacement_alignment"] for K in probe_ks])
            axes[0].plot(Ks, knn_t, "o-", color=color, label=modality, lw=2)
            axes[1].plot(Ks, mipd, "o-", color=color, label=modality, lw=2)
            axes[2].plot(Ks, disp, "o-", color=color, label=modality, lw=2)
        axes[0].set_xlabel("Iteration $K$"); axes[0].set_ylabel("kNN-to-training cosine dist")
        axes[1].set_xlabel("Iteration $K$"); axes[1].set_ylabel("Cohort MIPD (cosine)")
        axes[2].set_xlabel("Iteration $K$"); axes[2].set_ylabel("Displacement alignment")
        axes[2].axhline(0, color="gray", ls=":", lw=1)
        for ax in axes:
            ax.grid(True, alpha=0.3); ax.legend(fontsize=10)
        fig.suptitle("Block I — geometric mixing across K", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "I_geometric_mixing.pdf")); plt.close(fig)
        logger.info(f"  → {fig_dir}/I_geometric_mixing.pdf")

    # Figure J — autocorrelation + kNN persistence vs LAG (HEADLINE)
    if "J" in results:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for modality, color in [("image", "#2962FF"), ("text", "#E65100")]:
            d = results["J"][modality]
            lags_arr = np.array(d["lags"])
            ac_late = np.array(d["autocorr_late_mean"])
            ac_late_sem = np.array(d["autocorr_late_sem"])
            ac_full = np.array(d["autocorr_full_mean"])
            knn_late = np.array(d["knn_jaccard_late_mean"])
            knn_late_sem = np.array(d["knn_jaccard_late_sem"])

            # Autocorr panel: late window (solid), full window (dashed faint)
            axes[0].plot(lags_arr, ac_late, "o-", color=color,
                          label=f"{modality} (late)", lw=2)
            axes[0].fill_between(lags_arr, ac_late - ac_late_sem,
                                   ac_late + ac_late_sem, color=color, alpha=0.2)
            axes[0].plot(lags_arr, ac_full, ":", color=color, alpha=0.5,
                          lw=1.2, label=f"{modality} (full)")
            axes[0].axhline(d["random_pair_autocorr"], color=color, ls="--",
                              alpha=0.6, lw=1.2)
            if d["tau_mix_iters"] is not None:
                axes[0].axvline(d["tau_mix_iters"], color=color, ls=":", lw=1.4)
                axes[0].text(d["tau_mix_iters"], 0.05,
                              f"τ_mix={d['tau_mix_iters']}",
                              color=color, rotation=90, fontsize=9)

            # kNN panel: late window only (most relevant for asymptotic)
            axes[1].plot(lags_arr, knn_late, "o-", color=color,
                          label=f"{modality}", lw=2)
            axes[1].fill_between(lags_arr, knn_late - knn_late_sem,
                                   knn_late + knn_late_sem, color=color, alpha=0.2)
            axes[1].axhline(d["random_pair_jaccard"], color=color, ls="--",
                              alpha=0.6, lw=1.2)
        axes[0].set_xlabel("Lag $\\ell$ (iterations)")
        axes[0].set_ylabel("Trajectory autocorrelation $\\langle z_k \\cdot z_{k+\\ell} \\rangle$")
        axes[0].set_title("Memory decay\n(solid: late window, dotted: full window)")
        axes[0].grid(True, alpha=0.3); axes[0].legend(fontsize=9, loc="upper right")
        axes[1].set_xlabel("Lag $\\ell$ (iterations)")
        axes[1].set_ylabel(f"kNN Jaccard persistence (k_nn={results['J']['image']['k_nn_used']})")
        axes[1].set_title("Neighborhood persistence (late window)")
        axes[1].grid(True, alpha=0.3); axes[1].legend(fontsize=9, loc="upper right")
        fig.suptitle("Block J — memory decay + neighborhood persistence vs lag (K=100)",
                       fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "J_persistence_long.pdf")); plt.close(fig)
        logger.info(f"  → {fig_dir}/J_persistence_long.pdf")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--long_dir", required=True,
                    help="Path to chexgen_long results directory")
    p.add_argument("--ref_dir", required=True,
                    help="Reference-embedding directory (for blocks H, I)")
    p.add_argument("--out_dir", required=True,
                    help="Output directory for results JSON + figures")
    p.add_argument("--data_csv", type=str,
                    default="/n/groups/training/bmif203/AIM2/processed_data/processed_data.csv",
                    help="Master CSV with CheXpert columns (Block G GT labels).")
    p.add_argument("--use_chexpert", choices=["auto", "extractor", "rulebased", "none"],
                    default="auto",
                    help="Block G CheXpert backend. 'auto' = project extractor "
                         "with rule-based fallback. 'extractor' = require project. "
                         "'rulebased' = always use rules. 'none' = skip pathology metrics.")
    p.add_argument("--K_max", type=int, default=100)
    p.add_argument("--blocks", type=str, default="A,C,E,G,H,I,J",
                    help="Comma-separated list of blocks to run.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("AIM2 Long-horizon Analysis (K=100, N=48)")
    logger.info("=" * 60)
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")

    # Filter probe_ks to only those <= K_max actually loaded
    probe_ks = [k for k in PROBE_KS if k <= args.K_max]
    logger.info(f"  Probing at K = {probe_ks}")

    # Load data
    Z_img, Z_txt, A_img, A_txt, sids, findings, gt_findings = load_long_trajectories(
        args.long_dir, K_max=args.K_max)
    ref_img, ref_txt = load_reference(args.ref_dir)
    logger.info(f"  N = {len(sids)} valid trajectories")

    blocks_to_run = [b.strip() for b in args.blocks.split(",")]
    results = {"args": vars(args), "n_trajectories": len(sids),
                "probe_ks": probe_ks, "lags": LAGS_FOR_BLOCK_J}

    if "A" in blocks_to_run:
        results["A"] = block_A_long(Z_img, Z_txt, A_img, A_txt, probe_ks)
    if "C" in blocks_to_run:
        results["C"] = block_C_long(Z_img, Z_txt, probe_ks)
    if "E" in blocks_to_run:
        results["E"] = block_E_long(Z_img, Z_txt, probe_ks)
    if "G" in blocks_to_run:
        # Load CheXpert extractor + GT labels (mirroring analysis_surface_form.py)
        chx_fn, chx_name = _get_chexpert_extractor(args.use_chexpert)
        logger.info(f"  Block G CheXpert extractor: {chx_name}")
        if os.path.exists(args.data_csv):
            gt_labels_map, available_labels = load_gt_labels(args.data_csv)
            logger.info(f"  GT labels: {len(gt_labels_map)} studies, "
                        f"{len(available_labels)} available label columns")
        else:
            logger.warning(f"  data_csv not found: {args.data_csv}")
            gt_labels_map, available_labels = {}, []
        results["G"] = block_G_long(
            findings, gt_findings, gt_labels_map, sids, probe_ks,
            chx_fn, chx_name, available_labels,
        )
    if "H" in blocks_to_run:
        results["H"] = block_H_long(Z_img, Z_txt, ref_img, ref_txt, probe_ks)
    if "I" in blocks_to_run:
        results["I"] = block_I_long(Z_img, Z_txt, ref_img, ref_txt, probe_ks)
    if "J" in blocks_to_run:
        results["J"] = block_J_long(Z_img, Z_txt, K_max=args.K_max,
                                       lags=LAGS_FOR_BLOCK_J)

    # Save JSON
    out_json = os.path.join(args.out_dir, "long_horizon_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else str(o))
    logger.info(f"\nResults → {out_json}")

    # Figures
    make_figures(results, args.out_dir, probe_ks, LAGS_FOR_BLOCK_J)

    logger.info(f"\nAll outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()