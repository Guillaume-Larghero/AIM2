#!/usr/bin/env python3
"""
AIM2 — Generate clean PNG figures for the 6-minute final presentation (v2).

CHANGES FROM v1:
  • Smarter pair selection — uses 2D UMAP distance at iter 0 and iter K
    (not just 256-d cosine) so the visual story matches the message.
  • For Fig 2 (Lyapunov): generates MULTIPLE candidate pairs with
    different visual properties; user picks the best for the talk.
  • Added zoom-in inset panels showing the region of interest at higher
    UMAP resolution alongside the overall map.
  • For Fig 2 λ_a panel: requires monotone-divergence pairs (rejects
    pairs where distance non-monotonically dips then rises — those
    make a confusing story).
  • For Fig 3 (persistence): generates MULTIPLE candidate trios so user
    can pick the most visually striking persistent neighborhood.

OUTPUTS (in --out_dir):
  talk_fig1_mi_collapse.png                   — single fig (no candidates)
  talk_fig2_lyapunov_umap_v{N}.png            — N=1..n_candidates_lyap
  talk_fig3_persistence_umap_v{N}.png         — N=1..n_candidates_pers

USAGE:
  python make_presentation_figures.py \\
      --main_dir   .../results/chexgen_main \\
      --ref_dir    .../reference_embeddings \\
      --out_dir    .../figures_for_talk \\
      --n_candidates_lyap 4 \\
      --n_candidates_pers 4
"""

import argparse
import logging
import os
import pickle
import sys
from glob import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size":        14,
    "axes.titlesize":   16,
    "axes.labelsize":   14,
    "xtick.labelsize":  12,
    "ytick.labelsize":  12,
    "legend.fontsize":  11,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.linewidth":      1.2,
    "lines.linewidth":     2.4,
    "lines.markersize":    7,
    "savefig.bbox":        "tight",
    "savefig.dpi":         200,
    "figure.facecolor":    "white",
    "axes.facecolor":      "white",
})

COLOR = {
    "primary":   "#065A82",
    "secondary": "#1C7293",
    "accent":    "#21295C",
    "warning":   "#C73E1D",
    "good":      "#2E7D32",
    "muted":     "#5C6B7A",
    "light":     "#E8EEF3",
    "bg_anchor": "#FFD166",
    "training":  "#B8C5D6",
}

# Module-level so the persistence selector helper can use it
A_img_global = None


# ══════════════════════════════════════════════════════════════════════════════
#  Loaders
# ══════════════════════════════════════════════════════════════════════════════

def load_trajectories(main_dir):
    metric_files = sorted(glob(os.path.join(main_dir, "*", "metrics.json")))
    logger.info(f"Found {len(metric_files)} trajectories")
    Z_img, Z_txt, sids, A_img, A_txt = [], [], [], [], []
    K_canonical = None
    for f in metric_files:
        sdir = os.path.dirname(f)
        sid  = os.path.basename(sdir)
        a_i = os.path.join(sdir, "anchor_img_embed.npy")
        a_t = os.path.join(sdir, "anchor_text_embed.npy")
        img_files = sorted(glob(os.path.join(sdir, "img_embed_iter_*.npy")))
        txt_files = sorted(glob(os.path.join(sdir, "text_embed_iter_*.npy")))
        if not img_files or len(img_files) != len(txt_files):
            continue
        if not (os.path.exists(a_i) and os.path.exists(a_t)):
            continue
        try:
            zi = np.stack([np.load(p) for p in img_files])
            zt = np.stack([np.load(p) for p in txt_files])
        except Exception:
            continue
        if K_canonical is None:
            K_canonical = zi.shape[0]
        if zi.shape[0] != K_canonical or zt.shape[0] != K_canonical:
            continue
        Z_img.append(zi); Z_txt.append(zt)
        A_img.append(np.load(a_i)); A_txt.append(np.load(a_t))
        sids.append(sid)
    Z_img = np.stack(Z_img); Z_txt = np.stack(Z_txt)
    A_img = np.stack(A_img); A_txt = np.stack(A_txt)
    logger.info(f"  Loaded N={len(sids)}, K={K_canonical}, D={Z_img.shape[-1]}")
    return Z_img, Z_txt, sids, A_img, A_txt


def load_umap(ref_dir):
    paths = {
        "ru_img":     os.path.join(ref_dir, "umap_img.pkl"),
        "ru_txt":     os.path.join(ref_dir, "umap_txt.pkl"),
        "img_2d":     os.path.join(ref_dir, "umap_img_2d.npy"),
        "txt_2d":     os.path.join(ref_dir, "umap_txt_2d.npy"),
        "img_embed":  os.path.join(ref_dir, "img_embeds.npy"),
        "txt_embed":  os.path.join(ref_dir, "txt_embeds.npy"),
    }
    # Embeddings are required for the modality-asymmetry figure but optional
    # for the rest; warn if missing rather than failing hard.
    for k in ("ru_img", "ru_txt", "img_2d", "txt_2d"):
        if not os.path.exists(paths[k]):
            logger.error(f"Missing {paths[k]}"); sys.exit(1)
    with open(paths["ru_img"], "rb") as f: reducer_img = pickle.load(f)
    with open(paths["ru_txt"], "rb") as f: reducer_txt = pickle.load(f)
    img_2d = np.load(paths["img_2d"])
    txt_2d = np.load(paths["txt_2d"])
    ref_img_256 = ref_txt_256 = None
    if os.path.exists(paths["img_embed"]) and os.path.exists(paths["txt_embed"]):
        ref_img_256 = np.load(paths["img_embed"])
        ref_txt_256 = np.load(paths["txt_embed"])
        logger.info(f"  UMAP+embed refs loaded: img_2d {img_2d.shape}, "
                    f"img_256 {ref_img_256.shape}")
    else:
        logger.warning(f"  256-d ref embeddings not found in {ref_dir}; "
                        f"Fig 4 modality histograms will fall back to UMAP-2D.")
        logger.info(f"  UMAP refs loaded: img_2d {img_2d.shape}")
    return reducer_img, reducer_txt, img_2d, txt_2d, ref_img_256, ref_txt_256


def normalize(X):
    n = np.linalg.norm(X, axis=-1, keepdims=True)
    return X / np.maximum(n, 1e-12)


# ══════════════════════════════════════════════════════════════════════════════
#  MI estimator (Block E)
# ══════════════════════════════════════════════════════════════════════════════

def estimate_mi_per_iter(Z, n_components=8, k=4):
    from sklearn.decomposition import PCA
    from scipy.special import digamma
    N, K, D = Z.shape
    fit_data = np.vstack([Z[:, 0, :], Z[:, -1, :]])
    pca = PCA(n_components=n_components).fit(fit_data)
    Z_pca = np.stack([pca.transform(Z[:, i, :]) for i in range(K)], axis=1)

    def ksg_mi(X, Y, k=4):
        from sklearn.neighbors import NearestNeighbors as NN
        N = len(X)
        XY = np.hstack([X, Y])
        knn_xy = NN(n_neighbors=k + 1, metric="chebyshev").fit(XY)
        dists, _ = knn_xy.kneighbors(XY)
        eps = dists[:, -1]
        nx = np.zeros(N, dtype=int); ny = np.zeros(N, dtype=int)
        knn_x = NN(metric="chebyshev").fit(X)
        knn_y = NN(metric="chebyshev").fit(Y)
        for i in range(N):
            r = eps[i] - 1e-12
            nx[i] = len(knn_x.radius_neighbors([X[i]], radius=r, return_distance=False)[0]) - 1
            ny[i] = len(knn_y.radius_neighbors([Y[i]], radius=r, return_distance=False)[0]) - 1
        nx = np.maximum(nx, 1); ny = np.maximum(ny, 1)
        return digamma(k) + digamma(N) - np.mean(digamma(nx + 1) + digamma(ny + 1))

    mi = np.zeros(K)
    for ki in range(K):
        mi[ki] = max(0.0, ksg_mi(Z_pca[:, 0, :], Z_pca[:, ki, :], k=k))
    return mi


# ══════════════════════════════════════════════════════════════════════════════
#  Smarter pair / trio selection — UMAP-aware
# ══════════════════════════════════════════════════════════════════════════════

def find_lambda_sys_candidates(Z_img, A_img, sids, reducer, n_candidates=4):
    """Find pairs that VISUALLY converge in 2D UMAP:
       large UMAP distance at anchor, small UMAP distance at iter K.

    Score each pair by anchor_umap_dist - endpoint_umap_dist.
    Spread out the picks so user gets diverse candidates.
    """
    logger.info("  Projecting all anchors + endpoints to UMAP for λ_sys selection...")
    N = len(sids)
    anchor_2d   = reducer.transform(A_img)              # (N, 2)
    endpoint_2d = reducer.transform(Z_img[:, -1, :])    # iter K

    rng = np.random.default_rng(0)
    n_sample = min(400, N)
    sample = rng.choice(N, n_sample, replace=False)

    anc_d = np.linalg.norm(
        anchor_2d[sample, None, :] - anchor_2d[None, sample, :], axis=-1
    )
    end_d = np.linalg.norm(
        endpoint_2d[sample, None, :] - endpoint_2d[None, sample, :], axis=-1
    )
    score = anc_d - end_d   # large positive = strong contraction in UMAP

    np.fill_diagonal(score, -np.inf)
    np.fill_diagonal(anc_d, 0)

    # Require minimum anchor separation (so the visual story shows starting far)
    anc_d_thresh = np.percentile(anc_d[anc_d > 0], 75)
    score[anc_d < anc_d_thresh] = -np.inf

    # Pick top-N pairs with no shared trajectories
    chosen = []
    used = set()
    flat_idx = np.argsort(score.ravel())[::-1]
    for fi in flat_idx:
        ii, jj = np.unravel_index(fi, score.shape)
        if score[ii, jj] == -np.inf: break
        i_orig, j_orig = int(sample[ii]), int(sample[jj])
        if i_orig in used or j_orig in used: continue
        chosen.append((i_orig, j_orig, float(anc_d[ii, jj]), float(end_d[ii, jj])))
        used.add(i_orig); used.add(j_orig)
        if len(chosen) >= n_candidates: break
    for k, (i, j, d0, dK) in enumerate(chosen):
        logger.info(f"    λ_sys cand #{k+1}: {sids[i]} vs {sids[j]}  "
                    f"UMAP(anchor)={d0:.2f}  UMAP(iter K)={dK:.2f}")
    return chosen


def find_lambda_a_candidates(Z_img, A_img, sids, reducer, n_candidates=4,
                              cosine_thresh=0.75):
    """Find pairs with NEAR-IDENTICAL anchors (cosine ≥ thresh) whose
    trajectory pairwise distance grows MONOTONICALLY ENOUGH (not too noisy).

    Score: 256-d distance growth ratio = ||z_K-z_K'|| / ||z_0-z_0'|| .
    Reject pairs where distance dips dramatically below iter-0 (<0.6×) at
    some intermediate k — those are confusing visual stories.
    """
    logger.info(f"  Finding λ_a candidates (anchor cosine ≥ {cosine_thresh})...")
    N = len(sids)
    A = normalize(A_img)
    K = Z_img.shape[1]

    rng = np.random.default_rng(1)
    sample = rng.choice(N, min(500, N), replace=False)
    A_sample = A[sample]

    sims = A_sample @ A_sample.T
    np.fill_diagonal(sims, -2)
    candidate_pairs = []
    for ii in range(len(sample)):
        for jj in range(ii + 1, len(sample)):
            if sims[ii, jj] >= cosine_thresh:
                candidate_pairs.append((int(sample[ii]), int(sample[jj]),
                                         float(sims[ii, jj])))
    logger.info(f"    Found {len(candidate_pairs)} pairs with cosine ≥ {cosine_thresh}")

    if not candidate_pairs:
        logger.warning(f"  No pairs at cosine ≥ {cosine_thresh}; lowering to 0.5")
        return find_lambda_a_candidates(Z_img, A_img, sids, reducer,
                                          n_candidates, cosine_thresh=0.5)

    # Score each candidate by clean monotonic divergence
    scored = []
    for i, j, c0 in candidate_pairs:
        traj_d = np.array([np.linalg.norm(Z_img[i, k] - Z_img[j, k])
                            for k in range(K)])
        d0, dK = traj_d[0], traj_d[-1]
        if d0 < 1e-6: continue
        growth = dK / d0
        # Clean growth: distance shouldn't drop way below d0 in middle
        min_mid = traj_d[1:-1].min() if K > 2 else d0
        cleanliness = min(min_mid, d0) / d0   # 1 = never dips, <1 = dips
        if growth > 1.2 and cleanliness > 0.6:
            score = growth * cleanliness
            scored.append((i, j, c0, growth, cleanliness, score))

    scored.sort(key=lambda x: x[-1], reverse=True)
    chosen = scored[:n_candidates]
    for k, (i, j, c0, g, cl, sc) in enumerate(chosen):
        logger.info(f"    λ_a cand #{k+1}: {sids[i]} vs {sids[j]}  "
                    f"cos₀={c0:.3f}  growth={g:.2f}×  clean={cl:.2f}")
    return [(i, j) for (i, j, c0, g, cl, sc) in chosen]


def find_persistence_candidates(Z_img, sids, reducer, n_candidates=4, k_nn=10,
                                 trio_size=3):
    """Find trios of trajectories that ended in the same UMAP neighborhood
    AND stayed together over the LAST 3 iterations. Prefer trios with:
      - SPREAD-OUT anchors (visually distinct starts)
      - COMPACT endpoint cluster (clear local convergence)
      - HIGH triple-overlap of kNN sets at iter K-2, K-1, K
    """
    logger.info(f"  Finding persistence trios (k_nn={k_nn})...")
    N, K, D = Z_img.shape
    Z_n = normalize(Z_img.reshape(-1, D)).reshape(N, K, D)

    knn = NearestNeighbors(n_neighbors=k_nn + 1, metric="cosine",
                            algorithm="brute", n_jobs=-1)
    iter_neighbors = []
    for k_ in [K - 3, K - 2, K - 1]:
        knn.fit(Z_n[:, k_, :])
        _, idx = knn.kneighbors(Z_n[:, k_, :])
        iter_neighbors.append([set(idx[i, 1:]) for i in range(N)])

    triple_overlap = np.zeros(N, dtype=int)
    for i in range(N):
        triple_overlap[i] = len(iter_neighbors[0][i] & iter_neighbors[1][i] & iter_neighbors[2][i])

    candidates_by_persistence = np.argsort(triple_overlap)[::-1]
    logger.info(f"    Top triple-overlap counts: "
                f"{triple_overlap[candidates_by_persistence[:5]].tolist()}")

    endpoint_2d = reducer.transform(Z_img[:, -1, :])
    if A_img_global is not None:
        anchor_2d = reducer.transform(A_img_global)
    else:
        anchor_2d = reducer.transform(Z_img[:, 0, :])

    chosen_trios = []
    used = set()
    for c in candidates_by_persistence:
        if int(c) in used: continue
        if triple_overlap[c] < 4: break
        triple = (iter_neighbors[0][c] & iter_neighbors[1][c] & iter_neighbors[2][c])
        if len(triple) < trio_size - 1: continue
        triple_arr = list(triple)
        # Prefer buddies with SPREAD-OUT anchors
        anchor_dists_to_c = np.linalg.norm(
            anchor_2d[triple_arr] - anchor_2d[c], axis=1
        )
        order = np.argsort(anchor_dists_to_c)[::-1]
        buddies = [triple_arr[k] for k in order[:trio_size - 1]]
        trio = [int(c)] + [int(b) for b in buddies]

        anc_pts = anchor_2d[trio]
        anc_spread = np.linalg.norm(anc_pts.std(axis=0))
        if anc_spread < 0.5: continue

        end_pts = endpoint_2d[trio]
        end_spread = np.linalg.norm(end_pts.std(axis=0))
        chosen_trios.append({
            "indices":         trio,
            "anchor_spread":   float(anc_spread),
            "endpoint_spread": float(end_spread),
            "triple_overlap":  int(triple_overlap[c]),
        })
        for t in trio: used.add(t)
        if len(chosen_trios) >= n_candidates: break

    for k, t in enumerate(chosen_trios):
        ids_short = [sids[i][:8] for i in t['indices']]
        logger.info(f"    Persistence cand #{k+1}: {ids_short}  "
                    f"anc_spread={t['anchor_spread']:.2f}  "
                    f"end_spread={t['endpoint_spread']:.2f}  "
                    f"triple_overlap={t['triple_overlap']}")
    return chosen_trios


# ══════════════════════════════════════════════════════════════════════════════
#  Plot helpers — UMAP overview + zoom
# ══════════════════════════════════════════════════════════════════════════════

def _draw_overview(ax, ref_2d, traj_pts_list, anchor_pts_list, colors, labels):
    sub = np.random.default_rng(0).choice(len(ref_2d),
                                            size=min(20000, len(ref_2d)),
                                            replace=False)
    ax.scatter(ref_2d[sub, 0], ref_2d[sub, 1], s=2, c=COLOR["training"],
                alpha=0.3, rasterized=True)
    for traj_pts, anchor_pt, color, label in zip(
            traj_pts_list, anchor_pts_list, colors, labels):
        ax.plot(traj_pts[:, 0], traj_pts[:, 1], "-o",
                color=color, lw=2, ms=5, mec="white", mew=0.7,
                alpha=0.85, zorder=3)
        ax.scatter(anchor_pt[0], anchor_pt[1], s=240, marker="*",
                    c=COLOR["bg_anchor"], edgecolors=color, linewidths=2,
                    zorder=4, label=label)
        ax.scatter(traj_pts[-1, 0], traj_pts[-1, 1], s=110, marker="s",
                    c=color, edgecolors="white", linewidths=1.4, zorder=4)
    ax.set_xticks([]); ax.set_yticks([])


def _draw_zoom(ax, ref_2d, traj_pts_list, anchor_pts_list, colors,
                zoom_padding=0.6, draw_anchors=True, only_late_iters=False):
    all_pts = np.vstack(traj_pts_list + [a[None, :] for a in anchor_pts_list])
    xmin, xmax = all_pts[:, 0].min() - zoom_padding, all_pts[:, 0].max() + zoom_padding
    ymin, ymax = all_pts[:, 1].min() - zoom_padding, all_pts[:, 1].max() + zoom_padding

    in_zoom = ((ref_2d[:, 0] >= xmin) & (ref_2d[:, 0] <= xmax) &
               (ref_2d[:, 1] >= ymin) & (ref_2d[:, 1] <= ymax))
    ref_zoom = ref_2d[in_zoom]
    if len(ref_zoom) > 5000:
        sub2 = np.random.default_rng(1).choice(len(ref_zoom), 5000, replace=False)
        ref_zoom = ref_zoom[sub2]
    ax.scatter(ref_zoom[:, 0], ref_zoom[:, 1], s=4, c=COLOR["training"],
                alpha=0.35, rasterized=True)

    K = len(traj_pts_list[0])
    for traj_pts, anchor_pt, color in zip(traj_pts_list, anchor_pts_list, colors):
        if only_late_iters:
            seg = traj_pts[-3:]
            ax.plot(seg[:, 0], seg[:, 1], "-o", color=color, lw=2.4, ms=10,
                    mec="white", mew=1, alpha=0.92, zorder=3)
            ax.annotate(f"k={K-3}", seg[0], xytext=(6, 6),
                         textcoords="offset points", fontsize=9, color=color,
                         fontweight="bold")
            ax.annotate(f"k={K-1}", seg[-1], xytext=(6, 6),
                         textcoords="offset points", fontsize=9, color=color,
                         fontweight="bold")
        else:
            ax.plot(traj_pts[:, 0], traj_pts[:, 1], "-o", color=color, lw=2.4,
                    ms=8, mec="white", mew=0.9, alpha=0.92, zorder=3)
        if draw_anchors:
            ax.scatter(anchor_pt[0], anchor_pt[1], s=380, marker="*",
                        c=COLOR["bg_anchor"], edgecolors=color, linewidths=2.2,
                        zorder=4)
        ax.scatter(traj_pts[-1, 0], traj_pts[-1, 1], s=180, marker="s",
                    c=color, edgecolors="white", linewidths=1.6, zorder=4)
        if not only_late_iters and draw_anchors:
            ax.annotate("0", anchor_pt, xytext=(5, 5),
                          textcoords="offset points", fontsize=10,
                          color=color, fontweight="bold")
            ax.annotate(f"{K}", traj_pts[-1], xytext=(5, 5),
                          textcoords="offset points", fontsize=10,
                          color=color, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    return xmin, xmax, ymin, ymax


def _draw_zoom_rect(ax_full, xmin, xmax, ymin, ymax):
    rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                          fill=False, ec=COLOR["accent"], lw=1.6, ls="--",
                          alpha=0.8, zorder=5)
    ax_full.add_patch(rect)


def plot_pair_distance(ax, d, color, title, regime):
    iters = np.arange(len(d))
    ax.plot(iters, d, "o-", color=color, lw=2.5, ms=7, mec="white", mew=1)
    ax.set_xlabel("Iteration $k$")
    ax.set_ylabel(r"$\|z_k^{(a)} - z_k^{(b)}\|_2$")
    ax.set_title(title, fontsize=12)
    ax.text(0.04, 0.94, regime, transform=ax.transAxes,
            fontsize=11, fontweight="bold", color=color, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=1.2))
    ax.grid(True, alpha=0.3)


# ══════════════════════════════════════════════════════════════════════════════
#  FIG 1 — MI collapse
# ══════════════════════════════════════════════════════════════════════════════

def figure_mi_collapse(Z_img, Z_txt, out_path):
    logger.info("[FIG 1] MI collapse...")
    mi_img = estimate_mi_per_iter(Z_img, n_components=8, k=4)
    mi_txt = estimate_mi_per_iter(Z_txt, n_components=8, k=4)
    K = len(mi_img)
    iters = np.arange(K)
    pct_img = 100 * (1 - mi_img[1] / max(mi_img[0], 1e-9))
    pct_txt = 100 * (1 - mi_txt[1] / max(mi_txt[0], 1e-9))
    logger.info(f"  Image MI: {mi_img[0]:.2f} → {mi_img[1]:.2f} → {mi_img[-1]:.2f}  (–{pct_img:.0f}%)")
    logger.info(f"  Text  MI: {mi_txt[0]:.2f} → {mi_txt[1]:.2f} → {mi_txt[-1]:.2f}  (–{pct_txt:.0f}%)")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    panels = [
        (axes[0], mi_img, COLOR["primary"], "Image embedding", pct_img),
        (axes[1], mi_txt, COLOR["secondary"], "Text embedding", pct_txt),
    ]
    for ax, mi, color, label, drop in panels:
        ax.plot(iters, mi, "o-", color=color, lw=3, ms=9, mec="white", mew=1.2)
        ax.annotate("", xy=(1, mi[1]), xytext=(0, mi[0]),
                    arrowprops=dict(arrowstyle="->", color=COLOR["warning"],
                                     lw=2.5, shrinkA=8, shrinkB=8, mutation_scale=20))
        ax.text(0.5, (mi[0] + mi[1]) / 2, f"–{drop:.0f}%",
                ha="center", va="center", fontsize=20, fontweight="bold",
                color=COLOR["warning"],
                bbox=dict(boxstyle="round,pad=0.35", fc="white",
                          ec=COLOR["warning"], lw=1.6))
        ax.set_xlabel("Iteration $k$")
        ax.set_ylabel(r"$I(z_0;\,z_k)$  [nats]")
        ax.set_title(label, fontweight="bold")
        ax.set_xticks(iters)
        ax.grid(True, alpha=0.25)
        ymax = max(mi[0] * 1.18, 0.3)
        ax.set_ylim(-0.05, ymax)
        ax.text(0.02, mi[0] + ymax * 0.04, f"{mi[0]:.2f}",
                fontsize=11, color=color, fontweight="bold")
        ax.text(K - 1.05, mi[-1] + ymax * 0.04, f"{mi[-1]:.2f}",
                fontsize=11, color=color, fontweight="bold")

    fig.suptitle("Anchor information collapses in ONE iteration",
                  fontsize=18, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path); plt.close(fig)
    logger.info(f"  → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  FIG 2 — Lyapunov UMAP (multiple candidates)
# ══════════════════════════════════════════════════════════════════════════════

def figure_lyapunov_candidates(Z_img, A_img, sids, reducer, ref_2d,
                                 out_dir, n_candidates):
    logger.info("[FIG 2] Lyapunov candidates...")
    sys_pairs = find_lambda_sys_candidates(Z_img, A_img, sids, reducer,
                                              n_candidates=n_candidates)
    a_pairs   = find_lambda_a_candidates(Z_img, A_img, sids, reducer,
                                            n_candidates=n_candidates)

    n_show = min(len(sys_pairs), len(a_pairs))
    if n_show == 0:
        logger.warning("  No valid candidates found.")
        return

    K = Z_img.shape[1]
    for v in range(n_show):
        i_sys, j_sys = sys_pairs[v][0], sys_pairs[v][1]
        i_a,   j_a   = a_pairs[v][0],   a_pairs[v][1]

        # Project trajectories
        anc_sys_a = reducer.transform(A_img[i_sys][None, :])[0]
        anc_sys_b = reducer.transform(A_img[j_sys][None, :])[0]
        traj_sys_a = reducer.transform(Z_img[i_sys])
        traj_sys_b = reducer.transform(Z_img[j_sys])

        anc_a_a = reducer.transform(A_img[i_a][None, :])[0]
        anc_a_b = reducer.transform(A_img[j_a][None, :])[0]
        traj_a_a = reducer.transform(Z_img[i_a])
        traj_a_b = reducer.transform(Z_img[j_a])

        d_sys = np.array([np.linalg.norm(Z_img[i_sys, k] - Z_img[j_sys, k])
                            for k in range(K)])
        d_a = np.array([np.linalg.norm(Z_img[i_a, k] - Z_img[j_a, k])
                          for k in range(K)])

        # Build figure: 2 rows × 3 cols (full UMAP, zoom UMAP, distance plot)
        fig = plt.figure(figsize=(17, 10))
        gs = fig.add_gridspec(2, 3, width_ratios=[1.5, 1.3, 1.0],
                                hspace=0.35, wspace=0.32)

        # ── Row 0: λ_sys ─────────────────────────────────────────────────────
        ax_full = fig.add_subplot(gs[0, 0])
        ax_zoom = fig.add_subplot(gs[0, 1])
        labels_sys = [f"Anchor A ({sids[i_sys][:8]})",
                       f"Anchor B ({sids[j_sys][:8]})"]
        colors_sys = [COLOR["primary"], COLOR["warning"]]
        traj_list_sys = [traj_sys_a, traj_sys_b]
        anc_list_sys  = [anc_sys_a, anc_sys_b]

        _draw_overview(ax_full, ref_2d, traj_list_sys, anc_list_sys,
                        colors_sys, labels_sys)
        ax_full.legend(loc="upper right", framealpha=0.9, fontsize=10)
        ax_full.set_xlabel("UMAP 1"); ax_full.set_ylabel("UMAP 2")
        ax_full.set_title(r"$\lambda_{\rm sys}$:  Different anchors $\to$ trajectories CONVERGE",
                           fontsize=13)
        xmin, xmax, ymin, ymax = _draw_zoom(
            ax_zoom, ref_2d, traj_list_sys, anc_list_sys, colors_sys)
        ax_zoom.set_xlabel("UMAP 1"); ax_zoom.set_ylabel("UMAP 2")
        ax_zoom.set_title("Zoom: trajectory action region", fontsize=13)
        _draw_zoom_rect(ax_full, xmin, xmax, ymin, ymax)

        ax_d = fig.add_subplot(gs[0, 2])
        plot_pair_distance(ax_d, d_sys, COLOR["accent"],
                            title="Distance vs iteration",
                            regime=r"$\|z_k^{(a)} - z_k^{(b)}\|$ contracts")

        # ── Row 1: λ_a ───────────────────────────────────────────────────────
        ax_full2 = fig.add_subplot(gs[1, 0])
        ax_zoom2 = fig.add_subplot(gs[1, 1])
        labels_a = [f"Near-twin a ({sids[i_a][:8]})",
                     f"Near-twin b ({sids[j_a][:8]})"]
        colors_a = [COLOR["primary"], COLOR["secondary"]]
        traj_list_a = [traj_a_a, traj_a_b]
        anc_list_a  = [anc_a_a, anc_a_b]

        _draw_overview(ax_full2, ref_2d, traj_list_a, anc_list_a,
                        colors_a, labels_a)
        ax_full2.legend(loc="upper right", framealpha=0.9, fontsize=10)
        ax_full2.set_xlabel("UMAP 1"); ax_full2.set_ylabel("UMAP 2")
        ax_full2.set_title(r"$\lambda_a$:  Near-identical anchors $\to$ trajectories DIVERGE",
                            fontsize=13)
        xmin2, xmax2, ymin2, ymax2 = _draw_zoom(
            ax_zoom2, ref_2d, traj_list_a, anc_list_a, colors_a)
        ax_zoom2.set_xlabel("UMAP 1"); ax_zoom2.set_ylabel("UMAP 2")
        ax_zoom2.set_title("Zoom: trajectory action region", fontsize=13)
        _draw_zoom_rect(ax_full2, xmin2, xmax2, ymin2, ymax2)

        ax_d2 = fig.add_subplot(gs[1, 2])
        plot_pair_distance(ax_d2, d_a, COLOR["warning"],
                            title="Distance vs iteration",
                            regime=r"$\|z_k^{(a)} - z_k^{(b)}\|$ expands")

        fig.suptitle(f"Two Lyapunov regimes (candidate v{v+1})",
                      fontsize=16, fontweight="bold", y=1.00)
        out_path = os.path.join(out_dir, f"talk_fig2_lyapunov_umap_v{v+1}.png")
        plt.savefig(out_path); plt.close(fig)
        logger.info(f"  → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  FIG 3 — Persistence (multiple candidates)
# ══════════════════════════════════════════════════════════════════════════════

def figure_persistence_candidates(Z_img, A_img, sids, reducer, ref_2d,
                                    out_dir, n_candidates):
    logger.info("[FIG 3] Persistence candidates...")
    global A_img_global
    A_img_global = A_img
    trios = find_persistence_candidates(Z_img, sids, reducer,
                                          n_candidates=n_candidates)

    if not trios:
        logger.warning("  No trios found.")
        return

    # Compute global Block J statistics for the stat panel
    K = Z_img.shape[1]
    Z_n = normalize(Z_img.reshape(-1, Z_img.shape[-1])).reshape(Z_img.shape)
    knn = NearestNeighbors(n_neighbors=11, metric="cosine",
                            algorithm="brute", n_jobs=-1)
    jaccards = []
    for k_ in range(K - 1):
        knn.fit(Z_n[:, k_, :])
        _, ka = knn.kneighbors(Z_n[:, k_, :])
        knn.fit(Z_n[:, k_ + 1, :])
        _, kb = knn.kneighbors(Z_n[:, k_ + 1, :])
        ja = []
        for i in range(len(Z_n)):
            a = set(ka[i, 1:]); b = set(kb[i, 1:])
            ja.append(len(a & b) / max(len(a | b), 1))
        jaccards.append(np.mean(ja))
    j_mean_late = np.mean(jaccards[-5:])
    Nn = len(Z_n)
    random_baseline = 10 * 10 / (Nn - 1) / (2 * 10 - 100/(Nn - 1))
    times_random = j_mean_late / max(random_baseline, 1e-9)
    lag1 = []
    for k_ in range(K - 1):
        sims = (Z_n[:, k_, :] * Z_n[:, k_ + 1, :]).sum(axis=1)
        lag1.append(float(sims.mean()))
    lag1_mean = np.mean(lag1[-5:])
    logger.info(f"  Global stats: kNN Jaccard {j_mean_late:.3f} ({times_random:.0f}× rnd), "
                f"lag-1 cosine {lag1_mean:.2f}")

    palette = [COLOR["primary"], COLOR["warning"], COLOR["good"]]

    for v, trio in enumerate(trios):
        indices = trio["indices"]
        anchors = []
        trajs = []
        for idx in indices:
            anchors.append(reducer.transform(A_img[idx][None, :])[0])
            trajs.append(reducer.transform(Z_img[idx]))
        labels = [f"{sids[i][:8]}" for i in indices]
        colors = palette[:len(indices)]

        # Layout: 2 rows × 3 cols
        # Top:    full UMAP (col 0) + zoom UMAP (col 1) + memory box (col 2)
        # Bottom: per-traj step trace (cols 0-1) + Jaccard bar (col 2)
        fig = plt.figure(figsize=(17, 10))
        gs = fig.add_gridspec(2, 3, width_ratios=[1.4, 1.4, 1.0],
                                hspace=0.32, wspace=0.30)

        ax_full = fig.add_subplot(gs[0, 0])
        ax_zoom = fig.add_subplot(gs[0, 1])

        # Full overview — fade-in by iter for the iteration progression sense
        sub = np.random.default_rng(0).choice(len(ref_2d),
                                                size=min(20000, len(ref_2d)),
                                                replace=False)
        ax_full.scatter(ref_2d[sub, 0], ref_2d[sub, 1], s=2,
                         c=COLOR["training"], alpha=0.3, rasterized=True)
        for traj_pts, anc_pt, color, label in zip(trajs, anchors, colors, labels):
            ax_full.plot(traj_pts[:, 0], traj_pts[:, 1], "-",
                          color=color, lw=1.6, alpha=0.55, zorder=3)
            for k in range(K):
                alpha = 0.3 + 0.7 * (k / max(K - 1, 1))
                size  = 25 + 70 * (k / max(K - 1, 1))
                ax_full.scatter(traj_pts[k, 0], traj_pts[k, 1], s=size, c=color,
                                 alpha=alpha, edgecolors="white", linewidths=0.6,
                                 zorder=4)
            ax_full.scatter(anc_pt[0], anc_pt[1], s=280, marker="*",
                             c=COLOR["bg_anchor"], edgecolors=color, linewidths=2,
                             zorder=5, label=label)
        ax_full.legend(loc="upper right", framealpha=0.9, fontsize=10)
        ax_full.set_xlabel("UMAP 1"); ax_full.set_ylabel("UMAP 2")
        ax_full.set_title("Trajectories trapped in a local neighborhood",
                           fontsize=13)
        ax_full.set_xticks([]); ax_full.set_yticks([])

        # Zoom — only late iterations + persistent-neighborhood circle
        xmin, xmax, ymin, ymax = _draw_zoom(
            ax_zoom, ref_2d, trajs, anchors, colors,
            zoom_padding=0.4, draw_anchors=False, only_late_iters=True)
        endpoints = np.array([t[-1] for t in trajs])
        cx, cy = endpoints.mean(axis=0)
        radius = max(np.std(endpoints, axis=0).max(), 0.3) * 2.5
        circ = plt.Circle((cx, cy), radius, fill=False, ls="--", lw=2.0,
                           ec=COLOR["accent"], alpha=0.8, zorder=2)
        ax_zoom.add_patch(circ)
        ax_zoom.set_xlabel("UMAP 1"); ax_zoom.set_ylabel("UMAP 2")
        ax_zoom.set_title("Zoom: persistent neighborhood\n(last 3 iterations)",
                           fontsize=13)
        _draw_zoom_rect(ax_full, xmin, xmax, ymin, ymax)

        # Right column row 0: trajectory memory box
        ax_a = fig.add_subplot(gs[0, 2])
        ax_a.axis("off")
        ax_a.text(0.05, 0.92, "Trajectory memory", fontsize=14, fontweight="bold",
                   transform=ax_a.transAxes, color=COLOR["accent"])
        ax_a.text(0.05, 0.74,
                   f"lag-1 cosine\n  ≈  {lag1_mean:.2f}",
                   fontsize=14, transform=ax_a.transAxes, color=COLOR["primary"],
                   fontweight="bold")
        ax_a.text(0.05, 0.55, "Random-pair\nbaseline ≈ 0.14",
                   fontsize=11, transform=ax_a.transAxes, color=COLOR["muted"])
        ax_a.text(0.05, 0.32,
                   "Trajectories REMEMBER\nwhere they were several\niterations earlier.",
                   fontsize=11, transform=ax_a.transAxes,
                   style="italic", color=COLOR["accent"])
        ax_a.add_patch(plt.Rectangle((0.02, 0.04), 0.96, 0.94,
                                       fill=False, ec=COLOR["accent"], lw=1,
                                       transform=ax_a.transAxes))

        # Bottom row: per-traj step size in UMAP + Jaccard bar
        ax_steps = fig.add_subplot(gs[1, 0:2])
        for c, anc, traj in zip(colors, anchors, trajs):
            full = np.vstack([anc[None, :], traj])
            steps = np.linalg.norm(np.diff(full, axis=0), axis=1)
            ax_steps.plot(np.arange(1, len(steps) + 1), steps,
                           "o-", color=c, lw=2, ms=6, mec="white", mew=0.8,
                           alpha=0.85)
        ax_steps.set_xlabel("Iteration $k$")
        ax_steps.set_ylabel("UMAP step size")
        ax_steps.set_title("Per-trajectory step size — small late-iter steps = staying put",
                            fontsize=12)
        ax_steps.grid(True, alpha=0.3)

        ax_j = fig.add_subplot(gs[1, 2])
        bars = ax_j.bar([0, 1, 2],
                         [random_baseline, j_mean_late, 1.0],
                         color=[COLOR["muted"], COLOR["primary"], COLOR["good"]],
                         edgecolor="white", linewidth=1.5)
        ax_j.set_xticks([0, 1, 2])
        ax_j.set_xticklabels(["Random", "Observed", "Perfect"], fontsize=10)
        ax_j.set_ylabel("kNN Jaccard")
        ax_j.set_title("Neighborhood persistence", fontsize=12)
        ax_j.set_ylim(0, 1.05)
        for b, val in zip(bars, [random_baseline, j_mean_late, 1.0]):
            ax_j.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                       f"{val:.3f}" if val < 0.05 else f"{val:.2f}",
                       ha="center", fontsize=10, fontweight="bold")
        ax_j.text(0.5, 0.85, f"~{times_random:.0f}× above random",
                   transform=ax_j.transAxes, ha="center",
                   fontsize=11, fontweight="bold", color=COLOR["primary"],
                   bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec=COLOR["primary"], lw=1.3))

        fig.suptitle(f"Inside the attractor: trajectories drift slowly through "
                      f"local neighborhoods (candidate v{v+1})",
                      fontsize=15, fontweight="bold", y=0.99)

        out_path = os.path.join(out_dir, f"talk_fig3_persistence_umap_v{v+1}.png")
        plt.savefig(out_path); plt.close(fig)
        logger.info(f"  → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  FIG 4 — Modality asymmetry (NEW): image drifts IN, text drifts OUT
# ══════════════════════════════════════════════════════════════════════════════

def _knn_distance_to_training(query_2d, ref_2d, k=10):
    """For each query point, distance to its k-th nearest training neighbor in
    UMAP space. Smaller = closer to dense training region. Returns (Q,) array.
    """
    knn = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(ref_2d)
    dists, _ = knn.kneighbors(query_2d)
    return dists[:, -1]


def _knn_cosine_distance_to_training_256d(query, ref_256, k=10):
    """For each query point in 256-d, MEAN cosine distance to its k nearest
    training neighbors. This is the same metric that Block I uses to detect
    modality-asymmetric drift; it survives in 256-d but collapses to noise
    in the UMAP-2D projection. Smaller = closer to training distribution.
    Returns (Q,) array.
    """
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

    role='image_in':  iter-0 should start far from training core (high d), iter-K close (low d)
    role='text_out':  iter-0 should start close (low d), iter-K far (high d)
    """
    d0 = _knn_cosine_distance_to_training_256d(A_emb,         ref_256, k=10)
    dK = _knn_cosine_distance_to_training_256d(Z_traj[:, -1, :], ref_256, k=10)
    if role == "image_in":
        score = d0 - dK
    elif role == "text_out":
        score = dK - d0
    else:
        raise ValueError(f"unknown role {role}")

    # Monotone-drift filter using all intermediate iterations.
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
    """Find an exemplar trajectory whose iter-0→iter-K UMAP path matches the
    expected modality drift direction.

    role='image_in':  iter-0 should start far from training core, iter-K close
    role='text_out':  iter-0 should start close, iter-K far

    Scores by (raw drift in expected direction) − (drift in wrong direction).
    """
    A_2d = reducer.transform(A_emb)
    K_2d = reducer.transform(Z_traj[:, -1, :])

    d0 = _knn_distance_to_training(A_2d,  ref_2d, k=10)
    dK = _knn_distance_to_training(K_2d,  ref_2d, k=10)

    if role == "image_in":
        # We want d0 large (peripheral) and dK small (core).
        score = d0 - dK
    elif role == "text_out":
        # We want d0 small (core) and dK large (peripheral).
        score = dK - d0
    else:
        raise ValueError(f"unknown role {role}")

    # Filter to trajectories where the drift is monotone enough — at least
    # 70% of intermediate points should also be on the right side of the
    # baseline d0.
    K_iters = Z_traj.shape[1]
    inter_2d = np.stack([reducer.transform(Z_traj[:, k, :]) for k in range(K_iters)],
                         axis=1)   # (N, K, 2)
    monotone_score = np.zeros(len(Z_traj))
    for i in range(len(Z_traj)):
        d_per_iter = _knn_distance_to_training(inter_2d[i], ref_2d, k=10)
        if role == "image_in":
            # Each later iter should be ≤ d0
            ok = (d_per_iter[1:] <= d_per_iter[0] + 0.5).sum() / (K_iters - 1)
        else:
            ok = (d_per_iter[1:] >= d_per_iter[0] - 0.5).sum() / (K_iters - 1)
        monotone_score[i] = ok

    combined = score * (monotone_score > 0.6).astype(float)
    rank = np.argsort(combined)[::-1]

    # Return top candidate index + diagnostic info
    best = int(rank[0])
    return best, float(d0[best]), float(dK[best])


def figure_modality_asymmetry(Z_img, Z_txt, A_img, A_txt, sids,
                                reducer_img, ref_img_2d,
                                reducer_txt, ref_txt_2d,
                                ref_img_256, ref_txt_256,
                                out_path):
    """Two-row figure: image trajectory drifting INTO MedCLIP training
    distribution, text trajectory drifting OUT of it. Plus cohort-level
    histograms of 256-d cosine kNN distance to training, iter 0 vs iter K.

    Story: ChexGen pulls images toward its training prior (Block I:
    image kNN-cos 0.194 → 0.172); MAIRA-2 pushes text outside MedCLIP's
    text training manifold (text kNN-cos 0.133 → 0.170).

    The asymmetry signal lives in 256-d cosine geometry — the UMAP-2D
    projection compresses it away. So we use 256-d cosine kNN for the
    quantitative cohort histograms (right column) and use the 2D UMAP
    only for the visual exemplar trajectory plot (left column).

    If ref_img_256/ref_txt_256 are None, we fall back to UMAP-2D
    distances and warn the user.
    """
    logger.info("[FIG 4] Modality asymmetry — picking exemplar trajectories...")

    have_256d = (ref_img_256 is not None) and (ref_txt_256 is not None)
    if have_256d:
        i_img, d0_img, dK_img, d_per_img = _pick_modality_asymmetry_traj_256d(
            Z_img, A_img, ref_img_256, role="image_in")
        i_txt, d0_txt, dK_txt, d_per_txt = _pick_modality_asymmetry_traj_256d(
            Z_txt, A_txt, ref_txt_256, role="text_out")
        metric_label = "256-d cosine"
    else:
        logger.warning("  Falling back to UMAP-2D picker (256-d ref embeddings missing)")
        i_img, d0_img, dK_img = _pick_modality_asymmetry_traj(
            Z_img, A_img, reducer_img, ref_img_2d, role="image_in")
        i_txt, d0_txt, dK_txt = _pick_modality_asymmetry_traj(
            Z_txt, A_txt, reducer_txt, ref_txt_2d, role="text_out")
        d_per_img = d_per_txt = None
        metric_label = "UMAP-2D euclidean (FALLBACK)"

    logger.info(f"  Image-IN  exemplar: idx={i_img} ({sids[i_img][:8]})  "
                f"d_kNN(iter 0)={d0_img:.3f} → d_kNN(iter K)={dK_img:.3f} ({metric_label})")
    logger.info(f"  Text-OUT  exemplar: idx={i_txt} ({sids[i_txt][:8]})  "
                f"d_kNN(iter 0)={d0_txt:.3f} → d_kNN(iter K)={dK_txt:.3f} ({metric_label})")

    # The UMAP visual stays — it shows ONE trajectory making its drift through
    # the embedding cloud. The cohort claim comes from the 256-d histogram.
    A_img_2d   = reducer_img.transform(A_img[i_img][None, :])[0]
    traj_img_2d = reducer_img.transform(Z_img[i_img])
    A_txt_2d   = reducer_txt.transform(A_txt[i_txt][None, :])[0]
    traj_txt_2d = reducer_txt.transform(Z_txt[i_txt])

    # Cohort-level kNN distances. Use 256-d cosine if available (real signal),
    # else fall back to UMAP-2D euclidean (weaker, possibly washed out).
    if have_256d:
        cohort_d0_img = _knn_cosine_distance_to_training_256d(A_img,         ref_img_256, k=10)
        cohort_dK_img = _knn_cosine_distance_to_training_256d(Z_img[:, -1, :], ref_img_256, k=10)
        cohort_d0_txt = _knn_cosine_distance_to_training_256d(A_txt,         ref_txt_256, k=10)
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
    logger.info(f"  Cohort image: median {med_d0_img:.3f} → {med_dK_img:.3f}  "
                f"(Δ={delta_img:+.3f}; {'IN' if delta_img < 0 else 'OUT'})")
    logger.info(f"  Cohort text : median {med_d0_txt:.3f} → {med_dK_txt:.3f}  "
                f"(Δ={delta_txt:+.3f}; {'OUT' if delta_txt > 0 else 'IN'})")

    # ─── Plot: 2 rows × 2 cols (UMAP exemplar, cohort histogram) ─────────────
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.7, 1], hspace=0.35, wspace=0.25)

    # Row 0 — Image: drift IN
    ax_img = fig.add_subplot(gs[0, 0])
    sub = np.random.default_rng(0).choice(len(ref_img_2d),
                                            size=min(20000, len(ref_img_2d)),
                                            replace=False)
    ax_img.scatter(ref_img_2d[sub, 0], ref_img_2d[sub, 1], s=3,
                    c=COLOR["training"], alpha=0.35, rasterized=True)
    # Iter dots fade light → saturated as we approach core
    K_n = len(traj_img_2d)
    for k in range(K_n):
        alpha = 0.3 + 0.7 * (k / max(K_n - 1, 1))
        size = 40 + 100 * (k / max(K_n - 1, 1))
        ax_img.scatter(traj_img_2d[k, 0], traj_img_2d[k, 1],
                        s=size, c=COLOR["primary"], alpha=alpha,
                        edgecolors="white", linewidths=0.8, zorder=4)
    ax_img.plot(traj_img_2d[:, 0], traj_img_2d[:, 1], "-",
                 color=COLOR["primary"], lw=2.5, alpha=0.7, zorder=3)
    ax_img.scatter(A_img_2d[0], A_img_2d[1], s=380, marker="*",
                    c=COLOR["bg_anchor"], edgecolors=COLOR["primary"],
                    linewidths=2.4, zorder=5,
                    label=f"Anchor (iter 0)")
    ax_img.scatter(traj_img_2d[-1, 0], traj_img_2d[-1, 1], s=180, marker="s",
                    c=COLOR["primary"], edgecolors="white", linewidths=2,
                    zorder=5, label="Iter 10")
    ax_img.legend(loc="upper right", framealpha=0.92, fontsize=11)
    ax_img.set_xlabel("UMAP 1"); ax_img.set_ylabel("UMAP 2")
    ax_img.set_title("Image trajectory: drifts INTO MedCLIP training distribution",
                      fontweight="bold", fontsize=14)
    ax_img.set_xticks([]); ax_img.set_yticks([])

    # Row 0 right — cohort histogram, image
    ax_img_h = fig.add_subplot(gs[0, 1])
    bins = np.linspace(0, max(cohort_d0_img.max(), cohort_dK_img.max()) * 1.05, 35)
    ax_img_h.hist(cohort_d0_img, bins=bins, color=COLOR["muted"], alpha=0.55,
                   edgecolor="white", label=f"Iter 0  (med {med_d0_img:.3f})")
    ax_img_h.hist(cohort_dK_img, bins=bins, color=COLOR["primary"], alpha=0.75,
                   edgecolor="white", label=f"Iter 10 (med {med_dK_img:.3f})")
    ax_img_h.axvline(med_d0_img, color=COLOR["muted"], ls="--", lw=1.5)
    ax_img_h.axvline(med_dK_img, color=COLOR["primary"], ls="--", lw=1.5)
    ax_img_h.set_xlabel(x_axis_label, fontsize=10)
    ax_img_h.set_ylabel("Trajectories")
    ax_img_h.set_title(f"Image cohort ($n={len(cohort_d0_img)}$): "
                         f"Δmedian = {delta_img:+.3f}", fontsize=12)
    ax_img_h.legend(fontsize=10, framealpha=0.9)
    arrow_word = "← LEFT (closer)" if delta_img < 0 else "RIGHT (farther) →"
    ax_img_h.text(0.97, 0.85,
                   f"distribution shifts\n{arrow_word}",
                   transform=ax_img_h.transAxes, ha="right",
                   fontsize=11, color=COLOR["primary"], fontweight="bold",
                   bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec=COLOR["primary"], lw=1.4))

    # Row 1 — Text: drift OUT
    ax_txt = fig.add_subplot(gs[1, 0])
    sub_t = np.random.default_rng(1).choice(len(ref_txt_2d),
                                              size=min(20000, len(ref_txt_2d)),
                                              replace=False)
    ax_txt.scatter(ref_txt_2d[sub_t, 0], ref_txt_2d[sub_t, 1], s=3,
                    c=COLOR["training"], alpha=0.35, rasterized=True)
    for k in range(K_n):
        alpha = 0.3 + 0.7 * (k / max(K_n - 1, 1))
        size = 40 + 100 * (k / max(K_n - 1, 1))
        ax_txt.scatter(traj_txt_2d[k, 0], traj_txt_2d[k, 1],
                        s=size, c=COLOR["warning"], alpha=alpha,
                        edgecolors="white", linewidths=0.8, zorder=4)
    ax_txt.plot(traj_txt_2d[:, 0], traj_txt_2d[:, 1], "-",
                 color=COLOR["warning"], lw=2.5, alpha=0.7, zorder=3)
    ax_txt.scatter(A_txt_2d[0], A_txt_2d[1], s=380, marker="*",
                    c=COLOR["bg_anchor"], edgecolors=COLOR["warning"],
                    linewidths=2.4, zorder=5, label="Anchor (iter 0)")
    ax_txt.scatter(traj_txt_2d[-1, 0], traj_txt_2d[-1, 1], s=180, marker="s",
                    c=COLOR["warning"], edgecolors="white", linewidths=2,
                    zorder=5, label="Iter 10")
    ax_txt.legend(loc="upper right", framealpha=0.92, fontsize=11)
    ax_txt.set_xlabel("UMAP 1"); ax_txt.set_ylabel("UMAP 2")
    ax_txt.set_title("Text trajectory: drifts OUT of MedCLIP training distribution",
                      fontweight="bold", fontsize=14)
    ax_txt.set_xticks([]); ax_txt.set_yticks([])

    # Row 1 right — cohort histogram, text
    ax_txt_h = fig.add_subplot(gs[1, 1])
    bins = np.linspace(0, max(cohort_d0_txt.max(), cohort_dK_txt.max()) * 1.05, 35)
    ax_txt_h.hist(cohort_d0_txt, bins=bins, color=COLOR["muted"], alpha=0.55,
                   edgecolor="white", label=f"Iter 0  (med {med_d0_txt:.3f})")
    ax_txt_h.hist(cohort_dK_txt, bins=bins, color=COLOR["warning"], alpha=0.75,
                   edgecolor="white", label=f"Iter 10 (med {med_dK_txt:.3f})")
    ax_txt_h.axvline(med_d0_txt, color=COLOR["muted"], ls="--", lw=1.5)
    ax_txt_h.axvline(med_dK_txt, color=COLOR["warning"], ls="--", lw=1.5)
    ax_txt_h.set_xlabel(x_axis_label, fontsize=10)
    ax_txt_h.set_ylabel("Trajectories")
    ax_txt_h.set_title(f"Text cohort ($n={len(cohort_d0_txt)}$): "
                         f"Δmedian = {delta_txt:+.3f}", fontsize=12)
    ax_txt_h.legend(fontsize=10, framealpha=0.9)
    arrow_word = "RIGHT (farther) →" if delta_txt > 0 else "← LEFT (closer)"
    ax_txt_h.text(0.97, 0.85,
                   f"distribution shifts\n{arrow_word}",
                   transform=ax_txt_h.transAxes, ha="right",
                   fontsize=11, color=COLOR["warning"], fontweight="bold",
                   bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec=COLOR["warning"], lw=1.4))

    fig.suptitle("Asymmetric drift:  ChexGen pulls images IN, "
                  "MAIRA-2 pushes text OUT",
                  fontsize=16, fontweight="bold", y=1.00)
    plt.savefig(out_path); plt.close(fig)
    logger.info(f"  → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  FIG 5 — Improved kNN persistence (REPLACES Fig 3 in talk)
# ══════════════════════════════════════════════════════════════════════════════

def _pick_persistent_pair(Z_img, k_nn=10, n_candidates=4):
    """Find pairs of trajectories that ended up as mutual kNN neighbors at
    BOTH iter K-2 and iter K (max overlap = persistence). Returns sorted list
    of (i, j, persistence_score) tuples.
    """
    N, K, D = Z_img.shape
    Z_n = normalize(Z_img.reshape(-1, D)).reshape(N, K, D)

    knn = NearestNeighbors(n_neighbors=k_nn + 1, metric="cosine",
                            algorithm="brute", n_jobs=-1)
    knn.fit(Z_n[:, -1, :])
    _, idx_K = knn.kneighbors(Z_n[:, -1, :])
    knn.fit(Z_n[:, -3, :])
    _, idx_Km2 = knn.kneighbors(Z_n[:, -3, :])

    # For each trajectory, find buddies present at both iter K-2 and K
    pairs = []
    used = set()
    overlap_count = np.zeros(N, dtype=int)
    for i in range(N):
        common = set(idx_K[i, 1:]) & set(idx_Km2[i, 1:])
        overlap_count[i] = len(common)
    rank = np.argsort(overlap_count)[::-1]
    for c in rank:
        if int(c) in used: continue
        if overlap_count[c] < 4: break
        # Pick the buddy that's also a high-overlap trajectory
        for j_cand in (set(idx_K[c, 1:]) & set(idx_Km2[c, 1:])):
            if int(j_cand) in used: continue
            pairs.append((int(c), int(j_cand), int(overlap_count[c])))
            used.add(int(c)); used.add(int(j_cand))
            break
        if len(pairs) >= n_candidates: break
    return pairs


def figure_persistence_v3(Z_img, A_img, sids, reducer, ref_2d,
                            out_dir, n_candidates=4, k_nn=10):
    """Improved kNN persistence figure for the talk: lead with intuition.

    Three panels:
      Left: two trajectories shown across iter K-4..K with explicit
            "buddy lines" connecting them at each iteration. Visual
            story: "these two trajectories are close AND stay close."
      Middle: kNN Jaccard bar chart with prominent multiplier callout.
      Right: autocorrelation curve vs lag, comparing observed to random
             baseline. Visual story: trajectories have memory.
    """
    logger.info("[FIG 5] Persistence v3 — picking buddy pairs...")
    pairs = _pick_persistent_pair(Z_img, k_nn=k_nn, n_candidates=n_candidates)
    if not pairs:
        logger.warning("  No pairs found.")
        return
    logger.info(f"  Selected {len(pairs)} candidate pairs")
    for k, (i, j, ovl) in enumerate(pairs):
        logger.info(f"    cand #{k+1}: {sids[i][:8]} & {sids[j][:8]}  "
                    f"persistence_overlap={ovl}")

    # Compute global stats
    K = Z_img.shape[1]
    Z_n = normalize(Z_img.reshape(-1, Z_img.shape[-1])).reshape(Z_img.shape)
    knn = NearestNeighbors(n_neighbors=k_nn + 1, metric="cosine",
                            algorithm="brute", n_jobs=-1)
    jaccards_per_iter = []
    for k_ in range(K - 1):
        knn.fit(Z_n[:, k_, :])
        _, ka = knn.kneighbors(Z_n[:, k_, :])
        knn.fit(Z_n[:, k_ + 1, :])
        _, kb = knn.kneighbors(Z_n[:, k_ + 1, :])
        ja = []
        for i in range(len(Z_n)):
            a = set(ka[i, 1:]); b = set(kb[i, 1:])
            ja.append(len(a & b) / max(len(a | b), 1))
        jaccards_per_iter.append(np.mean(ja))
    j_mean_late = np.mean(jaccards_per_iter[-5:])
    Nn = len(Z_n)
    random_baseline = k_nn * k_nn / (Nn - 1) / (2 * k_nn - k_nn * k_nn / (Nn - 1))
    times_random = j_mean_late / max(random_baseline, 1e-9)

    # Autocorrelation at increasing lags. For each lag ℓ we average ⟨z_k · z_{k+ℓ}⟩
    # over the LATE-regime iters where both k and k+ℓ exist (i.e., the asymptotic
    # attractor regime, not the early transient where MI is still falling).
    # Window: average over the last (K-1-lag)//2 + 1 valid pairs at minimum,
    # and over all valid late pairs from the second half onwards if available.
    max_lag = min(5, K - 1)
    autocorr_obs = []
    for lag in range(1, max_lag + 1):
        # Valid k indices: 0..K-1-lag inclusive. Restrict to late half.
        k_lo = max((K - 1) // 2, 0)
        k_hi = K - 1 - lag           # inclusive upper bound for k_
        if k_hi < k_lo:
            # Fall back to using whatever valid pairs exist
            k_lo = 0
        sims = []
        for k_ in range(k_lo, k_hi + 1):
            sims.append((Z_n[:, k_, :] * Z_n[:, k_ + lag, :]).sum(axis=1).mean())
        autocorr_obs.append(float(np.mean(sims)) if sims else float("nan"))
    # Random-pair baseline at any lag
    rng = np.random.default_rng(0)
    perm = rng.permutation(Nn)
    rand_sims = (Z_n[:, -1, :] * Z_n[perm, -1, :]).sum(axis=1).mean()
    autocorr_rand = float(rand_sims)
    logger.info(f"  Late kNN Jaccard {j_mean_late:.3f} ({times_random:.0f}× rnd)")
    logger.info(f"  Autocorr lags 1..{max_lag}: {[f'{a:.2f}' for a in autocorr_obs]}")
    logger.info(f"  Random-pair baseline: {autocorr_rand:.3f}")

    for v, (i, j, ovl) in enumerate(pairs):
        # Project trajectories — show only LAST 5 iters for clarity
        K_show = 5
        traj_i = reducer.transform(Z_img[i, -K_show:])
        traj_j = reducer.transform(Z_img[j, -K_show:])

        fig = plt.figure(figsize=(16, 5.5))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.6, 1, 1], wspace=0.3)

        # ─── Left: buddy diagram ─────────────────────────────────────────────
        ax = fig.add_subplot(gs[0, 0])
        all_pts = np.vstack([traj_i, traj_j])
        pad = 0.6
        xmin, xmax = all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad
        ymin, ymax = all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad

        in_zoom = ((ref_2d[:, 0] >= xmin) & (ref_2d[:, 0] <= xmax) &
                   (ref_2d[:, 1] >= ymin) & (ref_2d[:, 1] <= ymax))
        ref_zoom = ref_2d[in_zoom]
        if len(ref_zoom) > 5000:
            sub2 = np.random.default_rng(1).choice(len(ref_zoom), 5000, replace=False)
            ref_zoom = ref_zoom[sub2]
        ax.scatter(ref_zoom[:, 0], ref_zoom[:, 1], s=4, c=COLOR["training"],
                    alpha=0.3, rasterized=True)

        # Each trajectory as a fading line + iter dots
        for traj, color, label in [(traj_i, COLOR["primary"], sids[i][:8]),
                                      (traj_j, COLOR["warning"], sids[j][:8])]:
            ax.plot(traj[:, 0], traj[:, 1], "-", color=color, lw=2.4,
                     alpha=0.85, zorder=3)
            for kk in range(K_show):
                alpha = 0.4 + 0.6 * (kk / max(K_show - 1, 1))
                size = 80 + 90 * (kk / max(K_show - 1, 1))
                ax.scatter(traj[kk, 0], traj[kk, 1], s=size, c=color,
                            alpha=alpha, edgecolors="white", linewidths=0.9,
                            zorder=4, label=label if kk == K_show - 1 else None)

        # "Buddy lines" — connect the two trajectories at each iteration
        for kk in range(K_show):
            ax.plot([traj_i[kk, 0], traj_j[kk, 0]],
                     [traj_i[kk, 1], traj_j[kk, 1]],
                     "--", color=COLOR["accent"], lw=1.0, alpha=0.45,
                     zorder=2)

        # Iter labels at start and end of each trajectory
        for traj, color in [(traj_i, COLOR["primary"]),
                              (traj_j, COLOR["warning"])]:
            ax.annotate(f"k={K - K_show}", traj[0], xytext=(7, 7),
                          textcoords="offset points", fontsize=10,
                          color=color, fontweight="bold")
            ax.annotate(f"k={K - 1}", traj[-1], xytext=(7, 7),
                          textcoords="offset points", fontsize=10,
                          color=color, fontweight="bold")

        ax.legend(loc="upper right", framealpha=0.9, fontsize=10)
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        ax.set_title("Two trajectories: close at iter $K-4$, "
                      "still close at iter $K$\n(dashed = pair-distance per step)",
                      fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)

        # ─── Middle: kNN Jaccard bars ─────────────────────────────────────────
        ax_b = fig.add_subplot(gs[0, 1])
        bars = ax_b.bar([0, 1, 2],
                         [random_baseline, j_mean_late, 1.0],
                         color=[COLOR["muted"], COLOR["primary"], COLOR["good"]],
                         edgecolor="white", linewidth=2)
        ax_b.set_xticks([0, 1, 2])
        ax_b.set_xticklabels(["Random\nshuffle", "Observed", "Perfect"],
                               fontsize=10)
        ax_b.set_ylabel("kNN Jaccard (k=10)")
        ax_b.set_title("Neighborhood persistence", fontsize=12)
        ax_b.set_ylim(0, 1.05)
        for b, val in zip(bars, [random_baseline, j_mean_late, 1.0]):
            ax_b.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                       f"{val:.3f}" if val < 0.05 else f"{val:.2f}",
                       ha="center", fontsize=11, fontweight="bold")
        # Big multiplier callout in the middle
        ax_b.text(0.5, 0.62, f"~{times_random:.0f}×\nabove\nrandom",
                   transform=ax_b.transAxes, ha="center", va="center",
                   fontsize=18, fontweight="bold", color=COLOR["primary"],
                   bbox=dict(boxstyle="round,pad=0.5", fc="white",
                              ec=COLOR["primary"], lw=2))

        # ─── Right: autocorrelation decay curve ──────────────────────────────
        ax_a = fig.add_subplot(gs[0, 2])
        lags = np.arange(1, max_lag + 1)
        ax_a.plot(lags, autocorr_obs, "o-", color=COLOR["primary"], lw=2.5,
                   ms=10, mec="white", mew=1.4, label="Observed")
        ax_a.axhline(autocorr_rand, color=COLOR["muted"], lw=1.6, ls="--",
                      label=f"Random pair ({autocorr_rand:.2f})")
        ax_a.fill_between([0, max_lag + 0.3], autocorr_rand, 1.0,
                            color=COLOR["primary"], alpha=0.06)
        ax_a.set_xlabel("Lag $\\ell$ (iterations)")
        ax_a.set_ylabel("Trajectory autocorrelation")
        ax_a.set_title("Memory decay", fontsize=12)
        ax_a.set_xlim(0.5, max_lag + 0.5)
        ax_a.set_ylim(0, max(1.05, max(autocorr_obs) * 1.15))
        ax_a.legend(fontsize=10, loc="upper right", framealpha=0.92)
        ax_a.grid(True, alpha=0.3)
        # Highlight that observed at lag 5 is still above random
        ax_a.annotate(
            f"lag-{max_lag} still\n{autocorr_obs[-1]/autocorr_rand:.0f}× random",
            xy=(max_lag, autocorr_obs[-1]),
            xytext=(max_lag - 1.8, autocorr_obs[-1] + 0.18),
            fontsize=10, color=COLOR["accent"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=COLOR["accent"], lw=1.4))

        fig.suptitle(f"Inside the attractor:  trajectories drift slowly through "
                      f"local neighborhoods (candidate v{v+1})",
                      fontsize=15, fontweight="bold", y=1.02)
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"talk_fig5_persistence_v3_v{v+1}.png")
        plt.savefig(out_path); plt.close(fig)
        logger.info(f"  → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  FIG 6 — Clean uniform-length vector field (REPLACES Block D's D3 for talk)
# ══════════════════════════════════════════════════════════════════════════════

def _kernel_weighted_field(grid_pts, starts, deltas, h, chunk=200):
    """Nadaraya–Watson kernel regression of a 2D vector field.

    For each grid point g, returns the Gaussian-weighted mean of the input
    deltas (weighted by squared distance from g to each start point), plus
    the total kernel weight at g.

    Args:
      grid_pts: (G, 2) — evaluation locations
      starts:   (S, 2) — observed start positions
      deltas:   (S, 2) — observed displacements
      h:        scalar — Gaussian bandwidth (in same units as grid)
      chunk:    int    — chunk size for memory management

    Returns:
      U_grid, V_grid, W_total — each shape (G,)
    """
    G = len(grid_pts)
    U = np.zeros(G); V = np.zeros(G); W_total = np.zeros(G)
    for c in range(0, G, chunk):
        gp = grid_pts[c:c + chunk]
        diff_x = gp[:, 0:1] - starts[None, :, 0]
        diff_y = gp[:, 1:2] - starts[None, :, 1]
        d2 = diff_x ** 2 + diff_y ** 2
        w = np.exp(-d2 / (2 * h ** 2))
        wsum = w.sum(axis=1)
        U[c:c + chunk] = (w * deltas[None, :, 0]).sum(axis=1) / np.maximum(wsum, 1e-12)
        V[c:c + chunk] = (w * deltas[None, :, 1]).sum(axis=1) / np.maximum(wsum, 1e-12)
        W_total[c:c + chunk] = wsum
    return U, V, W_total


def _draw_quiver_panel(ax, GX, GY, U, V, W, cell_w, weight_threshold,
                        ref_2d, modality_label, title,
                        attractor_loc=None, centroid_2d=None,
                        traj_overlay=None):
    """Draw a single uniform-length, color-coded vector-field panel.

    Optional overlays:
      attractor_loc: (x, y) — visual attractor marker (gold star)
      centroid_2d:   (x, y) — cohort-mean iter-K endpoint projected to 2D
                              (red diamond marker; sanity check vs attractor)
      traj_overlay:  list of (K+1, 2) projected trajectory paths to draw
                     as faint colored lines with iter-0 / iter-K markers
    """
    grid_n = U.shape[0]
    mag = np.sqrt(U ** 2 + V ** 2)
    valid = W >= weight_threshold
    U_unit = np.where(valid & (mag > 1e-9), U / np.maximum(mag, 1e-9), np.nan)
    V_unit = np.where(valid & (mag > 1e-9), V / np.maximum(mag, 1e-9), np.nan)
    mag_for_color = np.where(valid, mag, np.nan)

    # Background: training-distribution density
    sub = np.random.default_rng(0).choice(len(ref_2d),
                                            size=min(40000, len(ref_2d)),
                                            replace=False)
    ax.hexbin(ref_2d[sub, 0], ref_2d[sub, 1], gridsize=100,
                cmap="Greys", mincnt=2, alpha=0.30)

    arrow_len = 0.75 * cell_w
    q = ax.quiver(
        GX, GY,
        U_unit * arrow_len, V_unit * arrow_len,
        mag_for_color,
        cmap="viridis",
        scale_units="xy", angles="xy", scale=1.0,
        width=0.0030, headwidth=4.0, headlength=5.0, headaxislength=4.5,
        alpha=0.9, edgecolor="white", linewidth=0.3,
    )

    # Optional cohort-centroid marker (where iter-K endpoints actually
    # converge to in 256-d, projected to 2D). Critical sanity check:
    # if this aligns with the visual attractor, the attractor is real.
    if centroid_2d is not None:
        ax.scatter(centroid_2d[0], centroid_2d[1], s=300, marker="D",
                    c=COLOR["warning"], edgecolors="white", linewidths=2.2,
                    zorder=10,
                    label=f"256-d cohort iter-$K$ centroid")

    # Optional visual-attractor marker (where the field magnitude is smallest)
    if attractor_loc is not None:
        ax.scatter(attractor_loc[0], attractor_loc[1], s=400, marker="*",
                    c=COLOR["bg_anchor"], edgecolors=COLOR["accent"],
                    linewidths=2.2, zorder=11,
                    label=f"Visual flow minimum")

    # Optional sample trajectory overlays
    if traj_overlay is not None:
        traj_palette = ["#C73E1D", "#1C7293", "#2E7D32", "#A23B72",
                          "#F18F01", "#5D2E8C", "#048A81", "#B7245C"]
        for ti, traj in enumerate(traj_overlay):
            color = traj_palette[ti % len(traj_palette)]
            ax.plot(traj[:, 0], traj[:, 1], "-", color=color, lw=1.5,
                     alpha=0.7, zorder=8)
            ax.scatter(traj[0, 0], traj[0, 1], s=150, marker="o",
                        c=color, edgecolors="white", linewidths=1.5,
                        alpha=0.95, zorder=9)
            ax.scatter(traj[-1, 0], traj[-1, 1], s=120, marker="s",
                        c=color, edgecolors="white", linewidths=1.5,
                        alpha=0.95, zorder=9)

    if attractor_loc is not None or centroid_2d is not None or traj_overlay is not None:
        ax.legend(loc="upper right", framealpha=0.9, fontsize=9)

    ax.set_xlabel(f"UMAP-1 ({modality_label})")
    ax.set_ylabel(f"UMAP-2 ({modality_label})")
    ax.set_title(title, fontsize=12)
    ax.set_xlim(ref_2d[:, 0].min() - 0.5, ref_2d[:, 0].max() + 0.5)
    ax.set_ylim(ref_2d[:, 1].min() - 0.5, ref_2d[:, 1].max() + 0.5)
    ax.set_aspect("equal", adjustable="box")
    return q, valid


def figure_clean_vector_field(Z_emb, A_emb, reducer, ref_2d, out_path,
                                 grid_n=30, kernel_bw_factor=1.2,
                                 weight_threshold=2.0, modality_label="image",
                                 n_overlay_trajectories=6,
                                 do_permutation_null=True,
                                 rng_seed=0):
    """Clean displacement vector field with sanity checks.

    The figure has two panels by default:
      LEFT  — Real flow field (kernel-regressed mean of observed
              trajectory deltas) with overlays:
                • cohort iter-K centroid (256-d → 2D), marked as a
                  red diamond. Its alignment with the visual attractor
                  validates the field's structure.
                • sample trajectory paths (faint colored lines + start /
                  end markers).
      RIGHT — Permutation-null field. Same start positions, but each
              trajectory's deltas are randomly shuffled across the cohort
              (every step at every iteration is a draw from the cohort's
              global step distribution). Any structure here is purely
              kernel-interpolation artifact — it shows what the figure
              would look like if there were NO real flow. If the LEFT
              panel shows a clear attractor that the RIGHT panel does
              not, the attractor is genuine.

    Higher grid_n gives finer resolution. Default raised to 30
    (was 22) for sharper structure.

    Args:
      Z_emb:   (N, K, 256) trajectory embeddings (image OR text)
      A_emb:   (N, 256) anchor embeddings (modality must match Z_emb)
      reducer: fitted UMAP reducer for this modality
      ref_2d:  (R, 2) UMAP coords of training distribution

    Outputs the figure to `out_path` and also logs:
      • Centroid-vs-attractor alignment distance (sanity check)
      • Real field max magnitude vs null field max magnitude
        (signal-to-noise check)
      • Trajectory-toward-attractor angle test
    """
    logger.info(f"[FIG 6] Clean vector field with sanity checks ({modality_label})...")

    # Project trajectories to 2D ───────────────────────────────────────────────
    N, K, D = Z_emb.shape
    proj = np.stack([reducer.transform(Z_emb[:, k, :]) for k in range(K)],
                     axis=1)              # (N, K, 2)
    starts = proj[:, :-1, :].reshape(-1, 2)
    deltas = proj[:, 1:, :] - proj[:, :-1, :]
    deltas = deltas.reshape(-1, 2)
    logger.info(f"  {len(starts)} (start, delta) pairs from {N} trajectories")

    # Define grid & bandwidth ──────────────────────────────────────────────────
    pad = 0.5
    xmin, xmax = ref_2d[:, 0].min() - pad, ref_2d[:, 0].max() + pad
    ymin, ymax = ref_2d[:, 1].min() - pad, ref_2d[:, 1].max() + pad
    gx = np.linspace(xmin, xmax, grid_n)
    gy = np.linspace(ymin, ymax, grid_n)
    GX, GY = np.meshgrid(gx, gy)
    grid_pts = np.column_stack([GX.ravel(), GY.ravel()])
    cell_w = (xmax - xmin) / max(grid_n - 1, 1)
    h = kernel_bw_factor * cell_w
    logger.info(f"  Grid {grid_n}×{grid_n}, cell_w={cell_w:.3f}, kernel h={h:.3f}")

    # Real field ───────────────────────────────────────────────────────────────
    U, V, W_total = _kernel_weighted_field(grid_pts, starts, deltas, h)
    U = U.reshape(grid_n, grid_n)
    V = V.reshape(grid_n, grid_n)
    W_total = W_total.reshape(grid_n, grid_n)
    mag = np.sqrt(U ** 2 + V ** 2)
    valid = W_total >= weight_threshold
    logger.info(f"  Real field: valid {valid.sum()}/{grid_n*grid_n} cells; "
                f"mag range [{mag[valid].min():.3f}, {mag[valid].max():.3f}]")

    # Visual attractor location: minimum-magnitude cell among valid cells
    if valid.any():
        masked_mag = np.where(valid, mag, np.inf)
        min_idx = np.unravel_index(np.argmin(masked_mag), masked_mag.shape)
        attractor_loc = (float(GX[min_idx]), float(GY[min_idx]))
    else:
        attractor_loc = None

    # Cohort iter-K centroid (in 256-d → projected to 2D)
    centroid_256 = Z_emb[:, -1, :].mean(axis=0)             # (256,)
    centroid_2d_raw = reducer.transform(centroid_256[None, :])[0]
    centroid_2d = (float(centroid_2d_raw[0]), float(centroid_2d_raw[1]))

    # Sanity check: how far is the visual attractor from the 256-d centroid?
    if attractor_loc is not None:
        dx = attractor_loc[0] - centroid_2d[0]
        dy = attractor_loc[1] - centroid_2d[1]
        cd = float(np.sqrt(dx ** 2 + dy ** 2))
        logger.info(f"  Visual attractor at ({attractor_loc[0]:+.2f}, "
                    f"{attractor_loc[1]:+.2f})")
        logger.info(f"  256-d cohort centroid (UMAP) at ({centroid_2d[0]:+.2f}, "
                    f"{centroid_2d[1]:+.2f})")
        logger.info(f"  Distance attractor → centroid: {cd:.2f} UMAP units "
                    f"(cell width = {cell_w:.2f}, so {cd/cell_w:.1f} cells apart)")

    # Permutation null ─────────────────────────────────────────────────────────
    if do_permutation_null:
        rng = np.random.default_rng(rng_seed)
        delta_perm = deltas[rng.permutation(len(deltas))]
        Un, Vn, Wn = _kernel_weighted_field(grid_pts, starts, delta_perm, h)
        Un = Un.reshape(grid_n, grid_n)
        Vn = Vn.reshape(grid_n, grid_n)
        Wn = Wn.reshape(grid_n, grid_n)
        mag_n = np.sqrt(Un ** 2 + Vn ** 2)
        valid_n = Wn >= weight_threshold
        logger.info(f"  Null  field: valid {valid_n.sum()}/{grid_n*grid_n} cells; "
                    f"mag range [{mag_n[valid_n].min():.3f}, {mag_n[valid_n].max():.3f}]")

        # Real-vs-null magnitude ratio: high = strong signal
        if valid.any() and valid_n.any():
            real_max = float(mag[valid].max())
            null_max = float(mag_n[valid_n].max())
            real_p90 = float(np.percentile(mag[valid], 90))
            null_p90 = float(np.percentile(mag_n[valid_n], 90))
            logger.info(f"  Signal/null ratio: max {real_max/max(null_max,1e-9):.1f}×, "
                        f"p90 {real_p90/max(null_p90,1e-9):.1f}×")

    # Sanity check: do trajectories actually flow toward the attractor? ────────
    # For each (start, delta) pair, compute cosine angle between delta and
    # the direction (attractor_loc - start). Average across all pairs.
    if attractor_loc is not None:
        target_dir = np.array(attractor_loc)[None, :] - starts        # (S, 2)
        target_norm = np.linalg.norm(target_dir, axis=1, keepdims=True)
        delta_norm = np.linalg.norm(deltas, axis=1, keepdims=True)
        valid_dirs = (target_norm > 1e-6).flatten() & (delta_norm > 1e-6).flatten()
        if valid_dirs.sum() > 0:
            target_dir_n = target_dir[valid_dirs] / target_norm[valid_dirs]
            delta_n = deltas[valid_dirs] / delta_norm[valid_dirs]
            cos_to_attractor = (target_dir_n * delta_n).sum(axis=1)
            mean_cos = float(cos_to_attractor.mean())
            frac_pos = float((cos_to_attractor > 0).mean())
            logger.info(f"  Mean cos(δ, target→attractor) = {mean_cos:+.3f}  "
                        f"({100*frac_pos:.0f}% of steps point toward attractor; "
                        f"random baseline = 50%)")

    # Pick representative trajectories for overlay ─────────────────────────────
    # Strategy: spread across the iter-0 cloud so the overlay isn't redundant.
    rng = np.random.default_rng(rng_seed)
    if n_overlay_trajectories > 0 and N > 0:
        # Cluster iter-0 positions, pick one trajectory from each cluster
        from sklearn.cluster import KMeans
        n_pick = min(n_overlay_trajectories, N)
        kmeans = KMeans(n_clusters=n_pick, random_state=rng_seed, n_init=10)
        cluster_labels = kmeans.fit_predict(proj[:, 0, :])
        chosen_idx = []
        for c in range(n_pick):
            members = np.where(cluster_labels == c)[0]
            if len(members) > 0:
                # Pick the trajectory closest to the cluster centroid
                centroid_c = proj[members, 0, :].mean(axis=0)
                dists = np.linalg.norm(proj[members, 0, :] - centroid_c, axis=1)
                chosen_idx.append(int(members[np.argmin(dists)]))
        traj_overlay = [proj[i] for i in chosen_idx]
    else:
        traj_overlay = None

    # Plot ─────────────────────────────────────────────────────────────────────
    if do_permutation_null:
        fig, axes = plt.subplots(1, 2, figsize=(20, 8.5),
                                   gridspec_kw={"wspace": 0.18})
        ax_real, ax_null = axes
    else:
        fig, ax_real = plt.subplots(figsize=(11, 8.5))
        ax_null = None

    q_real, _ = _draw_quiver_panel(
        ax_real, GX, GY, U, V, W_total, cell_w, weight_threshold,
        ref_2d, modality_label,
        title=f"Real flow field — {modality_label}\n"
              f"Arrow direction = local flow; color = local speed",
        attractor_loc=attractor_loc,
        centroid_2d=centroid_2d,
        traj_overlay=traj_overlay,
    )
    cbar_q = plt.colorbar(q_real, ax=ax_real, fraction=0.04, pad=0.04)
    cbar_q.set_label("Mean local step speed (UMAP units)", fontsize=10)
    cbar_q.ax.tick_params(labelsize=9)

    if do_permutation_null and ax_null is not None:
        q_null, _ = _draw_quiver_panel(
            ax_null, GX, GY, Un, Vn, Wn, cell_w, weight_threshold,
            ref_2d, modality_label,
            title=f"Permutation-null field — {modality_label}\n"
                  f"Same start positions, deltas shuffled (no real flow)",
            attractor_loc=None, centroid_2d=None, traj_overlay=None,
        )
        cbar_n = plt.colorbar(q_null, ax=ax_null, fraction=0.04, pad=0.04)
        cbar_n.set_label("Mean local step speed (UMAP units)", fontsize=10)
        cbar_n.ax.tick_params(labelsize=9)

    fig.suptitle(
        f"Phase portrait sanity check ({modality_label}): real flow vs. "
        f"permutation null",
        fontsize=15, fontweight="bold", y=1.00,
    )
    plt.savefig(out_path); plt.close(fig)
    logger.info(f"  → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  FIG 7 — Phase portrait computed in 256-d, projected to 2D
# ══════════════════════════════════════════════════════════════════════════════

def figure_phase_portrait_256d(
        Z_emb, reducer, ref_2d, ref_256, out_path,
        grid_n=60, kernel_bw_factor=2.0, weight_threshold=0.5,
        modality_label="image", finite_diff_eps=0.05,
        n_overlay_trajectories=8, do_permutation_null=True,
        rng_seed=0):
    """Phase portrait built in 256-d, projected to 2D for visualization.

    Differences from figure_clean_vector_field:
      • The mean velocity field is computed in the FULL 256-d embedding
        space, not in 2D UMAP space. UMAP nonlinearity could distort or
        weaken signal that exists in 256-d; this version preserves it.
      • For each 2D grid cell, we find a representative 256-d location
        by averaging the 256-d coordinates of the nearest training-
        distribution points (in 2D). This means cells in 2D ``whitespace''
        that still correspond to specific 256-d regions get a real
        representative point and can therefore display an arrow.
      • The 256-d displacement is projected back to 2D via local
        finite-difference of UMAP, anchored at the cell's representative
        location.

    Algorithm at each grid cell g:
      1. Get representative 256-d location p(g) via kNN against training
         distribution in 2D, then average their 256-d coords.
      2. For all observed (start_256, delta_256) pairs, compute kernel
         weights w_i = exp(-||start_256_i - p(g)||² / 2 h²)  in 256-d.
      3. Weighted mean displacement δ̄(g) = Σ w_i δ_i / Σ w_i in 256-d.
      4. Project p(g) to 2D as g_2D = umap(p(g)). Project p(g) + ε δ̄(g)
         to 2D as g_2D_step. The 2D arrow is (g_2D_step - g_2D) / ε.
      5. Mask cells where Σ w_i is below `weight_threshold`.

    Higher grid_n (60 by default; can go to 80) captures finer structure.
    """
    logger.info(f"[FIG 7] Phase portrait via 256-d computation ({modality_label})...")
    if ref_256 is None:
        logger.error("  256-d reference embeddings required for this figure")
        return

    # Build (start_256, delta_256) pairs in original 256-d space ────────────────
    N, K, D = Z_emb.shape
    starts_256 = Z_emb[:, :-1, :].reshape(-1, D)
    deltas_256 = Z_emb[:, 1:, :] - Z_emb[:, :-1, :]
    deltas_256 = deltas_256.reshape(-1, D)
    logger.info(f"  {len(starts_256)} (start, delta) pairs in 256-d, "
                f"D={D}, N={N}, K={K}")

    # Project starts to 2D for grid construction (one-time UMAP call) ─────────
    proj_starts = reducer.transform(starts_256)              # (S, 2)
    # Also project the trajectory iter-K endpoints for the centroid marker
    centroid_256 = Z_emb[:, -1, :].mean(axis=0)
    centroid_2d_raw = reducer.transform(centroid_256[None, :])[0]
    centroid_2d = (float(centroid_2d_raw[0]), float(centroid_2d_raw[1]))

    # Define grid in 2D over training-distribution support ────────────────────
    pad = 0.5
    xmin, xmax = ref_2d[:, 0].min() - pad, ref_2d[:, 0].max() + pad
    ymin, ymax = ref_2d[:, 1].min() - pad, ref_2d[:, 1].max() + pad
    gx = np.linspace(xmin, xmax, grid_n)
    gy = np.linspace(ymin, ymax, grid_n)
    GX, GY = np.meshgrid(gx, gy)
    grid_pts_2d = np.column_stack([GX.ravel(), GY.ravel()])  # (G², 2)
    G = len(grid_pts_2d)
    cell_w = (xmax - xmin) / max(grid_n - 1, 1)
    logger.info(f"  Grid {grid_n}×{grid_n} = {G} cells, cell_w={cell_w:.3f}")

    # Step 1: find a representative 256-d location for each grid cell. ────────
    # We do kNN against the training distribution in 2D, then average 256-d
    # coords. This is what allows whitespace cells (where no trajectories
    # visited but training points exist) to still get a 256-d representative.
    knn_ref_2d = NearestNeighbors(n_neighbors=20, algorithm="auto")
    knn_ref_2d.fit(ref_2d)
    nn_dists, nn_idx = knn_ref_2d.kneighbors(grid_pts_2d)     # (G, 20)

    # Mask grid cells whose nearest-20 training points are too far (cells
    # outside the training-distribution support). Threshold: distance to
    # the nearest training point ≤ 1.5 cell widths.
    in_support = nn_dists[:, 0] <= 1.5 * cell_w

    # Representative 256-d location for each grid cell: distance-weighted
    # average of the 20 nearest training points' 256-d coords.
    grid_rep_256 = np.zeros((G, D))
    for c in range(G):
        if not in_support[c]:
            continue
        idx = nn_idx[c]
        # Down-weight by distance for a smoother representative
        w_inner = np.exp(-(nn_dists[c] ** 2) / (2 * cell_w ** 2))
        w_inner /= w_inner.sum()
        grid_rep_256[c] = (w_inner[:, None] * ref_256[idx]).sum(axis=0)
    logger.info(f"  Cells in training-dist support: {in_support.sum()}/{G}")

    # Step 2-3: kernel-weighted mean displacement in 256-d at each cell. ──────
    # 256-d bandwidth scales with cohort spread in 256-d (use median pairwise
    # distance among a sample of starts). This makes the kernel size
    # geometrically sensible in the original space.
    #
    # MEMORY NOTE: building the full (sample, sample, 256) tensor for the
    # median-pdist would allocate 8GB at sample=2000. We use scipy's pdist
    # which is condensed (1-D output, ~16MB at sample=2000) and far cheaper.
    rng = np.random.default_rng(rng_seed)
    sample_idx = rng.choice(len(starts_256), min(2000, len(starts_256)),
                              replace=False)
    sample_starts = starts_256[sample_idx]
    from scipy.spatial.distance import pdist as _pdist
    sample_pdist = _pdist(sample_starts, metric="euclidean")
    median_pdist_256 = float(np.median(sample_pdist))
    h_256 = kernel_bw_factor * 0.10 * median_pdist_256   # 10% of median × factor
    logger.info(f"  256-d median pairwise dist: {median_pdist_256:.3f}; "
                f"bandwidth h={h_256:.3f}")

    # Vectorized chunked computation. AVOIDS (Gc, S, D) intermediate by
    # computing distances via the inner-product identity
    #   ||a - b||² = ||a||² + ||b||² - 2 a·b
    # and computing the weighted mean delta as a matrix product W @ Δ
    # rather than (w[:,:,None] * Δ[None,:,:]).sum(axis=1).
    #
    # Memory at chunk=80 for grid_n=60: peak ~ 80 * 10810 * 8 bytes (≈7MB)
    # for the kernel matrix, plus the (Gc, D) result. Trivial.
    U_256 = np.zeros((G, D)); W_total = np.zeros(G)
    starts_sq = (starts_256 ** 2).sum(axis=1)              # (S,)
    chunk = 80
    inv_two_h2 = 1.0 / (2.0 * h_256 ** 2)
    n_chunks = int(np.ceil(G / chunk))
    log_every = max(1, n_chunks // 10)
    for ci, c0 in enumerate(range(0, G, chunk)):
        c1 = min(c0 + chunk, G)
        if not in_support[c0:c1].any():
            continue
        gp = grid_rep_256[c0:c1]                            # (Gc, D)
        gp_sq = (gp ** 2).sum(axis=1)                       # (Gc,)
        # ||g - s||² = ||g||² + ||s||² - 2 g·s    shape (Gc, S)
        cross = gp @ starts_256.T                           # (Gc, S)
        d2 = gp_sq[:, None] + starts_sq[None, :] - 2.0 * cross
        np.maximum(d2, 0.0, out=d2)                         # numerical safety
        w = np.exp(-d2 * inv_two_h2)                        # (Gc, S)
        wsum = w.sum(axis=1)                                # (Gc,)
        # Weighted mean delta = W @ Δ  shape (Gc, D)
        U_256[c0:c1] = (w @ deltas_256) / np.maximum(wsum[:, None], 1e-12)
        W_total[c0:c1] = wsum
        if (ci % log_every) == 0:
            logger.info(f"    256-d kernel: {ci+1}/{n_chunks} chunks done")

    # Step 4: project the 256-d displacement to 2D via local finite-diff. ─────
    # We project p(g) and p(g) + ε δ̄(g) through UMAP and take the difference.
    # ε is set so the perturbation is small in 256-d (won't leave the local
    # linearity regime).
    mask = in_support & (W_total > weight_threshold)
    n_valid = int(mask.sum())
    logger.info(f"  Valid cells (in support + sufficient weight): {n_valid}/{G}")

    if n_valid == 0:
        logger.error("  No valid cells; aborting")
        return

    valid_idx = np.where(mask)[0]
    rep_pts_valid = grid_rep_256[valid_idx]              # (Nv, 256)
    delta_norm = np.linalg.norm(U_256[valid_idx], axis=1, keepdims=True)
    eps_safe = finite_diff_eps / np.maximum(delta_norm, 1e-9)
    rep_pts_step = rep_pts_valid + eps_safe * U_256[valid_idx]

    # UMAP projects in batch — concatenate for a single call
    proj_in = np.vstack([rep_pts_valid, rep_pts_step])
    proj_out = reducer.transform(proj_in)
    rep_2d_valid = proj_out[:n_valid]
    rep_2d_step  = proj_out[n_valid:]
    arrows_2d = (rep_2d_step - rep_2d_valid) / np.maximum(eps_safe, 1e-12)

    # Reassemble into full grid arrays (NaN where invalid)
    U_proj = np.full(G, np.nan); V_proj = np.full(G, np.nan)
    U_proj[valid_idx] = arrows_2d[:, 0]
    V_proj[valid_idx] = arrows_2d[:, 1]
    U_proj = U_proj.reshape(grid_n, grid_n)
    V_proj = V_proj.reshape(grid_n, grid_n)
    mag_proj = np.sqrt(U_proj ** 2 + V_proj ** 2)
    valid_grid = ~np.isnan(U_proj)

    logger.info(f"  Real field magnitude range: "
                f"[{np.nanmin(mag_proj):.3f}, {np.nanmax(mag_proj):.3f}]")

    # Permutation null in 256-d ───────────────────────────────────────────────
    if do_permutation_null:
        perm = rng.permutation(len(deltas_256))
        deltas_perm = deltas_256[perm]
        Un_256 = np.zeros((G, D)); Wn_total = np.zeros(G)
        # Reuse starts_sq, h_256, inv_two_h2 from the real-field computation
        for ci, c0 in enumerate(range(0, G, chunk)):
            c1 = min(c0 + chunk, G)
            if not in_support[c0:c1].any(): continue
            gp = grid_rep_256[c0:c1]
            gp_sq = (gp ** 2).sum(axis=1)
            cross = gp @ starts_256.T
            d2 = gp_sq[:, None] + starts_sq[None, :] - 2.0 * cross
            np.maximum(d2, 0.0, out=d2)
            w = np.exp(-d2 * inv_two_h2)
            wsum = w.sum(axis=1)
            Un_256[c0:c1] = (w @ deltas_perm) / np.maximum(wsum[:, None], 1e-12)
            Wn_total[c0:c1] = wsum
            if (ci % log_every) == 0:
                logger.info(f"    256-d null kernel: {ci+1}/{n_chunks} chunks done")

        # Project null
        mask_n = in_support & (Wn_total > weight_threshold)
        n_valid_n = int(mask_n.sum())
        valid_idx_n = np.where(mask_n)[0]
        rep_n = grid_rep_256[valid_idx_n]
        d_norm_n = np.linalg.norm(Un_256[valid_idx_n], axis=1, keepdims=True)
        eps_n = finite_diff_eps / np.maximum(d_norm_n, 1e-9)
        rep_n_step = rep_n + eps_n * Un_256[valid_idx_n]
        proj_in_n = np.vstack([rep_n, rep_n_step])
        proj_out_n = reducer.transform(proj_in_n)
        arrows_n = (proj_out_n[n_valid_n:] - proj_out_n[:n_valid_n]) / \
                    np.maximum(eps_n, 1e-12)
        U_proj_n = np.full(G, np.nan); V_proj_n = np.full(G, np.nan)
        U_proj_n[valid_idx_n] = arrows_n[:, 0]
        V_proj_n[valid_idx_n] = arrows_n[:, 1]
        U_proj_n = U_proj_n.reshape(grid_n, grid_n)
        V_proj_n = V_proj_n.reshape(grid_n, grid_n)
        mag_proj_n = np.sqrt(U_proj_n ** 2 + V_proj_n ** 2)
        logger.info(f"  Null field magnitude range: "
                    f"[{np.nanmin(mag_proj_n):.3f}, {np.nanmax(mag_proj_n):.3f}]")
        # Signal/noise
        real_p90 = float(np.nanpercentile(mag_proj, 90))
        null_p90 = float(np.nanpercentile(mag_proj_n, 90))
        logger.info(f"  Signal/null p90 ratio: {real_p90 / max(null_p90, 1e-9):.1f}×")

    # Pick representative trajectories for overlay (k-means on iter-0) ────────
    if n_overlay_trajectories > 0 and N > 0:
        from sklearn.cluster import KMeans
        proj_iter0 = reducer.transform(Z_emb[:, 0, :])
        n_pick = min(n_overlay_trajectories, N)
        km = KMeans(n_clusters=n_pick, random_state=rng_seed, n_init=10)
        cluster_labels = km.fit_predict(proj_iter0)
        chosen_idx = []
        for c in range(n_pick):
            members = np.where(cluster_labels == c)[0]
            if len(members) > 0:
                ctr = proj_iter0[members].mean(axis=0)
                d = np.linalg.norm(proj_iter0[members] - ctr, axis=1)
                chosen_idx.append(int(members[np.argmin(d)]))
        # Project chosen trajectories all at once
        all_traj_pts_256 = Z_emb[chosen_idx].reshape(-1, D)
        all_traj_pts_2d = reducer.transform(all_traj_pts_256)
        traj_overlay = [all_traj_pts_2d[i*K:(i+1)*K] for i in range(len(chosen_idx))]
    else:
        traj_overlay = None

    # ── Plot: real-field panel + null-field panel ────────────────────────────
    if do_permutation_null:
        fig, axes = plt.subplots(1, 2, figsize=(22, 9.0),
                                   gridspec_kw={"wspace": 0.18})
        ax_real, ax_null = axes
    else:
        fig, ax_real = plt.subplots(figsize=(12, 9))
        ax_null = None

    q_real, _ = _draw_quiver_panel(
        ax_real, GX, GY, U_proj, V_proj,
        np.where(valid_grid, 1.0, 0.0),  # any nonzero → valid
        cell_w, 0.5,
        ref_2d, modality_label,
        title=f"Phase portrait — {modality_label} (computed in 256-d, projected)\n"
              f"Field built in original 256-d space; UMAP used only for display",
        attractor_loc=None,
        centroid_2d=centroid_2d,
        traj_overlay=traj_overlay,
    )
    cbar_q = plt.colorbar(q_real, ax=ax_real, fraction=0.04, pad=0.04)
    cbar_q.set_label("Local 2D-projected step speed", fontsize=10)
    cbar_q.ax.tick_params(labelsize=9)

    if do_permutation_null and ax_null is not None:
        q_null, _ = _draw_quiver_panel(
            ax_null, GX, GY, U_proj_n, V_proj_n,
            np.where(~np.isnan(U_proj_n), 1.0, 0.0),
            cell_w, 0.5,
            ref_2d, modality_label,
            title=f"Permutation null — {modality_label}\n"
                  f"Same 256-d positions, deltas shuffled (no real flow)",
            attractor_loc=None, centroid_2d=None, traj_overlay=None,
        )
        cbar_n = plt.colorbar(q_null, ax=ax_null, fraction=0.04, pad=0.04)
        cbar_n.set_label("Local 2D-projected step speed", fontsize=10)
        cbar_n.ax.tick_params(labelsize=9)

    fig.suptitle(
        f"Phase portrait in 256-d ({modality_label}): "
        f"grid {grid_n}×{grid_n}, kernel in original embedding space",
        fontsize=15, fontweight="bold", y=1.00,
    )
    plt.savefig(out_path); plt.close(fig)
    logger.info(f"  → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--main_dir", required=True)
    p.add_argument("--ref_dir",  required=True)
    p.add_argument("--out_dir",  required=True)
    p.add_argument("--n_candidates_lyap", type=int, default=4)
    p.add_argument("--n_candidates_pers", type=int, default=4)
    p.add_argument("--skip_fig1",  action="store_true",
                    help="Skip Fig 1 (MI cliff is locked).")
    p.add_argument("--skip_fig2",  action="store_true",
                    help="Skip Fig 2 (Lyapunov candidates).")
    p.add_argument("--skip_fig3",  action="store_true",
                    help="Skip Fig 3 (old persistence multi-trajectory).")
    p.add_argument("--skip_fig4",  action="store_true",
                    help="Skip Fig 4 (modality asymmetry, NEW).")
    p.add_argument("--skip_fig5",  action="store_true",
                    help="Skip Fig 5 (persistence v3, NEW).")
    p.add_argument("--skip_fig6",  action="store_true",
                    help="Skip Fig 6 (clean uniform-length vector field, NEW).")
    p.add_argument("--grid_n", type=int, default=30,
                    help="Vector-field grid resolution (default 30; try 40-50 for finer detail).")
    p.add_argument("--no_null",  action="store_true",
                    help="Disable Fig 6 permutation-null sanity panel (faster).")
    p.add_argument("--n_overlay", type=int, default=6,
                    help="Number of sample trajectories to overlay on Fig 6.")
    p.add_argument("--skip_fig7",  action="store_true",
                    help="Skip Fig 7 (256-d phase portrait, NEW).")
    p.add_argument("--grid_n_256d", type=int, default=60,
                    help="Grid resolution for Fig 7 (default 60; try 80 for finer).")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    logger.info("=" * 60)
    logger.info("AIM2 Presentation Figure Generator (v3)")
    logger.info("=" * 60)

    Z_img, Z_txt, sids, A_img, A_txt = load_trajectories(args.main_dir)
    (reducer_img, reducer_txt, ref_img_2d, ref_txt_2d,
     ref_img_256, ref_txt_256) = load_umap(args.ref_dir)

    if not args.skip_fig1:
        figure_mi_collapse(Z_img, Z_txt,
                            out_path=os.path.join(args.out_dir,
                                                    "talk_fig1_mi_collapse.png"))

    if not args.skip_fig2:
        figure_lyapunov_candidates(Z_img, A_img, sids, reducer_img, ref_img_2d,
                                      out_dir=args.out_dir,
                                      n_candidates=args.n_candidates_lyap)

    if not args.skip_fig3:
        figure_persistence_candidates(Z_img, A_img, sids, reducer_img, ref_img_2d,
                                         out_dir=args.out_dir,
                                         n_candidates=args.n_candidates_pers)

    if not args.skip_fig4:
        figure_modality_asymmetry(
            Z_img, Z_txt, A_img, A_txt, sids,
            reducer_img, ref_img_2d, reducer_txt, ref_txt_2d,
            ref_img_256, ref_txt_256,
            out_path=os.path.join(args.out_dir, "talk_fig4_modality_asymmetry.png"),
        )

    if not args.skip_fig5:
        figure_persistence_v3(Z_img, A_img, sids, reducer_img, ref_img_2d,
                                out_dir=args.out_dir,
                                n_candidates=args.n_candidates_pers)

    if not args.skip_fig6:
        figure_clean_vector_field(
            Z_img, A_img, reducer_img, ref_img_2d,
            out_path=os.path.join(args.out_dir,
                                    "talk_fig6_vector_field_image.png"),
            modality_label="image",
            grid_n=args.grid_n,
            do_permutation_null=not args.no_null,
            n_overlay_trajectories=args.n_overlay,
        )
        figure_clean_vector_field(
            Z_txt, A_txt, reducer_txt, ref_txt_2d,
            out_path=os.path.join(args.out_dir,
                                    "talk_fig6_vector_field_text.png"),
            modality_label="text",
            grid_n=args.grid_n,
            do_permutation_null=not args.no_null,
            n_overlay_trajectories=args.n_overlay,
        )

    if not args.skip_fig7:
        figure_phase_portrait_256d(
            Z_img, reducer_img, ref_img_2d, ref_img_256,
            out_path=os.path.join(args.out_dir,
                                    "talk_fig7_phase_portrait_256d_image.png"),
            modality_label="image",
            grid_n=args.grid_n_256d,
            do_permutation_null=not args.no_null,
            n_overlay_trajectories=args.n_overlay,
        )
        figure_phase_portrait_256d(
            Z_txt, reducer_txt, ref_txt_2d, ref_txt_256,
            out_path=os.path.join(args.out_dir,
                                    "talk_fig7_phase_portrait_256d_text.png"),
            modality_label="text",
            grid_n=args.grid_n_256d,
            do_permutation_null=not args.no_null,
            n_overlay_trajectories=args.n_overlay,
        )

    logger.info("\nDone. Figures written to " + args.out_dir)


if __name__ == "__main__":
    main()