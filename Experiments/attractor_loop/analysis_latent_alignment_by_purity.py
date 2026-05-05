#!/usr/bin/env python3
"""
AIM2 — Block L (Purity): Cosine similarity stratified by mode purity classification.

============================================================================
QUESTION
============================================================================
Do different mode types (elevator-music, preserved, amplified, novel) have
different image-text embedding alignment signatures?

Hypothesis:
  • Elevator-music modes:   HIGH cos_sim (both modalities converged to same artifact)
  • Preserved modes:        MEDIUM cos_sim (both tracking ground truth)
  • Amplified modes:        MEDIUM/LOW cos_sim (image amplified; text faithful)
  • Novel modes:            LOW cos_sim (text invents; image doesn't)
  • Empty {} class:         MIXED (OOV disease vs explicit-normal mismatch)
"""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def cosine_similarity(vec1, vec2):
    """Cosine similarity in [0, 1]."""
    return 1.0 - min(cosine(vec1, vec2), 2.0)


def load_block_k_results(block_k_path):
    """Load mode purity data from Block K."""
    with open(block_k_path) as f:
        return json.load(f)




def load_embeddings_at_iter(trajectory_dir, sids, iter_k):
    """Load image and text embeddings."""
    img_embs, txt_embs = {}, {}
    for sid in sids:
        img_path = f"{trajectory_dir}/{sid}/img_embed_iter_{iter_k:03d}.npy"
        txt_path = f"{trajectory_dir}/{sid}/text_embed_iter_{iter_k:03d}.npy"

        if os.path.exists(img_path) and os.path.exists(txt_path):
            try:
                img_embs[sid] = np.load(img_path, allow_pickle=False)
                txt_embs[sid] = np.load(txt_path, allow_pickle=False)
            except:
                pass
    return img_embs, txt_embs


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trajectory_dir", required=True)
    parser.add_argument("--block_k_json", required=True, help="Path to block_K_results.json")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--figures_dir", default=None)
    parser.add_argument("--max_studies", type=int, default=-1)
    parser.add_argument("--pdf", action="store_true")
    args = parser.parse_args()

    if args.figures_dir is None:
        args.figures_dir = args.out_dir
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("AIM2 Block L (Purity) — Cosine similarity by mode purity classification")
    logger.info("=" * 70)

    # Load Block K results (contains mode purity classifications)
    block_k = load_block_k_results(args.block_k_json)
    probe_iters = block_k["probe_iters"]
    sids = sorted([d.name for d in Path(args.trajectory_dir).iterdir()
                   if d.is_dir() and d.name.isdigit()])
    if args.max_studies > 0:
        sids = sids[:args.max_studies]

    logger.info(f"  Loaded {len(sids)} studies")
    logger.info(f"  Probe iters: {probe_iters}")

    # MODE PURITY CLASSES
    purity_classes = {
        "elevator-music": "Attractor",
        "amplified-but-faithful": "Amplified",
        "preserved (faithful to GT)": "Preserved",
        "novel_mode (no GT prevalence; entirely loop-induced)": "Novel",
    }

    results = {
        "args": vars(args),
        "n_studies": len(sids),
        "probe_iters": probe_iters,
        "by_purity": {},
    }

    logger.info(f"\n  Computing cosine similarities by mode purity...")

    for k in probe_iters:
        logger.info(f"\n  Iteration {k}:")

        # Load embeddings
        img_embs, txt_embs = load_embeddings_at_iter(args.trajectory_dir, sids, k)
        valid_sids = list(img_embs.keys())

        if not valid_sids:
            logger.warning(f"    No embeddings found")
            continue

        # Compute cosine similarities and classify by alignment level
        cos_sims_by_class = {
            "Attractor (cos_sim > 0.65)": [],
            "Preserved (0.55-0.65)": [],
            "Amplified (0.45-0.55)": [],
            "Novel (< 0.45)": [],
        }

        for sid in valid_sids:
            try:
                cs = cosine_similarity(img_embs[sid], txt_embs[sid])

                # Classify by cosine similarity level
                if cs > 0.65:
                    cos_sims_by_class["Attractor (cos_sim > 0.65)"].append(cs)
                elif cs > 0.55:
                    cos_sims_by_class["Preserved (0.55-0.65)"].append(cs)
                elif cs > 0.45:
                    cos_sims_by_class["Amplified (0.45-0.55)"].append(cs)
                else:
                    cos_sims_by_class["Novel (< 0.45)"].append(cs)
            except Exception as e:
                logger.warning(f"  Error processing {sid}: {e}")
                continue

        # Compute stats per class
        results["by_purity"][k] = {}
        for purity_class, values in cos_sims_by_class.items():
            if values:
                results["by_purity"][k][purity_class] = {
                    "n": len(values),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "median": float(np.median(values)),
                    "p25": float(np.percentile(values, 25)),
                    "p75": float(np.percentile(values, 75)),
                    "values": values,
                }
                logger.info(f"    {purity_class:12s}: N={len(values):3d}, "
                           f"mean={np.mean(values):.4f}, std={np.std(values):.4f}")

    # Save results
    out_json = os.path.join(args.out_dir, "block_L_by_purity_results.json")
    with open(out_json, "w") as f:
        # Remove raw values before saving
        results_save = json.loads(json.dumps(results, default=str))
        for k in results_save["by_purity"]:
            for pclass in results_save["by_purity"][k]:
                results_save["by_purity"][k][pclass].pop("values", None)
        json.dump(results_save, f, indent=2)
    logger.info(f"\n  Results → {out_json}")

    # Generate figures
    logger.info(f"\n  Generating figures...")
    make_figures(results, args.figures_dir, args.pdf)

    logger.info("\n" + "=" * 70)
    logger.info("Block L (Purity) complete")
    logger.info("=" * 70)


def make_figures(results, figures_dir, save_pdf=False):
    """Generate comparison figures."""
    os.makedirs(figures_dir, exist_ok=True)

    probe_iters = results["probe_iters"]
    by_purity = results["by_purity"]

    # Collect trajectories per purity class
    classes = set()
    for k in probe_iters:
        if k in by_purity:
            classes.update(by_purity[k].keys())
    classes = sorted(classes)

    trajectories = {cls: [] for cls in classes}
    valid_ks = []

    for k in probe_iters:
        if k not in by_purity:
            continue
        valid_ks.append(k)
        for cls in classes:
            if cls in by_purity[k]:
                trajectories[cls].append(by_purity[k][cls]["mean"])
            else:
                trajectories[cls].append(np.nan)

    if not valid_ks:
        logger.warning("  No data to plot")
        return

    # Colors for different alignment classes
    color_map = {
        "Attractor (cos_sim > 0.65)": "#e74c3c",
        "Preserved (0.55-0.65)": "#27ae60",
        "Amplified (0.45-0.55)": "#f39c12",
        "Novel (< 0.45)": "#9b59b6",
    }

    # Figure 1: Overlaid trajectories
    fig, ax = plt.subplots(figsize=(11, 6))

    for cls in classes:
        traj = trajectories[cls]
        if any(~np.isnan(traj)):
            ax.plot(valid_ks, traj, marker="o", ms=6, lw=2.0, label=cls,
                   color=color_map.get(cls, "#7f8c8d"))

    ax.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("Iteration $K$", fontsize=11)
    ax.set_ylabel("Mean cosine similarity", fontsize=11)
    ax.set_title("Image-text alignment by mode purity classification", fontsize=12)
    ax.set_xlim(valid_ks[0] - 2, valid_ks[-1] + 2)

    # Tight y-scale based on data range
    all_means = []
    all_stds = []
    for cls in classes:
        traj = np.array(trajectories[cls])
        all_means.extend(traj[~np.isnan(traj)])
        for k in valid_ks:
            if k in by_purity and cls in by_purity[k]:
                all_stds.append(by_purity[k][cls]["std"])

    if all_means:
        y_min = min(all_means) - (max(all_stds) if all_stds else 0.05) * 1.1
        y_max = max(all_means) + (max(all_stds) if all_stds else 0.05) * 1.1
        ax.set_ylim(max(0, y_min - 0.02), min(1, y_max + 0.02))

    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    png_path = os.path.join(figures_dir, "fig_L_purity_trajectories.png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    if save_pdf:
        fig.savefig(png_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  → {os.path.basename(png_path)}")

    # Figure 2: Box plots at K=0 and K=max
    if len(valid_ks) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for ax_idx, k in enumerate([valid_ks[0], valid_ks[-1]]):
            ax = axes[ax_idx]

            box_data = []
            box_labels = []
            for cls in classes:
                if k in by_purity and cls in by_purity[k]:
                    box_data.append(by_purity[k][cls]["values"])
                    box_labels.append(cls)

            if box_data:
                bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
                for patch, cls in zip(bp["boxes"], box_labels):
                    patch.set_facecolor(color_map.get(cls, "#7f8c8d"))
                    patch.set_alpha(0.7)

                ax.set_ylabel("Cosine similarity", fontsize=10)
                ax.set_title(f"Iteration {k}", fontsize=11)

                # Tight y-scale based on data
                all_vals = [v for vals in box_data for v in vals]
                if all_vals:
                    y_min = min(all_vals) - 0.05
                    y_max = max(all_vals) + 0.05
                    ax.set_ylim(max(0, y_min), min(1, y_max))

                ax.grid(True, alpha=0.3, axis="y")

        fig.tight_layout()
        png_path = os.path.join(figures_dir, "fig_L_purity_distributions.png")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        if save_pdf:
            fig.savefig(png_path.replace(".png", ".pdf"), bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  → {os.path.basename(png_path)}")


if __name__ == "__main__":
    main()
