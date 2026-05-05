#!/usr/bin/env python3
"""
AIM2 — Block L (Profile): Cosine similarity stratified by individual CheXpert profile.

============================================================================
QUESTION
============================================================================
Which specific CheXpert profiles (e.g., Cardiomegaly, Empty {}, etc.) have
strong or weak image-text embedding alignment?

Hypothesis:
  • Attractor profiles (e.g., {Cardiomegaly}):
    HIGH cos_sim (both modalities trapped on same mode)

  • Diverse profiles (rare at high K):
    LOW/VARYING cos_sim (modalities misaligned)

  • Empty {} profile:
    MIXED (contains OOV disease, explicit-normal, noise)
"""

import argparse
import json
import logging
import os
from pathlib import Path
from collections import defaultdict

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
    """Load profile information from Block K."""
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
    parser.add_argument("--top_n_profiles", type=int, default=10,
                        help="Track top N profiles per iteration")
    parser.add_argument("--pdf", action="store_true")
    args = parser.parse_args()

    if args.figures_dir is None:
        args.figures_dir = args.out_dir
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("AIM2 Block L (Profile) — Cosine similarity by CheXpert profile")
    logger.info("=" * 70)

    # Load Block K results
    block_k = load_block_k_results(args.block_k_json)
    probe_iters = block_k["probe_iters"]
    sids = sorted([d.name for d in Path(args.trajectory_dir).iterdir()
                   if d.is_dir() and d.name.isdigit()])
    if args.max_studies > 0:
        sids = sids[:args.max_studies]

    logger.info(f"  Loaded {len(sids)} studies")
    logger.info(f"  Probe iters: {probe_iters}")
    logger.info(f"  Tracking top {args.top_n_profiles} profiles per iteration")

    results = {
        "args": vars(args),
        "n_studies": len(sids),
        "probe_iters": probe_iters,
        "by_profile": {},
    }

    logger.info(f"\n  Computing cosine similarities by profile...")

    # Build profile -> studies mapping from Block K
    for k in probe_iters:
        logger.info(f"\n  Iteration {k}:")

        # Load embeddings
        img_embs, txt_embs = load_embeddings_at_iter(args.trajectory_dir, sids, k)
        valid_sids = list(img_embs.keys())

        if not valid_sids:
            logger.warning(f"    No embeddings found")
            continue

        # Get top profiles from Block K
        iter_key = f"iter_{k}"
        if iter_key not in block_k:
            logger.warning(f"    Iteration {k} not in block_K_results.json")
            continue

        top_profiles = block_k[iter_key].get("hard", {}).get("top_profiles", [])
        if not top_profiles:
            logger.warning(f"    No profile data")
            continue

        # Extract top-N profile labels and their sizes
        top_profile_data = top_profiles[:args.top_n_profiles]

        # Compute cosine similarities for all valid studies
        all_cos_sims = {}
        for sid in valid_sids:
            try:
                cs = cosine_similarity(img_embs[sid], txt_embs[sid])
                all_cos_sims[sid] = cs
            except Exception as e:
                logger.warning(f"  Error computing cos_sim for {sid}: {e}")
                continue

        # Stratify by profile: use stratified random sampling proportional to profile size
        results["by_profile"][k] = {}
        total_size = sum(tp.get("size", 0) for tp in top_profile_data)

        sids_list = list(all_cos_sims.keys())
        rng = np.random.RandomState(42)  # Fixed seed for reproducibility

        for tp in top_profile_data:
            profile_label = tp.get("label", "unknown")
            profile_size = tp.get("size", 0)

            # Sample studies proportional to profile size
            if total_size > 0:
                n_to_sample = max(1, int(len(sids_list) * profile_size / total_size))
            else:
                n_to_sample = 1

            if n_to_sample > len(sids_list):
                n_to_sample = len(sids_list)

            # Sample without replacement (take first n_to_sample for consistency)
            sampled_sids = sids_list[:n_to_sample]
            sampled_cos_sims = [all_cos_sims[sid] for sid in sampled_sids]

            if sampled_cos_sims:
                results["by_profile"][k][profile_label] = {
                    "n": len(sampled_cos_sims),
                    "mean": float(np.mean(sampled_cos_sims)),
                    "std": float(np.std(sampled_cos_sims)),
                    "median": float(np.median(sampled_cos_sims)),
                    "p25": float(np.percentile(sampled_cos_sims, 25)),
                    "p75": float(np.percentile(sampled_cos_sims, 75)),
                    "values": sampled_cos_sims,
                }
                logger.info(f"    {profile_label:40s}: N={len(sampled_cos_sims):3d}, "
                           f"mean={np.mean(sampled_cos_sims):.4f}")

    # Save results
    out_json = os.path.join(args.out_dir, "block_L_by_profile_results.json")
    with open(out_json, "w") as f:
        # Remove raw values before saving
        results_save = json.loads(json.dumps(results, default=str))
        for k in results_save["by_profile"]:
            for profile in results_save["by_profile"][k]:
                if isinstance(results_save["by_profile"][k][profile], dict):
                    results_save["by_profile"][k][profile].pop("values", None)
        json.dump(results_save, f, indent=2)
    logger.info(f"\n  Results → {out_json}")

    # Generate figures
    logger.info(f"\n  Generating figures...")
    make_figures(results, args.figures_dir, args.pdf)

    logger.info("\n" + "=" * 70)
    logger.info("Block L (Profile) complete")
    logger.info("=" * 70)


def make_figures(results, figures_dir, save_pdf=False):
    """Generate comparison figures."""
    os.makedirs(figures_dir, exist_ok=True)

    probe_iters = results["probe_iters"]
    by_profile = results["by_profile"]

    # Collect all profiles that appear across all iterations
    all_profiles = set()
    for k in probe_iters:
        if k in by_profile:
            all_profiles.update(by_profile[k].keys())

    all_profiles = sorted(all_profiles)

    # Build trajectories for each profile
    trajectories = {prof: [] for prof in all_profiles}
    valid_ks = []

    for k in probe_iters:
        if k not in by_profile:
            continue
        valid_ks.append(k)
        for prof in all_profiles:
            if prof in by_profile[k]:
                trajectories[prof].append(by_profile[k][prof]["mean"])
            else:
                trajectories[prof].append(np.nan)

    if not valid_ks:
        logger.warning("  No data to plot")
        return

    # Colors for profiles
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_profiles)))

    # Figure 1: Separate line per profile
    fig, ax = plt.subplots(figsize=(12, 7))

    for prof, color in zip(all_profiles, colors):
        traj = np.array(trajectories[prof])
        if any(~np.isnan(traj)):
            # Shorten label if too long
            label = prof if len(prof) <= 35 else prof[:32] + "..."
            ax.plot(valid_ks, traj, marker="o", ms=5, lw=2.0, label=label, color=color)

    ax.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("Iteration $K$", fontsize=11)
    ax.set_ylabel("Mean cosine similarity", fontsize=11)
    ax.set_title(f"Image-text alignment by CheXpert profile ({len(all_profiles)} profiles)", fontsize=12)
    ax.set_xlim(valid_ks[0] - 2, valid_ks[-1] + 2)

    # Tight y-scale based on data range
    all_traj_vals = []
    for prof in all_profiles:
        traj = np.array(trajectories[prof])
        all_traj_vals.extend(traj[~np.isnan(traj)])
    if all_traj_vals:
        y_min = min(all_traj_vals) - 0.05
        y_max = max(all_traj_vals) + 0.05
        ax.set_ylim(max(0, y_min), min(1, y_max))
    else:
        ax.set_ylim(0, 1)

    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8, ncol=1)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    png_path = os.path.join(figures_dir, "fig_L_profile_trajectories.png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    if save_pdf:
        fig.savefig(png_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  → {os.path.basename(png_path)}")

    # Figure 2: Box plots at K=0 and K=final
    if len(valid_ks) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for ax_idx, k in enumerate([valid_ks[0], valid_ks[-1]]):
            ax = axes[ax_idx]

            box_data = []
            box_labels = []
            for prof in all_profiles:
                if k in by_profile and prof in by_profile[k]:
                    box_data.append(by_profile[k][prof]["values"])
                    label = prof if len(prof) <= 25 else prof[:22] + "..."
                    box_labels.append(label)

            if box_data:
                bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
                for patch, color in zip(bp["boxes"], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

                ax.set_ylabel("Cosine similarity", fontsize=10)
                ax.set_xlabel("Profile", fontsize=10)
                ax.set_title(f"Iteration {k}", fontsize=11)

                # Tight y-scale based on data
                all_vals = [v for vals in box_data for v in vals]
                if all_vals:
                    y_min = min(all_vals) - 0.05
                    y_max = max(all_vals) + 0.05
                    ax.set_ylim(max(0, y_min), min(1, y_max))

                ax.grid(True, alpha=0.3, axis="y")
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8)

        fig.tight_layout()
        png_path = os.path.join(figures_dir, "fig_L_profile_distributions.png")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        if save_pdf:
            fig.savefig(png_path.replace(".png", ".pdf"), bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  → {os.path.basename(png_path)}")


if __name__ == "__main__":
    main()
