#!/usr/bin/env python3
"""
Evaluate learned embedding space with retrieval metrics and visualizations.

Usage:
    python evaluate_embeddings.py --checkpoint /checkpoints/best_model.pth
    python evaluate_embeddings.py --checkpoint best_model.pth --use_test --n_samples 1000
"""

import os
import json
import argparse

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.manifold import TSNE
import umap

from CLIP.config.config import Config
from CLIP.data.dataloader import create_dataloaders
from CLIP.model.clip_model import MedicalCLIP
from CLIP.utils.checkpoint import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate embedding space')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Output directory')
    parser.add_argument('--use_test', action='store_true', help='Use test set (default: validation)')
    parser.add_argument('--n_samples', type=int, default=None, help='Number of samples (None = all)')
    return parser.parse_args()


@torch.no_grad()
def extract_embeddings(model, dataloader, device, n_samples=None):
    """Extract image and text embeddings from a dataloader."""
    model.eval()

    all_image_embeds = []
    all_text_embeds = []
    all_labels = []
    all_metadata = []

    n_processed = 0
    for batch in tqdm(dataloader, desc="Extracting embeddings"):
        images = batch['images'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        image_embeds, text_embeds, _ = model(images, input_ids, attention_mask)

        all_image_embeds.append(image_embeds.cpu())
        all_text_embeds.append(text_embeds.cpu())
        all_labels.append(batch['chexpert_labels'])
        all_metadata.extend(batch['metadata'])

        n_processed += len(images)
        if n_samples and n_processed >= n_samples:
            break

    return (
        torch.cat(all_image_embeds, dim=0),
        torch.cat(all_text_embeds, dim=0),
        torch.cat(all_labels, dim=0),
        all_metadata,
    )


def compute_retrieval_metrics(image_embeds, text_embeds, k_values=[1, 5, 10]):
    """Compute retrieval metrics (Recall@k, MRR, median rank, mAP, similarity stats)."""
    N = len(image_embeds)

    image_embeds = F.normalize(image_embeds, p=2, dim=1)
    text_embeds = F.normalize(text_embeds, p=2, dim=1)

    similarity = image_embeds @ text_embeds.T
    targets = torch.arange(N)

    metrics = {}

    # Image-to-text retrieval
    i2t_sorted = similarity.argsort(dim=1, descending=True)
    for k in k_values:
        recall = (i2t_sorted[:, :k] == targets.unsqueeze(1)).any(dim=1).float().mean().item()
        metrics[f'i2t_R@{k}'] = recall

    i2t_ranks = (i2t_sorted == targets.unsqueeze(1)).nonzero(as_tuple=True)[1].float()
    metrics['i2t_MRR'] = (1.0 / (i2t_ranks + 1)).mean().item()
    metrics['i2t_median_rank'] = i2t_ranks.median().item() + 1
    metrics['i2t_mAP'] = metrics['i2t_MRR']

    # Text-to-image retrieval
    t2i_sorted = similarity.T.argsort(dim=1, descending=True)
    for k in k_values:
        recall = (t2i_sorted[:, :k] == targets.unsqueeze(1)).any(dim=1).float().mean().item()
        metrics[f't2i_R@{k}'] = recall

    t2i_ranks = (t2i_sorted == targets.unsqueeze(1)).nonzero(as_tuple=True)[1].float()
    metrics['t2i_MRR'] = (1.0 / (t2i_ranks + 1)).mean().item()
    metrics['t2i_median_rank'] = t2i_ranks.median().item() + 1
    metrics['t2i_mAP'] = metrics['t2i_MRR']

    # Averaged
    for k in k_values:
        metrics[f'avg_R@{k}'] = (metrics[f'i2t_R@{k}'] + metrics[f't2i_R@{k}']) / 2
    metrics['avg_MRR'] = (metrics['i2t_MRR'] + metrics['t2i_MRR']) / 2
    metrics['avg_median_rank'] = (metrics['i2t_median_rank'] + metrics['t2i_median_rank']) / 2
    metrics['avg_mAP'] = (metrics['i2t_mAP'] + metrics['t2i_mAP']) / 2

    # Similarity statistics
    pos_sim = torch.diagonal(similarity).mean().item()
    neg_mask = ~torch.eye(N, dtype=torch.bool)
    neg_sim = similarity[neg_mask].mean().item()

    metrics['pos_similarity'] = pos_sim
    metrics['neg_similarity'] = neg_sim
    metrics['similarity_gap'] = pos_sim - neg_sim

    return metrics, similarity


def create_metrics_table(metrics, output_dir):
    """Save retrieval metrics as CSV and a styled PNG table."""
    rows = []
    for k in [1, 5, 10]:
        rows.append({
            'Metric': f'Recall@{k}',
            'Image-to-Text': f"{metrics[f'i2t_R@{k}']*100:.2f}%",
            'Text-to-Image': f"{metrics[f't2i_R@{k}']*100:.2f}%",
            'Average': f"{metrics[f'avg_R@{k}']*100:.2f}%",
        })

    rows.append({
        'Metric': 'MRR',
        'Image-to-Text': f"{metrics['i2t_MRR']:.4f}",
        'Text-to-Image': f"{metrics['t2i_MRR']:.4f}",
        'Average': f"{metrics['avg_MRR']:.4f}",
    })
    rows.append({
        'Metric': 'Median Rank',
        'Image-to-Text': f"{metrics['i2t_median_rank']:.1f}",
        'Text-to-Image': f"{metrics['t2i_median_rank']:.1f}",
        'Average': f"{metrics['avg_median_rank']:.1f}",
    })
    rows.append({
        'Metric': 'mAP',
        'Image-to-Text': f"{metrics['i2t_mAP']:.4f}",
        'Text-to-Image': f"{metrics['t2i_mAP']:.4f}",
        'Average': f"{metrics['avg_mAP']:.4f}",
    })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, 'retrieval_metrics.csv'), index=False)

    # Styled table PNG
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('off')
    table = ax.table(
        cellText=df.values, colLabels=df.columns,
        cellLoc='center', loc='center',
        colWidths=[0.3, 0.23, 0.23, 0.23],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E7E6E6')

    plt.title('Cross-Modal Retrieval Performance', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(os.path.join(output_dir, 'retrieval_metrics_table.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Similarity metrics
    sim_df = pd.DataFrame({
        'Metric': ['Positive Similarity', 'Negative Similarity', 'Gap'],
        'Value': [f"{metrics['pos_similarity']:.4f}",
                  f"{metrics['neg_similarity']:.4f}",
                  f"{metrics['similarity_gap']:.4f}"],
    })
    sim_df.to_csv(os.path.join(output_dir, 'similarity_metrics.csv'), index=False)

    print("\nRetrieval metrics:")
    print(df.to_string(index=False))
    print(f"\n{sim_df.to_string(index=False)}")

    return df


def plot_similarity_heatmap(similarity, output_dir, n_display=100):
    """Plot similarity matrix heatmap for first n_display samples."""
    sim_subset = similarity[:n_display, :n_display].numpy()

    plt.figure(figsize=(12, 10))
    sns.heatmap(sim_subset, cmap='RdYlBu_r', center=0,
                xticklabels=False, yticklabels=False,
                cbar_kws={'label': 'Cosine Similarity'})
    plt.xlabel('Text Embeddings', fontsize=12)
    plt.ylabel('Image Embeddings', fontsize=12)
    plt.title(f'Image-Text Similarity Matrix (first {n_display} samples)', fontsize=14, fontweight='bold')

    for i in range(n_display):
        plt.plot([i, i+1], [i, i], 'g-', linewidth=2, alpha=0.5)
        plt.plot([i, i], [i, i+1], 'g-', linewidth=2, alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'similarity_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved similarity_heatmap.png")


def plot_similarity_distributions(similarity, output_dir):
    """Plot positive vs negative similarity distributions."""
    N = similarity.shape[0]

    pos_sims = torch.diagonal(similarity).numpy()
    neg_mask = ~torch.eye(N, dtype=torch.bool)
    neg_sims = similarity[neg_mask].numpy()

    if len(neg_sims) > 10000:
        neg_sims = np.random.choice(neg_sims, 10000, replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(pos_sims, bins=50, alpha=0.7, label='Positive', color='green', density=True)
    axes[0].hist(neg_sims, bins=50, alpha=0.7, label='Negative', color='red', density=True)
    axes[0].set_xlabel('Cosine Similarity')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Similarity Distribution')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].boxplot([pos_sims, neg_sims], labels=['Positive', 'Negative'],
                    patch_artist=True, boxprops=dict(facecolor='lightblue', alpha=0.7))
    axes[1].set_ylabel('Cosine Similarity')
    axes[1].set_title('Similarity Comparison')
    axes[1].grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'similarity_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved similarity_distributions.png")
    print(f"  Positive: mean={pos_sims.mean():.4f}, std={pos_sims.std():.4f}")
    print(f"  Negative: mean={neg_sims.mean():.4f}, std={neg_sims.std():.4f}")


def plot_embedding_space(image_embeds, text_embeds, labels, output_dir, method='tsne', n_vis=500):
    """Visualize embedding space with t-SNE or UMAP, colored by dominant CheXpert label."""
    label_names = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
        'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
        'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
        'Pneumonia', 'Pneumothorax', 'Support Devices',
    ]

    if len(image_embeds) > n_vis:
        idx = np.random.choice(len(image_embeds), n_vis, replace=False)
        image_embeds = image_embeds[idx]
        text_embeds = text_embeds[idx]
        labels = labels[idx]

    all_embeds = torch.cat([image_embeds, text_embeds], dim=0).numpy()
    n = len(image_embeds)

    print(f"  Running {method.upper()}...")
    if method == 'tsne':
        embeds_2d = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(all_embeds)
    else:
        embeds_2d = umap.UMAP(n_components=2, random_state=42).fit_transform(all_embeds)

    img_2d = embeds_2d[:n]
    txt_2d = embeds_2d[n:]

    # Dominant label per sample
    colors = []
    for i in range(n):
        pos_idx = (labels[i] == 1.0).nonzero(as_tuple=True)[0]
        colors.append(pos_idx[0].item() if len(pos_idx) > 0 else 8)
    colors = np.array(colors)

    palette = plt.cm.tab20(np.linspace(0, 1, 14))

    fig, ax = plt.subplots(figsize=(14, 10))
    for li in range(14):
        mask = colors == li
        if mask.sum() == 0:
            continue
        ax.scatter(img_2d[mask, 0], img_2d[mask, 1],
                   c=[palette[li]], marker='o', s=50, alpha=0.6,
                   edgecolors='black', linewidth=0.5,
                   label=f'{label_names[li]} (img)')
        ax.scatter(txt_2d[mask, 0], txt_2d[mask, 1],
                   c=[palette[li]], marker='^', s=50, alpha=0.6,
                   edgecolors='black', linewidth=0.5,
                   label=f'{label_names[li]} (txt)')

    for i in range(min(50, n)):
        ax.plot([img_2d[i, 0], txt_2d[i, 0]],
                [img_2d[i, 1], txt_2d[i, 1]],
                'gray', alpha=0.2, linewidth=0.5)

    ax.set_xlabel(f'{method.upper()} 1')
    ax.set_ylabel(f'{method.upper()} 2')
    ax.set_title(f'Embedding Space ({method.upper()}) — o=image, ^=text', fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'embedding_space_{method}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved embedding_space_{method}.png")


def plot_rank_distribution(similarity, output_dir):
    """Plot histogram of retrieval ranks for correct matches."""
    N = similarity.shape[0]
    targets = torch.arange(N)

    i2t_sorted = similarity.argsort(dim=1, descending=True)
    i2t_ranks = (i2t_sorted == targets.unsqueeze(1)).nonzero(as_tuple=True)[1].numpy() + 1

    t2i_sorted = similarity.T.argsort(dim=1, descending=True)
    t2i_ranks = (t2i_sorted == targets.unsqueeze(1)).nonzero(as_tuple=True)[1].numpy() + 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(i2t_ranks, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0].axvline(np.median(i2t_ranks), color='red', linestyle='--', linewidth=2,
                    label=f'Median: {np.median(i2t_ranks):.1f}')
    axes[0].set_xlabel('Rank of Correct Text')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Image-to-Text Retrieval Ranks')
    axes[0].legend()
    axes[0].grid(alpha=0.3, axis='y')

    axes[1].hist(t2i_ranks, bins=50, edgecolor='black', alpha=0.7, color='lightcoral')
    axes[1].axvline(np.median(t2i_ranks), color='red', linestyle='--', linewidth=2,
                    label=f'Median: {np.median(t2i_ranks):.1f}')
    axes[1].set_xlabel('Rank of Correct Image')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Text-to-Image Retrieval Ranks')
    axes[1].legend()
    axes[1].grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rank_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved rank_distributions.png")


def main():
    args = parse_args()

    config = Config()
    device = torch.device(config.system.DEVICE)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("Loading data...")
    train_loader, val_loader, test_loader = create_dataloaders(config)

    dataloader = test_loader if args.use_test else val_loader
    split_name = "test" if args.use_test else "validation"
    print(f"Using {split_name} set: {len(dataloader.dataset)} samples")

    print("Loading model...")
    model = MedicalCLIP(config).to(device)
    epoch, _ = load_checkpoint(args.checkpoint, model, device=device)
    print(f"Loaded checkpoint from epoch {epoch}")

    print(f"Extracting embeddings...")
    image_embeds, text_embeds, labels, metadata = extract_embeddings(
        model, dataloader, device, n_samples=args.n_samples
    )
    print(f"Extracted {len(image_embeds)} pairs")

    print("Computing retrieval metrics...")
    metrics, similarity = compute_retrieval_metrics(image_embeds, text_embeds)

    print("\nGenerating visualizations...")
    create_metrics_table(metrics, output_dir)
    plot_similarity_heatmap(similarity, output_dir)
    plot_similarity_distributions(similarity, output_dir)
    plot_rank_distribution(similarity, output_dir)

    try:
        plot_embedding_space(image_embeds, text_embeds, labels, output_dir, method='tsne')
    except Exception as e:
        print(f"  t-SNE failed: {e}")

    try:
        plot_embedding_space(image_embeds, text_embeds, labels, output_dir, method='umap')
    except Exception as e:
        print(f"  UMAP failed: {e} (pip install umap-learn)")

    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nAll results saved to {output_dir}/")


if __name__ == "__main__":
    main()