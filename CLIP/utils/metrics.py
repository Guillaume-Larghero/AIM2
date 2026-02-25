"""Retrieval and classification metrics for Medical CLIP."""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def compute_retrieval_metrics(image_embeds, text_embeds, k_values=[1, 5, 10]):
    """Compute image-text retrieval metrics (Recall@k, MRR, median rank, similarity stats)."""
    batch_size = image_embeds.shape[0]
    device = image_embeds.device

    image_embeds = F.normalize(image_embeds, p=2, dim=1)
    text_embeds = F.normalize(text_embeds, p=2, dim=1)

    similarity = image_embeds @ text_embeds.T
    targets = torch.arange(batch_size, device=device)

    metrics = {}

    # Image-to-text retrieval
    i2t_sorted = similarity.argsort(dim=1, descending=True)
    for k in k_values:
        if k <= batch_size:
            recall = (i2t_sorted[:, :k] == targets.unsqueeze(1)).any(dim=1).float().mean().item()
            metrics[f'i2t_R@{k}'] = recall

    i2t_ranks = (i2t_sorted == targets.unsqueeze(1)).nonzero(as_tuple=True)[1].float()
    metrics['i2t_MRR'] = (1.0 / (i2t_ranks + 1)).mean().item()
    metrics['i2t_median_rank'] = i2t_ranks.median().item() + 1

    # Text-to-image retrieval
    t2i_sorted = similarity.T.argsort(dim=1, descending=True)
    for k in k_values:
        if k <= batch_size:
            recall = (t2i_sorted[:, :k] == targets.unsqueeze(1)).any(dim=1).float().mean().item()
            metrics[f't2i_R@{k}'] = recall

    t2i_ranks = (t2i_sorted == targets.unsqueeze(1)).nonzero(as_tuple=True)[1].float()
    metrics['t2i_MRR'] = (1.0 / (t2i_ranks + 1)).mean().item()
    metrics['t2i_median_rank'] = t2i_ranks.median().item() + 1

    # Averaged metrics
    for k in k_values:
        if k <= batch_size:
            metrics[f'avg_R@{k}'] = (metrics[f'i2t_R@{k}'] + metrics[f't2i_R@{k}']) / 2
    metrics['avg_MRR'] = (metrics['i2t_MRR'] + metrics['t2i_MRR']) / 2

    metrics['i2t_acc'] = metrics.get('i2t_R@1', 0.0)
    metrics['t2i_acc'] = metrics.get('t2i_R@1', 0.0)
    metrics['avg_acc'] = (metrics['i2t_acc'] + metrics['t2i_acc']) / 2

    # Similarity statistics
    pos_sim = torch.diagonal(similarity).mean().item()
    neg_mask = ~torch.eye(batch_size, device=device, dtype=torch.bool)
    neg_sim = similarity[neg_mask].mean().item()

    metrics['pos_sim'] = pos_sim
    metrics['neg_sim'] = neg_sim
    metrics['sim_gap'] = pos_sim - neg_sim

    return metrics


def compute_chexpert_metrics(logits, labels, label_names=None, return_per_label=True):
    """Compute classification metrics for CheXpert, handling NaN-masked labels."""
    batch_size, num_labels = logits.shape
    device = logits.device

    if label_names is None:
        label_names = [f'label_{i}' for i in range(num_labels)]

    valid_mask = ~torch.isnan(labels)
    labels_cleaned = torch.nan_to_num(labels, nan=0.0)

    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    metrics = {}
    num_valid = valid_mask.float().sum().item()

    if num_valid > 0:
        valid_preds = preds[valid_mask]
        valid_labels = labels_cleaned[valid_mask]
        valid_probs = probs[valid_mask]

        metrics['accuracy'] = (valid_preds == valid_labels).float().mean().item()

        tp = ((valid_preds == 1) & (valid_labels == 1)).sum().float().item()
        fp = ((valid_preds == 1) & (valid_labels == 0)).sum().float().item()
        fn = ((valid_preds == 0) & (valid_labels == 1)).sum().float().item()
        tn = ((valid_preds == 0) & (valid_labels == 0)).sum().float().item()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)

        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1'] = 2 * precision * recall / (precision + recall + 1e-8)
        metrics['specificity'] = specificity
        metrics['balanced_accuracy'] = (recall + specificity) / 2
        metrics['num_valid'] = num_valid
        metrics['num_positive'] = (valid_labels == 1).sum().item()
        metrics['num_negative'] = (valid_labels == 0).sum().item()

        try:
            valid_probs_np = valid_probs.cpu().numpy()
            valid_labels_np = valid_labels.cpu().numpy()
            if len(np.unique(valid_labels_np)) > 1:
                metrics['auroc'] = roc_auc_score(valid_labels_np, valid_probs_np)
                metrics['auprc'] = average_precision_score(valid_labels_np, valid_probs_np)
        except Exception:
            pass
    else:
        for key in ['accuracy', 'precision', 'recall', 'f1', 'specificity', 'balanced_accuracy']:
            metrics[key] = 0.0
        metrics['num_valid'] = 0

    # Per-label metrics
    if return_per_label:
        per_label = {}
        for i, name in enumerate(label_names):
            mask = valid_mask[:, i]
            n = mask.sum().item()
            if n == 0:
                continue

            lp = preds[:, i][mask]
            ll = labels_cleaned[:, i][mask]
            lprobs = probs[:, i][mask]

            label_tp = ((lp == 1) & (ll == 1)).sum().float().item()
            label_fp = ((lp == 1) & (ll == 0)).sum().float().item()
            label_fn = ((lp == 0) & (ll == 1)).sum().float().item()
            label_tn = ((lp == 0) & (ll == 0)).sum().float().item()

            label_prec = label_tp / (label_tp + label_fp + 1e-8)
            label_rec = label_tp / (label_tp + label_fn + 1e-8)

            entry = {
                'accuracy': (lp == ll).float().mean().item(),
                'precision': label_prec,
                'recall': label_rec,
                'f1': 2 * label_prec * label_rec / (label_prec + label_rec + 1e-8),
                'specificity': label_tn / (label_tn + label_fp + 1e-8),
                'n_valid': n,
                'n_positive': (ll == 1).sum().item(),
                'n_negative': (ll == 0).sum().item(),
            }

            try:
                lp_np = lprobs.cpu().numpy()
                ll_np = ll.cpu().numpy()
                if len(np.unique(ll_np)) > 1:
                    entry['auroc'] = roc_auc_score(ll_np, lp_np)
            except Exception:
                pass

            per_label[name] = entry

        metrics['per_label'] = per_label

    return metrics