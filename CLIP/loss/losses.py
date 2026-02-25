"""Loss functions for Medical CLIP training."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.metrics import compute_retrieval_metrics, compute_chexpert_metrics


class CLIPLoss(nn.Module):
    """Contrastive loss with optional hard negative mining based on embedding similarity."""

    def __init__(self, config):
        super().__init__()
        self.temperature = config.training.TEMPERATURE
        self.use_hard_negatives = config.training.USE_HARD_NEGATIVES
        self.hard_neg_ratio = config.training.HARD_NEG_RATIO
        self.retrieval_k_values = config.evaluation.RETRIEVAL_K_VALUES

    def forward(self, image_embeds, text_embeds):
        """Compute symmetric contrastive loss with retrieval metrics."""
        batch_size = image_embeds.shape[0]
        device = image_embeds.device

        image_embeds = F.normalize(image_embeds, p=2, dim=1)
        text_embeds = F.normalize(text_embeds, p=2, dim=1)

        logits = image_embeds @ text_embeds.T / self.temperature
        targets = torch.arange(batch_size, device=device)

        if self.use_hard_negatives:
            loss_i2t = self._hard_negative_loss(logits, targets)
            loss_t2i = self._hard_negative_loss(logits.T, targets)
        else:
            loss_i2t = F.cross_entropy(logits, targets)
            loss_t2i = F.cross_entropy(logits.T, targets)

        loss = (loss_i2t + loss_t2i) / 2

        with torch.no_grad():
            metrics = compute_retrieval_metrics(
                image_embeds, text_embeds, k_values=self.retrieval_k_values
            )
            metrics['i2t_loss'] = loss_i2t.item()
            metrics['t2i_loss'] = loss_t2i.item()
            metrics['temperature'] = self.temperature

        return loss, metrics

    def _hard_negative_loss(self, logits, targets):
        """Weight samples by hardness of their most similar negatives."""
        batch_size = logits.shape[0]
        device = logits.device

        loss_per_sample = F.cross_entropy(logits, targets, reduction='none')

        eye_mask = torch.eye(batch_size, device=device).bool()
        neg_logits = logits.masked_fill(eye_mask, float('-inf'))

        k = max(1, int(self.hard_neg_ratio * (batch_size - 1)))
        hard_neg_values, _ = neg_logits.topk(k, dim=1)

        hardness = hard_neg_values.mean(dim=1)
        weights = 1.0 + torch.sigmoid(hardness)

        return (loss_per_sample * weights.detach()).mean()


class CheXpertLoss(nn.Module):
    """BCE loss for CheXpert multi-label classification, masking NaN labels."""

    def __init__(self, config):
        super().__init__()
        self.pos_weight = None
        self.use_focal = config.training.USE_FOCAL_LOSS
        self.focal_gamma = config.training.FOCAL_GAMMA
        self.label_smoothing = config.training.LABEL_SMOOTHING
        self.label_names = config.data.CHEXPERT_LABELS

    def forward(self, logits, labels):
        """Compute masked BCE loss. Only 1.0 and 0.0 labels contribute; NaN is ignored."""
        valid_mask = ~torch.isnan(labels)
        labels_cleaned = torch.nan_to_num(labels, nan=0.0)

        if self.label_smoothing > 0:
            labels_cleaned = labels_cleaned * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        bce_loss = F.binary_cross_entropy_with_logits(
            logits, labels_cleaned, pos_weight=self.pos_weight, reduction='none'
        )

        if self.use_focal:
            probs = torch.sigmoid(logits)
            pt = torch.where(labels_cleaned > 0.5, probs, 1 - probs)
            bce_loss = ((1 - pt) ** self.focal_gamma) * bce_loss

        masked_loss = bce_loss * valid_mask.float()
        num_valid = valid_mask.float().sum()

        if num_valid > 0:
            loss = masked_loss.sum() / num_valid
        else:
            loss = torch.tensor(0.0, device=logits.device, requires_grad=True)

        with torch.no_grad():
            if num_valid > 0:
                metrics = compute_chexpert_metrics(
                    logits, labels, label_names=self.label_names, return_per_label=True
                )
            else:
                metrics = {k: 0.0 for k in ['accuracy', 'precision', 'recall', 'f1', 'num_valid']}

        return loss, metrics


def compute_total_loss(image_embeds, text_embeds, chexpert_logits, chexpert_labels,
                       config, contrastive_loss_fn, chexpert_loss_fn):
    """Combine contrastive and CheXpert losses with configured weighting."""
    loss_dict = {}

    contrastive_loss, contrastive_metrics = contrastive_loss_fn(image_embeds, text_embeds)
    loss_dict['contrastive_loss'] = contrastive_loss.item()
    loss_dict.update({f'contrastive_{k}': v for k, v in contrastive_metrics.items()})

    total_loss = config.training.CONTRASTIVE_WEIGHT * contrastive_loss

    if config.model.USE_CHEXPERT_AUX and chexpert_logits is not None:
        chexpert_loss, chexpert_metrics = chexpert_loss_fn(chexpert_logits, chexpert_labels)
        loss_dict['chexpert_loss'] = chexpert_loss.item()
        loss_dict.update({f'chexpert_{k}': v for k, v in chexpert_metrics.items()})
        total_loss += config.training.CHEXPERT_WEIGHT * chexpert_loss

    loss_dict['total_loss'] = total_loss.item()

    return total_loss, loss_dict