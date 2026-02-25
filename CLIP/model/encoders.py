"""Image and text encoders, projection heads, and CheXpert classification head."""

import torch
import torch.nn as nn
import torchvision.models as models
from transformers import AutoModel
import timm


class ImageEncoder(nn.Module):
    """Image encoder supporting ViT and ResNet backbones with optional attention pooling."""

    def __init__(self, config):
        super().__init__()

        self.model_name = config.model.IMAGE_ENCODER
        self.use_attention_pooling = config.model.USE_ATTENTION_POOLING
        self.feature_dim = config.model.IMAGE_FEATURE_DIM

        if self.model_name.startswith('vit'):
            self.encoder = timm.create_model(
                self.model_name,
                pretrained=config.model.IMAGE_PRETRAINED,
                num_classes=0,
                global_pool='',
            )
            if self.use_attention_pooling:
                self.attention_pool = AttentionPool(
                    self.feature_dim,
                    config.model.ATTENTION_POOL_HIDDEN_DIM_RATIO,
                )

        elif self.model_name in ('resnet50', 'resnet18'):
            resnet_fn = models.resnet50 if self.model_name == 'resnet50' else models.resnet18
            resnet = resnet_fn(pretrained=config.model.IMAGE_PRETRAINED)
            self.encoder = nn.Sequential(*list(resnet.children())[:-2])

            if self.use_attention_pooling:
                self.attention_pool = SpatialAttentionPool(
                    self.feature_dim,
                    config.model.SPATIAL_ATTENTION_REDUCTION_RATIO,
                )
            else:
                self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            raise ValueError(f"Unknown image encoder: {self.model_name}")

        print(f"Image Encoder: {self.model_name} (dim={self.feature_dim}, "
              f"attention_pool={self.use_attention_pooling})")

    def forward(self, images):
        """(B, 3, H, W) -> (B, feature_dim)"""
        features = self.encoder(images)

        if self.use_attention_pooling:
            features = self.attention_pool(features)
        elif len(features.shape) > 2:
            if hasattr(self, 'global_pool'):
                features = self.global_pool(features)
            features = features.flatten(1)

        return features


class AttentionPool(nn.Module):
    """Learned attention pooling over ViT patch tokens."""

    def __init__(self, feature_dim, hidden_dim_ratio=0.25):
        super().__init__()
        hidden_dim = int(feature_dim * hidden_dim_ratio)
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        """(B, num_patches, D) -> (B, D)"""
        attn_weights = torch.softmax(self.attention(x), dim=1)
        return (x * attn_weights).sum(dim=1)


class SpatialAttentionPool(nn.Module):
    """Learned spatial attention pooling for CNN feature maps."""

    def __init__(self, feature_dim, reduction_ratio=8):
        super().__init__()
        hidden_dim = feature_dim // reduction_ratio
        self.attention_conv = nn.Sequential(
            nn.Conv2d(feature_dim, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        """(B, C, H, W) -> (B, C)"""
        attn_map = self.attention_conv(x)
        attended = x * attn_map
        return self.global_pool(attended).flatten(1)


class TextEncoder(nn.Module):
    """ClinicalBERT text encoder with weighted multi-layer [CLS] fusion."""

    def __init__(self, config):
        super().__init__()

        self.model_name = config.model.TEXT_ENCODER
        self.use_last_n_layers = config.model.TEXT_NUM_LAYERS
        self.feature_dim = config.model.TEXT_FEATURE_DIM

        self.encoder = AutoModel.from_pretrained(
            self.model_name,
            output_hidden_states=True,
        )

        if self.use_last_n_layers > 1:
            self.layer_weights = nn.Parameter(torch.ones(self.use_last_n_layers))

        print(f"Text Encoder: {self.model_name} (dim={self.feature_dim}, "
              f"fused_layers={self.use_last_n_layers})")

    def forward(self, input_ids, attention_mask):
        """(B, seq_len) -> (B, feature_dim)"""
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        if self.use_last_n_layers > 1:
            hidden_states = outputs.hidden_states[-self.use_last_n_layers:]
            cls_embeddings = torch.stack([h[:, 0, :] for h in hidden_states], dim=0)
            weights = torch.softmax(self.layer_weights, dim=0)
            return (cls_embeddings * weights.view(-1, 1, 1)).sum(dim=0)
        else:
            return outputs.last_hidden_state[:, 0, :]


class ProjectionHead(nn.Module):
    """MLP projection head with residual connection."""

    def __init__(self, config, input_dim=None):
        super().__init__()

        if input_dim is None:
            input_dim = config.model.PROJECTION_DIM

        hidden_dim = config.model.PROJECTION_HIDDEN_DIM
        output_dim = config.model.PROJECTION_DIM
        dropout = config.model.PROJECTION_DROPOUT
        num_layers = config.model.PROJECTION_NUM_LAYERS
        self.normalize = config.model.NORMALIZE_EMBEDDINGS

        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.extend([
                    nn.Linear(input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ])
            elif i == num_layers - 1:
                layers.append(nn.Linear(hidden_dim, output_dim))
            else:
                layers.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ])

        self.mlp = nn.Sequential(*layers)

        if input_dim == output_dim:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Linear(input_dim, output_dim)

        print(f"Projection: {input_dim} -> {hidden_dim} -> {output_dim} ({num_layers} layers)")

    def forward(self, x):
        """(B, input_dim) -> (B, output_dim), optionally L2-normalized."""
        embeddings = self.mlp(x) + self.residual(x)
        if self.normalize:
            embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings


class CheXpertHead(nn.Module):
    """Multi-label classification head for CheXpert auxiliary task."""

    def __init__(self, config):
        super().__init__()

        input_dim = config.model.PROJECTION_DIM
        num_labels = config.model.NUM_CHEXPERT_LABELS
        hidden_dim = int(input_dim * config.model.CHEXPERT_HIDDEN_DIM_RATIO)
        dropout = config.model.PROJECTION_DROPOUT

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

        print(f"CheXpert Head: {input_dim} -> {hidden_dim} -> {num_labels}")

    def forward(self, embeddings):
        """(B, input_dim) -> (B, num_labels) raw logits."""
        return self.classifier(embeddings)