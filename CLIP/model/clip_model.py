"""Medical CLIP model for aligning chest X-rays and radiology reports."""

import torch
import torch.nn as nn
from .encoders import ImageEncoder, TextEncoder, ProjectionHead, CheXpertHead


class MedicalCLIP(nn.Module):
    """Dual-encoder CLIP model with optional CheXpert auxiliary head."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Image pathway
        self.image_encoder = ImageEncoder(config)
        self.image_projection = ProjectionHead(config, input_dim=config.model.IMAGE_FEATURE_DIM)

        # Text pathway
        self.text_encoder = TextEncoder(config)
        self.text_projection = ProjectionHead(config, input_dim=config.model.TEXT_FEATURE_DIM)

        # CheXpert auxiliary head
        if config.model.USE_CHEXPERT_AUX:
            self.chexpert_head = CheXpertHead(config)
        else:
            self.chexpert_head = None

        # Learnable temperature (log-parameterized)
        self.logit_scale = nn.Parameter(
            torch.ones([]) * torch.log(torch.tensor(1.0 / config.training.TEMPERATURE))
        )

        self._print_model_info()

    def forward(self, images, input_ids, attention_mask):
        """Full forward pass returning embeddings and optional CheXpert logits."""
        image_features = self.image_encoder(images)
        image_embeds = self.image_projection(image_features)

        text_features = self.text_encoder(input_ids, attention_mask)
        text_embeds = self.text_projection(text_features)

        chexpert_logits = None
        if self.chexpert_head is not None:
            chexpert_logits = self.chexpert_head(image_embeds)

        return image_embeds, text_embeds, chexpert_logits

    def encode_image(self, images):
        """Encode images only (inference)."""
        with torch.no_grad():
            return self.image_projection(self.image_encoder(images))

    def encode_text(self, input_ids, attention_mask):
        """Encode text only (inference)."""
        with torch.no_grad():
            return self.text_projection(self.text_encoder(input_ids, attention_mask))

    def _print_model_info(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\nModel: {total:,} params ({trainable:,} trainable)")
        print(f"  Image: {self.config.model.IMAGE_ENCODER}")
        print(f"  Text:  {self.config.model.TEXT_ENCODER}")
        print(f"  Projection dim: {self.config.model.PROJECTION_DIM}")
        print(f"  CheXpert aux: {self.config.model.USE_CHEXPERT_AUX}")

    def get_param_groups(self, config):
        """Parameter groups with differential learning rates."""
        param_groups = [
            {'params': self.image_encoder.parameters(), 'lr': config.training.IMAGE_ENCODER_LR, 'name': 'image_encoder'},
            {'params': self.text_encoder.parameters(), 'lr': config.training.TEXT_ENCODER_LR, 'name': 'text_encoder'},
            {'params': list(self.image_projection.parameters()) + list(self.text_projection.parameters()),
             'lr': config.training.PROJECTION_LR, 'name': 'projection_heads'},
            {'params': [self.logit_scale], 'lr': config.training.PROJECTION_LR, 'name': 'temperature'},
        ]

        if self.chexpert_head is not None:
            param_groups.append({
                'params': self.chexpert_head.parameters(),
                'lr': config.training.PROJECTION_LR,
                'name': 'chexpert_head',
            })

        return param_groups