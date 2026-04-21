"""CLIP embedder: loads trained CLIP model and generates embeddings."""

import os
import pickle
import logging
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from CLIP.model.clip_model import MedicalCLIP
from CLIP.config.config import Config as CLIPConfig

logger = logging.getLogger(__name__)


class CLIPEmbedder:
    """Load a trained Medical CLIP model and generate image/text embeddings."""

    def __init__(self, config, clip_config=None):
        """
        Args:
            config: RAGConfig object.
            clip_config: CLIP Config object (created fresh if None).
        """
        self.config = config
        self.device = torch.device(config.embedder.DEVICE)
        self.model = self._load_clip_model(config, clip_config)
        self.model.eval()
        self.model.to(self.device)
        self.transform = self._get_transform()

        self.cache_path = os.path.join(config.paths.CACHE_DIR, "embeddings_cache.pkl")
        self.embedding_cache: Dict[str, np.ndarray] = {}

        if config.embedder.CACHE_EMBEDDINGS and os.path.exists(self.cache_path):
            self._load_cache()

        logger.info(f"CLIPEmbedder initialized on {self.device}")

    def _load_clip_model(self, config, clip_config=None):
        """Load trained Medical CLIP model from checkpoint."""
        checkpoint_path = config.paths.CLIP_CHECKPOINT
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"CLIP checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading CLIP model from: {checkpoint_path}")

        if clip_config is None:
            clip_config = CLIPConfig()

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Infer the image size the model was trained with from the pos_embed shape.
        # pos_embed: [1, num_tokens, dim] where num_tokens = (img_size // patch_size)^2 + 1
        # vit_base_patch16 uses patch_size=16.
        state = checkpoint['model_state_dict']
        pos_embed_key = next(
            (k for k in state if 'pos_embed' in k and 'image_encoder' in k), None
        )
        if pos_embed_key is not None:
            num_tokens = state[pos_embed_key].shape[1]
            inferred_img_size = int((num_tokens - 1) ** 0.5) * 16
            if inferred_img_size != clip_config.data.IMAGE_SIZE:
                logger.info(
                    f"Checkpoint pos_embed has {num_tokens} tokens → img_size={inferred_img_size}px. "
                    f"Overriding CLIPConfig IMAGE_SIZE {clip_config.data.IMAGE_SIZE} → {inferred_img_size}."
                )
                clip_config.data.IMAGE_SIZE = inferred_img_size

        # Store the resolved size so _get_transform uses the correct crop size.
        self._model_image_size = clip_config.data.IMAGE_SIZE

        model = MedicalCLIP(clip_config)
        model.load_state_dict(checkpoint['model_state_dict'])

        logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        return model

    def _get_transform(self):
        """Validation transform matching CLIP training: Resize then CenterCrop to model size."""
        crop_size = getattr(self, '_model_image_size', self.config.embedder.IMAGE_SIZE)
        # Scale the resize proportionally (standard: resize to crop_size * 256/224).
        resize_size = int(round(crop_size * 256 / 224))
        return transforms.Compose([
            transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config.embedder.IMAGE_MEAN,
                std=self.config.embedder.IMAGE_STD,
            ),
        ])

    @torch.no_grad()
    def encode_image(self, image: Union[Image.Image, torch.Tensor]) -> np.ndarray:
        """Generate embedding for a single image. Returns shape (embedding_dim,)."""
        if isinstance(image, Image.Image):
            image = self.transform(image)
        if image.ndim == 3:
            image = image.unsqueeze(0)
        image = image.to(self.device)
        return self.model.encode_image(image).cpu().numpy()[0]

    @torch.no_grad()
    def encode_image_from_path(self, image_path: str, use_cache: bool = True) -> np.ndarray:
        """Generate embedding for an image file path."""
        if use_cache and image_path in self.embedding_cache:
            return self.embedding_cache[image_path]

        image = Image.open(image_path).convert('RGB')
        embedding = self.encode_image(image)

        if self.config.embedder.CACHE_EMBEDDINGS:
            self.embedding_cache[image_path] = embedding
        return embedding

    def generate_embeddings_for_dataset(
        self,
        image_paths: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Generate embeddings for a list of image paths. Returns {path: embedding}."""
        if batch_size is None:
            batch_size = self.config.embedder.BATCH_SIZE

        dataset = ImagePathDataset(image_paths, self.transform)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=self.config.embedder.NUM_WORKERS, pin_memory=True,
        )

        embeddings_dict = {}
        for batch_images, batch_paths in tqdm(dataloader, desc="Image embeddings",
                                               disable=not show_progress):
            batch_images = batch_images.to(self.device)
            with torch.no_grad():
                batch_emb = self.model.encode_image(batch_images).cpu().numpy()
            for path, emb in zip(batch_paths, batch_emb):
                embeddings_dict[path] = emb

        if self.config.embedder.CACHE_EMBEDDINGS:
            self.embedding_cache.update(embeddings_dict)

        logger.info(f"Generated {len(embeddings_dict)} image embeddings")
        return embeddings_dict

    def generate_embeddings_from_dataframe(
        self,
        df,
        image_path_col: str = 'image_path',
        batch_size: Optional[int] = None,
        show_progress: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Generate image embeddings for paths in a DataFrame column."""
        image_paths = df[image_path_col].tolist()
        return self.generate_embeddings_for_dataset(
            image_paths, batch_size=batch_size, show_progress=show_progress,
        )

    def save_embeddings(self, embeddings_dict: Dict[str, np.ndarray],
                        save_path: Optional[str] = None):
        """Save embeddings dict to pickle."""
        if save_path is None:
            save_path = self.cache_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(embeddings_dict, f)
        logger.info(f"Saved {len(embeddings_dict)} embeddings to {save_path}")

    def load_embeddings(self, load_path: Optional[str] = None) -> Dict[str, np.ndarray]:
        """Load embeddings dict from pickle."""
        if load_path is None:
            load_path = self.cache_path
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Embeddings file not found: {load_path}")
        with open(load_path, 'rb') as f:
            embeddings_dict = pickle.load(f)
        logger.info(f"Loaded {len(embeddings_dict)} embeddings from {load_path}")
        return embeddings_dict

    def _load_cache(self):
        try:
            self.embedding_cache = self.load_embeddings()
            logger.info(f"Loaded {len(self.embedding_cache)} cached embeddings")
        except Exception as e:
            logger.warning(f"Could not load cache: {e}")
            self.embedding_cache = {}

    def save_cache(self):
        """Persist current cache to disk."""
        if self.embedding_cache:
            self.save_embeddings(self.embedding_cache, self.cache_path)

    @torch.no_grad()
    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> np.ndarray:
        """Generate text embeddings from tokenized input."""
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        return self.model.encode_text(input_ids, attention_mask).cpu().numpy()

    @torch.no_grad()
    def encode_text_from_string(self, text: str, tokenizer, max_length: int = 512) -> np.ndarray:
        """Generate embedding for a raw text string. Returns shape (embedding_dim,)."""
        encoded = tokenizer(
            text, max_length=max_length, padding='max_length',
            truncation=True, return_tensors='pt',
        )
        return self.encode_text(encoded['input_ids'], encoded['attention_mask'])[0]

    def generate_text_embeddings_from_dataframe(
        self,
        df,
        tokenizer,
        text_col: str = 'text',
        max_length: int = 512,
        batch_size: Optional[int] = None,
        show_progress: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Generate text embeddings for reports in a DataFrame. Returns {study_id: embedding}."""
        if batch_size is None:
            batch_size = self.config.embedder.BATCH_SIZE

        study_ids = df['study_id'].tolist()
        texts = df[text_col].tolist()
        embeddings_dict = {}

        for i in tqdm(range(0, len(texts), batch_size), desc="Text embeddings",
                      disable=not show_progress):
            batch_texts = texts[i:i + batch_size]
            batch_ids = study_ids[i:i + batch_size]

            encoded = tokenizer(
                batch_texts, max_length=max_length, padding='max_length',
                truncation=True, return_tensors='pt',
            )
            batch_emb = self.encode_text(encoded['input_ids'], encoded['attention_mask'])

            for sid, emb in zip(batch_ids, batch_emb):
                embeddings_dict[str(sid)] = emb

        logger.info(f"Generated {len(embeddings_dict)} text embeddings")
        return embeddings_dict

    def get_embedding_dim(self) -> int:
        return self.model.config.model.PROJECTION_DIM

    def __repr__(self):
        return f"CLIPEmbedder(device={self.device}, cached={len(self.embedding_cache)})"


class ImagePathDataset(Dataset):
    """Simple dataset for loading images from file paths."""

    def __init__(self, image_paths: List[str], transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, image_path
        except Exception as e:
            logger.error(f"Error loading {image_path}: {e}")
            if self.transform:
                image = self.transform(Image.new('RGB', (224, 224), (0, 0, 0)))
            else:
                image = torch.zeros(3, 224, 224)
            return image, image_path