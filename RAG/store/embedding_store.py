"""Embedding store: manages image and text embeddings with HDF5 persistence."""

import os
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import h5py

logger = logging.getLogger(__name__)


class EmbeddingStore:
    """Store for managing image and text embeddings with efficient retrieval."""

    def __init__(self, config):
        self.config = config
        self.image_embeddings: Dict[str, np.ndarray] = {}
        self.text_embeddings: Dict[str, np.ndarray] = {}
        self.store_path = os.path.join(config.paths.INDEX_DIR, "embeddings.h5")
        logger.info("Initialized EmbeddingStore")

    def add_image_embeddings(self, embeddings_dict: Dict[str, np.ndarray]):
        self.image_embeddings.update(embeddings_dict)
        logger.info(f"Added {len(embeddings_dict)} image embeddings (total: {len(self.image_embeddings)})")

    def add_text_embeddings(self, embeddings_dict: Dict[str, np.ndarray]):
        self.text_embeddings.update(embeddings_dict)
        logger.info(f"Added {len(embeddings_dict)} text embeddings (total: {len(self.text_embeddings)})")

    def get_image_embedding(self, study_id: str) -> Optional[np.ndarray]:
        return self.image_embeddings.get(study_id)

    def get_text_embedding(self, study_id: str) -> Optional[np.ndarray]:
        return self.text_embeddings.get(study_id)

    def get_all_image_embeddings(
        self, study_ids: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Get image embeddings as (matrix, valid_study_ids)."""
        if study_ids is None:
            study_ids = list(self.image_embeddings.keys())

        embeddings, valid_ids = [], []
        for sid in study_ids:
            emb = self.get_image_embedding(sid)
            if emb is not None:
                embeddings.append(emb)
                valid_ids.append(sid)

        if not embeddings:
            logger.error(f"No valid image embeddings found (requested {len(study_ids)})")
            return np.array([]), []

        arr = np.vstack(embeddings)
        logger.info(f"Retrieved {len(valid_ids)} image embeddings, shape {arr.shape}")
        return arr, valid_ids

    def get_all_text_embeddings(
        self, study_ids: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Get text embeddings as (matrix, valid_study_ids)."""
        if study_ids is None:
            study_ids = list(self.text_embeddings.keys())

        embeddings, valid_ids = [], []
        for sid in study_ids:
            emb = self.get_text_embedding(sid)
            if emb is not None:
                embeddings.append(emb)
                valid_ids.append(sid)

        if not embeddings:
            logger.error(f"No valid text embeddings found (requested {len(study_ids)})")
            return np.array([]), []

        arr = np.vstack(embeddings)
        logger.info(f"Retrieved {len(valid_ids)} text embeddings, shape {arr.shape}")
        return arr, valid_ids

    def save(self, path: Optional[str] = None):
        """Save embeddings to HDF5."""
        if path is None:
            path = self.store_path
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with h5py.File(path, 'w') as f:
            img_grp = f.create_group('image_embeddings')
            for sid, emb in self.image_embeddings.items():
                img_grp.create_dataset(sid, data=emb)
            txt_grp = f.create_group('text_embeddings')
            for sid, emb in self.text_embeddings.items():
                txt_grp.create_dataset(sid, data=emb)

        logger.info(f"Saved embeddings to {path} "
                     f"(images: {len(self.image_embeddings)}, texts: {len(self.text_embeddings)})")

    def load(self, path: Optional[str] = None):
        """Load embeddings from HDF5."""
        if path is None:
            path = self.store_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"Embedding store not found: {path}")

        with h5py.File(path, 'r') as f:
            if 'image_embeddings' in f:
                for sid in f['image_embeddings'].keys():
                    self.image_embeddings[sid] = f['image_embeddings'][sid][:]
            if 'text_embeddings' in f:
                for sid in f['text_embeddings'].keys():
                    self.text_embeddings[sid] = f['text_embeddings'][sid][:]

        logger.info(f"Loaded embeddings from {path} "
                     f"(images: {len(self.image_embeddings)}, texts: {len(self.text_embeddings)})")

    def get_statistics(self) -> Dict:
        stats = {
            'num_image_embeddings': len(self.image_embeddings),
            'num_text_embeddings': len(self.text_embeddings),
        }
        if self.image_embeddings:
            first = next(iter(self.image_embeddings.values()))
            stats['image_embedding_dim'] = first.shape[0] if hasattr(first, 'shape') else 0
        else:
            stats['image_embedding_dim'] = 0
        if self.text_embeddings:
            first = next(iter(self.text_embeddings.values()))
            stats['text_embedding_dim'] = first.shape[0] if hasattr(first, 'shape') else 0
        else:
            stats['text_embedding_dim'] = 0
        return stats

    def clear(self):
        self.image_embeddings.clear()
        self.text_embeddings.clear()
        logger.info("Cleared all embeddings")

    def __len__(self):
        return len(self.image_embeddings)

    def __repr__(self):
        return f"EmbeddingStore(images={len(self.image_embeddings)}, texts={len(self.text_embeddings)})"