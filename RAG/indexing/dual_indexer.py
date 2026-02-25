"""Dual FAISS indexer: separate indices for image and text retrieval."""

import os
import logging
from typing import List, Tuple, Optional, Dict

import numpy as np

from .faiss_indexer import FaissIndexer

logger = logging.getLogger(__name__)


class DualFaissIndexer:
    """Manages two FAISS indices: one for images, one for texts."""

    def __init__(self, config, embedding_dim: int = 256):
        self.config = config
        self.embedding_dim = embedding_dim
        self.image_indexer = FaissIndexer(config, embedding_dim)
        self.text_indexer = FaissIndexer(config, embedding_dim)
        logger.info("Initialized DualFaissIndexer")

    def build_image_index(self, embeddings: np.ndarray, study_ids: List[str]):
        logger.info("Building image index...")
        self.image_indexer.build_index(embeddings, study_ids)

    def build_text_index(self, embeddings: np.ndarray, study_ids: List[str]):
        logger.info("Building text index...")
        self.text_indexer.build_index(embeddings, study_ids)

    def search_by_image(self, query_embedding: np.ndarray, top_k: int = 5,
                        return_distances: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        return self.image_indexer.search(query_embedding, top_k, return_distances)

    def search_by_text(self, query_embedding: np.ndarray, top_k: int = 5,
                       return_distances: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        return self.text_indexer.search(query_embedding, top_k, return_distances)

    def batch_search_by_image(self, query_embeddings: np.ndarray,
                              top_k: int = 5) -> Tuple[np.ndarray, List[List[str]]]:
        return self.image_indexer.batch_search(query_embeddings, top_k)

    def batch_search_by_text(self, query_embeddings: np.ndarray,
                             top_k: int = 5) -> Tuple[np.ndarray, List[List[str]]]:
        return self.text_indexer.batch_search(query_embeddings, top_k)

    def save(self, save_dir: Optional[str] = None):
        if save_dir is None:
            save_dir = self.config.paths.INDEX_DIR
        image_dir = os.path.join(save_dir, "image_index")
        text_dir = os.path.join(save_dir, "text_index")
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(text_dir, exist_ok=True)
        self.image_indexer.save(image_dir)
        self.text_indexer.save(text_dir)
        logger.info(f"Saved dual indices to {save_dir}")

    def load(self, load_dir: Optional[str] = None):
        if load_dir is None:
            load_dir = self.config.paths.INDEX_DIR
        self.image_indexer.load(os.path.join(load_dir, "image_index"))
        self.text_indexer.load(os.path.join(load_dir, "text_index"))
        logger.info(f"Loaded dual indices from {load_dir}")

    def get_stats(self) -> Dict:
        return {
            'image_index': self.image_indexer.get_stats(),
            'text_index': self.text_indexer.get_stats(),
        }

    def __repr__(self):
        return f"DualFaissIndexer(images={len(self.image_indexer)}, texts={len(self.text_indexer)})"