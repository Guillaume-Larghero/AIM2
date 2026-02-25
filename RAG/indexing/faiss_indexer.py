"""FAISS indexer for efficient similarity search."""

import os
import pickle
import logging
from typing import List, Tuple, Optional, Union

import numpy as np
import faiss

logger = logging.getLogger(__name__)


class FaissIndexer:
    """FAISS-based indexer supporting Flat, HNSW, and IVF index types."""

    def __init__(self, config, embedding_dim: int = 256):
        self.config = config
        self.embedding_dim = embedding_dim
        self.index = None
        self.study_ids: List[str] = []
        self.index_type = config.retrieval.INDEX_TYPE
        self.metric_type = config.retrieval.METRIC_TYPE
        self.hnsw_params = {}
        self.ivf_params = {}
        logger.info(f"FaissIndexer: type={self.index_type}, metric={self.metric_type}")

    def build_index(self, embeddings: np.ndarray, study_ids: List[str]):
        """Build FAISS index from embeddings array of shape (N, embedding_dim)."""
        logger.info(f"Building index: shape={embeddings.shape}, dtype={embeddings.dtype}")

        if embeddings.size == 0:
            raise ValueError("Embeddings array is empty")

        # Handle accidental 1D array
        if embeddings.ndim == 1:
            if embeddings.size % self.embedding_dim == 0:
                n = embeddings.size // self.embedding_dim
                logger.warning(f"Reshaping 1D array to ({n}, {self.embedding_dim})")
                embeddings = embeddings.reshape(n, self.embedding_dim)
            else:
                raise ValueError(
                    f"Cannot reshape 1D array of size {embeddings.size} "
                    f"to (N, {self.embedding_dim})"
                )

        if embeddings.ndim != 2:
            raise ValueError(f"Embeddings must be 2D, got shape {embeddings.shape}")
        if embeddings.shape[0] != len(study_ids):
            raise ValueError(
                f"Count mismatch: {embeddings.shape[0]} embeddings vs {len(study_ids)} study_ids"
            )
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Dimension mismatch: {embeddings.shape[1]} vs expected {self.embedding_dim}"
            )

        embeddings = embeddings.astype(np.float32)
        if self.metric_type == "INNER_PRODUCT":
            faiss.normalize_L2(embeddings)

        builders = {
            "FlatIP": self._build_flat_ip,
            "FlatL2": self._build_flat_l2,
            "HNSW": self._build_hnsw,
            "IVFFlat": self._build_ivf,
        }
        if self.index_type not in builders:
            raise ValueError(f"Unknown index type: {self.index_type}")

        self.index = builders[self.index_type](embeddings)
        self.study_ids = study_ids
        logger.info(f"Built {self.index_type} index: {len(study_ids)} embeddings, ntotal={self.index.ntotal}")

    def _build_flat_ip(self, embeddings: np.ndarray) -> faiss.Index:
        index = faiss.IndexFlatIP(self.embedding_dim)
        index.add(embeddings)
        return index

    def _build_flat_l2(self, embeddings: np.ndarray) -> faiss.Index:
        index = faiss.IndexFlatL2(self.embedding_dim)
        index.add(embeddings)
        return index

    def _build_hnsw(self, embeddings: np.ndarray) -> faiss.Index:
        M = self.config.retrieval.HNSW_M
        ef_construction = self.config.retrieval.HNSW_EF_CONSTRUCTION
        ef_search = self.config.retrieval.HNSW_EF_SEARCH
        self.hnsw_params = {'M': M, 'efConstruction': ef_construction, 'efSearch': ef_search}

        metric = (faiss.METRIC_INNER_PRODUCT if self.metric_type == "INNER_PRODUCT"
                  else faiss.METRIC_L2)
        index = faiss.IndexHNSWFlat(self.embedding_dim, M, metric)
        index.hnsw.efConstruction = ef_construction
        index.hnsw.efSearch = ef_search
        index.add(embeddings)

        logger.info(f"  HNSW: M={M}, efConstruction={ef_construction}, efSearch={ef_search}")
        return index

    def _build_ivf(self, embeddings: np.ndarray) -> faiss.Index:
        nlist = self.config.retrieval.NLIST
        nprobe = self.config.retrieval.NPROBE
        self.ivf_params = {'nlist': nlist, 'nprobe': nprobe}

        metric = (faiss.METRIC_INNER_PRODUCT if self.metric_type == "INNER_PRODUCT"
                  else faiss.METRIC_L2)
        quantizer = (faiss.IndexFlatIP(self.embedding_dim) if self.metric_type == "INNER_PRODUCT"
                     else faiss.IndexFlatL2(self.embedding_dim))
        index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist, metric)

        logger.info(f"  Training IVF with {nlist} clusters...")
        index.train(embeddings)
        index.add(embeddings)
        index.nprobe = nprobe

        logger.info(f"  IVF: nlist={nlist}, nprobe={nprobe}")
        return index

    def add_embeddings(self, embeddings: np.ndarray, study_ids: List[str]):
        """Add embeddings to existing index incrementally."""
        if self.index is None:
            raise ValueError("Index not initialized. Call build_index() first.")
        if embeddings.shape[0] != len(study_ids):
            raise ValueError(f"Count mismatch: {embeddings.shape[0]} vs {len(study_ids)}")

        embeddings = embeddings.astype(np.float32)
        if self.metric_type == "INNER_PRODUCT":
            faiss.normalize_L2(embeddings)

        self.index.add(embeddings)
        self.study_ids.extend(study_ids)
        logger.info(f"Added {len(study_ids)} embeddings (total: {len(self.study_ids)})")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        return_distances: bool = True,
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """Search for top_k nearest neighbors. Returns (distances, study_ids) or just study_ids."""
        if self.index is None:
            raise ValueError("Index not initialized")

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding.astype(np.float32)
        if self.metric_type == "INNER_PRODUCT":
            faiss.normalize_L2(query_embedding)

        distances, indices = self.index.search(query_embedding, top_k)
        study_ids = np.array([self.study_ids[idx] for idx in indices[0]])

        if return_distances:
            return distances[0], study_ids
        return study_ids

    def batch_search(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 5,
    ) -> Tuple[np.ndarray, List[List[str]]]:
        """Batch search. Returns (distances, list_of_study_id_lists)."""
        if self.index is None:
            raise ValueError("Index not initialized")

        query_embeddings = query_embeddings.astype(np.float32)
        if self.metric_type == "INNER_PRODUCT":
            faiss.normalize_L2(query_embeddings)

        distances, indices = self.index.search(query_embeddings, top_k)
        study_ids_batch = [
            [self.study_ids[idx] for idx in row] for row in indices
        ]
        return distances, study_ids_batch

    def save(self, save_dir: Optional[str] = None):
        """Save FAISS index and study ID mapping."""
        if self.index is None:
            raise ValueError("Index not initialized")
        if save_dir is None:
            save_dir = self.config.paths.INDEX_DIR

        os.makedirs(save_dir, exist_ok=True)
        index_path = os.path.join(save_dir, f"faiss_{self.index_type.lower()}.index")
        faiss.write_index(self.index, index_path)

        mapping_path = os.path.join(save_dir, "study_id_mapping.pkl")
        with open(mapping_path, 'wb') as f:
            pickle.dump({
                'study_ids': self.study_ids,
                'embedding_dim': self.embedding_dim,
                'index_type': self.index_type,
                'metric_type': self.metric_type,
                'hnsw_params': self.hnsw_params,
                'ivf_params': self.ivf_params,
            }, f)

        logger.info(f"Saved FAISS index to {save_dir}")

    def load(self, load_dir: Optional[str] = None):
        """Load FAISS index and study ID mapping."""
        if load_dir is None:
            load_dir = self.config.paths.INDEX_DIR

        mapping_path = os.path.join(load_dir, "study_id_mapping.pkl")
        with open(mapping_path, 'rb') as f:
            data = pickle.load(f)

        self.study_ids = data['study_ids']
        self.embedding_dim = data['embedding_dim']
        self.index_type = data['index_type']
        self.metric_type = data['metric_type']
        self.hnsw_params = data.get('hnsw_params', {})
        self.ivf_params = data.get('ivf_params', {})

        index_path = os.path.join(load_dir, f"faiss_{self.index_type.lower()}.index")
        self.index = faiss.read_index(index_path)

        if self.index_type == "HNSW":
            self.index.hnsw.efSearch = self.config.retrieval.HNSW_EF_SEARCH
        if self.index_type == "IVFFlat":
            self.index.nprobe = self.config.retrieval.NPROBE

        logger.info(f"Loaded FAISS index from {load_dir}: "
                     f"type={self.index_type}, n={len(self.study_ids)}")

    def get_stats(self) -> dict:
        if self.index is None:
            return {"initialized": False}
        stats = {
            "initialized": True,
            "index_type": self.index_type,
            "metric_type": self.metric_type,
            "embedding_dim": self.embedding_dim,
            "total_embeddings": len(self.study_ids),
            "ntotal": self.index.ntotal,
        }
        if self.index_type == "HNSW" and self.hnsw_params:
            stats["hnsw_M"] = self.hnsw_params.get('M')
            stats["hnsw_efSearch"] = self.hnsw_params.get('efSearch')
        elif self.index_type == "IVFFlat" and self.ivf_params:
            stats["ivf_nlist"] = self.ivf_params.get('nlist')
            stats["ivf_nprobe"] = self.ivf_params.get('nprobe')
        return stats

    def __len__(self):
        return len(self.study_ids)

    def __repr__(self):
        if self.index is None:
            return "FaissIndexer(uninitialized)"
        return f"FaissIndexer(type={self.index_type}, n={len(self.study_ids)})"