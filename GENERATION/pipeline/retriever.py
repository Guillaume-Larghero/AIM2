"""Retriever for the generation pipeline: image-to-report using CLIP embeddings and FAISS."""

import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

CHEXPERT_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
    'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
    'Pneumonia', 'Pneumothorax', 'Support Devices'
]


@dataclass
class RetrievalResult:
    """A single retrieval result."""
    study_id: str
    similarity_score: float
    findings: str
    impression: str
    indication: str = ""
    chexpert_labels: Optional[List[float]] = None
    positive_findings: Optional[List[str]] = None
    image_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'study_id': self.study_id,
            'similarity_score': self.similarity_score,
            'findings': self.findings,
            'impression': self.impression,
            'indication': self.indication,
            'chexpert_labels': self.chexpert_labels,
            'positive_findings': self.positive_findings,
            'image_path': self.image_path
        }


class RAGRetriever:
    """Retrieval component using CLIP embeddings and FAISS indices."""

    def __init__(self, config, clip_embedder, faiss_indexer, metadata_db):
        self.config = config
        self.clip_embedder = clip_embedder
        self.indexer = faiss_indexer
        self.metadata_db = metadata_db

        retrieval_config = getattr(config, 'retrieval', None)
        if retrieval_config:
            self.top_k = retrieval_config.TOP_K
            self.min_similarity = retrieval_config.MIN_SIMILARITY
            self.use_chexpert_filtering = retrieval_config.USE_CHEXPERT_FILTERING
            self.chexpert_match_threshold = retrieval_config.CHEXPERT_MATCH_THRESHOLD
            self.metric_type = getattr(retrieval_config, 'METRIC_TYPE', 'INNER_PRODUCT')
        else:
            self.top_k = 5
            self.min_similarity = 0.0
            self.use_chexpert_filtering = False
            self.chexpert_match_threshold = 2
            self.metric_type = "INNER_PRODUCT"

        self.chexpert_labels = getattr(
            getattr(config, 'data', None), 'CHEXPERT_LABELS', CHEXPERT_LABELS
        )
        logger.info("RAGRetriever initialized")

    def retrieve_by_image_path(
        self, image_path: str, top_k: Optional[int] = None, return_embeddings: bool = False
    ):
        query_embedding = self.clip_embedder.encode_image_from_path(image_path)
        results = self.retrieve_by_embedding(query_embedding, top_k=top_k)
        if return_embeddings:
            return results, query_embedding
        return results

    def retrieve_by_embedding(
        self,
        query_embedding: np.ndarray,
        top_k: Optional[int] = None,
        query_chexpert_labels: Optional[List[float]] = None
    ) -> List[RetrievalResult]:
        if top_k is None:
            top_k = self.top_k

        fetch_k = top_k * 2 if self.use_chexpert_filtering else top_k
        distances, study_ids = self.indexer.search(
            query_embedding, top_k=fetch_k, return_distances=True
        )

        if self.metric_type == "INNER_PRODUCT":
            similarity_scores = distances
        else:
            similarity_scores = 1.0 / (1.0 + distances)

        results = []
        for study_id, sim_score in zip(study_ids, similarity_scores):
            if sim_score < self.min_similarity:
                continue
            metadata = self.metadata_db.get_study(study_id)
            if metadata is None:
                continue
            positive_findings = self.metadata_db.get_positive_findings(study_id)
            results.append(RetrievalResult(
                study_id=study_id,
                similarity_score=float(sim_score),
                findings=metadata['findings'],
                impression=metadata['impression'],
                indication=metadata.get('indication', ''),
                chexpert_labels=metadata['chexpert_labels'],
                positive_findings=positive_findings,
                image_path=metadata.get('image_path', '')
            ))

        if self.use_chexpert_filtering and query_chexpert_labels:
            results = self._filter_by_chexpert(results, query_chexpert_labels)

        return results[:top_k]

    def _filter_by_chexpert(self, results: List[RetrievalResult], query_labels: List[float]):
        query_positive = {i for i, val in enumerate(query_labels) if val == 1.0}
        if not query_positive:
            return results
        filtered = []
        for result in results:
            if result.chexpert_labels is None:
                continue
            result_positive = {i for i, val in enumerate(result.chexpert_labels) if val == 1.0}
            if len(query_positive & result_positive) >= self.chexpert_match_threshold:
                filtered.append(result)
        return filtered

    def __repr__(self):
        return f"RAGRetriever(index={len(self.indexer)}, db={len(self.metadata_db)}, top_k={self.top_k})"