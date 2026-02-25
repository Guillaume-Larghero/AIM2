"""RAG retrieval pipeline for inference."""

import logging
from typing import List, Dict, Optional, Union

from PIL import Image

logger = logging.getLogger(__name__)


class RAGRetriever:
    """Query with image or text and retrieve similar reports/images."""

    def __init__(self, embedder, metadata_db, indexer, tokenizer=None):
        self.embedder = embedder
        self.metadata_db = metadata_db
        self.indexer = indexer
        self.tokenizer = tokenizer
        logger.info("RAGRetriever initialized")

    def retrieve_by_image(
        self,
        image: Union[str, Image.Image],
        top_k: int = 5,
        min_similarity: Optional[float] = None,
        return_full_metadata: bool = True,
    ) -> List[Dict]:
        """Retrieve similar reports given an image (path or PIL Image)."""
        if isinstance(image, str):
            query_emb = self.embedder.encode_image_from_path(image, use_cache=False)
        else:
            query_emb = self.embedder.encode_image(image)

        distances, study_ids = self.indexer.search_by_image(
            query_emb, top_k=top_k, return_distances=True,
        )

        results = []
        for dist, sid in zip(distances, study_ids):
            if min_similarity is not None and dist < min_similarity:
                continue
            if return_full_metadata:
                metadata = self.metadata_db.get_study(sid)
                if metadata:
                    metadata['similarity'] = float(dist)
                    results.append(metadata)
            else:
                results.append({
                    'study_id': sid,
                    'report': self.metadata_db.get_report_text(sid),
                    'similarity': float(dist),
                })
        return results

    def retrieve_by_text(
        self,
        text: str,
        top_k: int = 5,
        min_similarity: Optional[float] = None,
        return_full_metadata: bool = True,
    ) -> List[Dict]:
        """Retrieve similar images given a text query."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer required for text retrieval")

        query_emb = self.embedder.encode_text_from_string(
            text, self.tokenizer,
            max_length=self.embedder.config.embedder.TEXT_MAX_LENGTH,
        )

        distances, study_ids = self.indexer.search_by_text(
            query_emb, top_k=top_k, return_distances=True,
        )

        results = []
        for dist, sid in zip(distances, study_ids):
            if min_similarity is not None and dist < min_similarity:
                continue
            if return_full_metadata:
                metadata = self.metadata_db.get_study(sid)
                if metadata:
                    metadata['similarity'] = float(dist)
                    results.append(metadata)
            else:
                study = self.metadata_db.get_study(sid)
                results.append({
                    'study_id': sid,
                    'image_path': study['image_path'] if study else '',
                    'similarity': float(dist),
                })
        return results

    def format_retrieved_reports(
        self,
        results: List[Dict],
        include_findings: bool = True,
        include_impression: bool = True,
        max_reports: Optional[int] = None,
    ) -> str:
        """Format retrieved reports for an LLM prompt."""
        if max_reports:
            results = results[:max_reports]

        formatted = []
        for i, result in enumerate(results, 1):
            sections = []
            if include_findings and result.get('findings'):
                sections.append(f"Findings: {result['findings']}")
            if include_impression and result.get('impression'):
                sections.append(f"Impression: {result['impression']}")
            report_text = "\n".join(sections)
            formatted.append(
                f"Report {i} (similarity: {result.get('similarity', 0):.3f}):\n{report_text}"
            )
        return "\n\n".join(formatted)

    def __repr__(self):
        return f"RAGRetriever(metadata={len(self.metadata_db)}, indexer={self.indexer})"