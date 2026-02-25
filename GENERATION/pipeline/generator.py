"""Report generator using RAG pipeline: retrieves similar cases and generates via LLM."""

import logging
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

from GENERATION.pipeline.retriever import RAGRetriever
from GENERATION.llm.wrapper import LLMWrapper
from GENERATION.llm.prompts import PromptBuilder, CHEXPERT_LABELS

logger = logging.getLogger(__name__)


@dataclass
class GeneratedReport:
    """Container for a generated report."""
    study_id: str
    query_image_path: str
    findings: str
    impression: str
    retrieved_study_ids: List[str]
    retrieval_scores: List[float]
    num_retrieved: int
    gt_findings: Optional[str] = None
    gt_impression: Optional[str] = None
    gt_chexpert_labels: Optional[List[float]] = None
    query_chexpert_labels: Optional[List[float]] = None
    query_positive_findings: Optional[List[str]] = None
    generation_time: float = 0.0
    model_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ReportGenerator:
    """Generates complete radiology reports (findings + impression) using RAG."""

    def __init__(self, config, retriever: RAGRetriever,
                 llm_wrapper: Optional[LLMWrapper] = None,
                 prompt_builder: Optional[PromptBuilder] = None):
        self.config = config
        self.retriever = retriever
        self.llm = llm_wrapper or LLMWrapper(config)
        self.prompt_builder = prompt_builder or PromptBuilder(config)

        gen_config = getattr(config, 'generation', None)
        self.llm_model = gen_config.LLM_MODEL if gen_config else "gpt-4o-mini"
        self.llm_provider = gen_config.LLM_PROVIDER if gen_config else "openai"
        self.chexpert_labels = getattr(
            getattr(config, 'data', None), 'CHEXPERT_LABELS', CHEXPERT_LABELS
        )
        logger.info(f"ReportGenerator initialized: {self.llm_provider}/{self.llm_model}")

    def generate_report(
        self,
        image_path: str,
        study_id: Optional[str] = None,
        ground_truth_findings: Optional[str] = None,
        ground_truth_impression: Optional[str] = None,
        query_chexpert_labels: Optional[List[float]] = None,
        return_detailed_info: bool = False
    ) -> GeneratedReport:
        start_time = time.time()

        retrieval_results = self.retriever.retrieve_by_image_path(image_path)
        if not retrieval_results:
            logger.warning(f"No retrieval results for {study_id}. Using zero-shot generation.")

        query_positive_findings = None
        if query_chexpert_labels:
            query_positive_findings = [
                self.chexpert_labels[i] for i, val in enumerate(query_chexpert_labels) if val == 1.0
            ]

        prompts = self.prompt_builder.build_complete_report_prompt(
            retrieval_results, query_chexpert_labels, query_positive_findings
        )
        full_report = self.llm.generate_complete_report(
            system_prompt=prompts['system'], user_prompt=prompts['user']
        )

        generation_time = time.time() - start_time

        return GeneratedReport(
            study_id=study_id or Path(image_path).stem,
            query_image_path=image_path,
            findings=full_report.get('findings', '').strip(),
            impression=full_report.get('impression', '').strip(),
            retrieved_study_ids=[r.study_id for r in retrieval_results],
            retrieval_scores=[r.similarity_score for r in retrieval_results],
            num_retrieved=len(retrieval_results),
            gt_findings=ground_truth_findings,
            gt_impression=ground_truth_impression,
            gt_chexpert_labels=query_chexpert_labels,
            query_chexpert_labels=query_chexpert_labels,
            query_positive_findings=query_positive_findings,
            generation_time=generation_time,
            model_name=self.llm_model
        )

    def __repr__(self):
        return f"ReportGenerator(retriever={self.retriever}, model={self.llm_model})"