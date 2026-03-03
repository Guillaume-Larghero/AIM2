"""Prompt templates for RAG-based report generation."""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

CHEXPERT_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
    'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
    'Pneumonia', 'Pneumothorax', 'Support Devices'
]

SYSTEM_PROMPT = """You are an expert radiologist specializing in chest X-ray interpretation.

Your task is to generate a complete, concise radiology report with FINDINGS and IMPRESSION sections based on similar cases provided as reference.

Guidelines:
- Generate a SHORTER, focused report with only the most likely and relevant findings
- Use standard radiological terminology
- Be precise and avoid speculation
- Maintain professional medical language
- Base your report on the similar cases provided, but adapt appropriately to the specific image

Format:
FINDINGS: [Describe the observed anatomical structures and any abnormalities]

IMPRESSION: [Provide a concise clinical summary and interpretation]"""


class PromptBuilder:
    """Builds prompts for complete report generation with structured similar cases."""

    def __init__(self, config):
        if hasattr(config, 'data'):
            self.chexpert_labels = config.data.CHEXPERT_LABELS
        else:
            self.chexpert_labels = CHEXPERT_LABELS

    def build_complete_report_prompt(
        self,
        retrieval_results: List[Any],
        query_chexpert_labels: Optional[List[float]] = None,
        query_positive_findings: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        user = self._build_user_prompt(
            retrieval_results, query_chexpert_labels, query_positive_findings
        )
        return {"system": SYSTEM_PROMPT, "user": user}

    def _build_user_prompt(
        self,
        retrieval_results: List[Any],
        query_chexpert_labels: Optional[List[float]] = None,
        query_positive_findings: Optional[List[str]] = None,
    ) -> str:
        parts = []

        if query_positive_findings:
            parts.append(f"Query Image Detected Findings: {', '.join(query_positive_findings)}\n")
        else:
            parts.append("Query Image: No specific abnormalities detected by automated labeler.\n")

        if retrieval_results:
            parts.append(f"The {len(retrieval_results)} most similar chest X-ray cases:\n")
            for i, result in enumerate(retrieval_results, 1):
                case = [f"\n{'='*60}", f"Case {i} (Similarity: {result.similarity_score:.3f})", f"{'='*60}"]
                if result.chexpert_labels:
                    case.append("\nLabels:")
                    case.append(self._format_chexpert_labels(result.chexpert_labels))
                if result.indication and result.indication.strip():
                    case.append(f"\nIndication: {result.indication.strip()}")
                if result.findings and result.findings.strip():
                    case.append(f"\nFindings: {result.findings.strip()}")
                if result.impression and result.impression.strip():
                    case.append(f"\nImpression: {result.impression.strip()}")
                parts.append('\n'.join(case))

        parts.append(f"\n{'='*60}")
        parts.append("\nBased on these similar cases, generate a complete radiology report.")
        parts.append("Keep it CONCISE. Provide both FINDINGS and IMPRESSION sections.")
        return '\n'.join(parts)

    def _format_chexpert_labels(self, chexpert_labels: Optional[List[float]]) -> str:
        if not chexpert_labels or len(chexpert_labels) != len(self.chexpert_labels):
            return "  Labels: Not available"
        positive = [n for v, n in zip(chexpert_labels, self.chexpert_labels) if v == 1.0]
        negative = [n for v, n in zip(chexpert_labels, self.chexpert_labels) if v == 0.0]
        lines = []
        if positive:
            lines.append(f"  Has: {', '.join(positive)}")
        if negative:
            lines.append(f"  Doesn't have: {', '.join(negative)}")
        return '\n'.join(lines) if lines else "  Labels: All uncertain or unknown"