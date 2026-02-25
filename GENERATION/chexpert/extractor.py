"""CheXpert label extraction from generated radiology reports using rule-based NLP."""

import re
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

CHEXPERT_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
    'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
    'Pneumonia', 'Pneumothorax', 'Support Devices'
]

POSITIVE_PATTERNS = {
    'Atelectasis': [
        r'\batelecta\w*\b', r'\bcollapse[d]?\b.*\b(lung|lobe)\b',
        r'\b(lung|lobe)\b.*\bcollapse[d]?\b', r'\bvolume loss\b',
    ],
    'Cardiomegaly': [
        r'\bcardiomegal\w*\b', r'\benlarged\s+(heart|cardiac)\b',
        r'\bheart\s+(is\s+)?enlarged\b', r'\bcardiac\s+enlargement\b',
        r'\bheart\s+size\s+(is\s+)?(mildly\s+|moderately\s+|severely\s+)?increased\b',
        r'\bincreased\s+(heart|cardiac)\s+size\b',
    ],
    'Consolidation': [
        r'\bconsolidat\w*\b', r'\bairspace\s+(disease|opacity)\b',
        r'\bair\s*space\s+opacit\w*\b', r'\blobar\s+opaci\w*\b',
    ],
    'Edema': [
        r'\bedema\b', r'\bpulmonary\s+(vascular\s+)?congestion\b',
        r'\bvascular\s+congestion\b', r'\binterstitial\s+edema\b',
        r'\balveolar\s+edema\b', r'\bfluid\s+overload\b', r'\bcephalization\b',
    ],
    'Enlarged Cardiomediastinum': [
        r'\bwidened\s+mediastinum\b', r'\bmediastinal\s+widening\b',
        r'\benlarged\s+mediastinum\b', r'\bmediastinal\s+enlargement\b',
    ],
    'Fracture': [
        r'\bfracture[ds]?\b', r'\brib\s+fracture\b', r'\bbroken\s+(rib|bone)\b',
    ],
    'Lung Lesion': [
        r'\b(lung|pulmonary)\s+(mass|lesion|nodule)\b',
        r'\b(mass|lesion|nodule)\s+(in|of|within)\s+(the\s+)?(lung|pulmonary)\b',
        r'\bpulmonary\s+nodule\b', r'\blung\s+nodule\b',
    ],
    'Lung Opacity': [
        r'\bopacit\w+\b', r'\bopaque\b', r'\bhaziness\b', r'\bhazy\b',
        r'\binfiltrat\w*\b', r'\bairspace\s+disease\b',
    ],
    'No Finding': [
        r'\bno\s+(acute\s+)?(cardiopulmonary\s+)?(abnormalit|finding|disease)\w*\b',
        r'\bnormal\s+(chest|study|exam|radiograph)\b', r'\bunremarkable\b',
        r'\bclear\s+lungs?\b', r'\blungs?\s+(are\s+)?clear\b',
        r'\bno\s+acute\s+process\b', r'\bno\s+significant\s+abnormality\b',
    ],
    'Pleural Effusion': [
        r'\bpleural\s+effusion\w*\b', r'\beffusion\w*\b', r'\bpleural\s+fluid\b',
        r'\bhydrothorax\b', r'\bblunting\s+of\s+(the\s+)?(costophrenic|cp)\b',
    ],
    'Pleural Other': [
        r'\bpleural\s+thickening\b', r'\bpleural\s+plaque\w*\b',
        r'\bpleural\s+abnormalit\w*\b', r'\bpleural\s+calcification\b',
    ],
    'Pneumonia': [
        r'\bpneumonia\b', r'\bpneumonitis\b', r'\binfection\b',
        r'\binfectious\s+process\b', r'\bbronchopneumonia\b', r'\baspiration\b',
    ],
    'Pneumothorax': [
        r'\bpneumothorax\b', r'\bptx\b', r'\bcollapsed\s+lung\b',
        r'\bair\s+in\s+(the\s+)?pleural\b',
    ],
    'Support Devices': [
        r'\b(et|endotracheal)\s+tube\b', r'\bett\b', r'\bng\s+tube\b',
        r'\bnasogastric\b', r'\bcentral\s+(venous\s+)?(line|catheter)\b',
        r'\bpicc\b', r'\bpacemaker\b', r'\bicd\b', r'\baicd\b',
        r'\bchest\s+tube\b', r'\btracheo(stomy)?\b', r'\bstent\b',
        r'\bcatheter\b', r'\bport\b', r'\bdrain\b',
    ],
}

NEGATION_PATTERNS = [
    r'\bno\s+(evidence\s+of\s+)?', r'\bwithout\s+(evidence\s+of\s+)?',
    r'\bnegative\s+for\b', r'\babsent\b', r'\brule[ds]?\s+out\b',
    r'\bnot\s+(seen|identified|present|demonstrated|evident)\b',
    r'\bfree\s+of\b', r'\bno\s+definite\b', r'\bcleared?\b',
    r'\bresolved?\b', r'\bimproved?\b',
]

UNCERTAINTY_PATTERNS = [
    r'\bpossible\b', r'\bprobable\b', r'\bsuspect\w*\b',
    r'\bmay\s+represent\b', r'\bcannot\s+(be\s+)?(excluded|ruled\s+out)\b',
    r'\buncertain\b', r'\bcould\s+(be|represent)\b',
    r'\bsuggestive\s+of\b',
]


@dataclass
class LabelExtractionResult:
    """Result of CheXpert label extraction."""
    labels: List[float]
    label_names: List[str]
    evidence: Dict[str, List[str]]


class CheXpertLabelExtractor:
    """Rule-based CheXpert label extractor from radiology report text."""

    def __init__(self, label_names: Optional[List[str]] = None):
        self.label_names = label_names or CHEXPERT_LABELS
        self._compiled_positive = {
            label: [re.compile(p, re.IGNORECASE) for p in patterns]
            for label, patterns in POSITIVE_PATTERNS.items()
        }
        self._compiled_negation = [re.compile(p, re.IGNORECASE) for p in NEGATION_PATTERNS]
        self._compiled_uncertainty = [re.compile(p, re.IGNORECASE) for p in UNCERTAINTY_PATTERNS]

    def extract_labels(self, report_text: str) -> LabelExtractionResult:
        if not report_text or not report_text.strip():
            return LabelExtractionResult(
                labels=[0.0] * len(self.label_names),
                label_names=self.label_names,
                evidence={}
            )

        text = report_text.lower()
        sentences = self._split_sentences(text)
        labels = []
        evidence = {}

        for label_name in self.label_names:
            value, ev = self._extract_single_label(label_name, sentences)
            labels.append(value)
            if ev:
                evidence[label_name] = ev

        # If positive findings exist, override "No Finding"
        no_finding_idx = self.label_names.index('No Finding') if 'No Finding' in self.label_names else -1
        if no_finding_idx >= 0 and labels[no_finding_idx] == 1.0:
            other_positive = any(
                labels[i] == 1.0
                for i in range(len(labels))
                if i != no_finding_idx and self.label_names[i] != 'Support Devices'
            )
            if other_positive:
                labels[no_finding_idx] = 0.0

        return LabelExtractionResult(labels=labels, label_names=self.label_names, evidence=evidence)

    def _extract_single_label(self, label_name: str, sentences: List[str]) -> Tuple[float, List[str]]:
        if label_name not in self._compiled_positive:
            return 0.0, []

        for sentence in sentences:
            for pattern in self._compiled_positive[label_name]:
                match = pattern.search(sentence)
                if match:
                    if self._is_negated(sentence, match.start()):
                        continue
                    if self._is_uncertain(sentence):
                        return -1.0, [f"[uncertain] {sentence.strip()}"]
                    return 1.0, [sentence.strip()]
        return 0.0, []

    def _is_negated(self, sentence: str, match_pos: int) -> bool:
        prefix = sentence[:match_pos]
        for pattern in self._compiled_negation:
            neg_match = pattern.search(prefix)
            if neg_match and (match_pos - neg_match.end()) < 30:
                return True
        return False

    def _is_uncertain(self, sentence: str) -> bool:
        return any(p.search(sentence) for p in self._compiled_uncertainty)

    def _split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'[.!?]\s+', text)
        expanded = []
        for s in sentences:
            expanded.extend(re.split(r'\n+', s))
        return [s.strip() for s in expanded if s.strip()]

    def extract_labels_batch(self, report_texts: List[str]) -> List[LabelExtractionResult]:
        return [self.extract_labels(text) for text in report_texts]


def extract_chexpert_from_reports(
    generated_reports: List[Dict[str, str]],
    label_names: Optional[List[str]] = None,
    show_progress: bool = False
) -> List[List[float]]:
    """Extract CheXpert labels from a list of report dicts with 'findings' and 'impression'."""
    extractor = CheXpertLabelExtractor(label_names)
    combined_texts = []
    for report in generated_reports:
        parts = []
        if report.get('findings'):
            parts.append(report['findings'])
        if report.get('impression'):
            parts.append(report['impression'])
        combined_texts.append(' '.join(parts).strip())

    results = extractor.extract_labels_batch(combined_texts)
    return [r.labels for r in results]