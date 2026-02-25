"""Metadata database: stores report text, CheXpert labels, and study metadata.

CheXpert label encoding:
  1.0  = positive (finding present)
  0.0  = negative (explicitly absent)
 -1.0  = uncertain
  None = blank (not mentioned; NaN in original data)

None and 0.0 have different clinical meanings and must be preserved.
"""

import json
import os
import logging
from typing import Dict, List, Optional, Any
from collections import defaultdict

import pandas as pd

logger = logging.getLogger(__name__)


class MetadataDB:
    """Metadata database for storing and retrieving study information."""

    def __init__(self, config):
        self.config = config
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.db_path = os.path.join(config.paths.INDEX_DIR, "metadata_db.json")
        logger.info("Initialized MetadataDB")

    def add_study(
        self,
        study_id: str,
        findings: str = "",
        impression: str = "",
        indication: str = "",
        chexpert_labels: Optional[List[float]] = None,
        image_path: str = "",
        split: str = "train",
        **kwargs,
    ):
        """Add a study to the database."""
        if not findings and not impression:
            findings = self.config.data.TEXT_FALLBACK

        self.metadata[study_id] = {
            'study_id': study_id,
            'findings': findings,
            'impression': impression,
            'indication': indication,
            'chexpert_labels': chexpert_labels if chexpert_labels is not None else [None] * 14,
            'image_path': image_path,
            'split': split,
            **kwargs,
        }

    def add_from_dataframe(self, df: pd.DataFrame):
        """Bulk add studies from a DataFrame. NaN labels are preserved as None."""
        chexpert_cols = self.config.data.CHEXPERT_LABELS
        logger.info(f"Adding {len(df)} studies to metadata database...")

        for _, row in df.iterrows():
            study_id = str(row['study_id'])
            findings = row.get('findings', '') or ''
            impression = row.get('impression', '') or ''
            indication = row.get('indication', '') or ''

            chexpert_labels = []
            for col in chexpert_cols:
                value = row.get(col, None)
                chexpert_labels.append(None if pd.isna(value) else float(value))

            self.add_study(
                study_id=study_id,
                findings=findings,
                impression=impression,
                indication=indication,
                chexpert_labels=chexpert_labels,
                image_path=row.get('image_path', ''),
                split=row.get('split', 'train'),
            )

        logger.info(f"Added {len(df)} studies (NaN labels preserved)")

    def get_study(self, study_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve study metadata by ID."""
        return self.metadata.get(study_id)

    def get_report_text(
        self,
        study_id: str,
        combine_sections: bool = True,
        include_indication: bool = False,
    ) -> str:
        """Get report text for a study."""
        study = self.get_study(study_id)
        if study is None:
            return ""

        sections = []
        if include_indication and study['indication']:
            sections.append(study['indication'])
        if study['findings']:
            sections.append(study['findings'])
        if study['impression']:
            sections.append(study['impression'])

        if not sections:
            return self.config.data.TEXT_FALLBACK
        if combine_sections:
            return self.config.data.SECTION_SEPARATOR.join(sections)
        return {
            'findings': study['findings'],
            'impression': study['impression'],
            'indication': study['indication'],
        }

    def get_findings(self, study_id: str) -> str:
        study = self.get_study(study_id)
        return study['findings'] if study else ""

    def get_impression(self, study_id: str) -> str:
        study = self.get_study(study_id)
        return study['impression'] if study else ""

    def get_chexpert_labels(self, study_id: str) -> Optional[List[float]]:
        study = self.get_study(study_id)
        return study['chexpert_labels'] if study else None

    def get_chexpert_labels_as_dict(self, study_id: str) -> Optional[Dict[str, float]]:
        """Get CheXpert labels as {label_name: value}."""
        study = self.get_study(study_id)
        if study is None:
            return None
        return dict(zip(self.config.data.CHEXPERT_LABELS, study['chexpert_labels']))

    def get_positive_findings(self, study_id: str) -> List[str]:
        """Get names of positive CheXpert findings (value == 1.0)."""
        labels_dict = self.get_chexpert_labels_as_dict(study_id)
        if labels_dict is None:
            return []
        return [name for name, value in labels_dict.items() if value == 1.0]

    def filter_by_split(self, split: str) -> List[str]:
        """Get all study IDs for a given split."""
        return [
            sid for sid, meta in self.metadata.items()
            if meta['split'] == split
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics. Only counts positive findings (1.0)."""
        if not self.metadata:
            return {"total_studies": 0}

        split_counts = defaultdict(int)
        for meta in self.metadata.values():
            split_counts[meta['split']] += 1

        finding_counts = defaultdict(int)
        label_names = self.config.data.CHEXPERT_LABELS
        for meta in self.metadata.values():
            for i, val in enumerate(meta['chexpert_labels']):
                if val == 1.0:
                    finding_counts[label_names[i]] += 1

        has_findings = sum(1 for m in self.metadata.values() if m['findings'])
        has_impression = sum(1 for m in self.metadata.values() if m['impression'])

        return {
            "total_studies": len(self.metadata),
            "split_counts": dict(split_counts),
            "has_findings": has_findings,
            "has_impression": has_impression,
            "finding_counts": dict(sorted(finding_counts.items(),
                                          key=lambda x: x[1], reverse=True)),
        }

    def save(self, path: Optional[str] = None):
        if path is None:
            path = self.db_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        logger.info(f"Saved metadata database ({len(self.metadata)} studies) to {path}")

    def load(self, path: Optional[str] = None):
        if path is None:
            path = self.db_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"Metadata database not found: {path}")
        with open(path, 'r') as f:
            self.metadata = json.load(f)
        logger.info(f"Loaded metadata database ({len(self.metadata)} studies) from {path}")

    def __len__(self):
        return len(self.metadata)

    def __contains__(self, study_id: str):
        return study_id in self.metadata

    def __repr__(self):
        return f"MetadataDB(studies={len(self.metadata)})"