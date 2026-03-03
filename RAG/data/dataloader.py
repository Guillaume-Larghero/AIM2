"""MIMIC-CXR data loader for RAG pipeline."""

import logging
from typing import Tuple, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class MIMICDataLoader:
    """Load and prepare MIMIC-CXR data for the RAG pipeline."""

    def __init__(self, config):
        self.config = config
        self.chexpert_cols = config.data.CHEXPERT_LABELS

    def load_csv(self, csv_path: Optional[str] = None) -> pd.DataFrame:
        """Load MIMIC-CXR CSV."""
        if csv_path is None:
            csv_path = self.config.paths.DATA_CSV
        logger.info(f"Loading data from: {csv_path}")
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} samples")
        logger.info(f"Splits: {df['split'].value_counts().to_dict()}")
        return df

    def prepare_dataframe(
        self,
        df: pd.DataFrame,
        combine_sections: bool = True,
        include_indication: bool = False,
    ) -> pd.DataFrame:
        """Prepare dataframe: combine text sections, standardize splits."""
        df = df.copy()
        df['study_id'] = df['study_id'].astype(str)

        df['indication'] = df['indication'].fillna('')
        df['findings'] = df['findings'].fillna('')
        df['impression'] = df['impression'].fillna('')

        if combine_sections:
            separator = self.config.data.SECTION_SEPARATOR
            text_parts = []
            if include_indication:
                text_parts.append(df['indication'])
            text_parts.append(df['findings'])
            text_parts.append(df['impression'])

            df['text'] = text_parts[0].astype(str)
            for part in text_parts[1:]:
                df['text'] = df['text'] + separator + part.astype(str)

            df['text'] = df['text'].str.replace(
                f'{separator}{separator}+', separator, regex=True
            ).str.strip()

            empty_mask = df['text'].str.strip() == ''
            df.loc[empty_mask, 'text'] = self.config.data.TEXT_FALLBACK

        # CheXpert labels: NaN values are preserved (not filled to 0.0)
        # NaN = "not mentioned", 0.0 = "explicitly negative" — different semantics

        df['split'] = df['split'].replace({'validate': 'val'})

        logger.info("Dataframe prepared")
        logger.info(f"  Non-empty text: {(df['text'] != self.config.data.TEXT_FALLBACK).sum()}")
        return df

    def get_train_val_test_splits(
        self, df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split dataframe by train/val/test."""
        train_df = df[df['split'] == 'train'].reset_index(drop=True)
        val_df = df[df['split'] == 'val'].reset_index(drop=True)
        test_df = df[df['split'] == 'test'].reset_index(drop=True)
        logger.info(f"Splits — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")
        return train_df, val_df, test_df

    def get_statistics(self, df: pd.DataFrame) -> dict:
        """Compute dataset statistics."""
        stats = {
            'total_samples': len(df),
            'splits': df['split'].value_counts().to_dict(),
            'has_indication': df['has_indication'].sum(),
            'has_findings': df['has_findings'].sum(),
            'has_impression': df['has_impression'].sum(),
        }
        if 'ViewPosition' in df.columns:
            stats['view_positions'] = df['ViewPosition'].value_counts().to_dict()

        chexpert_stats = {}
        for col in self.chexpert_cols:
            if col in df.columns:
                chexpert_stats[col] = int((df[col] >= 1.0).sum())
        stats['chexpert_positive_counts'] = chexpert_stats
        return stats

    def print_statistics(self, df: pd.DataFrame):
        """Print formatted dataset statistics."""
        stats = self.get_statistics(df)

        print("\n" + "=" * 80)
        print("DATASET STATISTICS")
        print("=" * 80)
        print(f"\nTotal samples: {stats['total_samples']}")

        print("\nSplit distribution:")
        for split, count in stats['splits'].items():
            print(f"  {split}: {count}")

        print("\nText sections:")
        print(f"  Has indication: {stats['has_indication']}")
        print(f"  Has findings: {stats['has_findings']}")
        print(f"  Has impression: {stats['has_impression']}")

        if stats.get('view_positions'):
            print("\nView positions:")
            for view, count in stats['view_positions'].items():
                print(f"  {view}: {count}")

        print("\nCheXpert positive findings (top 10):")
        for label, count in sorted(
            stats['chexpert_positive_counts'].items(),
            key=lambda x: x[1], reverse=True,
        )[:10]:
            print(f"  {label}: {count}")
        print("=" * 80)