"""Utility functions for the GENERATION pipeline."""

import logging
from typing import Optional

import pandas as pd

from RAG.data.dataloader import MIMICDataLoader
from RAG.config.config import RAGConfig

logger = logging.getLogger(__name__)


def load_test_data(config, n_samples: Optional[int] = None) -> pd.DataFrame:
    """Load and filter test data for evaluation.

    Args:
        config: GenerationPipelineConfig object.
        n_samples: Subsample size (None = all).

    Returns:
        Filtered DataFrame with test samples.
    """
    rag_config = RAGConfig()
    rag_config.paths.DATA_CSV = config.paths.DATA_CSV
    rag_config.paths.BASE_DIR = config.paths.BASE_DIR
    if hasattr(config, 'system'):
        rag_config.system.SEED = config.system.SEED

    loader = MIMICDataLoader(rag_config)
    df = loader.load_csv()
    df = loader.prepare_dataframe(df)

    eval_split = config.data.EVAL_SPLIT if hasattr(config, 'data') else 'test'
    test_df = df[df['split'] == eval_split].copy()

    if config.data.REQUIRE_FINDINGS:
        test_df = test_df[test_df['has_findings'] == True]
    if config.data.REQUIRE_IMPRESSION:
        test_df = test_df[test_df['has_impression'] == True]

    seed = config.system.SEED if hasattr(config, 'system') else 42
    if n_samples is not None and n_samples < len(test_df):
        test_df = test_df.sample(n=n_samples, random_state=seed)

    test_df = test_df.reset_index(drop=True)
    logger.info(f"Loaded {len(test_df)} test samples")
    return test_df