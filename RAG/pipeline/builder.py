"""RAG database builder: orchestrates embedding generation, storage, and index building."""

import logging
from typing import Optional

import pandas as pd
from transformers import AutoTokenizer

from ..embedder.embedder import CLIPEmbedder
from ..metadata.metadata_db import MetadataDB
from ..store.embedding_store import EmbeddingStore
from ..indexing.dual_indexer import DualFaissIndexer

logger = logging.getLogger(__name__)


class RAGDatabaseBuilder:
    """Build complete RAG database with embeddings and FAISS indices."""

    def __init__(self, config):
        self.config = config
        logger.info("Initializing RAG Database Builder...")

        self.embedder = CLIPEmbedder(config)
        self.metadata_db = MetadataDB(config)
        self.embedding_store = EmbeddingStore(config)

        embedding_dim = self.embedder.get_embedding_dim()
        self.indexer = DualFaissIndexer(config, embedding_dim)
        self.tokenizer = None

        logger.info("RAG Database Builder initialized")

    def _load_tokenizer(self):
        if self.tokenizer is None:
            model_name = self.config.embedder.TEXT_ENCODER
            logger.info(f"Loading tokenizer: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        return self.tokenizer

    def build_from_dataframe(
        self,
        df: pd.DataFrame,
        image_path_col: str = 'image_path',
        text_col: str = 'text',
    ):
        """Build complete RAG database from a DataFrame."""
        logger.info(f"Building RAG database from {len(df)} samples")

        # 1. Metadata
        logger.info("[1/5] Building metadata database...")
        self.metadata_db.add_from_dataframe(df)
        self.metadata_db.save()

        # 2. Image embeddings
        logger.info("[2/5] Generating image embeddings...")
        image_embeddings_by_path = self.embedder.generate_embeddings_from_dataframe(
            df, image_path_col=image_path_col, show_progress=True,
        )
        image_embeddings = {}
        for _, row in df.iterrows():
            sid = str(row['study_id'])
            path = row[image_path_col]
            if path in image_embeddings_by_path:
                image_embeddings[sid] = image_embeddings_by_path[path]
        logger.info(f"  Mapped {len(image_embeddings)} image embeddings to study_ids")
        self.embedding_store.add_image_embeddings(image_embeddings)

        # 3. Text embeddings
        logger.info("[3/5] Generating text embeddings...")
        tokenizer = self._load_tokenizer()
        text_embeddings = self.embedder.generate_text_embeddings_from_dataframe(
            df, tokenizer=tokenizer, text_col=text_col, show_progress=True,
        )
        self.embedding_store.add_text_embeddings(text_embeddings)
        self.embedding_store.save()

        # 4. Image index (image query -> retrieve reports)
        logger.info("[4/5] Building image index...")
        study_ids = df['study_id'].astype(str).tolist()
        image_emb_matrix, valid_ids = self.embedding_store.get_all_image_embeddings(study_ids)
        if len(valid_ids) == 0:
            raise ValueError("No valid image embeddings found")
        self.indexer.build_image_index(image_emb_matrix, valid_ids)

        # 5. Text index (text query -> retrieve images)
        logger.info("[5/5] Building text index...")
        text_emb_matrix, valid_ids = self.embedding_store.get_all_text_embeddings(study_ids)
        if len(valid_ids) == 0:
            raise ValueError("No valid text embeddings found")
        self.indexer.build_text_index(text_emb_matrix, valid_ids)

        self.indexer.save()

        logger.info("RAG database build complete")
        self._print_stats()

    def build_from_splits(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
        image_path_col: str = 'image_path',
        text_col: str = 'text',
    ):
        """Build database using configured train/val splits."""
        dfs = []
        if self.config.data.USE_TRAIN_FOR_INDEX:
            train_df = train_df.copy()
            train_df['split'] = 'train'
            dfs.append(train_df)
        if self.config.data.USE_VAL_FOR_INDEX and val_df is not None:
            val_df = val_df.copy()
            val_df['split'] = 'val'
            dfs.append(val_df)
        if not dfs:
            raise ValueError("No data selected for index building")

        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Index data: {len(combined_df)} samples")
        self.build_from_dataframe(combined_df, image_path_col, text_col)

    def load_existing_database(self):
        """Load a previously built RAG database from disk."""
        logger.info("Loading existing RAG database...")
        self.metadata_db.load()
        self.embedding_store.load()
        self.indexer.load()
        logger.info("Loaded existing database")
        self._print_stats()

    def _print_stats(self):
        meta = self.metadata_db.get_statistics()
        emb = self.embedding_store.get_statistics()
        idx = self.indexer.get_stats()
        print(f"\nDatabase statistics:")
        print(f"  Metadata entries:  {meta['total_studies']}")
        print(f"  Image embeddings:  {emb['num_image_embeddings']}")
        print(f"  Text embeddings:   {emb['num_text_embeddings']}")
        print(f"  Image index size:  {idx['image_index']['total_embeddings']}")
        print(f"  Text index size:   {idx['text_index']['total_embeddings']}")
        print(f"  Embedding dim:     {emb['image_embedding_dim']}")

    def get_metadata_db(self):
        return self.metadata_db

    def get_embedding_store(self):
        return self.embedding_store

    def get_indexer(self):
        return self.indexer

    def __repr__(self):
        return f"RAGDatabaseBuilder(metadata={len(self.metadata_db)}, embeddings={len(self.embedding_store)})"