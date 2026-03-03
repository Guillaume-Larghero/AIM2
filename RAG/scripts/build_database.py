#!/usr/bin/env python3
"""
Build complete RAG database for image-to-report and text-to-image retrieval.

Usage:
    python build_database.py [--val-in-index] [--include-indication] [--debug]
"""

import os
import sys
import argparse
import logging
from datetime import datetime
import traceback
import torch

from RAG.config.config import RAGConfig
from RAG.data.dataloader import MIMICDataLoader
from RAG.pipeline.builder import RAGDatabaseBuilder


def setup_logging(config, debug=False):
    """Set up file and console logging."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(config.paths.LOG_DIR, f"build_database_{timestamp}.log")
    level = logging.DEBUG if debug else logging.INFO

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    logging.info(f"Logging to: {log_file}")
    return log_file


def validate_environment(config):
    """Check that required files, directories, and packages exist."""
    logging.info("Validating environment...")
    errors = []

    if not os.path.exists(config.paths.CLIP_CHECKPOINT):
        errors.append(f"CLIP checkpoint not found: {config.paths.CLIP_CHECKPOINT}")
    else:
        logging.info(f"  CLIP checkpoint: {config.paths.CLIP_CHECKPOINT}")

    # Verify CLIP package is importable
    try:
        from CLIP.model.clip_model import MedicalCLIP  # noqa: F401
        logging.info("  CLIP package: importable")
    except ImportError as e:
        errors.append(f"CLIP package not importable: {e}")

    if not os.path.exists(config.paths.DATA_CSV):
        errors.append(f"Data CSV not found: {config.paths.DATA_CSV}")
    else:
        logging.info(f"  Data CSV: {config.paths.DATA_CSV}")

    if torch.cuda.is_available():
        logging.info(f"  CUDA: {torch.cuda.get_device_name(0)} "
                     f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    else:
        logging.warning("  CUDA not available, using CPU")

    for package, install_name in {'faiss': 'faiss-cpu', 'h5py': 'h5py',
                                   'transformers': 'transformers'}.items():
        try:
            __import__(package)
            logging.info(f"  {package}: installed")
        except ImportError:
            errors.append(f"Missing package: {install_name}")

    if errors:
        logging.error("Validation failed:")
        for e in errors:
            logging.error(f"  - {e}")
        return False

    logging.info("Environment validation passed\n")
    return True


def load_data(config, include_indication=False):
    """Load and prepare MIMIC-CXR data."""
    loader = MIMICDataLoader(config)
    df = loader.load_csv()
    df = loader.prepare_dataframe(df, include_indication=include_indication)
    loader.print_statistics(df)
    return loader.get_train_val_test_splits(df)


def run_sanity_checks(builder):
    """Run sanity checks on built database."""
    logging.info("Running sanity checks...")

    metadata_db = builder.get_metadata_db()
    embedding_store = builder.get_embedding_store()
    indexer = builder.get_indexer()

    n_meta = len(metadata_db)
    n_img = len(embedding_store.image_embeddings)
    n_txt = len(embedding_store.text_embeddings)
    n_img_idx = len(indexer.image_indexer)
    n_txt_idx = len(indexer.text_indexer)

    logging.info(f"  Metadata: {n_meta}, Image emb: {n_img}, Text emb: {n_txt}, "
                 f"Image idx: {n_img_idx}, Text idx: {n_txt_idx}")

    if n_meta == n_img == n_txt == n_img_idx == n_txt_idx:
        logging.info("  Counts match")
    else:
        logging.warning("  Count mismatch detected")

    # Self-retrieval test
    sample_sid = list(embedding_store.image_embeddings.keys())[0]

    sample_emb = embedding_store.get_image_embedding(sample_sid)
    dists, sids = indexer.search_by_image(sample_emb, top_k=5)
    logging.info(f"  Image self-retrieval: top={sids[0]} (sim={dists[0]:.3f}), "
                 f"{'OK' if sids[0] == sample_sid else 'FAILED'}")

    sample_emb = embedding_store.get_text_embedding(sample_sid)
    dists, sids = indexer.search_by_text(sample_emb, top_k=5)
    logging.info(f"  Text self-retrieval: top={sids[0]} (sim={dists[0]:.3f}), "
                 f"{'OK' if sids[0] == sample_sid else 'FAILED'}")

    # Metadata retrieval
    study = metadata_db.get_study(sample_sid)
    if study:
        logging.info(f"  Metadata retrieval: OK (findings={bool(study['findings'])})")
    else:
        logging.error("  Metadata retrieval: FAILED")

    # File existence
    idx_type = builder.config.retrieval.INDEX_TYPE.lower()
    files = [
        os.path.join(builder.config.paths.INDEX_DIR, "metadata_db.json"),
        os.path.join(builder.config.paths.INDEX_DIR, "embeddings.h5"),
        os.path.join(builder.config.paths.INDEX_DIR, "image_index", f"faiss_{idx_type}.index"),
        os.path.join(builder.config.paths.INDEX_DIR, "text_index", f"faiss_{idx_type}.index"),
    ]
    all_exist = True
    for fp in files:
        if os.path.exists(fp):
            logging.info(f"  {os.path.basename(fp)}: {os.path.getsize(fp) / 1e6:.1f} MB")
        else:
            logging.error(f"  Missing: {fp}")
            all_exist = False

    return all_exist


def main():
    parser = argparse.ArgumentParser(description="Build RAG database")
    parser.add_argument("--debug", action="store_true", help="Debug logging")
    parser.add_argument("--val-in-index", action="store_true",
                        help="Include validation set in index")
    parser.add_argument("--include-indication", action="store_true",
                        help="Include indication section in text")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to custom config JSON")
    args = parser.parse_args()

    config = RAGConfig.load(args.config) if args.config else RAGConfig()
    if args.val_in_index:
        config.data.USE_VAL_FOR_INDEX = True

    log_file = setup_logging(config, debug=args.debug)

    try:
        logging.info(f"Base dir: {config.paths.BASE_DIR}")
        logging.info(f"CLIP checkpoint: {config.paths.CLIP_CHECKPOINT}")
        logging.info(f"Index type: {config.retrieval.INDEX_TYPE}")
        logging.info(f"Batch size: {config.embedder.BATCH_SIZE}")
        logging.info(f"Val in index: {config.data.USE_VAL_FOR_INDEX}")

        if not validate_environment(config):
            sys.exit(1)

        train_df, val_df, test_df = load_data(config, args.include_indication)

        builder = RAGDatabaseBuilder(config)
        builder.build_from_splits(
            train_df=train_df, val_df=val_df, test_df=test_df,
            image_path_col='image_path', text_col='text',
        )

        checks_passed = run_sanity_checks(builder)

        logging.info(f"\nLog: {log_file}")
        logging.info(f"Output: {config.paths.OUTPUT_DIR}")
        if checks_passed:
            logging.info("Database built successfully, all checks passed")
        else:
            logging.warning("Database built but some checks failed")

    except Exception as e:
        logging.error(f"BUILD FAILED: {e}")
        logging.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()