"""Configuration for the GENERATION pipeline."""

import os
import random
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch


@dataclass
class PathConfig:
    """File paths for the generation pipeline."""
    BASE_DIR: str = "/n/groups/training/bmif203/AIM2"

    # CLIP model dependency
    CLIP_CHECKPOINT: str = "CLIP/outputs/checkpoints/best_model.pth"

    # RAG dependencies (pre-built indices and metadata)
    INDEX_DIR: str = "RAG/outputs/indices"
    IMAGE_INDEX_DIR: str = "RAG/outputs/indices/image_index"
    TEXT_INDEX_DIR: str = "RAG/outputs/indices/text_index"
    EMBEDDINGS_H5: str = "RAG/outputs/indices/embeddings.h5"
    METADATA_DB: str = "RAG/outputs/indices/metadata_db.json"

    # Data paths
    DATA_CSV: str = "processed_data/processed_data.csv"
    IMAGE_ROOT: str = "cxr_jpg" # needs to be changed to the common repo when download is done

    # Output directories
    OUTPUT_DIR: str = "GENERATION/outputs"
    EVALUATION_DIR: str = "GENERATION/outputs/evaluation"
    LOG_DIR: str = "GENERATION/outputs/logs"
    EXPERIMENTS_DIR: str = "GENERATION/outputs/experiments"

    def __post_init__(self):
        self.CLIP_CHECKPOINT = os.path.join(self.BASE_DIR, self.CLIP_CHECKPOINT)
        self.INDEX_DIR = os.path.join(self.BASE_DIR, self.INDEX_DIR)
        self.IMAGE_INDEX_DIR = os.path.join(self.BASE_DIR, self.IMAGE_INDEX_DIR)
        self.TEXT_INDEX_DIR = os.path.join(self.BASE_DIR, self.TEXT_INDEX_DIR)
        self.EMBEDDINGS_H5 = os.path.join(self.BASE_DIR, self.EMBEDDINGS_H5)
        self.METADATA_DB = os.path.join(self.BASE_DIR, self.METADATA_DB)
        self.DATA_CSV = os.path.join(self.BASE_DIR, self.DATA_CSV)
        self.IMAGE_ROOT = "/n/scratch/users/g/gul075/AIM_PHD/Foundation_in_clinical_data3/cxr_jpg"  # TODO: update to /n/groups/training/bmif203/AIM2/dataset/physionet.org/files/mimic-cxr-jpg/2.1.0/files
        self.OUTPUT_DIR = os.path.join(self.BASE_DIR, self.OUTPUT_DIR)
        self.EVALUATION_DIR = os.path.join(self.BASE_DIR, self.EVALUATION_DIR)
        self.LOG_DIR = os.path.join(self.BASE_DIR, self.LOG_DIR)
        self.EXPERIMENTS_DIR = os.path.join(self.BASE_DIR, self.EXPERIMENTS_DIR)

        for d in [self.OUTPUT_DIR, self.EVALUATION_DIR, self.LOG_DIR, self.EXPERIMENTS_DIR]:
            os.makedirs(d, exist_ok=True)


@dataclass
class RetrievalConfig:
    """Retrieval component settings."""
    TOP_K: int = 10
    MIN_SIMILARITY: float = 0.0
    USE_CHEXPERT_FILTERING: bool = False
    CHEXPERT_MATCH_THRESHOLD: int = 2
    METRIC_TYPE: str = "INNER_PRODUCT"


@dataclass
class GenerationConfig:
    """LLM-based report generation settings."""
    LLM_PROVIDER: str = "openai"
    LLM_MODEL: str = "gpt-4o"
    MAX_TOKENS: int = 512
    TEMPERATURE: float = 0.0
    TOP_P: float = 1.0
    FREQUENCY_PENALTY: float = 0.0
    PRESENCE_PENALTY: float = 0.0
    API_KEY_ENV: str = "OPENAI_API_KEY"
    API_TIMEOUT: int = 60
    MAX_RETRIES: int = 3
    INCLUDE_CHEXPERT_LABELS: bool = True


@dataclass
class SystemConfig:
    """System and hardware settings."""
    DEVICE: str = "cuda"
    SEED: int = 42
    DETERMINISTIC: bool = True
    BATCH_SIZE: int = 64
    NUM_WORKERS: int = 4


@dataclass
class DataConfig:
    """Data processing settings."""
    EVAL_SPLIT: str = "train"
    REQUIRE_FINDINGS: bool = True
    REQUIRE_IMPRESSION: bool = True
    CHEXPERT_LABELS: List[str] = field(default_factory=lambda: [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
        'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
        'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
        'Pneumonia', 'Pneumothorax', 'Support Devices'
    ])


class GenerationPipelineConfig:
    """Master configuration for the GENERATION pipeline."""

    def __init__(self):
        self.paths = PathConfig()
        self.retrieval = RetrievalConfig()
        self.generation = GenerationConfig()
        self.system = SystemConfig()
        self.data = DataConfig()
        self._configure_system()

    def _configure_system(self):
        random.seed(self.system.SEED)
        np.random.seed(self.system.SEED)
        torch.manual_seed(self.system.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.system.SEED)
        if self.system.DETERMINISTIC:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def validate(self) -> bool:
        errors = []
        if not os.path.exists(self.paths.CLIP_CHECKPOINT):
            errors.append(f"CLIP checkpoint not found: {self.paths.CLIP_CHECKPOINT}")
        if not os.path.exists(self.paths.IMAGE_INDEX_DIR):
            errors.append(f"Image index not found: {self.paths.IMAGE_INDEX_DIR}")
        if not os.path.exists(self.paths.METADATA_DB):
            errors.append(f"Metadata DB not found: {self.paths.METADATA_DB}")
        if not os.path.exists(self.paths.DATA_CSV):
            errors.append(f"Data CSV not found: {self.paths.DATA_CSV}")
        api_key = os.getenv(self.generation.API_KEY_ENV)
        if not api_key:
            errors.append(f"API key not found: {self.generation.API_KEY_ENV}")
        if errors:
            for e in errors:
                print(f"  {e}")
            return False
        return True