"""Configuration for RAG pipeline."""

import os
import json
import random
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch


@dataclass
class PathConfig:
    """File paths and directories."""
    BASE_DIR: str = "/n/scratch/users/g/gul075/Temp_AIM2" # "/n/groups/training/bmif203/AIM2"

    # CLIP model dependency
    CLIP_CHECKPOINT: str = "CLIP/outputs/checkpoints/best_model.pth"

    # Data paths
    DATA_CSV: str = "processed_data/processed_data.csv"
    IMAGE_ROOT: str = "cxr_jpg" # needs to be changed to the common repo when download is done
    REPORT_ROOT: str = "cxr_reports" #same (and modify back the lines below as well)

    # Output directories
    OUTPUT_DIR: str = "RAG/outputs"
    INDEX_DIR: str = "RAG/outputs/indices"
    EVALUATION_DIR: str = "RAG/outputs/evaluation"
    LOG_DIR: str = "RAG/outputs/logs"
    CACHE_DIR: str = "RAG/outputs/cache"

    def __post_init__(self):
        self.CLIP_CHECKPOINT = os.path.join(self.BASE_DIR, self.CLIP_CHECKPOINT)
        self.DATA_CSV = os.path.join(self.BASE_DIR, self.DATA_CSV)
        self.IMAGE_ROOT = "/n/scratch/users/g/gul075/AIM_PHD/Foundation_in_clinical_data3/cxr_jpg" #os.path.join(self.BASE_DIR, self.IMAGE_ROOT)
        self.REPORT_ROOT = "/n/scratch/users/g/gul075/AIM_PHD/Foundation_in_clinical_data3/cxr_reports" #os.path.join(self.BASE_DIR, self.REPORT_ROOT)
        self.OUTPUT_DIR = os.path.join(self.BASE_DIR, self.OUTPUT_DIR)
        self.INDEX_DIR = os.path.join(self.BASE_DIR, self.INDEX_DIR)
        self.EVALUATION_DIR = os.path.join(self.BASE_DIR, self.EVALUATION_DIR)
        self.LOG_DIR = os.path.join(self.BASE_DIR, self.LOG_DIR)
        self.CACHE_DIR = os.path.join(self.BASE_DIR, self.CACHE_DIR)

        for d in [self.OUTPUT_DIR, self.INDEX_DIR, self.EVALUATION_DIR,
                  self.LOG_DIR, self.CACHE_DIR]:
            os.makedirs(d, exist_ok=True)


@dataclass
class EmbedderConfig:
    """CLIP embedder settings. Image transforms must match CLIP validation transforms."""
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE: int = 64
    NUM_WORKERS: int = 4
    IMAGE_SIZE: int = 224
    IMAGE_MEAN: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    IMAGE_STD: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    TEXT_ENCODER: str = "emilyalsentzer/Bio_ClinicalBERT"
    TEXT_MAX_LENGTH: int = 512
    CACHE_EMBEDDINGS: bool = True


@dataclass
class RetrievalConfig:
    """FAISS index and retrieval settings."""
    INDEX_TYPE: str = "HNSW"
    METRIC_TYPE: str = "INNER_PRODUCT"

    # IVF parameters
    NLIST: int = 316
    NPROBE: int = 32

    # HNSW parameters
    HNSW_M: int = 48
    HNSW_EF_CONSTRUCTION: int = 200
    HNSW_EF_SEARCH: int = 128

    TOP_K: int = 5


@dataclass
class GenerationConfig:
    """LLM-based report generation settings (used by downstream generation scripts)."""
    LLM_PROVIDER: str = "openai"
    LLM_MODEL: str = "gpt-4o-mini"
    MAX_TOKENS: int = 512
    TEMPERATURE: float = 0.0
    TOP_P: float = 1.0
    FREQUENCY_PENALTY: float = 0.0
    PRESENCE_PENALTY: float = 0.0
    API_KEY_ENV: str = "OPENAI_API_KEY"
    API_TIMEOUT: int = 60
    MAX_RETRIES: int = 3
    INCLUDE_RETRIEVED_REPORTS: bool = True
    INCLUDE_CHEXPERT_LABELS: bool = True
    INCLUDE_STUDY_METADATA: bool = False
    GENERATE_FINDINGS: bool = True
    GENERATE_IMPRESSION: bool = True
    GENERATE_COMBINED: bool = False
    SECTION_SEPARATOR: str = "\n\n"
    USE_SECTION_HEADERS: bool = True


@dataclass
class EvaluationConfig:
    """Evaluation metrics settings (used by downstream evaluation scripts)."""
    COMPUTE_BLEU: bool = True
    COMPUTE_ROUGE: bool = True
    COMPUTE_BERTSCORE: bool = True
    COMPUTE_CHEXPERT_F1: bool = True
    COMPUTE_RADGRAPH_F1: bool = False
    BERTSCORE_MODEL: str = "microsoft/deberta-xlarge-mnli"
    BERTSCORE_DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    CHEXPERT_LABELER_PATH: Optional[str] = None
    EVAL_LEVEL: str = "both"
    SAVE_PER_SAMPLE_RESULTS: bool = True
    SAVE_PREDICTIONS: bool = True
    SAVE_ERROR_ANALYSIS: bool = True


@dataclass
class SystemConfig:
    """System and reproducibility settings."""
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    SEED: int = 42
    DETERMINISTIC: bool = True


@dataclass
class DataConfig:
    """Data processing settings."""
    USE_TRAIN_FOR_INDEX: bool = True
    USE_VAL_FOR_INDEX: bool = False
    EVAL_ON_SPLIT: str = "test"
    COMBINE_SECTIONS: bool = True
    SECTION_SEPARATOR: str = " [SEP] "
    TEXT_FALLBACK: str = "No findings reported."
    CHEXPERT_LABELS: List[str] = field(default_factory=lambda: [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
        'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
        'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
        'Pneumonia', 'Pneumothorax', 'Support Devices',
    ])


class RAGConfig:
    """Master configuration for RAG pipeline."""

    def __init__(self):
        self.paths = PathConfig()
        self.embedder = EmbedderConfig()
        self.retrieval = RetrievalConfig()
        self.generation = GenerationConfig()
        self.evaluation = EvaluationConfig()
        self.data = DataConfig()
        self.system = SystemConfig()

        self._set_seeds()

    def _set_seeds(self):
        random.seed(self.system.SEED)
        np.random.seed(self.system.SEED)
        torch.manual_seed(self.system.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.system.SEED)
        if self.system.DETERMINISTIC:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def print_config(self):
        print("=" * 80)
        print("RAG CONFIGURATION")
        print("=" * 80)
        sections = [
            ("PATHS", self.paths), ("EMBEDDER", self.embedder),
            ("RETRIEVAL", self.retrieval), ("GENERATION", self.generation),
            ("EVALUATION", self.evaluation), ("DATA", self.data),
            ("SYSTEM", self.system),
        ]
        for name, section in sections:
            print(f"\n{name}:")
            for key, value in section.__dict__.items():
                if not key.startswith('_'):
                    if 'API_KEY' in key and isinstance(value, str):
                        value = "***" if value else "Not set"
                    print(f"  {key}: {value}")
        print("=" * 80)

    def save(self, path: str):
        config_dict = {}
        for attr in ['paths', 'embedder', 'retrieval', 'generation',
                      'evaluation', 'data', 'system']:
            section = getattr(self, attr)
            config_dict[attr] = {
                k: v for k, v in section.__dict__.items() if not k.startswith('_')
            }
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"Configuration saved to {path}")

    @classmethod
    def load(cls, path: str):
        with open(path, 'r') as f:
            config_dict = json.load(f)
        config = cls()
        for section_name, section_dict in config_dict.items():
            section = getattr(config, section_name)
            for key, value in section_dict.items():
                setattr(section, key, value)
        print(f"Configuration loaded from {path}")
        return config

    def validate(self):
        errors = []
        if not os.path.exists(self.paths.CLIP_CHECKPOINT):
            errors.append(f"CLIP checkpoint not found: {self.paths.CLIP_CHECKPOINT}")
        if not os.path.exists(self.paths.DATA_CSV):
            errors.append(f"Data CSV not found: {self.paths.DATA_CSV}")
        if self.generation.LLM_PROVIDER in ['openai', 'anthropic']:
            if not os.getenv(self.generation.API_KEY_ENV):
                errors.append(f"API key not found: {self.generation.API_KEY_ENV}")
        if self.retrieval.TOP_K <= 0:
            errors.append(f"TOP_K must be > 0, got {self.retrieval.TOP_K}")
        if errors:
            print("Configuration validation failed:")
            for e in errors:
                print(f"  - {e}")
            return False
        print("Configuration validation passed")
        return True