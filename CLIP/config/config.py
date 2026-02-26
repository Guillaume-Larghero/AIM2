"""Configuration for CLIP cross-modal alignment training."""

import os
from dataclasses import dataclass, field
from typing import List, Optional
import torch
import json
import random
import numpy as np


@dataclass
class PathConfig:
    """File paths and directories."""
    BASE_DIR: str = "/n/groups/training/bmif203/AIM2"

    DATA_CSV: str = "processed_data/processed_data.csv"
    IMAGE_ROOT: str = "cxr_jpg" # needs to be changed to the common repo when download is done
    REPORT_ROOT: str = "cxr_reports" #same (and modify back the lines below as well)

    OUTPUT_DIR: str = "CLIP/outputs"
    CHECKPOINT_DIR: str = "CLIP/outputs/checkpoints"
    LOG_DIR: str = "CLIP/outputs/logs"
    VIZ_DIR: str = "CLIP/outputs/visualizations"

    def __post_init__(self):
        self.DATA_CSV = os.path.join(self.BASE_DIR, self.DATA_CSV)
        self.IMAGE_ROOT = "/n/scratch/users/g/gul075/AIM_PHD/Foundation_in_clinical_data3/cxr_jpg" #os.path.join(self.BASE_DIR, self.IMAGE_ROOT)
        self.REPORT_ROOT = "/n/scratch/users/g/gul075/AIM_PHD/Foundation_in_clinical_data3/cxr_reports" #os.path.join(self.BASE_DIR, self.REPORT_ROOT)
        self.OUTPUT_DIR = os.path.join(self.BASE_DIR, self.OUTPUT_DIR)
        self.CHECKPOINT_DIR = os.path.join(self.BASE_DIR, self.CHECKPOINT_DIR)
        self.LOG_DIR = os.path.join(self.BASE_DIR, self.LOG_DIR)
        self.VIZ_DIR = os.path.join(self.BASE_DIR, self.VIZ_DIR)

        for dir_path in [self.OUTPUT_DIR, self.CHECKPOINT_DIR, self.LOG_DIR, self.VIZ_DIR]:
            os.makedirs(dir_path, exist_ok=True)


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Image encoder
    IMAGE_ENCODER: str = "vit_base_patch16_224"
    IMAGE_PRETRAINED: bool = True
    IMAGE_FEATURE_DIM: int = 768
    USE_ATTENTION_POOLING: bool = True
    ATTENTION_POOL_HIDDEN_DIM_RATIO: float = 0.25
    SPATIAL_ATTENTION_REDUCTION_RATIO: int = 8

    # Text encoder
    TEXT_ENCODER: str = "emilyalsentzer/Bio_ClinicalBERT"
    TEXT_MAX_LENGTH: int = 512
    TEXT_FEATURE_DIM: int = 768
    TEXT_NUM_LAYERS: int = 4

    # Projection heads
    PROJECTION_DIM: int = 256
    PROJECTION_HIDDEN_DIM: int = 512
    PROJECTION_DROPOUT: float = 0.2
    PROJECTION_NUM_LAYERS: int = 2
    NORMALIZE_EMBEDDINGS: bool = True

    # CheXpert auxiliary head
    USE_CHEXPERT_AUX: bool = True
    NUM_CHEXPERT_LABELS: int = 14
    CHEXPERT_HIDDEN_DIM_RATIO: float = 0.5


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    BATCH_SIZE: int = 32
    NUM_EPOCHS: int = 100
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 1e-4

    # Warmup and schedule
    USE_WARMUP: bool = True
    WARMUP_EPOCHS: int = 10
    SCHEDULER: str = "cosine"
    SCHEDULER_ETA_MIN: float = 1e-6

    # Differential learning rates
    IMAGE_ENCODER_LR: float = 5e-6
    TEXT_ENCODER_LR: float = 5e-6
    PROJECTION_LR: float = 1e-4

    # Loss weights (I barely used Chexperts labeling in the latest version of training)
    CONTRASTIVE_WEIGHT: float = 0.99
    CHEXPERT_WEIGHT: float = 0.01
    TEMPERATURE: float = 0.07

    # Hard negative mining
    USE_HARD_NEGATIVES: bool = True
    HARD_NEG_RATIO: float = 0.25

    # Focal loss
    USE_FOCAL_LOSS: bool = True
    FOCAL_GAMMA: float = 2.0
    LABEL_SMOOTHING: float = 0.0

    # Optimization
    OPTIMIZER: str = "adamw"
    GRADIENT_CLIP: float = 1.0
    GRADIENT_ACCUMULATION_STEPS: int = 4
    USE_AMP: bool = True
    DROPOUT: float = 0.2


@dataclass
class DataConfig:
    """Data loading and augmentation configuration."""
    IMAGE_SIZE: int = 224
    IMAGE_MEAN: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    IMAGE_STD: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])

    # Augmentation
    USE_AUGMENTATION: bool = True
    RANDOM_CROP_SCALE_MIN: float = 0.7 # weirdly big cropping worked better idk why
    RANDOM_CROP_SCALE_MAX: float = 1.0
    RANDOM_CROP_RATIO_MIN: float = 0.9
    RANDOM_CROP_RATIO_MAX: float = 1.1

    RANDOM_ROTATION: bool = True
    RANDOM_ROTATION_DEGREES: float = 5.0

    COLOR_JITTER: bool = True
    COLOR_JITTER_BRIGHTNESS: float = 0.2
    COLOR_JITTER_CONTRAST: float = 0.2
    COLOR_JITTER_SATURATION: float = 0.0
    COLOR_JITTER_HUE: float = 0.0

    RANDOM_ERASING: bool = True
    RANDOM_ERASING_P: float = 0.2
    RANDOM_ERASING_SCALE_MIN: float = 0.02
    RANDOM_ERASING_SCALE_MAX: float = 0.1
    RANDOM_ERASING_RATIO_MIN: float = 0.3
    RANDOM_ERASING_RATIO_MAX: float = 3.3

    # Text preprocessing
    COMBINE_SECTIONS: bool = True
    SECTION_SEPARATOR: str = " [SEP] "
    TEXT_FALLBACK: str = "No findings reported."

    # DataLoader
    NUM_WORKERS: int = 2
    PIN_MEMORY: bool = True
    PREFETCH_FACTOR: int = 2
    PERSISTENT_WORKERS: bool = True

    CHEXPERT_LABELS: List[str] = field(default_factory=lambda: [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
        'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
        'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
        'Pneumonia', 'Pneumothorax', 'Support Devices'
    ])


@dataclass
class EvaluationConfig:
    """Evaluation and metrics."""
    EVAL_EVERY_N_EPOCHS: int = 3
    SAVE_EVERY_N_EPOCHS: int = 10
    RETRIEVAL_K_VALUES: List[int] = field(default_factory=lambda: [1, 5, 10])


@dataclass
class SystemConfig:
    """System and hardware configuration."""
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    SEED: int = 42
    DETERMINISTIC: bool = True
    BENCHMARK_CUDNN: bool = True
    USE_TENSORBOARD: bool = True
    LOG_INTERVAL: int = 10


class Config:
    """Master configuration class.

    Usage:
        from CLIP.config.config import Config
        config = Config()
    """

    def __init__(self):
        self.paths = PathConfig()
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.data = DataConfig()
        self.evaluation = EvaluationConfig()
        self.system = SystemConfig()

        self._set_seeds()
        self._configure_pytorch()

    def _set_seeds(self):
        """Set random seeds for reproducibility."""

        random.seed(self.system.SEED)
        np.random.seed(self.system.SEED)
        torch.manual_seed(self.system.SEED)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.system.SEED)
            torch.cuda.manual_seed_all(self.system.SEED)

        if self.system.DETERMINISTIC:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        elif self.system.BENCHMARK_CUDNN:
            torch.backends.cudnn.benchmark = True

    def _configure_pytorch(self):
        torch.set_num_threads(self.data.NUM_WORKERS)

    def print_config(self):
        """Print configuration summary."""
        print("=" * 80)
        print("CONFIGURATION SUMMARY")
        print("=" * 80)

        sections = [
            ("PATHS", self.paths),
            ("MODEL", self.model),
            ("TRAINING", self.training),
            ("DATA", self.data),
            ("EVALUATION", self.evaluation),
            ("SYSTEM", self.system),
        ]

        for name, section in sections:
            print(f"\n{name}:")
            for key, value in section.__dict__.items():
                if not key.startswith('_'):
                    print(f"  {key}: {value}")

        print("=" * 80)

    def save(self, path: str):
        """Save configuration to JSON."""
    
        config_dict = {
            'paths': {k: v for k, v in self.paths.__dict__.items() if not k.startswith('_')},
            'model': {k: v for k, v in self.model.__dict__.items() if not k.startswith('_')},
            'training': {k: v for k, v in self.training.__dict__.items() if not k.startswith('_')},
            'data': {k: v for k, v in self.data.__dict__.items() if not k.startswith('_')},
            'evaluation': {k: v for k, v in self.evaluation.__dict__.items() if not k.startswith('_')},
            'system': {k: v for k, v in self.system.__dict__.items() if not k.startswith('_')},
        }

        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)

        print(f"Configuration saved to {path}")


if __name__ == "__main__":
    config = Config()
    config.print_config()