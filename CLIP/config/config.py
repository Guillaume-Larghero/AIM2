"""Configuration for Medical CLIP cross-modal alignment training."""

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

    DATA_CSV:    str = "processed_data/processed_data.csv"
    IMAGE_ROOT:  str = "cxr_jpg"
    REPORT_ROOT: str = "cxr_reports"

    OUTPUT_DIR:     str = "CLIP/outputs"
    CHECKPOINT_DIR: str = "CLIP/outputs/checkpoints"
    LOG_DIR:        str = "CLIP/outputs/logs"
    VIZ_DIR:        str = "CLIP/outputs/visualizations"

    def __post_init__(self):
        self.DATA_CSV    = os.path.join(self.BASE_DIR, self.DATA_CSV)
        self.IMAGE_ROOT  = "/n/scratch/users/g/gul075/AIM_PHD/Foundation_in_clinical_data3/cxr_jpg"
        self.REPORT_ROOT = "/n/scratch/users/g/gul075/AIM_PHD/Foundation_in_clinical_data3/cxr_reports"
        self.OUTPUT_DIR     = os.path.join(self.BASE_DIR, self.OUTPUT_DIR)
        self.CHECKPOINT_DIR = os.path.join(self.BASE_DIR, self.CHECKPOINT_DIR)
        self.LOG_DIR        = os.path.join(self.BASE_DIR, self.LOG_DIR)
        self.VIZ_DIR        = os.path.join(self.BASE_DIR, self.VIZ_DIR)
        for d in [self.OUTPUT_DIR, self.CHECKPOINT_DIR, self.LOG_DIR, self.VIZ_DIR]:
            os.makedirs(d, exist_ok=True)


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # ── IMAGE ENCODER ──────────────────────────────────────────────────────────
    # Timm model string carries the ImageNet pretrained weights.
    # img_size=512 is passed at create_model() time so timm interpolates the
    # pretrained 14×14 positional embeddings (196 tokens) → 32×32 (1024 tokens).
    # Patch projection and attention weights are reused as-is.
    # IMAGE_FEATURE_DIM stays 768 — the ViT-B hidden dim is resolution-independent.
    IMAGE_ENCODER:    str  = "vit_base_patch16_224"
    IMAGE_PRETRAINED: bool = True
    IMAGE_FEATURE_DIM: int = 768
    USE_ATTENTION_POOLING: bool = True
    ATTENTION_POOL_HIDDEN_DIM_RATIO: float = 0.25
    SPATIAL_ATTENTION_REDUCTION_RATIO: int = 8

    # ── TEXT ENCODER ───────────────────────────────────────────────────────────
    TEXT_ENCODER:     str = "emilyalsentzer/Bio_ClinicalBERT"
    TEXT_MAX_LENGTH:  int = 512
    TEXT_FEATURE_DIM: int = 768
    TEXT_NUM_LAYERS:  int = 4

    # ── PROJECTION HEADS ───────────────────────────────────────────────────────
    PROJECTION_DIM:        int   = 256
    PROJECTION_HIDDEN_DIM: int   = 512
    PROJECTION_DROPOUT:    float = 0.2
    PROJECTION_NUM_LAYERS: int   = 2
    NORMALIZE_EMBEDDINGS:  bool  = True

    # ── CHEXPERT AUXILIARY HEAD ─────────────────────────────────────────────────
    USE_CHEXPERT_AUX:       bool  = True
    NUM_CHEXPERT_LABELS:    int   = 14
    CHEXPERT_HIDDEN_DIM_RATIO: float = 0.5


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # CHANGED 32 → 8:  At IMAGE_SIZE=512, ViT-B/16 produces 1024 patch tokens.
    # Attention memory is O(n²): 1024 tokens requires ~27× more memory than
    # 196 tokens (224px). Batch 8 keeps peak VRAM safely under 40 GB on A100.
    BATCH_SIZE:  int = 8
    NUM_EPOCHS:  int = 100
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY:  float = 1e-4

    USE_WARMUP:    bool  = True
    WARMUP_EPOCHS: int   = 10
    SCHEDULER:     str   = "cosine"
    SCHEDULER_ETA_MIN: float = 1e-6

    IMAGE_ENCODER_LR: float = 5e-6
    TEXT_ENCODER_LR:  float = 5e-6
    PROJECTION_LR:    float = 1e-4

    CONTRASTIVE_WEIGHT: float = 0.99
    CHEXPERT_WEIGHT:    float = 0.01
    TEMPERATURE:        float = 0.07

    USE_HARD_NEGATIVES: bool  = True
    HARD_NEG_RATIO:     float = 0.25

    USE_FOCAL_LOSS:   bool  = True
    FOCAL_GAMMA:      float = 2.0
    LABEL_SMOOTHING:  float = 0.0

    OPTIMIZER:      str   = "adamw"
    GRADIENT_CLIP:  float = 1.0

    # CHANGED 4 → 16:  Keeps effective batch identical: 8×16 = 128 (was 32×4=128).
    # Scheduler T_max is now set in OPTIMIZER STEPS (not mini-batch steps) so
    # the cosine decay is not diluted by the larger accumulation factor. See trainer.py.
    GRADIENT_ACCUMULATION_STEPS: int = 16

    USE_AMP: bool = True
    DROPOUT: float = 0.2


@dataclass
class DataConfig:
    """Data loading and augmentation configuration."""
    # CHANGED 224 → 512:
    # FLUX.2-dev outputs 512×512. At IMAGE_SIZE=512, FLUX.2 outputs can be
    # passed directly to MedCLIP with zero extra resize. GT images (~2735×2790,
    # nearly square) are LANCZOS-downsampled to 512×512 — identical transform
    # for all inputs, GT or generated.
    IMAGE_SIZE: int = 512

    # ImageNet normalisation — kept as-is. ViT-B was pretrained with these stats;
    # positional embeddings are interpolated but patch/attention weights reused.
    IMAGE_MEAN: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    IMAGE_STD:  List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])

    USE_AUGMENTATION: bool = True
    RANDOM_CROP_SCALE_MIN: float = 0.7
    RANDOM_CROP_SCALE_MAX: float = 1.0

    # CHANGED (0.9, 1.1) → (1.0, 1.0):
    # Always produce a square crop during training, matching the square
    # inference resize. Dataset aspect ratio is 0.995 — no content lost.
    RANDOM_CROP_RATIO_MIN: float = 1.0
    RANDOM_CROP_RATIO_MAX: float = 1.0

    RANDOM_ROTATION: bool = True
    RANDOM_ROTATION_DEGREES: float = 5.0

    COLOR_JITTER: bool = True
    COLOR_JITTER_BRIGHTNESS: float = 0.2
    COLOR_JITTER_CONTRAST:   float = 0.2
    COLOR_JITTER_SATURATION: float = 0.0
    COLOR_JITTER_HUE:        float = 0.0

    RANDOM_ERASING: bool = True
    RANDOM_ERASING_P:         float = 0.2
    RANDOM_ERASING_SCALE_MIN: float = 0.02
    RANDOM_ERASING_SCALE_MAX: float = 0.1
    RANDOM_ERASING_RATIO_MIN: float = 0.3
    RANDOM_ERASING_RATIO_MAX: float = 3.3

    # NEW FLAG — USE_FINDINGS_ONLY:
    # MAIRA-2 generates FINDINGS only (not IMPRESSION). Using FINDINGS+IMPRESSION
    # from GT reports against MAIRA-2 FINDINGS-only outputs would be an unfair
    # comparison in the MedCLIP embedding space. This flag embeds FINDINGS only
    # on both sides, and filters the dataset to has_findings=True rows.
    USE_FINDINGS_ONLY: bool = True

    # COMBINE_SECTIONS is irrelevant when USE_FINDINGS_ONLY=True. Kept for
    # backward compatibility only.
    COMBINE_SECTIONS:  bool = False
    SECTION_SEPARATOR: str  = " [SEP] "
    TEXT_FALLBACK:     str  = "No findings reported."

    NUM_WORKERS:        int  = 2
    PIN_MEMORY:         bool = True
    PREFETCH_FACTOR:    int  = 2
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
    EVAL_EVERY_N_EPOCHS:  int = 3
    SAVE_EVERY_N_EPOCHS:  int = 10
    RETRIEVAL_K_VALUES: List[int] = field(default_factory=lambda: [1, 5, 10])


@dataclass
class SystemConfig:
    """System and hardware configuration."""
    DEVICE:           str  = "cuda" if torch.cuda.is_available() else "cpu"
    SEED:             int  = 42
    DETERMINISTIC:    bool = True
    BENCHMARK_CUDNN:  bool = True
    USE_TENSORBOARD:  bool = True
    LOG_INTERVAL:     int  = 10


class Config:
    """Master configuration class."""

    def __init__(self):
        self.paths      = PathConfig()
        self.model      = ModelConfig()
        self.training   = TrainingConfig()
        self.data       = DataConfig()
        self.evaluation = EvaluationConfig()
        self.system     = SystemConfig()
        self._set_seeds()
        self._configure_pytorch()

    def _set_seeds(self):
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
        print("=" * 80)
        print("CONFIGURATION SUMMARY")
        print("=" * 80)
        for name, section in [
            ("PATHS", self.paths), ("MODEL", self.model),
            ("TRAINING", self.training), ("DATA", self.data),
            ("EVALUATION", self.evaluation), ("SYSTEM", self.system),
        ]:
            print(f"\n{name}:")
            for k, v in section.__dict__.items():
                if not k.startswith('_'):
                    print(f"  {k}: {v}")
        print("=" * 80)

    def save(self, path: str):
        config_dict = {
            s: {k: v for k, v in getattr(self, s).__dict__.items() if not k.startswith('_')}
            for s in ('paths', 'model', 'training', 'data', 'evaluation', 'system')
        }
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"Configuration saved to {path}")


if __name__ == "__main__":
    config = Config()
    config.print_config()