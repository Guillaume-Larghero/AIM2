"""Bidirectional generation loop experiment using MAIRA-2 for image-to-text.

Replaces the RAG + LLM report generator with direct MAIRA-2 inference.
The text-to-image half of the loop (RoentGen-v2 diffusion) is unchanged.

Cycles: Report -> Image -> Report -> Image -> ... (N iterations)
Tracks semantic drift via CLIP embeddings, BLEU scores, and CheXpert label preservation.

Usage:
    python -m GENERATION.scripts.run_loop_experiment_maira2 \\
        --study_id 50000014 --n_iterations 5 --start_from report \\
        --use_grounding --visualize
"""

import os
os.environ['HF_HOME'] = '/n/groups/training/bmif203/AIM2/.cache'
import shutil
import argparse
import logging
import json
import datetime
import hashlib
import time
import random
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict, field
import traceback

import numpy as np
from PIL import Image
from transformers import AutoTokenizer
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
except ImportError:
    from collections import Counter
    import math

    class SmoothingFunction:
        @staticmethod
        def method1(*args, **kwargs):
            return None

    def sentence_bleu(references, hypothesis, smoothing_function=None):
        ref = references[0]
        if not ref or not hypothesis:
            return 0.0
        scores = []
        for n in range(1, 5):
            ref_ngrams = Counter(tuple(ref[i:i+n]) for i in range(len(ref)-n+1))
            hyp_ngrams = Counter(tuple(hypothesis[i:i+n]) for i in range(len(hypothesis)-n+1))
            clipped = sum(min(hyp_ngrams[ng], ref_ngrams[ng]) for ng in hyp_ngrams)
            total = max(sum(hyp_ngrams.values()), 1)
            scores.append((clipped + 1) / (total + 1))
        bp = min(1.0, math.exp(1 - len(ref) / max(len(hypothesis), 1)))
        return bp * math.exp(sum(math.log(s) for s in scores) / 4)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

try:
    from umap import UMAP
except ImportError:
    UMAP = None

try:
    import h5py
except ImportError:
    h5py = None

from GENERATION.config.config import GenerationPipelineConfig
from GENERATION.chexpert.extractor import extract_chexpert_from_reports
from GENERATION.pipeline.text_to_image import (
    TextToImageRetriever, DiffusionImageGenerator, TextToImagePipeline
)
from GENERATION.utils.utils import load_test_data
from RAG.config.config import RAGConfig
from RAG.embedder.embedder import CLIPEmbedder
from RAG.indexing.dual_indexer import DualFaissIndexer
from RAG.metadata.metadata_db import MetadataDB

# MAIRA-2 replaces the RAG + LLM generator
from MAIRA.maira import MAIRAReportGenerator, load_mimic_study

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

CHEXPERT_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
    'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
    'Pneumonia', 'Pneumothorax', 'Support Devices'
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_study_id(study_id: str, add_prefix: bool = False) -> str:
    study_id = str(study_id).strip()
    if add_prefix:
        if not study_id.startswith('s'):
            study_id = f's{study_id}'
    else:
        if study_id.startswith('s'):
            study_id = study_id[1:]
    return study_id


def get_positive_labels(chexpert_labels: List[float], label_names: List[str]) -> List[str]:
    if not chexpert_labels:
        return []
    return [label_names[i] for i, val in enumerate(chexpert_labels)
            if i < len(label_names) and val == 1.0]


# ---------------------------------------------------------------------------
# Data classes (identical to original loop experiment)
# ---------------------------------------------------------------------------

@dataclass
class LoopStep:
    iteration: int
    step_type: str  # "report", "image", or "ground_truth"
    content_path: Optional[str] = None
    findings: str = ""
    impression: str = ""
    text_embedding: Optional[List[float]] = None
    image_embedding: Optional[List[float]] = None
    retrieved_study_ids: List[str] = field(default_factory=list)
    retrieval_scores: List[float] = field(default_factory=list)
    chexpert_labels: Optional[List[float]] = None
    positive_labels: Optional[List[str]] = None
    generation_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LoopTrace:
    trace_id: str
    seed_study_id: str
    start_from: str
    n_iterations: int
    gt_findings: str = ""
    gt_impression: str = ""
    gt_image_path: str = ""
    gt_chexpert_labels: Optional[List[float]] = None
    steps: List[LoopStep] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['steps'] = [s.to_dict() if hasattr(s, 'to_dict') else s for s in self.steps]
        return d

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# ---------------------------------------------------------------------------
# Main experiment class
# ---------------------------------------------------------------------------

class SemanticLoopExperimentMAIRA2:
    """Loop experiment with MAIRA-2 replacing the RAG + LLM image-to-text step.

    The text-to-image direction (RoentGen-v2 diffusion) is unchanged.
    MAIRA-2 generates the findings section directly from the CXR image.

    Args:
        config: GenerationPipelineConfig.
        text_to_image_pipeline: TextToImagePipeline (unchanged from original).
        maira_generator: MAIRAReportGenerator instance.
        clip_embedder: CLIPEmbedder for computing embedding drift metrics.
        metadata_db: MetadataDB for resolving ground-truth data.
        data_csv: Path to processed_data.csv (for MAIRA lateral/indication lookup).
        include_lateral: Pass lateral view to MAIRA-2 when available.
        include_indication: Pass indication text to MAIRA-2 when available.
        chexpert_labels: List of CheXpert label names.
    """

    def __init__(
        self,
        config,
        text_to_image_pipeline,
        maira_generator: MAIRAReportGenerator,
        clip_embedder,
        metadata_db,
        data_csv: str,
        include_lateral: bool = True,
        include_indication: bool = True,
        chexpert_labels=None,
    ):
        self.config = config
        self.t2i_pipeline = text_to_image_pipeline
        self.maira = maira_generator
        self.clip_embedder = clip_embedder
        self.metadata_db = metadata_db
        self.data_csv = data_csv
        self.include_lateral = include_lateral
        self.include_indication = include_indication
        self.chexpert_labels = chexpert_labels or CHEXPERT_LABELS
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        logger.info("SemanticLoopExperimentMAIRA2 initialized")

    def _text_embedding(self, findings: str, impression: str) -> np.ndarray:
        text = f"{findings} {impression}".strip()
        return self.clip_embedder.encode_text_from_string(
            text=text, tokenizer=self.tokenizer, max_length=512
        )

    def _image_embedding(self, image_path: str) -> np.ndarray:
        return self.clip_embedder.encode_image_from_path(image_path, use_cache=False)

    def _extract_chexpert(self, findings: str, impression: str) -> List[float]:
        try:
            reports = [{'findings': findings, 'impression': impression}]
            labels = extract_chexpert_from_reports(
                reports, label_names=self.chexpert_labels, show_progress=False
            )
            return labels[0] if labels else [0.0] * 14
        except Exception as e:
            logger.warning(f"CheXpert extraction failed: {e}")
            return [0.0] * 14

    def run_loop(
        self,
        seed_study_id: str,
        n_iterations: int = 5,
        start_from: str = "report",
        output_dir: str = None,
        save_intermediates: bool = True,
        seed: int = 42,
        fallback_row: Optional[Dict] = None,
    ) -> LoopTrace:
        normalized_id = normalize_study_id(seed_study_id, add_prefix=False)

        if output_dir is None:
            output_dir = os.path.join(
                self.config.paths.OUTPUT_DIR, "loop_experiments_maira2",
                f"{normalized_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        os.makedirs(output_dir, exist_ok=True)

        # Resolve ground truth from metadata DB (with CSV fallback)
        gt = self.metadata_db.get_study(normalized_id)
        if gt is None:
            gt = self.metadata_db.get_study(f's{normalized_id}')
        if gt is None and fallback_row is not None:
            logger.info(f"Study {normalized_id} not in metadata DB, using fallback")
            chexpert = []
            for lbl in self.chexpert_labels:
                val = fallback_row.get(lbl, None)
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    chexpert.append(None)
                else:
                    chexpert.append(float(val))
            gt = {
                'findings': fallback_row.get('findings', '') or '',
                'impression': fallback_row.get('impression', '') or '',
                'image_path': fallback_row.get('image_path', '') or '',
                'chexpert_labels': chexpert
            }
        if gt is None:
            available = list(self.metadata_db.metadata.keys())[:5]
            raise ValueError(f"Study {seed_study_id} not found. Try: {available}")

        gt_chexpert = gt.get('chexpert_labels', [])
        gt_positive = get_positive_labels(gt_chexpert, self.chexpert_labels)

        trace = LoopTrace(
            trace_id=hashlib.md5(f"{normalized_id}_{time.time()}".encode()).hexdigest()[:12],
            seed_study_id=normalized_id,
            start_from=start_from,
            n_iterations=n_iterations,
            gt_findings=gt.get('findings', ''),
            gt_impression=gt.get('impression', ''),
            gt_image_path=gt.get('image_path', ''),
            gt_chexpert_labels=gt_chexpert,
            timestamp=datetime.datetime.now().isoformat(),
            config={
                'seed': seed, 'n_iterations': n_iterations,
                'start_from': start_from, 'gt_positive_labels': gt_positive,
                'i2t_model': 'maira-2',
            }
        )

        logger.info(
            f"Loop {trace.trace_id} | Study: {normalized_id} | "
            f"Labels: {gt_positive} | Start: {start_from} | Iters: {n_iterations}"
        )

        current_findings = trace.gt_findings
        current_impression = trace.gt_impression
        current_image_path = trace.gt_image_path

        # Copy the input image into the output directory so it is archived
        # alongside generated images and always accessible for visualization.
        if current_image_path and os.path.exists(current_image_path):
            ext = Path(current_image_path).suffix or ".jpg"
            local_input = os.path.join(output_dir, f"iter0_input{ext}")
            shutil.copy2(current_image_path, local_input)
            current_image_path = local_input
            trace.gt_image_path = local_input

        # Ground truth step
        initial = LoopStep(
            iteration=0, step_type="ground_truth",
            content_path=current_image_path,
            findings=current_findings, impression=current_impression,
            chexpert_labels=trace.gt_chexpert_labels
        )
        if current_image_path and os.path.exists(current_image_path):
            initial.image_embedding = self._image_embedding(current_image_path).tolist()
        initial.text_embedding = self._text_embedding(current_findings, current_impression).tolist()
        trace.steps.append(initial)

        # Iterate
        for it in range(1, n_iterations + 1):
            logger.info(f"--- Iteration {it}/{n_iterations} ---")

            if start_from == "report":
                img_step = self._step_t2i(
                    current_findings, current_impression,
                    it, output_dir, save_intermediates, seed + it
                )
                trace.steps.append(img_step)
                current_image_path = img_step.content_path

                rpt_step = self._step_i2t(current_image_path, it)
                trace.steps.append(rpt_step)
                current_findings = rpt_step.findings
                current_impression = rpt_step.impression
            else:
                rpt_step = self._step_i2t(current_image_path, it)
                trace.steps.append(rpt_step)
                current_findings = rpt_step.findings
                current_impression = rpt_step.impression

                img_step = self._step_t2i(
                    current_findings, current_impression,
                    it, output_dir, save_intermediates, seed + it
                )
                trace.steps.append(img_step)
                current_image_path = img_step.content_path

        trace.metrics = self._compute_metrics(trace)
        trace_path = os.path.join(output_dir, f"trace_{trace.trace_id}.json")
        trace.save(trace_path)
        logger.info(f"Saved trace to: {trace_path}")
        return trace

    # -- step helpers --

    def _step_t2i(self, findings, impression, iteration, output_dir,
                  save_image, seed) -> LoopStep:
        t0 = time.time()
        save_path = os.path.join(output_dir, f"iter{iteration}_generated.png") if save_image else None
        result = self.t2i_pipeline.generate(
            findings=findings, impression=impression,
            study_id=f"loop_iter{iteration}", conditioning_strategy="image_guided",
            top_k=5, strength=0.4, seed=seed, save_path=save_path
        )
        dt = time.time() - t0
        step = LoopStep(
            iteration=iteration, step_type="image",
            content_path=result.generated_image_path,
            findings=findings, impression=impression,
            retrieved_study_ids=result.retrieved_study_ids,
            retrieval_scores=result.retrieval_scores, generation_time=dt
        )
        step.text_embedding = self._text_embedding(findings, impression).tolist()
        if result.generated_image_path and os.path.exists(result.generated_image_path):
            step.image_embedding = self._image_embedding(result.generated_image_path).tolist()
        logger.info(f"  [T->I] Generated in {dt:.2f}s")
        return step

    def _step_i2t(self, image_path: str, iteration: int) -> LoopStep:
        """MAIRA-2 direct inference: image -> findings."""
        t0 = time.time()

        # For generated images (iteration > 0) no lateral/indication is available.
        # For the GT image we could optionally fetch them from the CSV, but MAIRA
        # works well with just the frontal image in this loop context.
        result = self.maira.generate_report(
            image_path=image_path,
            study_id=f"loop_iter{iteration}",
        )

        dt = time.time() - t0
        chexpert = self._extract_chexpert(result.findings, result.impression)
        positive = get_positive_labels(chexpert, self.chexpert_labels)

        step = LoopStep(
            iteration=iteration, step_type="report",
            content_path=image_path,
            findings=result.findings, impression=result.impression,
            retrieved_study_ids=[],  # no retrieval with MAIRA-2
            retrieval_scores=[],
            chexpert_labels=chexpert, positive_labels=positive,
            generation_time=dt
        )
        step.text_embedding = self._text_embedding(result.findings, result.impression).tolist()
        if image_path and os.path.exists(image_path):
            step.image_embedding = self._image_embedding(image_path).tolist()
        logger.info(f"  [I->T] MAIRA-2 in {dt:.2f}s | Labels: {positive}")
        return step

    # -- metrics (identical to original) --

    def _compute_metrics(self, trace: LoopTrace) -> Dict[str, Any]:
        metrics = {
            'n_steps': len(trace.steps),
            'total_generation_time': sum(s.generation_time for s in trace.steps),
            'i2t_model': 'maira-2',
        }
        gt_positive = get_positive_labels(trace.gt_chexpert_labels or [], self.chexpert_labels)
        metrics['gt_positive_labels'] = gt_positive

        text_embs = [np.array(s.text_embedding) for s in trace.steps if s.text_embedding]
        image_embs = [np.array(s.image_embedding) for s in trace.steps if s.image_embedding]

        if len(text_embs) > 1:
            gt = text_embs[0]
            metrics['text_embedding_drift_cosine'] = [
                float(1 - np.dot(e, gt) / (np.linalg.norm(e) * np.linalg.norm(gt) + 1e-8))
                for e in text_embs
            ]
            metrics['text_embedding_drift_l2'] = [
                float(np.linalg.norm(e - gt)) for e in text_embs
            ]
        if len(image_embs) > 1:
            gt = image_embs[0]
            metrics['image_embedding_drift_cosine'] = [
                float(1 - np.dot(e, gt) / (np.linalg.norm(e) * np.linalg.norm(gt) + 1e-8))
                for e in image_embs
            ]
            metrics['image_embedding_drift_l2'] = [
                float(np.linalg.norm(e - gt)) for e in image_embs
            ]

        try:
            smoother = SmoothingFunction().method1
            gt_tokens = f"{trace.gt_findings} {trace.gt_impression}".split()
            bleu_scores = []
            for step in trace.steps[1:]:
                if step.step_type == "report":
                    gen_tokens = f"{step.findings} {step.impression}".split()
                    bleu_scores.append(float(sentence_bleu(
                        [gt_tokens], gen_tokens, smoothing_function=smoother)))
            metrics['bleu_scores'] = bleu_scores
        except Exception as e:
            logger.warning(f"BLEU computation failed: {e}")

        if gt_positive:
            label_preservation = {label: [] for label in gt_positive}
            recall_list, precision_list, jaccard_list = [], [], []

            for step in trace.steps[1:]:
                if step.positive_labels is None:
                    continue
                pred = set(step.positive_labels)
                gt_set = set(gt_positive)
                for label in gt_positive:
                    label_preservation[label].append(1 if label in pred else 0)
                if gt_set:
                    recall_list.append(float(len(gt_set & pred) / len(gt_set)))
                precision_list.append(
                    float(len(gt_set & pred) / len(pred)) if pred else 0.0
                )
                union = gt_set | pred
                if union:
                    jaccard_list.append(float(len(gt_set & pred) / len(union)))

            metrics['chexpert_recall_over_iterations'] = recall_list
            metrics['chexpert_precision_over_iterations'] = precision_list
            metrics['chexpert_jaccard_over_iterations'] = jaccard_list
            metrics['per_label_preservation'] = label_preservation
            if label_preservation:
                avg = {l: float(np.mean(v)) if v else 0.0
                       for l, v in label_preservation.items()}
                metrics['label_avg_preservation'] = avg
                metrics['best_preserved_label'] = max(avg, key=avg.get)
                metrics['worst_preserved_label'] = min(avg, key=avg.get)

        return metrics


# ---------------------------------------------------------------------------
# Visualization (identical to original)
# ---------------------------------------------------------------------------

class LoopVisualizer:
    """UMAP trajectory and drift curve visualizations."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def visualize_trace(self, trace: LoopTrace,
                        training_embeddings: Optional[np.ndarray] = None):
        text_embs = [np.array(s.text_embedding) for s in trace.steps if s.text_embedding]
        image_embs = [np.array(s.image_embedding) for s in trace.steps if s.image_embedding]

        if len(text_embs) >= 2:
            self._plot_umap_trajectory(
                np.array(text_embs),
                f"Text Embedding Trajectory (MAIRA-2) - {trace.seed_study_id}",
                f"{trace.trace_id}_text_trajectory.png",
                training_embeddings, trace
            )
        if len(image_embs) >= 2:
            self._plot_umap_trajectory(
                np.array(image_embs),
                f"Image Embedding Trajectory - {trace.seed_study_id}",
                f"{trace.trace_id}_image_trajectory.png",
                training_embeddings, trace
            )
        self._plot_drift_curves(trace)
        self._create_content_gallery(trace)
        self._save_metrics_summary(trace)
        logger.info(f"Saved visualizations to: {self.output_dir}")

    def _plot_umap_trajectory(self, embeddings, title, filename,
                              training_embeddings=None, trace=None):
        try:
            fig, ax = plt.subplots(figsize=(14, 11))

            if training_embeddings is not None and len(training_embeddings) > 0:
                max_train = min(5000, len(training_embeddings))
                idx = np.random.choice(len(training_embeddings), max_train, replace=False)
                train_sample = training_embeddings[idx]
                combined = np.vstack([train_sample, embeddings])
                combined_2d = UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(combined)
                train_2d = combined_2d[:len(train_sample)]
                loop_2d = combined_2d[len(train_sample):]
                ax.scatter(train_2d[:, 0], train_2d[:, 1], c='lightgray', s=5, alpha=0.3, label='Training')
            elif len(embeddings) >= 5:
                loop_2d = UMAP(n_neighbors=min(5, len(embeddings)-1), min_dist=0.1, random_state=42).fit_transform(embeddings)
            else:
                loop_2d = PCA(n_components=2).fit_transform(embeddings)

            colors = plt.cm.viridis(np.linspace(0, 1, len(loop_2d)))
            for i in range(len(loop_2d)):
                ax.scatter(loop_2d[i, 0], loop_2d[i, 1], c=[colors[i]], s=150,
                           edgecolors='black', linewidth=1.5, zorder=10)
                ax.annotate(f'{i}', (loop_2d[i, 0], loop_2d[i, 1]),
                            fontsize=10, ha='center', va='center', color='white', fontweight='bold')
                if i < len(loop_2d) - 1:
                    ax.annotate('', xy=(loop_2d[i+1, 0], loop_2d[i+1, 1]),
                                xytext=(loop_2d[i, 0], loop_2d[i, 1]),
                                arrowprops=dict(arrowstyle='->', color=colors[i], lw=2))

            ax.scatter(loop_2d[0, 0], loop_2d[0, 1], c='red', s=300, marker='*',
                       edgecolors='black', linewidth=2, zorder=15, label='Ground Truth')

            if trace and trace.gt_chexpert_labels:
                pos = get_positive_labels(trace.gt_chexpert_labels, CHEXPERT_LABELS)
                if pos:
                    lbl = ", ".join(pos[:3])
                    if len(pos) > 3:
                        lbl += f" (+{len(pos)-3})"
                    ax.annotate(f'GT: {lbl}', xy=(loop_2d[0, 0], loop_2d[0, 1]),
                                xytext=(10, 20), textcoords='offset points', fontsize=9, color='red',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Dim 1')
            ax.set_ylabel('Dim 2')
            ax.legend(loc='best')
            sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, len(loop_2d)-1))
            sm.set_array([])
            plt.colorbar(sm, ax=ax).set_label('Iteration')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, filename), dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.warning(f"UMAP plot failed: {e}")

    def _plot_drift_curves(self, trace):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        plot_data = [
            (axes[0, 0], 'text_embedding_drift_cosine', 'b', 'Text Cosine Drift'),
            (axes[0, 1], 'text_embedding_drift_l2', 'b', 'Text L2 Drift'),
            (axes[1, 0], 'image_embedding_drift_cosine', 'g', 'Image Cosine Drift'),
            (axes[1, 1], 'image_embedding_drift_l2', 'g', 'Image L2 Drift'),
        ]
        for ax, key, color, title in plot_data:
            if key in trace.metrics:
                drift = trace.metrics[key]
                ax.plot(range(len(drift)), drift, f'{color}-o', linewidth=2, markersize=8)
                ax.set_xlabel('Step')
                ax.set_ylabel('Distance from GT')
                ax.set_title(title)
                ax.grid(True, alpha=0.3)

        plt.suptitle(f"Semantic Drift (MAIRA-2) - {trace.seed_study_id}",
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{trace.trace_id}_drift_curves.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()

    def _create_content_gallery(self, trace):
        image_steps = [s for s in trace.steps if s.content_path and os.path.exists(s.content_path)]
        if not image_steps:
            return
        n = min(len(image_steps), 5)
        fig, axes = plt.subplots(2, n, figsize=(4*n, 8))
        if n == 1:
            axes = np.array([[axes[0]], [axes[1]]])
        for idx, step in enumerate(image_steps[:n]):
            try:
                img = Image.open(step.content_path)
                axes[0, idx].imshow(img, cmap='gray')
                axes[0, idx].set_title(f"Iter {step.iteration}", fontweight='bold')
            except Exception:
                axes[0, idx].text(0.5, 0.5, 'N/A', ha='center', va='center')
            axes[0, idx].axis('off')
            txt = f"Findings:\n{step.findings[:150]}...\n\nImpression:\n{step.impression[:100]}..."
            axes[1, idx].text(0.5, 0.5, txt, ha='center', va='center', fontsize=8,
                              wrap=True, transform=axes[1, idx].transAxes)
            axes[1, idx].axis('off')
        plt.suptitle(f"Gallery (MAIRA-2) - {trace.seed_study_id}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{trace.trace_id}_gallery.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()

    def _save_metrics_summary(self, trace):
        path = os.path.join(self.output_dir, f"{trace.trace_id}_summary.txt")
        gt_positive = get_positive_labels(trace.gt_chexpert_labels or [], CHEXPERT_LABELS)
        with open(path, 'w') as f:
            f.write(f"Loop Experiment Summary (MAIRA-2)\n{'='*60}\n\n")
            f.write(f"Trace ID: {trace.trace_id}\n")
            f.write(f"Seed Study: {trace.seed_study_id}\n")
            f.write(f"Start From: {trace.start_from}\n")
            f.write(f"I2T Model: MAIRA-2 (microsoft/maira-2)\n")
            f.write(f"Iterations: {trace.n_iterations}\n")
            f.write(f"Steps: {len(trace.steps)}\n\n")
            f.write(f"GT Labels ({len(gt_positive)} positive):\n")
            for lbl in gt_positive:
                f.write(f"  + {lbl}\n")
            f.write(f"\nGT Findings: {trace.gt_findings[:200]}...\n")
            f.write(f"GT Impression: {trace.gt_impression[:200]}...\n\n")
            f.write(f"Metrics:\n{'-'*40}\n")
            for key, value in trace.metrics.items():
                if isinstance(value, list):
                    f.write(f"{key}: {[round(v, 4) if isinstance(v, float) else v for v in value]}\n")
                elif isinstance(value, float):
                    f.write(f"{key}: {value:.4f}\n")
                else:
                    f.write(f"{key}: {value}\n")

    def visualize_multiple_traces(self, traces: List[LoopTrace],
                                  training_embeddings: Optional[np.ndarray] = None):
        try:
            fig, axes = plt.subplots(1, 2, figsize=(16, 7))
            all_embs, trace_labels, iter_labels = [], [], []
            for ti, trace in enumerate(traces):
                for step in trace.steps:
                    if step.text_embedding:
                        all_embs.append(np.array(step.text_embedding))
                        trace_labels.append(ti)
                        iter_labels.append(step.iteration)
            if len(all_embs) < 5:
                return
            all_embs = np.array(all_embs)

            if training_embeddings is not None and len(training_embeddings) > 0:
                n = min(3000, len(training_embeddings))
                ts = training_embeddings[np.random.choice(len(training_embeddings), n, replace=False)]
                combined_2d = UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(
                    np.vstack([ts, all_embs]))
                t2d = combined_2d[:n]
                l2d = combined_2d[n:]
                for ax in axes:
                    ax.scatter(t2d[:, 0], t2d[:, 1], c='lightgray', s=3, alpha=0.2)
            else:
                l2d = UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(all_embs)

            colors = plt.cm.tab10(np.array(trace_labels) / max(trace_labels))
            axes[0].scatter(l2d[:, 0], l2d[:, 1], c=colors, s=50, alpha=0.7)
            idx = 0
            for ti, trace in enumerate(traces):
                ns = len([s for s in trace.steps if s.text_embedding])
                t2 = l2d[idx:idx+ns]
                for i in range(len(t2)-1):
                    axes[0].annotate('', xy=(t2[i+1, 0], t2[i+1, 1]), xytext=(t2[i, 0], t2[i, 1]),
                                     arrowprops=dict(arrowstyle='->', color=plt.cm.tab10(ti/10), lw=1.5, alpha=0.7))
                idx += ns
            axes[0].set_title('By Seed Sample', fontweight='bold')

            colors = plt.cm.viridis(np.array(iter_labels) / max(iter_labels))
            axes[1].scatter(l2d[:, 0], l2d[:, 1], c=colors, s=50, alpha=0.7)
            sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, max(iter_labels)))
            sm.set_array([])
            plt.colorbar(sm, ax=axes[1]).set_label('Iteration')
            axes[1].set_title('By Iteration', fontweight='bold')

            plt.suptitle(f"Multi-Trace / MAIRA-2 ({len(traces)} traces)",
                         fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "multi_trace_umap.png"),
                        dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.warning(f"Multi-trace visualization failed: {e}")


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def generate_html_report(traces: List[LoopTrace], output_path: str):
    n_traces = len(traces)
    n_iters = traces[0].n_iterations if traces else 0
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f'''<!DOCTYPE html>
<html><head><title>Loop Experiment (MAIRA-2)</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
.container {{ max-width: 1400px; margin: 0 auto; }}
h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
.trace-card {{ background: white; border-radius: 10px; padding: 20px; margin: 20px 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
.step-container {{ display: flex; flex-wrap: wrap; gap: 20px; margin: 20px 0; }}
.step {{ background: #ecf0f1; border-radius: 8px; padding: 15px; min-width: 200px; flex: 1; }}
.step-image {{ border: 2px solid #3498db; }} .step-report {{ border: 2px solid #27ae60; }} .step-gt {{ border: 2px solid #e74c3c; }}
.step img {{ max-width: 100%; border-radius: 5px; }}
.metrics-table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
.metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
.metrics-table th {{ background: #3498db; color: white; }}
.findings {{ font-size: 0.9em; color: #555; max-height: 150px; overflow-y: auto; }}
.badge {{ background: #3498db; color: white; padding: 3px 10px; border-radius: 15px; font-size: 0.8em; display: inline-block; margin-bottom: 10px; }}
.gt-badge {{ background: #e74c3c; }}
.summary {{ background: #3498db; color: white; padding: 20px; border-radius: 10px; margin: 20px 0; }}
.label-tag {{ background: #27ae60; color: white; padding: 2px 8px; border-radius: 10px; font-size: 0.75em; margin: 2px; display: inline-block; }}
.model-badge {{ background: #8e44ad; color: white; padding: 3px 10px; border-radius: 15px; font-size: 0.85em; }}
</style></head><body><div class="container">
<h1>Semantic Loop Experiment <span class="model-badge">MAIRA-2</span></h1>
<div class="summary"><h3>Summary</h3>
<p><b>Traces:</b> {n_traces} | <b>Iterations:</b> {n_iters} | <b>I2T Model:</b> MAIRA-2 | <b>Generated:</b> {ts}</p></div>
'''

    for trace in traces:
        gt_labels = trace.gt_chexpert_labels or []
        pos = get_positive_labels(gt_labels, CHEXPERT_LABELS)
        labels_html = ''.join(f'<span class="label-tag">{l}</span>' for l in pos)

        html += f'''<div class="trace-card"><h2>Trace: {trace.seed_study_id}</h2>
<p><b>ID:</b> {trace.trace_id} | <b>Start:</b> {trace.start_from}</p>
<p><b>GT Labels:</b> {labels_html or "None"}</p>
<h3>Generation Loop</h3><div class="step-container">
'''
        for step in trace.steps:
            cls = "step-gt" if step.iteration == 0 else ("step-image" if step.step_type == "image" else "step-report")
            bcls = "gt-badge" if step.iteration == 0 else ""
            label = 'Ground Truth' if step.iteration == 0 else f'Iter {step.iteration} - {step.step_type.title()}'
            html += f'<div class="step {cls}"><span class="badge {bcls}">{label}</span>\n'
            if step.content_path and os.path.exists(step.content_path):
                html += f'<img src="file://{step.content_path}" alt="Image">\n'
            if step.findings:
                html += f'<div class="findings"><b>Findings:</b> {step.findings[:200]}...</div>\n'
            if step.chexpert_labels and step.step_type == "report":
                sp = get_positive_labels(step.chexpert_labels, CHEXPERT_LABELS)
                if sp:
                    html += '<div class="findings"><b>Labels:</b> '
                    html += ''.join(f'<span class="label-tag">{l}</span>' for l in sp)
                    html += '</div>\n'
            html += '</div>\n'

        html += '</div><h3>Metrics</h3><table class="metrics-table"><tr><th>Metric</th><th>Values</th></tr>\n'
        for key, value in trace.metrics.items():
            if isinstance(value, list):
                vs = ', '.join(f'{v:.4f}' if isinstance(v, float) else str(v) for v in value)
            elif isinstance(value, float):
                vs = f'{value:.4f}'
            else:
                vs = str(value)
            html += f'<tr><td>{key}</td><td>{vs}</td></tr>\n'
        html += '</table></div>\n'

    html += '</div></body></html>\n'
    with open(output_path, 'w') as f:
        f.write(html)
    logger.info(f"Saved HTML report to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Bidirectional Loop Experiment with MAIRA-2 image-to-text"
    )
    parser.add_argument('--study_id', type=str, default=None)
    parser.add_argument('--n_samples', type=int, default=1)
    parser.add_argument('--n_iterations', type=int, default=5)
    parser.add_argument('--start_from', type=str, choices=['report', 'image'], default='report')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--experiment_name', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda',
                        help='PyTorch device, e.g. cuda, cuda:3, cpu')
    # MAIRA-2 options
    parser.add_argument('--use_grounding', action='store_true',
                        help='Generate grounded reports (findings with bounding boxes)')
    parser.add_argument('--no_lateral', action='store_true',
                        help='Do not pass lateral image to MAIRA-2')
    parser.add_argument('--no_indication', action='store_true',
                        help='Do not pass indication text to MAIRA-2')
    # Visualization
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--visualize_with_training', action='store_true')
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    config = GenerationPipelineConfig()
    rag_config = RAGConfig()
    if args.device:
        config.system.DEVICE = args.device

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = args.experiment_name or f"loop_maira2_{timestamp}"
    output_dir = args.output_dir or os.path.join(
        config.paths.OUTPUT_DIR, "loop_experiments_maira2", experiment_name
    )
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output: {output_dir}")

    # ---- Shared components (still needed for T2I and embedding metrics) ----
    logger.info("Loading FAISS indices...")
    dual_indexer = DualFaissIndexer(rag_config, embedding_dim=256)
    dual_indexer.load(config.paths.INDEX_DIR)
    logger.info(f"  Image index: {len(dual_indexer.image_indexer)} | Text index: {len(dual_indexer.text_indexer)}")

    logger.info("Loading metadata...")
    metadata_db = MetadataDB(rag_config)
    metadata_db.load(config.paths.METADATA_DB)
    logger.info(f"  Metadata DB: {len(metadata_db)} studies")

    logger.info("Initializing CLIP embedder...")
    clip_embedder = CLIPEmbedder(rag_config)

    # ---- MAIRA-2 (replaces RAG + LLM for image -> text) ----
    logger.info("Loading MAIRA-2...")
    maira_generator = MAIRAReportGenerator(
        device=args.device,
        use_grounding=args.use_grounding,
    )

    # ---- Text -> image pipeline (unchanged) ----
    t2i_retriever = TextToImageRetriever(
        config=config, clip_embedder=clip_embedder,
        text_indexer=dual_indexer.text_indexer, metadata_db=metadata_db
    )
    t2i_generator = DiffusionImageGenerator(
        config=config,
        model_type="/n/scratch/users/g/gul075/models/RoentGen-v2",
        device=args.device
    )
    t2i_pipeline = TextToImagePipeline(
        config=config, retriever=t2i_retriever, generator=t2i_generator
    )

    # ---- Loop experiment ----
    experiment = SemanticLoopExperimentMAIRA2(
        config=config,
        text_to_image_pipeline=t2i_pipeline,
        maira_generator=maira_generator,
        clip_embedder=clip_embedder,
        metadata_db=metadata_db,
        data_csv=config.paths.DATA_CSV,
        include_lateral=not args.no_lateral,
        include_indication=not args.no_indication,
    )

    # ---- Select samples ----
    test_df = load_test_data(config)
    available_in_db = set(metadata_db.metadata.keys())
    test_df['study_id_normalized'] = test_df['study_id'].apply(
        lambda x: normalize_study_id(x, add_prefix=False)
    )

    if args.study_id:
        nid = normalize_study_id(args.study_id, add_prefix=False)
        study_ids = [nid]
        rows = test_df[test_df['study_id_normalized'] == nid]
        fallback_data = {nid: rows.iloc[0].to_dict()} if len(rows) > 0 else {}
    else:
        random.seed(args.seed)
        sample_df = test_df.sample(n=min(args.n_samples, len(test_df)), random_state=args.seed)
        study_ids = sample_df['study_id_normalized'].tolist()
        fallback_data = {row['study_id_normalized']: row.to_dict() for _, row in sample_df.iterrows()}

    logger.info(f"Running on {len(study_ids)} sample(s): {study_ids}")

    # ---- Run ----
    traces = []
    for study_id in study_ids:
        try:
            trace = experiment.run_loop(
                seed_study_id=str(study_id),
                n_iterations=args.n_iterations,
                start_from=args.start_from,
                output_dir=os.path.join(output_dir, str(study_id)),
                save_intermediates=True,
                seed=args.seed,
                fallback_row=fallback_data.get(study_id),
            )
            traces.append(trace)
        except Exception as e:
            logger.error(f"Error processing {study_id}: {e}")
            traceback.print_exc()

    # ---- Visualizations ----
    if args.visualize or args.visualize_with_training:
        logger.info("Generating visualizations...")
        training_embeddings = None
        if args.visualize_with_training:
            try:
                with h5py.File(config.paths.EMBEDDINGS_H5, 'r') as f:
                    key = 'text_embeddings' if 'text_embeddings' in f else 'embeddings'
                    training_embeddings = f[key][:]
                logger.info(f"  Loaded {len(training_embeddings)} training embeddings")
            except Exception as e:
                logger.warning(f"Could not load training embeddings: {e}")

        for trace in traces:
            try:
                vis = LoopVisualizer(os.path.join(output_dir, trace.seed_study_id))
                vis.visualize_trace(trace, training_embeddings=training_embeddings)
            except Exception as e:
                logger.warning(f"Visualization failed for {trace.seed_study_id}: {e}")
        if len(traces) > 1:
            LoopVisualizer(output_dir).visualize_multiple_traces(traces, training_embeddings)

    # ---- Save results ----
    html_path = os.path.join(output_dir, f"{experiment_name}_report.html")
    generate_html_report(traces, html_path)

    results_path = os.path.join(output_dir, f"{experiment_name}_results.json")
    with open(results_path, 'w') as f:
        json.dump({
            'experiment_name': experiment_name,
            'i2t_model': 'maira-2',
            'n_samples': len(traces),
            'n_iterations': args.n_iterations,
            'start_from': args.start_from,
            'use_grounding': args.use_grounding,
            'traces': [t.to_dict() for t in traces],
            'timestamp': datetime.datetime.now().isoformat()
        }, f, indent=2)

    logger.info(f"Processed {len(traces)} samples | Output: {output_dir}")
    for trace in traces:
        if 'text_embedding_drift_cosine' in trace.metrics:
            d = trace.metrics['text_embedding_drift_cosine']
            logger.info(f"  {trace.seed_study_id}: final text drift = {d[-1]:.4f}")


if __name__ == "__main__":
    main()
