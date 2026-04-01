"""Bidirectional generation loop experiment using MAIRA-2 (I→T) + SD3.5 LoRA (T→I).

Replaces RoentGen-v2 with the SD3.5 Medium model fine-tuned via LoRA on MIMIC-CXR.
The image-to-text half (MAIRA-2) is unchanged from run_loop_experiment_maira2.py.

Loop:  Report -> SD3.5-LoRA Image -> MAIRA-2 Report -> ... (N iterations)

Tracks semantic drift via CLIP embeddings, BLEU scores, and CheXpert label preservation.

Usage:
    python -m GENERATION.scripts.run_loop_experiment_sd35 \\
        --study_id 50000014 --n_iterations 5 --start_from report \\
        --lora_weights /n/groups/training/bmif203/AIM2/Experiments/finetune_lora/outputs/final_lora_weights/pytorch_lora_weights.safetensors \\
        --visualize
"""

import os
os.environ['HF_HOME'] = '/n/groups/training/bmif203/AIM2/.cache'
import argparse
import datetime
import json
import logging
import random
import time
import traceback
from pathlib import Path

import numpy as np

from GENERATION.config.config import GenerationPipelineConfig
from GENERATION.pipeline.text_to_image import (
    TextToImageRetriever, SD35LoRAImageGenerator, TextToImagePipeline
)
from GENERATION.utils.utils import load_test_data
from RAG.config.config import RAGConfig
from RAG.embedder.embedder import CLIPEmbedder
from RAG.indexing.dual_indexer import DualFaissIndexer
from RAG.metadata.metadata_db import MetadataDB
from MAIRA.maira import MAIRAReportGenerator

# Import experiment infrastructure from the MAIRA-2 script
from GENERATION.scripts.run_loop_experiment_maira2 import (
    SemanticLoopExperimentMAIRA2,
    LoopStep,
    LoopTrace,
    LoopVisualizer,
    generate_html_report,
    normalize_study_id,
    CHEXPERT_LABELS,
)

try:
    import h5py
except ImportError:
    h5py = None

logger = logging.getLogger(__name__)

SD35_DEFAULT_LORA_PATH = (
    "/n/groups/training/bmif203/AIM2/Experiments/finetune_lora/"
    "outputs/final_lora_weights/pytorch_lora_weights.safetensors"
)


class SemanticLoopExperimentSD35(SemanticLoopExperimentMAIRA2):
    """Loop experiment with MAIRA-2 (I→T) and SD3.5 LoRA (T→I).

    Inherits all metrics, visualization hooks, and loop logic from
    SemanticLoopExperimentMAIRA2. Only _step_t2i is overridden to use
    text-only conditioning (matching how the LoRA was trained).
    """

    def _step_t2i(self, findings, impression, iteration, output_dir,
                  save_image, seed) -> LoopStep:
        t0 = time.time()
        save_path = (
            os.path.join(output_dir, f"iter{iteration}_generated.png")
            if save_image else None
        )
        # SD3.5 LoRA is text-only — no reference image retrieval needed for generation.
        # The retriever still runs so retrieved_study_ids are logged in the trace.
        result = self.t2i_pipeline.generate(
            findings=findings,
            impression=impression,
            study_id=f"loop_iter{iteration}",
            conditioning_strategy="text_only",
            seed=seed,
            save_path=save_path,
        )
        dt = time.time() - t0
        step = LoopStep(
            iteration=iteration, step_type="image",
            content_path=result.generated_image_path,
            findings=findings, impression=impression,
            retrieved_study_ids=result.retrieved_study_ids,
            retrieval_scores=result.retrieval_scores,
            generation_time=dt,
        )
        step.text_embedding = self._text_embedding(findings, impression).tolist()
        if result.generated_image_path and os.path.exists(result.generated_image_path):
            step.image_embedding = self._image_embedding(result.generated_image_path).tolist()
        logger.info(f"  [T->I] SD3.5 LoRA generated in {dt:.2f}s")
        return step


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bidirectional Loop Experiment: MAIRA-2 (I→T) + SD3.5 LoRA (T→I)"
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
    # SD3.5 LoRA options
    parser.add_argument('--lora_weights', type=str, default=SD35_DEFAULT_LORA_PATH,
                        help='Path to pytorch_lora_weights.safetensors')
    parser.add_argument('--sd35_model_id', type=str,
                        default='stabilityai/stable-diffusion-3.5-medium',
                        help='SD3.5 base model ID or local path')
    parser.add_argument('--sd35_steps', type=int, default=28,
                        help='Number of diffusion inference steps')
    parser.add_argument('--sd35_guidance', type=float, default=7.0,
                        help='Classifier-free guidance scale')
    parser.add_argument('--sd35_resolution', type=int, default=768,
                        help='Output image resolution (square)')
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
    experiment_name = args.experiment_name or f"loop_sd35_{timestamp}"
    output_dir = args.output_dir or os.path.join(
        config.paths.OUTPUT_DIR, "loop_experiments_sd35", experiment_name
    )
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output: {output_dir}")

    # ---- Shared components ----
    logger.info("Loading FAISS indices...")
    dual_indexer = DualFaissIndexer(rag_config, embedding_dim=256)
    dual_indexer.load(config.paths.INDEX_DIR)
    logger.info(
        f"  Image index: {len(dual_indexer.image_indexer)} | "
        f"Text index: {len(dual_indexer.text_indexer)}"
    )

    logger.info("Loading metadata...")
    metadata_db = MetadataDB(rag_config)
    metadata_db.load(config.paths.METADATA_DB)
    logger.info(f"  Metadata DB: {len(metadata_db)} studies")

    logger.info("Initializing CLIP embedder...")
    clip_embedder = CLIPEmbedder(rag_config)

    # ---- MAIRA-2 (I→T) ----
    logger.info("Loading MAIRA-2...")
    maira_generator = MAIRAReportGenerator(
        device=args.device,
        use_grounding=args.use_grounding,
    )

    # ---- SD3.5 LoRA (T→I) ----
    logger.info(f"Initializing SD3.5 LoRA generator | lora: {args.lora_weights}")
    t2i_generator = SD35LoRAImageGenerator(
        config=config,
        lora_weights_path=args.lora_weights,
        model_id=args.sd35_model_id,
        device=args.device,
        num_inference_steps=args.sd35_steps,
        guidance_scale=args.sd35_guidance,
        image_size=args.sd35_resolution,
    )
    t2i_retriever = TextToImageRetriever(
        config=config, clip_embedder=clip_embedder,
        text_indexer=dual_indexer.text_indexer, metadata_db=metadata_db,
    )
    t2i_pipeline = TextToImagePipeline(
        config=config, retriever=t2i_retriever, generator=t2i_generator
    )

    # ---- Loop experiment ----
    experiment = SemanticLoopExperimentSD35(
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
        fallback_data = {
            row['study_id_normalized']: row.to_dict()
            for _, row in sample_df.iterrows()
        }

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
            't2i_model': 'sd35-lora',
            'lora_weights': args.lora_weights,
            'sd35_model_id': args.sd35_model_id,
            'sd35_steps': args.sd35_steps,
            'sd35_guidance': args.sd35_guidance,
            'sd35_resolution': args.sd35_resolution,
            'n_samples': len(traces),
            'n_iterations': args.n_iterations,
            'start_from': args.start_from,
            'use_grounding': args.use_grounding,
            'traces': [t.to_dict() for t in traces],
            'timestamp': datetime.datetime.now().isoformat(),
        }, f, indent=2)

    logger.info(f"Processed {len(traces)} samples | Output: {output_dir}")
    for trace in traces:
        if 'text_embedding_drift_cosine' in trace.metrics:
            d = trace.metrics['text_embedding_drift_cosine']
            logger.info(f"  {trace.seed_study_id}: final text drift = {d[-1]:.4f}")


if __name__ == "__main__":
    main()
