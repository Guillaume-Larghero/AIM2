"""Bidirectional generation loop experiment using MAIRA-2 (I→T) + ChexGen (T→I).

ChexGen is a text-conditioned chest X-ray generator. The image-to-text half
(MAIRA-2) is unchanged from run_loop_experiment_maira2.py.

Loop:  Report -> ChexGen Image -> MAIRA-2 Report -> ... (N iterations)

Prompt alignment: ChexGen is driven by findings only. The Medical CLIP model
tracks semantic drift (text/image embeddings) per iteration.

Usage:
    python -m GENERATION.scripts.run_loop_experiment_chexgen \
        --study_id 50000014 --n_iterations 5 --start_from report \
        --visualize
"""

import os
os.environ["HF_HOME"] = "/n/groups/training/bmif203/AIM2/.cache"
import argparse
import datetime
import json
import logging
import random
import time
import traceback

from GENERATION.config.config import GenerationPipelineConfig
from GENERATION.pipeline.text_to_image import (
    TextToImageRetriever, ChexGenImageGenerator, TextToImagePipeline
)
from GENERATION.utils.utils import load_test_data
from RAG.config.config import RAGConfig
from RAG.embedder.embedder import CLIPEmbedder
from RAG.indexing.dual_indexer import DualFaissIndexer
from RAG.metadata.metadata_db import MetadataDB
from MAIRA.maira import MAIRAReportGenerator

from GENERATION.scripts.run_loop_experiment_maira2 import (
    SemanticLoopExperimentMAIRA2,
    LoopStep,
    LoopVisualizer,
    generate_html_report,
    normalize_study_id,
)

try:
    import h5py
except ImportError:
    h5py = None

logger = logging.getLogger(__name__)

DEFAULT_CHEXGEN_CONFIG = "/n/groups/training/bmif203/AIM2/ChexGen/configs/model.py"
DEFAULT_CHEXGEN_CKPT = (
    "/n/groups/training/bmif203/AIM2/ChexGen/weights/finetune_impression_512.pth"
)
DEFAULT_CHEXGEN_VAE = "stabilityai/sd-vae-ft-ema"
DEFAULT_CHEXGEN_T5_CACHE_DIR = "/n/groups/training/bmif203/AIM2/.cache/IF_"


class SemanticLoopExperimentChexGen(SemanticLoopExperimentMAIRA2):
    """Loop experiment with MAIRA-2 (I→T) and ChexGen (T→I)."""

    def _offload_maira(self):
        import torch
        self.maira.model = self.maira.model.to("cpu")
        torch.cuda.empty_cache()
        logger.info("  [mem] MAIRA-2 offloaded to CPU")

    def _restore_maira(self):
        self.maira.model = self.maira.model.to(self.maira.device)
        logger.info("  [mem] MAIRA-2 restored to GPU")

    def _step_t2i(self, findings, impression, iteration, output_dir,
                  save_image, seed) -> LoopStep:
        self._offload_maira()

        t0 = time.time()
        save_path = (
            os.path.join(output_dir, f"iter{iteration}_generated.png")
            if save_image else None
        )
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
        logger.info(f"  [T->I] ChexGen generated in {dt:.2f}s")
        return step

    def _step_i2t(self, image_path: str, iteration: int) -> LoopStep:
        self._restore_maira()
        return super()._step_i2t(image_path, iteration)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bidirectional Loop Experiment: MAIRA-2 (I→T) + ChexGen (T→I)"
    )
    parser.add_argument("--study_id", type=str, default=None)
    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--n_iterations", type=int, default=5)
    parser.add_argument("--start_from", type=str, choices=["report", "image"], default="report")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda",
                        help="PyTorch device, e.g. cuda, cuda:3, cpu")
    parser.add_argument("--chexgen_ckpt", type=str, default=DEFAULT_CHEXGEN_CKPT,
                        help="Path to the ChexGen checkpoint file")
    parser.add_argument("--chexgen_config", type=str, default=DEFAULT_CHEXGEN_CONFIG,
                        help="Path to the ChexGen config file")
    parser.add_argument("--chexgen_vae", type=str, default=DEFAULT_CHEXGEN_VAE,
                        help="Local or cached VAE model ID/path for ChexGen decoding")
    parser.add_argument("--chexgen_steps", type=int, default=100,
                        help="Number of diffusion inference steps")
    parser.add_argument("--chexgen_guidance", type=float, default=4.0,
                        help="Classifier-free guidance scale")
    parser.add_argument("--chexgen_t5_cache_dir", type=str, default=DEFAULT_CHEXGEN_T5_CACHE_DIR,
                        help="Parent directory containing the local T5 cache folder `t5-v1_1-xxl`")
    parser.add_argument("--chexgen_resolution", type=int, default=512,
                        help="Output image resolution (square)")
    parser.add_argument("--use_grounding", action="store_true",
                        help="Generate grounded reports (findings with bounding boxes)")
    parser.add_argument("--no_lateral", action="store_true",
                        help="Do not pass lateral image to MAIRA-2")
    parser.add_argument("--no_indication", action="store_true",
                        help="Do not pass indication text to MAIRA-2")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--visualize_with_training", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    config = GenerationPipelineConfig()
    rag_config = RAGConfig()
    if args.device:
        config.system.DEVICE = args.device

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = args.experiment_name or f"loop_chexgen_{timestamp}"
    output_dir = args.output_dir or os.path.join(
        config.paths.OUTPUT_DIR, "loop_experiments_chexgen", experiment_name
    )
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output: {output_dir}")

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

    logger.info("Loading MAIRA-2...")
    maira_generator = MAIRAReportGenerator(
        device=args.device,
        use_grounding=args.use_grounding,
    )

    logger.info("Initializing ChexGen generator | ckpt: %s", args.chexgen_ckpt)
    t2i_generator = ChexGenImageGenerator(
        config=config,
        chexgen_config_path=args.chexgen_config,
        checkpoint_path=args.chexgen_ckpt,
        vae_model=args.chexgen_vae,
        device=args.device,
        num_inference_steps=args.chexgen_steps,
        guidance_scale=args.chexgen_guidance,
        image_size=args.chexgen_resolution,
        t5_cache_dir=args.chexgen_t5_cache_dir,
    )
    t2i_retriever = TextToImageRetriever(
        config=config, clip_embedder=clip_embedder,
        text_indexer=dual_indexer.text_indexer, metadata_db=metadata_db,
    )
    t2i_pipeline = TextToImagePipeline(
        config=config, retriever=t2i_retriever, generator=t2i_generator
    )

    experiment = SemanticLoopExperimentChexGen(
        config=config,
        text_to_image_pipeline=t2i_pipeline,
        maira_generator=maira_generator,
        clip_embedder=clip_embedder,
        metadata_db=metadata_db,
        data_csv=config.paths.DATA_CSV,
        include_lateral=not args.no_lateral,
        include_indication=not args.no_indication,
    )

    test_df = load_test_data(config)
    test_df["study_id_normalized"] = test_df["study_id"].apply(
        lambda x: normalize_study_id(x, add_prefix=False)
    )

    if args.study_id:
        nid = normalize_study_id(args.study_id, add_prefix=False)
        study_ids = [nid]
        rows = test_df[test_df["study_id_normalized"] == nid]
        fallback_data = {nid: rows.iloc[0].to_dict()} if len(rows) > 0 else {}
    else:
        random.seed(args.seed)
        sample_df = test_df.sample(n=min(args.n_samples, len(test_df)), random_state=args.seed)
        study_ids = sample_df["study_id_normalized"].tolist()
        fallback_data = {
            row["study_id_normalized"]: row.to_dict()
            for _, row in sample_df.iterrows()
        }

    logger.info(f"Running on {len(study_ids)} sample(s): {study_ids}")

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

    if args.visualize or args.visualize_with_training:
        logger.info("Generating visualizations...")
        training_embeddings = None
        if args.visualize_with_training:
            try:
                with h5py.File(config.paths.EMBEDDINGS_H5, "r") as f:
                    key = "text_embeddings" if "text_embeddings" in f else "embeddings"
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
            LoopVisualizer(output_dir).visualize_multiple_traces(
                traces, training_embeddings
            )

    html_path = os.path.join(output_dir, f"{experiment_name}_report.html")
    generate_html_report(traces, html_path)

    results_path = os.path.join(output_dir, f"{experiment_name}_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "experiment_name": experiment_name,
            "i2t_model": "maira-2",
            "t2i_model": "chexgen",
            "chexgen_checkpoint": args.chexgen_ckpt,
            "chexgen_config": args.chexgen_config,
            "chexgen_vae": args.chexgen_vae,
            "chexgen_steps": args.chexgen_steps,
            "chexgen_guidance": args.chexgen_guidance,
            "chexgen_t5_cache_dir": args.chexgen_t5_cache_dir,
            "chexgen_resolution": args.chexgen_resolution,
            "n_samples": len(traces),
            "n_iterations": args.n_iterations,
            "start_from": args.start_from,
            "use_grounding": args.use_grounding,
            "traces": [t.to_dict() for t in traces],
            "timestamp": datetime.datetime.now().isoformat(),
        }, f, indent=2)

    logger.info(f"Processed {len(traces)} samples | Output: {output_dir}")
    for trace in traces:
        if "text_embedding_drift_cosine" in trace.metrics:
            d = trace.metrics["text_embedding_drift_cosine"]
            logger.info(f"  {trace.seed_study_id}: final text drift = {d[-1]:.4f}")


if __name__ == "__main__":
    main()
