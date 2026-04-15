"""Text-to-image generation using RAG retrieval + RoentGen-v2 diffusion model.

Pipeline: report text -> retrieve similar reports -> get reference image -> img2img generation.
"""

import os
import logging
import time
from typing import List, Dict, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer
from diffusers import DiffusionPipeline, StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler

logger = logging.getLogger(__name__)


def _remap_peft_lora_keys(lora_path: str, prefix: str = "transformer") -> dict:
    """Convert PEFT-format LoRA keys to diffusers-format keys.

    PEFT saves LoRA weights as "base_model.model.<layer>.lora_A.weight".
    Diffusers' load_lora_weights() expects "<prefix>.<layer>.lora_A.weight".

    Args:
        lora_path: Path to the safetensors file.
        prefix: The diffusers component prefix (e.g. "transformer").

    Returns:
        State dict with remapped keys, ready for pipe.load_lora_weights().
    """
    from safetensors.torch import load_file
    state_dict = load_file(lora_path)
    peft_prefix = "base_model.model."
    remapped = {}
    n_remapped = 0
    for k, v in state_dict.items():
        if k.startswith(peft_prefix):
            new_k = f"{prefix}.{k[len(peft_prefix):]}"
            remapped[new_k] = v
            n_remapped += 1
        else:
            remapped[k] = v
    logger.info(f"LoRA key remapping: {n_remapped}/{len(state_dict)} keys remapped to '{prefix}.*'")
    return remapped


DEFAULT_ROENTGEN_PATH = "/n/groups/training/bmif203/AIM2/models/RoentGen-v2" # TO CHANGE FOR ROSHAN
FALLBACK_HF_MODEL = "stanfordmimi/RoentGen-v2"


@dataclass
class TextToImageResult:
    """Container for text-to-image generation result."""
    study_id: str
    input_findings: str
    input_impression: str
    num_retrieved: int
    retrieved_study_ids: List[str]
    retrieval_scores: List[float]
    retrieved_image_paths: List[str]
    retrieved_reports: List[Dict[str, str]]
    generated_image_path: str
    conditioning_strategy: str
    reference_image_used: Optional[str] = None
    gt_image_path: Optional[str] = None
    generation_time: float = 0.0
    model_name: str = ""
    seed: int = 42
    strength: float = 0.4

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TextToImageRetriever:
    """Retrieves similar reports via text embeddings, returns associated images."""

    def __init__(self, config, clip_embedder, text_indexer, metadata_db, tokenizer=None):
        self.config = config
        self.clip_embedder = clip_embedder
        self.text_indexer = text_indexer
        self.metadata_db = metadata_db
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

        retrieval_config = getattr(config, 'retrieval', None)
        self.top_k = retrieval_config.TOP_K if retrieval_config else 5
        self.min_similarity = retrieval_config.MIN_SIMILARITY if retrieval_config else 0.0
        self.max_length = 512
        logger.info("TextToImageRetriever initialized")

    def retrieve_by_report(
        self, findings: str, impression: str, top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        if top_k is None:
            top_k = self.top_k

        combined_text = f"{findings} {impression}".strip()
        if not combined_text:
            return []

        text_embedding = self.clip_embedder.encode_text_from_string(
            text=combined_text, tokenizer=self.tokenizer, max_length=self.max_length
        )
        distances, study_ids = self.text_indexer.search(
            text_embedding, top_k=top_k, return_distances=True
        )

        results = []
        for study_id, distance in zip(study_ids, distances):
            if distance < self.min_similarity:
                continue
            metadata = self.metadata_db.get_study(study_id)
            if metadata is None:
                continue
            results.append({
                'study_id': study_id,
                'similarity_score': float(distance),
                'image_path': metadata.get('image_path', ''),
                'findings': metadata.get('findings', ''),
                'impression': metadata.get('impression', ''),
                'chexpert_labels': metadata.get('chexpert_labels', [])
            })

        logger.info(f"Retrieved {len(results)} similar cases for text query")
        return results


class DiffusionImageGenerator:
    """Medical image generator using RoentGen-v2 with img2img conditioning."""

    def __init__(self, config, model_type: str = "roentgen-v2", device: str = "cuda"):
        self.config = config
        self.model_type = model_type
        self.device = device
        self.num_inference_steps = 50
        self.guidance_scale = 7.5
        self.image_size = 512
        self.default_strength = 0.4
        self.seed = getattr(getattr(config, 'system', None), 'SEED', 42)
        self.pipe = None
        self.img2img_pipe = None
        logger.info(f"DiffusionImageGenerator initialized (model: {model_type})")

    def load_model(self, model_id: Optional[str] = None):
        if model_id is None:
            model_id = self.model_type
        if os.path.isdir(model_id):
            load_path = model_id
        elif os.path.isdir(DEFAULT_ROENTGEN_PATH):
            load_path = DEFAULT_ROENTGEN_PATH
        else:
            load_path = FALLBACK_HF_MODEL

        logger.info(f"Loading RoentGen-v2 from: {load_path}")
        self.pipe = DiffusionPipeline.from_pretrained(
            load_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            local_files_only=os.path.isdir(load_path)
        )
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to(self.device)

        if self.device == "cuda":
            self.pipe.enable_attention_slicing()
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

        self.img2img_pipe = StableDiffusionImg2ImgPipeline(
            vae=self.pipe.vae, text_encoder=self.pipe.text_encoder,
            tokenizer=self.pipe.tokenizer, unet=self.pipe.unet,
            scheduler=self.pipe.scheduler, safety_checker=None,
            feature_extractor=None, requires_safety_checker=False
        ).to(self.device)

        logger.info("RoentGen-v2 loaded")

    def _build_medical_prompt(self, findings: str, impression: str) -> Tuple[str, str]:
        parts = []
        if impression and impression.strip():
            parts.append(impression.strip()[:300])
        if findings and findings.strip():
            parts.append(findings.strip()[:200])
        prompt = " ".join(parts) if parts else "Normal chest radiograph. No acute cardiopulmonary abnormality."
        return prompt, "blurry, low quality, artifacts, text overlay"

    def _prepare_reference_image(self, image_path: str) -> Image.Image:
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img.resize((self.image_size, self.image_size), Image.LANCZOS)

    def generate_text_only(self, findings: str, impression: str,
                           seed: Optional[int] = None) -> Image.Image:
        if self.pipe is None:
            self.load_model()
        prompt, negative_prompt = self._build_medical_prompt(findings, impression)
        generator = torch.Generator(device=self.device).manual_seed(seed or self.seed)

        with torch.autocast(self.device):
            result = self.pipe(
                prompt=prompt, negative_prompt=negative_prompt,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                height=self.image_size, width=self.image_size,
                generator=generator
            )
        return result.images[0].convert('L')

    def generate_image_guided(self, findings: str, impression: str,
                              reference_image: Union[str, Image.Image],
                              strength: Optional[float] = None,
                              seed: Optional[int] = None) -> Image.Image:
        if self.img2img_pipe is None:
            self.load_model()
        if strength is None:
            strength = self.default_strength

        if isinstance(reference_image, str):
            ref_img = self._prepare_reference_image(reference_image)
        else:
            ref_img = reference_image.resize((self.image_size, self.image_size), Image.LANCZOS)
            if ref_img.mode != 'RGB':
                ref_img = ref_img.convert('RGB')

        prompt, negative_prompt = self._build_medical_prompt(findings, impression)
        generator = torch.Generator(device=self.device).manual_seed(seed or self.seed)

        with torch.autocast(self.device):
            result = self.img2img_pipe(
                prompt=prompt, negative_prompt=negative_prompt,
                image=ref_img, strength=strength,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale, generator=generator
            )
        return result.images[0].convert('L')

    def __repr__(self):
        return f"DiffusionImageGenerator(model={self.model_type}, device={self.device})"


class TextToImagePipeline:
    """Complete text-to-image pipeline: RAG retrieval + RoentGen-v2 generation."""

    def __init__(self, config, retriever: TextToImageRetriever, generator: DiffusionImageGenerator):
        self.config = config
        self.retriever = retriever
        self.generator = generator
        logger.info("TextToImagePipeline initialized")

    def _select_best_reference(self, retrieved_cases: List[Dict[str, Any]]) -> Optional[str]:
        for case in retrieved_cases:
            image_path = case.get('image_path', '')
            if image_path and os.path.exists(image_path):
                return image_path
        return None

    def generate(
        self, findings: str, impression: str,
        study_id: Optional[str] = None, gt_image_path: Optional[str] = None,
        conditioning_strategy: str = "image_guided", top_k: int = 5,
        strength: Optional[float] = None, seed: Optional[int] = None,
        save_path: Optional[str] = None
    ) -> TextToImageResult:
        start_time = time.time()

        retrieved_cases = self.retriever.retrieve_by_report(
            findings=findings, impression=impression, top_k=top_k
        )
        best_reference = self._select_best_reference(retrieved_cases)

        actual_strength = 0.0
        reference_used = None

        if conditioning_strategy == "text_only" or best_reference is None:
            generated_image = self.generator.generate_text_only(
                findings=findings, impression=impression, seed=seed
            )
        elif conditioning_strategy == "image_guided":
            actual_strength = strength if strength is not None else self.generator.default_strength
            generated_image = self.generator.generate_image_guided(
                findings=findings, impression=impression,
                reference_image=best_reference, strength=actual_strength, seed=seed
            )
            reference_used = best_reference
        else:
            raise ValueError(f"Unknown conditioning strategy: {conditioning_strategy}")

        if save_path is None:
            save_dir = getattr(getattr(self.config, 'paths', None), 'OUTPUT_DIR', '.')
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"generated_{study_id or 'image'}.png")

        generated_image.save(save_path)
        generation_time = time.time() - start_time

        return TextToImageResult(
            study_id=study_id or "unknown",
            input_findings=findings, input_impression=impression,
            num_retrieved=len(retrieved_cases),
            retrieved_study_ids=[c['study_id'] for c in retrieved_cases],
            retrieval_scores=[c['similarity_score'] for c in retrieved_cases],
            retrieved_image_paths=[c['image_path'] for c in retrieved_cases if c.get('image_path')],
            retrieved_reports=[{'findings': c['findings'], 'impression': c['impression']} for c in retrieved_cases],
            generated_image_path=save_path,
            conditioning_strategy=conditioning_strategy,
            reference_image_used=reference_used,
            gt_image_path=gt_image_path,
            generation_time=generation_time,
            model_name="RoentGen-v2",
            seed=seed or self.generator.seed,
            strength=actual_strength
        )


class SD35LoRAImageGenerator:
    """Medical image generator using SD3.5 Medium fine-tuned with LoRA on MIMIC-CXR.

    Drop-in replacement for DiffusionImageGenerator in any TextToImagePipeline.
    The LoRA was trained text-only (MMDiT transformer only; text encoders frozen),
    so generate_image_guided falls back to text-only with a warning.

    Prompt format must match training exactly:
        "FINDINGS: <findings> IMPRESSION: <impression>"
    """

    DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
    DEFAULT_LORA_PATH = (
        "/n/groups/training/bmif203/AIM2/Experiments/finetune_lora_SD35/"
        "final_lora_weights/pytorch_lora_weights.safetensors"
    )

    def __init__(
        self,
        config,
        lora_weights_path: Optional[str] = None,
        model_id: Optional[str] = None,
        device: str = "cuda",
        num_inference_steps: int = 28,
        guidance_scale: float = 7.0,
        image_size: int = 768,
    ):
        self.config = config
        self.model_id = model_id or self.DEFAULT_MODEL_ID
        self.lora_weights_path = lora_weights_path or self.DEFAULT_LORA_PATH
        self.device = device
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.image_size = image_size
        self.default_strength = 0.0  # text-only; kept for interface compatibility
        self.seed = getattr(getattr(config, 'system', None), 'SEED', 42)
        self.pipe = None
        logger.info(
            f"SD35LoRAImageGenerator initialized | model: {self.model_id} | "
            f"lora: {self.lora_weights_path}"
        )

    def load_model(self):
        from diffusers import StableDiffusion3Pipeline
        # On HPC compute nodes there is no outbound internet — use the local cache only.
        # If the model is not yet cached, run the download command from a login node first.
        logger.info(f"Loading SD3.5 Medium from: {self.model_id}")
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            local_files_only=True,
        )
        self.pipe = self.pipe.to(self.device)

        if self.lora_weights_path and os.path.exists(self.lora_weights_path):
            logger.info(f"Loading LoRA weights from: {self.lora_weights_path}")
            self.pipe.load_lora_weights(
                _remap_peft_lora_keys(self.lora_weights_path, prefix="transformer")
            )
        else:
            logger.warning(
                f"LoRA weights not found at {self.lora_weights_path} — running base SD3.5"
            )

        if self.device == "cuda":
            self.pipe.enable_attention_slicing()
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

        logger.info("SD3.5 LoRA loaded")

    def _build_prompt(self, findings: str, impression: str) -> str:
        """Matches training prompt format exactly."""
        f = (findings or "").strip()
        i = (impression or "").strip()
        return f"FINDINGS: {f} IMPRESSION: {i}"

    def generate_text_only(
        self,
        findings: str,
        impression: str,
        seed: Optional[int] = None,
    ) -> Image.Image:
        if self.pipe is None:
            self.load_model()
        prompt = self._build_prompt(findings, impression)
        generator = torch.Generator(device=self.device).manual_seed(seed or self.seed)
        with torch.autocast(self.device):
            result = self.pipe(
                prompt=prompt,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                height=self.image_size,
                width=self.image_size,
                generator=generator,
            )
        return result.images[0].convert('L')

    def generate_image_guided(
        self,
        findings: str,
        impression: str,
        reference_image: Union[str, Image.Image],
        strength: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """LoRA was trained text-only — reference image is ignored."""
        logger.warning(
            "SD35LoRAImageGenerator was trained text-only; reference image ignored."
        )
        return self.generate_text_only(findings, impression, seed=seed)

    def __repr__(self):
        return f"SD35LoRAImageGenerator(model={self.model_id}, device={self.device})"


class Flux2LoRAImageGenerator:
    """Medical image generator using FLUX.2-dev fine-tuned with LoRA on MIMIC-CXR.

    Drop-in replacement for DiffusionImageGenerator in any TextToImagePipeline.

    Training details (from finetune_flux2_lora_cxr.py):
      - Single text encoder: Mistral Small 3.1 (no CLIP-L / T5-XXL)
      - Prompt format: "FINDINGS: <findings>" — impression deliberately excluded
        (MAIRA-2 generates findings only; training/inference match is exact)
      - Resolution: 512×512, bf16, guidance_scale=3.5 at inference
      - LoRA rank 32 applied to Flux2Transformer2DModel

    Requires diffusers main branch: pip install git+https://github.com/huggingface/diffusers
    """

    DEFAULT_MODEL_ID = "black-forest-labs/FLUX.2-dev"
    DEFAULT_LORA_PATH = (
        "/n/groups/training/bmif203/AIM2/Experiments/finetune_lora/"
        "outputs/checkpoint-004000/pytorch_lora_weights.safetensors"
    )

    def __init__(
        self,
        config,
        lora_weights_path: Optional[str] = None,
        model_id: Optional[str] = None,
        device: str = "cuda",
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        image_size: int = 512,
    ):
        self.config = config
        self.model_id = model_id or self.DEFAULT_MODEL_ID
        self.lora_weights_path = lora_weights_path or self.DEFAULT_LORA_PATH
        self.device = device
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.image_size = image_size
        self.default_strength = 0.0  # text-only; kept for interface compatibility
        self.seed = getattr(getattr(config, 'system', None), 'SEED', 42)
        self.pipe = None
        logger.info(
            f"Flux2LoRAImageGenerator initialized | model: {self.model_id} | "
            f"lora: {self.lora_weights_path}"
        )

    def load_model(self):
        try:
            from diffusers import Flux2Pipeline
        except ImportError:
            raise ImportError(
                "Flux2Pipeline not found. FLUX.2-dev requires diffusers from the main branch.\n"
                "Run: pip uninstall diffusers -y && "
                "pip install git+https://github.com/huggingface/diffusers -U"
            )
        # Compute nodes have no outbound internet — model must be pre-cached from a login node.
        logger.info(f"Loading FLUX.2-dev from: {self.model_id}")
        self.pipe = Flux2Pipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,  # trained with bf16
            local_files_only=True,
        )

        if self.lora_weights_path and os.path.exists(self.lora_weights_path):
            logger.info(f"Loading LoRA weights from: {self.lora_weights_path}")
            self.pipe.load_lora_weights(
                _remap_peft_lora_keys(self.lora_weights_path, prefix="transformer")
            )
        else:
            logger.warning(
                f"LoRA weights not found at {self.lora_weights_path} — running base FLUX.2"
            )

        if self.device == "cuda":
            # FLUX.2 transformer alone is ~64 GB. Use component-level CPU offloading so
            # only one sub-model (VAE / transformer / text encoder) occupies GPU at a time.
            # This avoids OOM when MAIRA-2 (~24 GB) is also present in the process.
            gpu_id = 0
            if self.device.startswith("cuda:"):
                gpu_id = int(self.device.split(":")[1])
            self.pipe.enable_model_cpu_offload(gpu_id=gpu_id)
        else:
            self.pipe = self.pipe.to(self.device)

        logger.info("FLUX.2 LoRA loaded")

    def _build_prompt(self, findings: str, impression: str) -> str:
        """Matches training prompt format exactly — findings only, impression excluded."""
        f = (findings or "").strip()
        if not f:
            return "No findings reported."
        return f"FINDINGS: {f}"

    def generate_text_only(
        self,
        findings: str,
        impression: str,
        seed: Optional[int] = None,
    ) -> Image.Image:
        if self.pipe is None:
            self.load_model()
        prompt = self._build_prompt(findings, impression)
        generator = torch.Generator(device=self.device).manual_seed(seed or self.seed)
        with torch.autocast(self.device, dtype=torch.bfloat16):
            result = self.pipe(
                prompt=prompt,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                height=self.image_size,
                width=self.image_size,
                generator=generator,
            )
        return result.images[0].convert('L')

    def generate_image_guided(
        self,
        findings: str,
        impression: str,
        reference_image: Union[str, Image.Image],
        strength: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """LoRA was trained text-only — reference image is ignored."""
        logger.warning(
            "Flux2LoRAImageGenerator was trained text-only; reference image ignored."
        )
        return self.generate_text_only(findings, impression, seed=seed)

    def __repr__(self):
        return f"Flux2LoRAImageGenerator(model={self.model_id}, device={self.device})"