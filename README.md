# AIM2: Attractor Dynamics in Medical Vision-Language Generation

A comprehensive study of generative AI stability in medical imaging through the lens of dynamical systems theory, applying Lyapunov analysis to understand how iterative vision-language generation loops converge to stable states.

## Overview

This project investigates the phenomenon of **model collapse and attractor dynamics** in medical generative AI. Specifically, we study how iterative loops of chest X-ray (CXR) generation and medical report generation converge to a finite set of clinical "attractors"—stable patterns in the embedding space.

### Key Questions

1. **Do medical vision-language generation loops exhibit attractor dynamics?** How do iterative image→text→image loops behave?
2. **Can we characterize attractors using dynamical systems metrics?** What do Lyapunov exponents, fixed points, and basins of attraction tell us?
3. **What is the clinical significance?** Do attractors correspond to high-frequency pathological patterns or are they artifacts of the models?
4. **Can we predict convergence?** Given an initial image, can we predict which attractor it will converge to?
5. **How does augmentation help?** What impact does Retrieval-Augmented Generation (RAG) have on attractor dynamics?

## Project Structure

```
AIM2/
├── CLIP/                    # Vision-language embedding model training
│   ├── config/              # Model configuration files
│   ├── data/                # Data loaders and preprocessing
│   ├── model/               # CLIP-based architecture
│   ├── loss/                # Contrastive loss functions
│   ├── training/            # Training scripts
│   └── scripts/             # Utility scripts
├── GENERATION/              # Image and text generation pipelines
│   ├── pipeline/            # Generation loops and orchestration
│   ├── llm/                 # LLM-based report generation
│   ├── chexpert/            # CheXpert label extraction and validation
│   └── scripts/             # Generation experiment scripts
├── DIFFUSION/               # Diffusion model components
│   ├── train_lora.py        # LoRA-based fine-tuning
│   ├── config.yaml          # Diffusion model configuration
│   └── TECHNIQUES.md        # Technical notes on prompt engineering
├── ChexGen/                 # Generative foundation model for CXR
│   ├── configs/             # Model and training configurations
│   ├── radiffuser/          # Core diffusion transformer implementation
│   ├── scripts/             # Sampling and generation scripts
│   └── tools/               # Entry points (sample.py, train.py)
├── RAG/                     # Retrieval-Augmented Generation
│   └── requirements.txt     # Dependencies for RAG pipeline
├── MAIRA/                   # Medical AI Report Assistant
│   └── maira.py             # Core MAIRA implementation
├── dataset/                 # Data processing and preparation
│   ├── dataset.py           # Dataset loading and management
│   └── preprocess.py        # Preprocessing pipelines
├── Experiments/             # Experimental configurations and logs
├── Results/                 # Analysis outputs and figures
├── models/                  # Pre-trained model weights
└── logs/                    # Training and experiment logs
```

## Quick Start

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- 500GB+ disk space (for datasets and model weights)

### Installation

1. **Clone the repository:**
   ```bash
   cd /n/groups/training/bmif203/AIM2
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   # Core dependencies
   pip install torch>=2.0 torchvision transformers diffusers
   
   # Install component-specific requirements
   pip install -r CLIP/requirements.txt
   pip install -r GENERATION/requirements.txt
   pip install -r RAG/requirements.txt
   
   # For ChexGen
   cd ChexGen && pip install -r requirements.txt && cd ..
   ```

## Components

### 1. CLIP Module
**Vision-Language Embedding Learning**

Trains a contrastive model on MIMIC-CXR data to learn joint embeddings of chest X-rays and their corresponding medical reports.

**Key files:**
- `CLIP/training/` - Main training loop
- `CLIP/model/` - Architecture definitions
- `CLIP/data/` - Dataset loaders

**Example usage:**
```bash
python CLIP/training/train.py --config CLIP/config/default.yaml
```

### 2. GENERATION Module
**Iterative Generation Pipelines**

Orchestrates iterative loops of image generation and text generation, tracking embeddings and trajectories through the embedding space.

**Supported configurations:**
- **Config A:** Image → Report → Image → ... (basic loop)
- **Config B:** Image → [RAG] → Report → [RAG] → Image → ... (RAG-augmented)
- **Config C:** Image → Report → Image (single modality fixed)

**Key files:**
- `GENERATION/pipeline/` - Generation loop orchestration
- `GENERATION/llm/` - Report generation (using LLaVA, Med-Flamingo, etc.)
- `GENERATION/scripts/` - Experiment runners

**Example usage:**
```bash
python GENERATION/scripts/run_experiment.py --config GENERATION/config/basic_loop.yaml
```

### 3. DIFFUSION Module
**Diffusion-based CXR Synthesis**

Fine-tunes diffusion models (Stable Diffusion, DiT) for medical image generation with LoRA adapters and classifier-free guidance.

**Key files:**
- `DIFFUSION/train_lora.py` - LoRA fine-tuning script
- `DIFFUSION/config.yaml` - Model and training hyperparameters

### 4. ChexGen Module
**Foundation Model for Chest Radiography**

A pre-trained latent diffusion transformer (DiT) specifically designed for realistic chest X-ray synthesis with text conditioning.

**Features:**
- Text-conditioned generation
- Mask-based inpainting (in development)
- Bounding-box-based editing (in development)

**Usage:**
```bash
cd ChexGen
bash scripts/sample.sh
```

See [ChexGen/README.md](ChexGen/README.md) for detailed documentation.

### 5. RAG Module
**Retrieval-Augmented Generation**

Implements RAG systems that retrieve relevant medical examples from a knowledge base to improve report generation quality and consistency.

### 6. MAIRA Module
**Medical AI Report Assistant**

Specialized LLM-based module for generating clinically coherent medical reports given chest X-ray embeddings.

## Experimental Workflow

### Phase 1: Setup
1. Prepare MIMIC-CXR dataset (requires credentialed access)
2. Train or load pre-trained CLIP model
3. Set up diffusion models for image generation
4. Configure LLM for report generation

### Phase 2: Trajectory Generation
1. Select diverse initial CXR images (stratified by pathology)
2. Run generation loops (100+ iterations per trajectory)
3. Log embeddings at each iteration
4. Track convergence metrics

### Phase 3: Dynamical Systems Analysis
1. **Lyapunov Exponent Estimation:** Measure sensitivity to initial conditions
2. **Fixed Point Identification:** Cluster final states to find attractors
3. **Basin of Attraction Mapping:** Determine which initial conditions lead to each attractor
4. **Phase Portrait Construction:** Visualize the embedding space dynamics

### Phase 4: Clinical Interpretation
1. Extract reports/images at identified attractors
2. Have radiologists annotate clinical content
3. Correlate with ICD codes and pathology labels
4. Analyze how different diseases behave

## Key Metrics

### Convergence Metrics
- **Inter-step distance:** `d(t) = ||e(t+1) - e(t)||`
- **Convergence rate:** `λ = lim_{t→∞} (1/t) log(d(t)/d(0))`
- **Time to convergence:** Iterations until stabilization

### Dynamical Systems Metrics
- **Lyapunov exponents:** Maximum Lyapunov exponent (λ_max)
- **Fixed point stability:** Distance to nearest attractor
- **Basin entropy:** Complexity of basin of attraction boundaries

### Clinical Metrics
- **Attractor purity:** Fraction of samples at each attractor belonging to same pathology class
- **Diagnostic coverage:** What percentage of conditions converge to attractors
- **RAG impact:** How does augmentation change attractor structure

## Dataset Requirements

- **MIMIC-CXR:** 377,110 images with free-text reports (requires PhysioNet credentialed access)
- **MIMIC-IV:** Associated clinical notes and ICD codes
- **CheXpert:** Optional for label extraction (some images overlap with MIMIC-CXR)

### Getting Access

1. Complete [CITI training](https://about.citiprogram.org/)
2. Request credentialed access via [PhysioNet](https://physionet.org/settings/credentialing/)
3. Download datasets and place in `dataset/` directory

## Computational Requirements

| Resource | Requirement |
|----------|------------|
| **GPU** | A100 40GB or RTX 6000 Ada (preferred) |
| **RAM** | 128GB+ |
| **Storage** | 500GB+ (for data, embeddings, checkpoints) |
| **Compute Time** | ~200 GPU-hours for full experiment suite |

## Configuration

Configuration files are YAML-based and located in subdirectory `config/` folders:

- `CLIP/config/` - Model architecture and training hyperparameters
- `GENERATION/config/` - Pipeline specifications and loop configurations
- `DIFFUSION/config.yaml` - Diffusion model settings
- `ChexGen/configs/` - ChexGen model configurations

## Output Structure

- **Experiments/:** Configuration files for each experimental run
- **Results/:** Analysis outputs, visualizations, attractor catalogs
- **logs/:** Training logs, tensorboard events, metrics
- **models/:** Checkpoints and pre-trained weights

## Key References

### Foundational Papers

1. **Hintze et al. (2025).** "Autonomous language-image generation loops converge to generic visual motifs." *Patterns* (Cell Press).

2. **Shumailov et al. (2024).** "AI models collapse when trained on recursively generated data." *Nature*.

3. **Chemnitz et al. (2025).** "A Dynamical Systems Perspective on the Analysis of Neural Networks." *arXiv*.

### Medical Imaging & Foundation Models

4. **Bluethgen et al. (2024).** "A vision-language foundation model for the generation of realistic chest X-ray images." *Nature Biomedical Engineering*.

5. **Johnson et al. (2019).** "MIMIC-CXR: A de-identified publicly available database of chest radiographs with free-text reports." *Scientific Data*.

6. **Ark+ (2025).** Foundation model for chest radiography. *Nature*.

### Technical Methods

7. **Radford et al. (2021).** "Learning Transferable Visual Models From Natural Language Supervision." *ICML* (CLIP).

8. **Ho et al. (2020).** "Denoising Diffusion Probabilistic Models." *NeurIPS*.

9. **Liang et al. (2022).** "Mind the gap: understanding the modality gap in multi-modal contrastive representation learning." *NeurIPS*.

10. **Chou et al. (2024).** "Embedding Geometries of Contrastive Language-Image Pre-Training." *arXiv*.

## Contributing

This is a class project for BMIF203 Training. For questions or contributions:
- Check existing Experiments/ and Results/ for prior work
- Document new experiments in Experiments/ with configuration and notes
- Update Results/ with analysis outputs and figures

## License

This project incorporates code from multiple sources:
- **ChexGen:** [Apache 2.0 License](ChexGen/LICENSE)
- **CLIP training:** Adapted from OpenAI's CLIP
- **Diffusion components:** Adapted from Hugging Face Diffusers

See individual component directories for specific license information.

## Acknowledgments

This project builds on:
- [DiT (Diffusion Transformers)](https://github.com/facebookresearch/DiT)
- [PixArt-alpha](https://github.com/PixArt-alpha/PixArt-alpha)
- [OpenAI CLIP](https://github.com/openai/CLIP)
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [MIMIC-CXR Dataset](https://physionet.org/content/mimic-cxr-jpg/)
