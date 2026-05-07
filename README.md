# Attractor

Code for training the MedCLIP-XR-512 observer and running the MAIRA-2 <-> ChexGen
attractor-loop experiments.

## What Is Here

```text
CLIP/                         Train and evaluate the MedCLIP-XR-512 model
GENERATION/chexpert/          CheXpert label extraction helpers
Experiments/attractor_loop/   Loop generation, analysis, and figure scripts
```

## Setup

Run from the repository root:

```bash
export PYTHONPATH="$PWD:${PYTHONPATH}"

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

You also need the ChexGen and MAIRA-2 dependencies/weights available locally.

Before running, update the local path constants near the top of the scripts if
needed, especially:

```text
BASE_DIR
DATA_CSV
MEDCLIP_CKPT
CHEXGEN_DIR
CHEXGEN_CKPT
HF_HOME
```

The expected CSV is `processed_data/processed_data.csv` and should include:

```text
split, study_id, subject_id, dicom_id, image_path, findings, has_findings
```

CheXpert label columns are optional but needed for the pathology-profile
analyses.

## 1. Train The CLIP Model

Edit `CLIP/config/config.py` so `PathConfig.BASE_DIR` points to the directory
containing `processed_data/processed_data.csv`.

```bash
python3 -m CLIP.scripts.train_CLIP \
  --epochs 100 \
  --batch_size 8
```

The main checkpoint is written to:

```text
CLIP/outputs/checkpoints/best_model.pth
```

Optional evaluation:

```bash
python3 -m CLIP.scripts.evaluate_embedding \
  --checkpoint CLIP/outputs/checkpoints/best_model.pth \
  --output_dir CLIP/outputs/evaluation \
  --use_test
```

## 2. Run The Main Attractor Loop

Main K=10 run:

```bash
python3 Experiments/attractor_loop/attractor_loop_chexgen.py \
  --data_csv processed_data/processed_data.csv \
  --use_all \
  --n_iters 10 \
  --num_steps 100 \
  --cfg_scale 4.0 \
  --seed 100 \
  --skip_self_test \
  --output_dir Experiments/attractor_loop/results/chexgen_main
```

For a cluster array, add chunking:

```bash
python3 Experiments/attractor_loop/attractor_loop_chexgen.py \
  --data_csv processed_data/processed_data.csv \
  --use_all \
  --chunk_idx 0 \
  --n_chunks 60 \
  --n_iters 10 \
  --num_steps 100 \
  --cfg_scale 4.0 \
  --seed 100 \
  --skip_self_test \
  --output_dir Experiments/attractor_loop/results/chexgen_main
```

## 3. Extend To K=100

This continues the completed K=10 run.

```bash
python3 Experiments/attractor_loop/attractor_loop_chexgen_long.py \
  --data_csv processed_data/processed_data.csv \
  --use_all \
  --K_existing 10 \
  --n_iters 100 \
  --num_steps 100 \
  --cfg_scale 4.0 \
  --base_seed 100 \
  --main_dir Experiments/attractor_loop/results/chexgen_main \
  --output_dir Experiments/attractor_loop/results/chexgen_long
```

For a cluster array, add:

```bash
--chunk_idx 0 --n_chunks 60
```

## 4. Run Seed Replicates

Used for the per-anchor divergence-rate analysis.

```bash
python3 Experiments/attractor_loop/attractor_lyapunov_seeds.py \
  --data_csv processed_data/processed_data.csv \
  --main_dir Experiments/attractor_loop/results/chexgen_main \
  --n_anchors 20 \
  --n_seeds 10 \
  --n_iters 100 \
  --num_steps 100 \
  --cfg_scale 4.0 \
  --base_seed 100 \
  --output_dir Experiments/attractor_loop/results/lyapunov_seeds
```

## 5. Build Reference Embeddings

```bash
python3 Experiments/attractor_loop/preflight_embed_corpus.py \
  --data_csv processed_data/processed_data.csv \
  --splits train validate \
  --batch_size 64 \
  --num_workers 4 \
  --seed 100 \
  --out_dir Experiments/attractor_loop/reference_embeddings
```

## 6. Run Analyses

```bash
mkdir -p Experiments/attractor_loop/analysis_long
```

Main dynamics:

```bash
python3 Experiments/attractor_loop/attractor_analysis.py \
  --main_dir Experiments/attractor_loop/results/chexgen_main \
  --lyapunov_dir Experiments/attractor_loop/results/lyapunov_seeds \
  --ref_dir Experiments/attractor_loop/reference_embeddings \
  --data_csv processed_data/processed_data.csv \
  --out_dir Experiments/attractor_loop/analysis_long \
  --blocks A B C E F
```

Attractor modes and OOV profiles:

```bash
python3 Experiments/attractor_loop/analysis_attractor_modes.py \
  --trajectory_dir Experiments/attractor_loop/results/chexgen_long \
  --data_csv processed_data/processed_data.csv \
  --out_dir Experiments/attractor_loop/analysis_long \
  --probe_iters 0,1,5,10,20,30,50,70,100 \
  --use_chexpert auto
```

Long-horizon geometry:

```bash
python3 Experiments/attractor_loop/analysis_long_horizon.py \
  --long_dir Experiments/attractor_loop/results/chexgen_long \
  --ref_dir Experiments/attractor_loop/reference_embeddings \
  --data_csv processed_data/processed_data.csv \
  --out_dir Experiments/attractor_loop/analysis_long \
  --K_max 100 \
  --blocks A,C,E,G,H,I,J
```

## 7. Make Figures

`all_figs.py` expects `Experiments/attractor_loop/figures/style.py` to be
present.

```bash
export MPLCONFIGDIR=/tmp/matplotlib-$USER
mkdir -p "$MPLCONFIGDIR"

python3 Experiments/attractor_loop/figures/all_figs.py \
  --panels all \
  --block_k Experiments/attractor_loop/analysis_long/block_K_results.json \
  --analysis_json Experiments/attractor_loop/analysis_long/analysis_results.json \
  --long_horizon Experiments/attractor_loop/analysis_long/long_horizon_results.json \
  --out_dir Experiments/attractor_loop/figures \
  --pdf
```

## Notes

- All example runs use seed `100`.
- Run scripts from the repo root.
- The loop scripts resume automatically when study outputs already exist.
- The K=100 script requires the K=10 outputs first.
