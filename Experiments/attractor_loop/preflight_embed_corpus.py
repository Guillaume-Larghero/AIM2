import argparse
import gc
import logging
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoTokenizer

BASE_DIR     = "../"
DATA_CSV     = f"{BASE_DIR}/processed_data/processed_data.csv"
MEDCLIP_CKPT = f"{BASE_DIR}/CLIP/outputs/checkpoints/best_model.pth"
HF_HOME      = "/.cache/huggingface"
OUT_DIR      = f"{BASE_DIR}/Experiments/attractor_loop/reference_embeddings"

os.environ["HF_HOME"]                = HF_HOME
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ── Same MedCLIP transform as the loop script (LANCZOS 512 + ImageNet norm) ───
MEDCLIP_TRANSFORM = transforms.Compose([
    transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.LANCZOS),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# 
#  DATASET
# 

class CXRDataset(Dataset):
    """Lazy-loading; returns (image_tensor, idx). Idx is for re-aligning batches."""

    def __init__(self, image_paths):
        self.paths = image_paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
            return MEDCLIP_TRANSFORM(img), idx
        except Exception as e:
            # Return a sentinel; we'll mark these in post-processing.
            logger.warning(f"Failed to load {self.paths[idx]}: {e}")
            return torch.zeros(3, 512, 512), -idx - 1  # negative idx flags failure


# 
#  MEDCLIP LOAD
# 

def load_medclip(device):
    logger.info(f"Loading MedCLIP from {MEDCLIP_CKPT}")
    if BASE_DIR not in sys.path:
        sys.path.insert(0, BASE_DIR)

    # Same shim as the loop script — bypass the torch>=2.6 safe-load check.
    try:
        import transformers.utils.import_utils as _tu
        import transformers.modeling_utils as _tmu
        _noop = lambda: None
        _tu.check_torch_load_is_safe  = _noop
        _tmu.check_torch_load_is_safe = _noop
    except Exception:
        pass

    from CLIP.config.config import Config
    from CLIP.model.clip_model import MedicalCLIP

    config = Config()
    config.data.IMAGE_SIZE        = 512
    config.data.COMBINE_SECTIONS  = False
    config.model.IMAGE_PRETRAINED = False

    model = MedicalCLIP(config)
    ckpt = torch.load(MEDCLIP_CKPT, map_location="cpu", weights_only=False)
    key = "model_state_dict" if "model_state_dict" in ckpt else "state_dict"
    model.load_state_dict(ckpt[key], strict=True)
    model = model.to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    # Sanity check
    with torch.no_grad():
        dummy = torch.randn(1, 3, 512, 512).to(device)
        emb = model.encode_image(dummy)
    assert emb.shape == (1, 256) and abs(emb.norm().item() - 1.0) < 0.01

    logger.info(f"  ✓ MedCLIP loaded: epoch={ckpt.get('epoch','?')}")
    return model, tokenizer


# 
#  IMAGE EMBEDDING
# 

def embed_images(model, image_paths, *, device, batch_size=64, num_workers=4):
    """Streaming embed. Returns (N, 256) float32 + (N,) bool mask of successes."""
    n = len(image_paths)
    embeds = np.zeros((n, 256), dtype=np.float32)
    valid  = np.zeros(n, dtype=bool)

    ds = CXRDataset(image_paths)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    t0 = time.time()
    n_done = 0
    with torch.no_grad():
        for imgs, idxs in loader:
            imgs = imgs.to(device, non_blocking=True)
            embs = model.encode_image(imgs)  # (B, 256), L2-normed
            embs = embs.float().cpu().numpy()
            idxs = idxs.numpy()

            ok = idxs >= 0
            for j in range(len(idxs)):
                if ok[j]:
                    embeds[idxs[j]] = embs[j]
                    valid[idxs[j]]  = True

            n_done += imgs.shape[0]
            if n_done % (batch_size * 50) == 0:
                rate = n_done / (time.time() - t0)
                eta  = (n - n_done) / max(rate, 1e-9) / 60.0
                logger.info(f"  images: {n_done}/{n} ({100*n_done/n:.1f}%)  "
                            f"rate={rate:.1f} img/s  ETA={eta:.1f} min")

    n_failed = (~valid).sum()
    logger.info(f"  Done. Embedded {valid.sum()}/{n}; {n_failed} failed.")
    return embeds, valid


# 
#  TEXT EMBEDDING
# 

@torch.no_grad()
def embed_texts(model, tokenizer, texts, *, device, batch_size=128):
    """Pre-prepared text strings → (N, 256) float32 L2-normed, plus valid mask.

    Caller is responsible for whitespace normalization and empty-string
    fallback (mirroring MIMICCXRDataset._prepare_text). This function just
    tokenizes and embeds in batches.
    """
    n = len(texts)
    embeds = np.zeros((n, 256), dtype=np.float32)
    valid  = np.zeros(n, dtype=bool)

    t0 = time.time()
    for i in range(0, n, batch_size):
        batch = texts[i:i+batch_size]

        enc = tokenizer(
            batch,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        embs = model.encode_text(
            enc["input_ids"].to(device),
            enc["attention_mask"].to(device),
        ).float().cpu().numpy()

        embeds[i:i+len(batch)] = embs
        valid[i:i+len(batch)]  = True

        if (i // batch_size) % 50 == 0 and i > 0:
            rate = (i + len(batch)) / (time.time() - t0)
            eta = (n - i - len(batch)) / max(rate, 1e-9) / 60.0
            logger.info(f"  texts: {i+len(batch)}/{n} ({100*(i+len(batch))/n:.1f}%)  "
                        f"rate={rate:.1f}/s  ETA={eta:.1f} min")

    logger.info(f"  Done. Embedded {valid.sum()}/{n}")
    return embeds, valid


# 
#  UMAP
# 

def fit_umap(embeds, valid, *, name, fit_subsample=50_000, seed=100):
    """
    Fit UMAP on a stratified random subsample for tractable .transform() later.
    Project ALL valid embeddings into 2D using the fitted reducer.
    Save the reducer (pickle) and the 2D coords (.npy).
    """
    import umap

    logger.info(f"Fitting UMAP for {name} space...")
    valid_idx = np.where(valid)[0]
    n_valid   = len(valid_idx)
    n_fit     = min(fit_subsample, n_valid)

    rng     = np.random.default_rng(seed)
    fit_idx = rng.choice(valid_idx, size=n_fit, replace=False)
    logger.info(f"  Fitting on {n_fit}/{n_valid} valid embeddings")

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=30,
        min_dist=0.1,
        metric="cosine",
        random_state=seed,
        verbose=True,
        low_memory=True,
    )
    t0 = time.time()
    reducer.fit(embeds[fit_idx])
    logger.info(f"  fit took {(time.time()-t0)/60:.1f} min")

    # Project all valid embeddings
    logger.info(f"  Projecting all {n_valid} valid embeddings to 2D")
    t0 = time.time()
    proj_2d_valid = reducer.transform(embeds[valid_idx])
    logger.info(f"  transform took {(time.time()-t0)/60:.1f} min")

    proj_2d = np.full((len(embeds), 2), np.nan, dtype=np.float32)
    proj_2d[valid_idx] = proj_2d_valid

    return reducer, proj_2d


# 
#  MAIN
# 

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_csv",     type=str, default=DATA_CSV)
    p.add_argument("--out_dir",      type=str, default=OUT_DIR)
    p.add_argument("--splits",       nargs="+", default=["train", "validate"])
    p.add_argument("--batch_size",   type=int, default=64)
    p.add_argument("--num_workers",  type=int, default=4)
    p.add_argument("--umap_subsample", type=int, default=50_000,
                   help="UMAP fit on a random sample. Set 0 to fit on all valid.")
    p.add_argument("--seed",         type=int, default=100)
    p.add_argument("--max_n",        type=int, default=None,
                   help="DEBUG: cap total studies (default: all)")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    img_path  = os.path.join(args.out_dir, "img_embeds.npy")
    txt_path  = os.path.join(args.out_dir, "txt_embeds.npy")
    meta_path = os.path.join(args.out_dir, "meta.csv")
    img_um_path = os.path.join(args.out_dir, "umap_img.pkl")
    txt_um_path = os.path.join(args.out_dir, "umap_txt.pkl")
    img_2d_path = os.path.join(args.out_dir, "umap_img_2d.npy")
    txt_2d_path = os.path.join(args.out_dir, "umap_txt_2d.npy")
    valid_path  = os.path.join(args.out_dir, "valid_masks.npz")

    logger.info("=" * 60)
    logger.info("Reference Embedding Preflight")
    logger.info("=" * 60)

    # ── Load metadata ─────────────────────────────────────────────────────────
    if os.path.exists(meta_path):
        logger.info(f"Loading cached meta from {meta_path}")
        df = pd.read_csv(meta_path)
    else:
        logger.info(f"Reading {args.data_csv}")
        df = pd.read_csv(args.data_csv, low_memory=False)
        df = df[df["split"].isin(args.splits)].copy()
        # Mirror MedCLIP training filter exactly: matches MIMICCXRDataset which
        # casts the column to string and tests case-insensitively. Pandas may
        # read the column as either bool or string depending on csv layout.
        if "has_findings" in df.columns:
            mask = df["has_findings"].astype(str).str.lower().isin(["true", "1", "1.0"])
            df = df[mask]
        # processed_data.csv contains exactly one row per (study_id, AP-view
        # image) pair — already deduped at preprocessing time. No groupby
        # needed. Each row is one sample, identical to what MedCLIP training
        # consumed.
        df = df.reset_index(drop=True)
        if args.max_n:
            df = df.head(args.max_n).copy()
        df["row_idx"] = np.arange(len(df))
        df.to_csv(meta_path, index=False)
        logger.info(f"  Saved meta: {len(df)} rows → {meta_path}")
    n = len(df)
    logger.info(f"  Total studies in scope: {n}  (1 row = 1 study = 1 AP image)")
    logger.info(f"  Splits: {df['split'].value_counts().to_dict()}")

    # ── Load MedCLIP ──────────────────────────────────────────────────────────
    need_image_embed = not os.path.exists(img_path)
    need_text_embed  = not os.path.exists(txt_path)
    if not (need_image_embed or need_text_embed):
        logger.info("Image and text embeddings already cached — skipping MedCLIP load")
        model, tokenizer = None, None
    else:
        model, tokenizer = load_medclip(device)

    # ── Image embeddings ──────────────────────────────────────────────────────
    if need_image_embed:
        logger.info("\n" + "─" * 60)
        logger.info("Embedding images")
        logger.info("─" * 60)
        img_paths = df["image_path"].tolist()
        img_embeds, img_valid = embed_images(
            model, img_paths,
            device=device, batch_size=args.batch_size, num_workers=args.num_workers,
        )
        np.save(img_path, img_embeds)
        np.savez(valid_path,
                 img_valid=img_valid,
                 txt_valid=np.zeros(n, dtype=bool) if need_text_embed
                           else np.load(valid_path)["txt_valid"]
                                if os.path.exists(valid_path) else np.zeros(n, dtype=bool))
        logger.info(f"  Saved → {img_path}  shape={img_embeds.shape}")
        gc.collect(); torch.cuda.empty_cache()
    else:
        logger.info(f"Loading cached image embeddings: {img_path}")
        img_embeds = np.load(img_path)

    # ── Text embeddings ───────────────────────────────────────────────────────
    if need_text_embed:
        logger.info("\n" + "─" * 60)
        logger.info("Embedding texts")
        logger.info("─" * 60)
        if model is None:
            model, tokenizer = load_medclip(device)

        # Mirror MIMICCXRDataset._prepare_text() exactly:
        #   1. .strip() the FINDINGS string
        #   2. Empty / NaN → fallback string "No findings reported."
        #   3. Whitespace runs collapsed via ' '.join(text.split())
        # This guarantees every embedded text lives in the same distributional
        # region as the training data — empties → fallback embedding, all
        # texts whitespace-normalized.
        TEXT_FALLBACK = "No findings reported."
        raw = df["findings"].tolist()

        def prepare(t):
            if t is None or (isinstance(t, float) and np.isnan(t)):
                t = ""
            t = str(t).strip()
            t = t if t else TEXT_FALLBACK
            return " ".join(t.split())

        texts = [prepare(t) for t in raw]
        n_fallback = sum(1 for t in texts if t == TEXT_FALLBACK)
        logger.info(f"  Empty/NaN findings → fallback: {n_fallback}/{len(texts)}")

        txt_embeds, txt_valid = embed_texts(
            model, tokenizer, texts, device=device, batch_size=128,
        )
        np.save(txt_path, txt_embeds)
        # Update the valid masks file
        if os.path.exists(valid_path):
            old = np.load(valid_path)
            img_valid_save = old["img_valid"]
        else:
            img_valid_save = np.ones(n, dtype=bool)
        np.savez(valid_path, img_valid=img_valid_save, txt_valid=txt_valid)
        logger.info(f"  Saved → {txt_path}  shape={txt_embeds.shape}")
    else:
        logger.info(f"Loading cached text embeddings: {txt_path}")
        txt_embeds = np.load(txt_path)

    # Free MedCLIP — done with embeddings
    if model is not None:
        del model
        gc.collect(); torch.cuda.empty_cache()

    # ── Load valid masks ──────────────────────────────────────────────────────
    if os.path.exists(valid_path):
        masks = np.load(valid_path)
        img_valid = masks["img_valid"]
        txt_valid = masks["txt_valid"]
    else:
        img_valid = np.ones(n, dtype=bool)
        txt_valid = np.ones(n, dtype=bool)

    logger.info(f"\n  Image: {img_valid.sum()}/{n} valid embeddings")
    logger.info(f"  Text:  {txt_valid.sum()}/{n} valid embeddings")

    # ── UMAP fitting (image) ──────────────────────────────────────────────────
    if not os.path.exists(img_um_path) or not os.path.exists(img_2d_path):
        logger.info("\n" + "─" * 60)
        logger.info("Fitting UMAP — image space")
        logger.info("─" * 60)
        sub = args.umap_subsample if args.umap_subsample > 0 else len(img_embeds)
        reducer_img, img_2d = fit_umap(
            img_embeds, img_valid,
            name="image", fit_subsample=sub, seed=args.seed,
        )
        with open(img_um_path, "wb") as f:
            pickle.dump(reducer_img, f)
        np.save(img_2d_path, img_2d)
        logger.info(f"  Saved → {img_um_path}")
        logger.info(f"  Saved → {img_2d_path}  shape={img_2d.shape}")
    else:
        logger.info(f"UMAP-img already cached: {img_um_path}")

    # ── UMAP fitting (text) ───────────────────────────────────────────────────
    if not os.path.exists(txt_um_path) or not os.path.exists(txt_2d_path):
        logger.info("\n" + "─" * 60)
        logger.info("Fitting UMAP — text space")
        logger.info("─" * 60)
        sub = args.umap_subsample if args.umap_subsample > 0 else len(txt_embeds)
        reducer_txt, txt_2d = fit_umap(
            txt_embeds, txt_valid,
            name="text", fit_subsample=sub, seed=args.seed,
        )
        with open(txt_um_path, "wb") as f:
            pickle.dump(reducer_txt, f)
        np.save(txt_2d_path, txt_2d)
        logger.info(f"  Saved → {txt_um_path}")
        logger.info(f"  Saved → {txt_2d_path}  shape={txt_2d.shape}")
    else:
        logger.info(f"UMAP-txt already cached: {txt_um_path}")

    logger.info("\n" + "=" * 60)
    logger.info("PREFLIGHT COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
