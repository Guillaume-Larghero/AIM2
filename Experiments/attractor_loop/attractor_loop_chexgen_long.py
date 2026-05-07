import argparse
import gc
import hashlib
import json
import logging
import os
import shutil
import sys
import time
import traceback
from glob import glob

import numpy as np
import pandas as pd
import torch
from PIL import Image

# Make this script importable from Experiments/attractor_loop/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from attractor_loop_chexgen import (
    load_medclip, load_maira, load_chexgen,
    embed_image, embed_text, run_maira,
    cosine_sim, ChexGenWrapper,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

BASE_DIR = "../"
DATA_CSV = f"{BASE_DIR}/processed_data/processed_data.csv"


# 
#  Anchor selection — quantile-based, deterministic
# 

def select_long_anchors(main_dir: str, n_anchors: int):
    """Pick n_anchors at evenly-spaced quantiles 0.02..0.98 of iter-0 image
    cosine from the merged main run summary. Returns list of dicts:
      [{"study_id": str, "iter0_cos": float}, ...].
    """
    summary_path = os.path.join(main_dir, "summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Main summary not found: {summary_path}")
    with open(summary_path) as f:
        summary = json.load(f)
    studies = summary["per_study"]
    valid = [(s["study_id"], s["image_cosine"][0]) for s in studies
             if len(s.get("image_cosine", [])) > 0]
    if len(valid) < n_anchors:
        raise ValueError(f"Only {len(valid)} studies in main run; "
                         f"requested {n_anchors} anchors.")
    sids = np.array([s for s, _ in valid])
    cos  = np.array([c for _, c in valid])

    qs = np.linspace(0.02, 0.98, n_anchors)
    targets = np.quantile(cos, qs)
    chosen_idx, used = [], set()
    for t in targets:
        order = np.argsort(np.abs(cos - t))
        for i in order:
            if int(i) not in used:
                chosen_idx.append(int(i))
                used.add(int(i))
                break
    return [{"study_id": str(sids[i]), "iter0_cos": float(cos[i])}
            for i in chosen_idx]


def select_long_anchors_all(main_dir: str):
    """Use ALL studies in main_dir, sorted by study_id (deterministic).

    Used for the N=1081 full-cohort long-horizon run. Returns the same
    dict format as select_long_anchors so downstream code is identical.
    Sorted-by-study_id ordering ensures reproducible chunking across reruns
    and that already-done studies are distributed predictably across chunks.
    """
    summary_path = os.path.join(main_dir, "summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Main summary not found: {summary_path}")
    with open(summary_path) as f:
        summary = json.load(f)
    studies = summary["per_study"]
    valid = [(s["study_id"], s["image_cosine"][0]) for s in studies
             if len(s.get("image_cosine", [])) > 0]
    # Sort by study_id (string sort, consistent across runs)
    valid_sorted = sorted(valid, key=lambda t: str(t[0]))
    return [{"study_id": str(sid), "iter0_cos": float(cos)}
            for sid, cos in valid_sorted]


# 
#  Reuse logic — copy iter 0..K_existing from main_dir to output_dir
# 

def copy_existing_iters(sid: str, main_dir: str, out_dir: str, K_existing: int):
    """Copy iter-0..K_existing artifacts from main_dir/<sid>/ to out_dir/<sid>/.
    Returns True if all expected files were present and copied."""
    src = os.path.join(main_dir, sid)
    dst = os.path.join(out_dir, sid)
    os.makedirs(dst, exist_ok=True)

    required_per_iter = ["gen_iter_{:03d}.png",
                          "img_embed_iter_{:03d}.npy",
                          "text_embed_iter_{:03d}.npy",
                          "findings_iter_{:03d}.txt"]
    required_global = ["anchor_img_embed.npy",
                        "anchor_text_embed.npy",
                        "gt_image.png",
                        "gt_findings.txt"]

    # Verify all expected files exist before any copies
    missing = []
    for f in required_global:
        if not os.path.exists(os.path.join(src, f)):
            missing.append(f)
    for k in range(K_existing + 1):
        for tmpl in required_per_iter:
            f = tmpl.format(k)
            if not os.path.exists(os.path.join(src, f)):
                missing.append(f)
    if missing:
        logger.warning(f"  [{sid}] missing {len(missing)} files from main_dir, "
                       f"first few: {missing[:3]}")
        return False

    # Copy global anchor files (skip if already present — supports resume)
    for f in required_global:
        sp, dp = os.path.join(src, f), os.path.join(dst, f)
        if not os.path.exists(dp):
            shutil.copy2(sp, dp)
    # Copy per-iter files for k=0..K_existing
    for k in range(K_existing + 1):
        for tmpl in required_per_iter:
            f = tmpl.format(k)
            sp, dp = os.path.join(src, f), os.path.join(dst, f)
            if not os.path.exists(dp):
                shutil.copy2(sp, dp)
    return True


def load_existing_metrics(sid: str, main_dir: str, K_existing: int):
    """Load the metrics.json from chexgen_main and return its content for
    seeding the long-run metrics dict (we extend its lists from k=K_existing+1
    onward).
    """
    p = os.path.join(main_dir, sid, "metrics.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        m = json.load(f)
    # Sanity check that we have at least K_existing+1 iterations
    if len(m.get("image_cosine", [])) < K_existing + 1:
        logger.warning(f"  [{sid}] metrics.json has only "
                       f"{len(m.get('image_cosine', []))} iters; expected "
                       f">= {K_existing + 1}")
        return None
    return m


# 
#  Long-horizon continuation loop
# 

def run_long_loop(
    *,
    sid: str,
    out_dir: str,
    main_dir: str,
    K_existing: int,
    K_target: int,
    medclip, tokenizer, maira, chexgen, device,
    num_steps: int, cfg_scale: float, base_seed: int,
):
    """Continue the per-study loop from iter K_existing+1 to K_target (inclusive).

    Reads iter-K_existing image and embeddings from out_dir (copied from
    main_dir), runs the standard MAIRA→ChexGen→MedCLIP step for each new k,
    and writes the same artifact set as run_loop in attractor_loop_chexgen.py.

    Updates metrics.json in-place at the end.
    """
    study_dir = os.path.join(out_dir, sid)
    if not os.path.isdir(study_dir):
        raise FileNotFoundError(f"Expected study dir at {study_dir}")

    # ── Load existing state ───────────────────────────────────────────────────
    metrics = load_existing_metrics(sid, main_dir, K_existing)
    if metrics is None:
        raise RuntimeError(f"[{sid}] no existing metrics.json to extend")

    # Update config to reflect the long run target
    metrics["config"]["n_iters"] = K_target
    metrics["config"]["base_seed_long"] = base_seed
    metrics["config"]["resumed_from_k"] = K_existing

    # Anchors (loaded from disk to avoid re-encoding GT)
    anchor_img  = torch.from_numpy(np.load(os.path.join(study_dir, "anchor_img_embed.npy")))
    anchor_text = torch.from_numpy(np.load(os.path.join(study_dir, "anchor_text_embed.npy")))

    # Iter-K_existing state — initialize "previous" embeddings and "current" image
    last_iter_img_path  = os.path.join(study_dir, f"gen_iter_{K_existing:03d}.png")
    last_iter_findings  = os.path.join(study_dir, f"findings_iter_{K_existing:03d}.txt")
    last_iter_img_emb   = os.path.join(study_dir, f"img_embed_iter_{K_existing:03d}.npy")
    last_iter_text_emb  = os.path.join(study_dir, f"text_embed_iter_{K_existing:03d}.npy")

    current_img   = Image.open(last_iter_img_path).convert("RGB")
    prev_img_emb  = torch.from_numpy(np.load(last_iter_img_emb))
    prev_text_emb = torch.from_numpy(np.load(last_iter_text_emb))

    logger.info(f"  [{sid}] resuming from iter {K_existing}: "
                f"gen_pil_size={current_img.size}  "
                f"prev_img_emb_norm={prev_img_emb.norm():.4f}  "
                f"prev_text_emb_norm={prev_text_emb.norm():.4f}")

    # ── Continuation loop iter K_existing+1 .. K_target ───────────────────────
    for k in range(K_existing + 1, K_target + 1):
        # Resume detection: if iter-k embeddings already exist, skip
        if (os.path.exists(os.path.join(study_dir, f"img_embed_iter_{k:03d}.npy")) and
            os.path.exists(os.path.join(study_dir, f"text_embed_iter_{k:03d}.npy")) and
            os.path.exists(os.path.join(study_dir, f"gen_iter_{k:03d}.png"))):
            # Load saved state and extend metrics (in case metrics.json was lost mid-run)
            with open(os.path.join(study_dir, f"findings_iter_{k:03d}.txt")) as f:
                findings_k = f.read()
            img_ek  = torch.from_numpy(np.load(os.path.join(study_dir, f"img_embed_iter_{k:03d}.npy")))
            text_ek = torch.from_numpy(np.load(os.path.join(study_dir, f"text_embed_iter_{k:03d}.npy")))
            current_img = Image.open(os.path.join(study_dir, f"gen_iter_{k:03d}.png")).convert("RGB")
            # If metrics already has this iter, don't re-append
            if len(metrics["image_cosine"]) > k:
                pass
            else:
                metrics["image_cosine"].append(cosine_sim(anchor_img, img_ek))
                metrics["text_cosine"].append(cosine_sim(anchor_text, text_ek))
                metrics["embed_l2"].append(float((anchor_img - img_ek).norm().item()))
                if prev_img_emb is not None:
                    metrics["image_evolution"].append(cosine_sim(prev_img_emb, img_ek))
                if prev_text_emb is not None:
                    metrics["text_evolution"].append(cosine_sim(prev_text_emb, text_ek))
                metrics["findings"].append(findings_k)
            prev_img_emb, prev_text_emb = img_ek, text_ek
            logger.info(f"  [{sid}] iter {k}: already done, skipping")
            continue

        iter_seed = base_seed + k
        iter_diag = {"k": k, "iter_seed": iter_seed,
                     "input_image_source": f"gen_iter_{k-1:03d}",
                     "input_image_size":   list(current_img.size)}

        # MAIRA
        t0 = time.time()
        findings_k = run_maira(maira, current_img)
        maira_dt = time.time() - t0
        iter_diag["maira"] = {"wallclock_s": round(maira_dt, 2),
                               "input_size":  [518, 518],
                               "output_chars": len(findings_k),
                               "output_words": len(findings_k.split())}
        logger.info(f"  [{sid}] iter {k}: MAIRA-2 ({maira_dt:.1f}s)  "
                    f"chars={len(findings_k)}  "
                    f"head=\"{findings_k[:80]}{'...' if len(findings_k) > 80 else ''}\"")

        # ChexGen
        try:
            gen_pil, gen_diag = chexgen.generate(
                findings_k, seed=iter_seed,
                num_steps=num_steps, cfg_scale=cfg_scale,
            )
        except Exception as e:
            logger.error(f"  [{sid}] iter {k}: ChexGen FAILED: {e}")
            iter_diag["chexgen_error"] = str(e)
            metrics["iterations"].append(iter_diag)
            raise
        iter_diag.update(gen_diag)

        # Persist
        gen_pil.convert("L").save(os.path.join(study_dir, f"gen_iter_{k:03d}.png"))
        with open(os.path.join(study_dir, f"findings_iter_{k:03d}.txt"), "w") as f:
            f.write(findings_k)

        # MedCLIP
        img_ek  = embed_image(medclip, gen_pil, device)
        text_ek = embed_text(medclip, tokenizer, findings_k, device)
        np.save(os.path.join(study_dir, f"img_embed_iter_{k:03d}.npy"),  img_ek.numpy())
        np.save(os.path.join(study_dir, f"text_embed_iter_{k:03d}.npy"), text_ek.numpy())

        iter_diag["medclip"] = {
            "img_embed_norm":  float(img_ek.norm().item()),
            "text_embed_norm": float(text_ek.norm().item()),
            "embed_dim":       int(img_ek.shape[0]),
        }

        # Metrics
        img_cos = cosine_sim(anchor_img, img_ek)
        txt_cos = cosine_sim(anchor_text, text_ek)
        l2_to_gt = float((anchor_img - img_ek).norm().item())
        img_evo = cosine_sim(prev_img_emb,  img_ek)
        txt_evo = cosine_sim(prev_text_emb, text_ek)
        iter_diag["metrics"] = {"image_cosine": img_cos, "text_cosine": txt_cos,
                                 "embed_l2": l2_to_gt,
                                 "image_evolution": img_evo,
                                 "text_evolution":  txt_evo}

        metrics["iterations"].append(iter_diag)
        metrics["findings"].append(findings_k)
        metrics["image_cosine"].append(img_cos)
        metrics["text_cosine"].append(txt_cos)
        metrics["embed_l2"].append(l2_to_gt)
        metrics["image_evolution"].append(img_evo)
        metrics["text_evolution"].append(txt_evo)

        logger.info(
            f"  [{sid}] iter {k}: ChexGen={gen_diag['chexgen']['sampling_wallclock_s']}s  "
            f"vae={gen_diag['chexgen']['vae_decode_wallclock_s']}s  "
            f"t5_tok={gen_diag['t5']['used_token_count']}/{gen_diag['t5']['natural_token_count']}"
            f"{' (TRUNC)' if gen_diag['t5']['was_truncated'] else ''}  "
            f"img_cos={img_cos:+.4f}  txt_cos={txt_cos:+.4f}  l2={l2_to_gt:.4f}  "
            f"img_evo={img_evo:+.4f}  txt_evo={txt_evo:+.4f}"
        )

        # Roll forward
        current_img   = gen_pil
        prev_img_emb  = img_ek
        prev_text_emb = text_ek

        # Snapshot metrics every 10 iters in case the job times out
        if k % 10 == 0:
            with open(os.path.join(study_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)

    # Final write
    with open(os.path.join(study_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


# 
#  Main
# 

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_anchors",  type=int, default=50,
                   help="Number of anchors to run (ignored if --use_all).")
    p.add_argument("--use_all",    action="store_true",
                   help="Use ALL studies in main_dir (full N=1081 cohort run); "
                        "overrides --n_anchors. Anchors selected in sorted-by-"
                        "study_id order for reproducible chunking. Studies "
                        "already complete in output_dir are detected via the "
                        "per-iter resume logic and skipped quickly.")
    p.add_argument("--n_iters",    type=int, default=100,
                   help="Target K (total iterations after iter 0).")
    p.add_argument("--K_existing", type=int, default=10,
                   help="Iterations already present in main_dir to reuse.")
    p.add_argument("--num_steps",  type=int, default=100)
    p.add_argument("--cfg_scale",  type=float, default=4.0)
    p.add_argument("--base_seed",  type=int, default=100,
                   help="Base seed offset (ADD TO SHA256(study_id) per study). "
                        "Must match what main run used: 100.")
    p.add_argument("--chunk_idx",  type=int, default=0,
                   help="Chunk index for SLURM array.")
    p.add_argument("--n_chunks",   type=int, default=1)
    p.add_argument("--main_dir",   type=str,
                   default=f"{BASE_DIR}/Experiments/attractor_loop/results/chexgen_main")
    p.add_argument("--output_dir", type=str,
                   default=f"{BASE_DIR}/Experiments/attractor_loop/results/chexgen_long")
    p.add_argument("--data_csv",   type=str, default=DATA_CSV)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 60)
    if args.use_all:
        logger.info("Long-Horizon Loop Extension (K=100, FULL N=1081 cohort)")
    else:
        logger.info(f"Long-Horizon Loop Extension (K={args.n_iters}, N={args.n_anchors})")
    logger.info("=" * 60)
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")

    # ── Anchor selection (deterministic across chunks) ────────────────────────
    if args.use_all:
        anchors = select_long_anchors_all(args.main_dir)
        logger.info(f"\n--use_all set: selected ALL {len(anchors)} valid studies "
                    f"(sorted by study_id for reproducible chunking)")
    else:
        anchors = select_long_anchors(args.main_dir, n_anchors=args.n_anchors)
        logger.info(f"\nSelected {len(anchors)} anchors at quantiles 0.02..0.98:")
    # Show only first 5 + last 2 to avoid spamming the log for N=1081
    if len(anchors) <= 10:
        for i, a in enumerate(anchors):
            logger.info(f"  [{i:4d}] {a['study_id']}  iter0_img_cos={a['iter0_cos']:+.4f}")
    else:
        for i, a in enumerate(anchors[:5]):
            logger.info(f"  [{i:4d}] {a['study_id']}  iter0_img_cos={a['iter0_cos']:+.4f}")
        logger.info(f"  [...{len(anchors)-7} more anchors...]")
        for i, a in enumerate(anchors[-2:]):
            logger.info(f"  [{len(anchors)-2+i:4d}] {a['study_id']}  iter0_img_cos={a['iter0_cos']:+.4f}")

    # Slice for chunking
    n_total = len(anchors)
    chunk_size = (n_total + args.n_chunks - 1) // args.n_chunks
    chunk_start = args.chunk_idx * chunk_size
    chunk_end   = min(chunk_start + chunk_size, n_total)
    chunk_anchors = anchors[chunk_start:chunk_end]
    logger.info(f"\nChunk {args.chunk_idx+1}/{args.n_chunks}: "
                f"anchors [{chunk_start}, {chunk_end}) → {len(chunk_anchors)} anchors")

    # ── Image-path lookup from CSV (anchors store study_id only) ──────────────
    df = pd.read_csv(args.data_csv, low_memory=False)
    df = df[df["split"] == "test"]
    if "has_findings" in df.columns:
        mask = df["has_findings"].astype(str).str.lower().isin(["true", "1", "1.0"])
        df = df[mask]
    df = df.groupby("study_id", as_index=False).first()
    df["study_id"] = df["study_id"].astype(str)
    df = df.set_index("study_id")

    # ── Load models ───────────────────────────────────────────────────────────
    logger.info("\nLoading models...")
    medclip, _, tokenizer = load_medclip(device)
    maira                 = load_maira()
    chexgen               = load_chexgen(device, num_steps=args.num_steps)

    os.makedirs(args.output_dir, exist_ok=True)

    failed = []
    completed = []
    skipped_done = []
    for i, a in enumerate(chunk_anchors):
        sid = a["study_id"]
        logger.info(f"\n{'─'*55}")
        logger.info(f"[{i+1}/{len(chunk_anchors)}] {sid}  "
                    f"(iter0_cos={a['iter0_cos']:+.4f})")
        logger.info(f"{'─'*55}")

        # ── Fast-path skip for already-fully-completed studies ──────────────
        # Avoids the ~30-60s overhead of reloading 90 iters of embeddings just
        # to confirm they're all there. For N=1081 this saves measurable time
        # on chunks that contain previously-completed N=48 studies.
        study_dir = os.path.join(args.output_dir, sid)
        last_iter_files_exist = all(
            os.path.exists(os.path.join(study_dir, f))
            for f in [
                f"img_embed_iter_{args.n_iters:03d}.npy",
                f"text_embed_iter_{args.n_iters:03d}.npy",
                f"gen_iter_{args.n_iters:03d}.png",
                f"findings_iter_{args.n_iters:03d}.txt",
                "metrics.json",
            ]
        )
        if last_iter_files_exist:
            try:
                with open(os.path.join(study_dir, "metrics.json")) as f:
                    m = json.load(f)
                if len(m.get("image_cosine", [])) >= args.n_iters + 1:
                    logger.info(f"  [{sid}] already fully complete "
                                f"(K={args.n_iters}); skipping.")
                    skipped_done.append(sid)
                    continue
            except (json.JSONDecodeError, KeyError):
                pass  # fall through to full processing

        # Per-study deterministic seed — must match attractor_loop_chexgen.py
        # (SHA-256-based, base + hash). This ensures iter k uses the same seed
        # as it would have if the original run had gone past k=10.
        per_study_seed = (
            args.base_seed +
            int(hashlib.sha256(sid.encode()).hexdigest()[:8], 16)
        ) & 0x7FFFFFFF

        # Step 1: copy existing iter 0..K_existing artifacts
        ok = copy_existing_iters(sid, args.main_dir, args.output_dir,
                                  args.K_existing)
        if not ok:
            logger.warning(f"  [{sid}] cannot reuse main_dir artifacts; skipping")
            failed.append(sid)
            continue

        # Step 2: continue iter K_existing+1 .. n_iters
        try:
            run_long_loop(
                sid=sid,
                out_dir=args.output_dir,
                main_dir=args.main_dir,
                K_existing=args.K_existing,
                K_target=args.n_iters,
                medclip=medclip, tokenizer=tokenizer, maira=maira,
                chexgen=chexgen, device=device,
                num_steps=args.num_steps,
                cfg_scale=args.cfg_scale,
                base_seed=per_study_seed,
            )
            completed.append(sid)
        except Exception as e:
            logger.error(f"  [{sid}] FAILED: {e}")
            traceback.print_exc()
            failed.append(sid)
            gc.collect()
            torch.cuda.empty_cache()

    logger.info(f"\n{'='*60}")
    logger.info(f"COMPLETE: {len(completed)} ran fresh, "
                f"{len(skipped_done)} already-done (skipped), "
                f"{len(failed)} failed")
    if failed:
        logger.warning(f"Failed: {failed}")


if __name__ == "__main__":
    main()
