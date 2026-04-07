#!/usr/bin/env python3
"""
Training script for Medical CLIP (512px, FINDINGS-only).

Issues fixed from original:
  1. --lr flag was silently ignored.
     config.training.LEARNING_RATE is set but the optimizer uses differential
     per-group LRs (IMAGE_ENCODER_LR, TEXT_ENCODER_LR, PROJECTION_LR) defined
     in get_param_groups(). LEARNING_RATE is never read by the optimizer.
     → --lr now sets all three group LRs proportionally so the flag has effect.

  2. --resume_checkpoint flag added.
     100-epoch runs on O2 may hit walltime limits. This lets you resume from
     any saved checkpoint without restarting training.

  3. test_loader was created but never used.
     → Final test evaluation added after trainer.fit().

  4. --use_findings_only flag added (mirrors config.data.USE_FINDINGS_ONLY).
     Allows overriding the config default from the command line for ablations.

Usage:
    python train_CLIP.py
    python train_CLIP.py --epochs 50 --batch_size 4
    python train_CLIP.py --resume_checkpoint CLIP/outputs/checkpoints/best_model.pth
    python train_CLIP.py --no_findings_only   # use FINDINGS+IMPRESSION (ablation)
"""

import os
import argparse

import torch

from CLIP.config.config import Config
from CLIP.data.dataloader import create_dataloaders
from CLIP.model.clip_model import MedicalCLIP
from CLIP.training.trainer import Trainer
from CLIP.utils.checkpoint import save_checkpoint, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Train Medical CLIP (512px, FINDINGS-only)')

    # Core overrides
    parser.add_argument('--epochs',      type=int,   default=None)
    parser.add_argument('--batch_size',  type=int,   default=None)

    # --lr sets IMAGE_ENCODER_LR, TEXT_ENCODER_LR, PROJECTION_LR proportionally.
    # The three group LRs keep their relative ratios from config; only the scale changes.
    # Example: --lr 5e-5 with default ratios (enc:5e-6, proj:1e-4) gives
    #          enc: 5e-5 * (5e-6/1e-4) = 2.5e-6, proj: 5e-5.
    parser.add_argument('--lr',          type=float, default=None,
                        help='Scale all per-group learning rates. Default: use config values.')

    # Resume
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help='Path to checkpoint .pth to resume training from.')

    # Text mode override (for ablation vs default USE_FINDINGS_ONLY=True)
    parser.add_argument('--no_findings_only', action='store_true',
                        help='Use FINDINGS+IMPRESSION instead of FINDINGS only. '
                             'Overrides config.data.USE_FINDINGS_ONLY=True.')

    return parser.parse_args()


def main():
    args = parse_args()
    config = Config()

    # ── CLI overrides ──────────────────────────────────────────────────────────
    if args.epochs is not None:
        config.training.NUM_EPOCHS = args.epochs

    if args.batch_size is not None:
        config.training.BATCH_SIZE = args.batch_size

    if args.lr is not None:
        # Scale all three per-group LRs proportionally to the new base lr.
        # The optimizer reads these directly from get_param_groups(); setting
        # config.training.LEARNING_RATE alone has no effect on training.
        ratio_enc  = config.training.IMAGE_ENCODER_LR / config.training.LEARNING_RATE
        ratio_txt  = config.training.TEXT_ENCODER_LR  / config.training.LEARNING_RATE
        ratio_proj = config.training.PROJECTION_LR    / config.training.LEARNING_RATE
        config.training.LEARNING_RATE    = args.lr
        config.training.IMAGE_ENCODER_LR = args.lr * ratio_enc
        config.training.TEXT_ENCODER_LR  = args.lr * ratio_txt
        config.training.PROJECTION_LR    = args.lr * ratio_proj
        print(f"[train_CLIP] LR override: "
              f"image_enc={config.training.IMAGE_ENCODER_LR:.2e} | "
              f"text_enc={config.training.TEXT_ENCODER_LR:.2e} | "
              f"proj={config.training.PROJECTION_LR:.2e}")

    if args.no_findings_only:
        config.data.USE_FINDINGS_ONLY = False
        config.data.COMBINE_SECTIONS  = True
        print("[train_CLIP] Text mode overridden: FINDINGS + IMPRESSION")

    # ── Print + save config ────────────────────────────────────────────────────
    config.print_config()
    config.save(os.path.join(config.paths.OUTPUT_DIR, 'config.json'))

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = create_dataloaders(config)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MedicalCLIP(config)

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )

    # ── Resume from checkpoint ────────────────────────────────────────────────
    start_epoch = 0
    if args.resume_checkpoint is not None:
        if os.path.exists(args.resume_checkpoint):
            start_epoch, metrics = load_checkpoint(
                args.resume_checkpoint,
                model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                device=config.system.DEVICE,
            )
            trainer.current_epoch = start_epoch
            print(f"[train_CLIP] Resumed from epoch {start_epoch} | "
                  f"metrics: {metrics}")
        else:
            print(f"[train_CLIP] WARNING: checkpoint not found at "
                  f"{args.resume_checkpoint}. Starting from scratch.")

    # ── Training ──────────────────────────────────────────────────────────────
    try:
        trainer.fit()
    except KeyboardInterrupt:
        print("\n[train_CLIP] Interrupted. Saving checkpoint...")
        save_checkpoint(
            model,
            trainer.optimizer,
            trainer.scheduler,
            trainer.current_epoch,
            {},
            os.path.join(config.paths.CHECKPOINT_DIR, 'interrupted.pth'),
        )

    # ── Final test evaluation ─────────────────────────────────────────────────
    print("\n[train_CLIP] Running final evaluation on test set...")
    model.eval()
    test_losses  = []
    test_metrics = {}

    from CLIP.loss.losses import CLIPLoss, CheXpertLoss, compute_total_loss
    clip_loss_fn     = CLIPLoss(config)
    chexpert_loss_fn = CheXpertLoss(config)

    device = torch.device(config.system.DEVICE)
    with torch.no_grad():
        for batch in test_loader:
            images          = batch['images'].to(device)
            input_ids       = batch['input_ids'].to(device)
            attention_mask  = batch['attention_mask'].to(device)
            chexpert_labels = batch['chexpert_labels'].to(device)

            image_embeds, text_embeds, chexpert_logits = model(
                images, input_ids, attention_mask)
            loss, metrics = compute_total_loss(
                image_embeds, text_embeds, chexpert_logits, chexpert_labels,
                config, clip_loss_fn, chexpert_loss_fn,
            )
            test_losses.append(loss.item())
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    test_metrics.setdefault(k, []).append(v)

    avg_test = {k: sum(v) / len(v) for k, v in test_metrics.items()}
    avg_test['test_loss'] = sum(test_losses) / len(test_losses)

    print(f"\n[train_CLIP] Test results:")
    for k, v in sorted(avg_test.items()):
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    import json
    results_path = os.path.join(config.paths.OUTPUT_DIR, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(avg_test, f, indent=2)
    print(f"  Saved to {results_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\nOutputs:      {config.paths.OUTPUT_DIR}")
    print(f"Checkpoints:  {config.paths.CHECKPOINT_DIR}")
    print(f"Logs:         {config.paths.LOG_DIR}")
    print(f"\nTensorBoard: tensorboard --logdir {config.paths.LOG_DIR}")


if __name__ == "__main__":
    main()