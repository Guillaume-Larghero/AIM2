#!/usr/bin/env python3
"""
Training script for Medical CLIP.

Usage:
    python train_CLIP.py [--epochs N] [--batch_size N] [--lr LR]
"""

import os
import argparse

from CLIP.config.config import Config
from CLIP.data.dataloader import create_dataloaders
from CLIP.model.clip_model import MedicalCLIP
from CLIP.training.trainer import Trainer
from CLIP.utils.checkpoint import save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Train Medical CLIP')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    return parser.parse_args()


def main():
    args = parse_args()

    config = Config()

    if args.epochs is not None:
        config.training.NUM_EPOCHS = args.epochs
    if args.batch_size is not None:
        config.training.BATCH_SIZE = args.batch_size
    if args.lr is not None:
        config.training.LEARNING_RATE = args.lr

    config.print_config()
    config.save(os.path.join(config.paths.OUTPUT_DIR, 'config.json'))

    train_loader, val_loader, test_loader = create_dataloaders(config)
    model = MedicalCLIP(config)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )

    try:
        trainer.fit()
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        save_checkpoint(
            model,
            trainer.optimizer,
            trainer.scheduler,
            trainer.current_epoch,
            {},
            os.path.join(config.paths.CHECKPOINT_DIR, 'interrupted.pth'),
        )

    print(f"\nOutputs: {config.paths.OUTPUT_DIR}")
    print(f"  Checkpoints: {config.paths.CHECKPOINT_DIR}")
    print(f"  Logs: {config.paths.LOG_DIR}")
    print(f"\nTensorBoard: tensorboard --logdir {config.paths.LOG_DIR}")


if __name__ == "__main__":
    main()