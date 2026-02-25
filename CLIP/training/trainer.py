"""Trainer for Medical CLIP."""

import os
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, StepLR
from tqdm import tqdm

from ..loss.losses import CLIPLoss, CheXpertLoss, compute_total_loss
from ..utils.logging import Logger
from ..utils.checkpoint import save_checkpoint


class Trainer:
    """Training orchestrator with mixed precision, gradient accumulation, and scheduling."""

    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.device = torch.device(config.system.DEVICE)
        self.model = self.model.to(self.device)

        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()

        self.contrastive_loss_fn = CLIPLoss(config)
        self.chexpert_loss_fn = CheXpertLoss(config)

        self.use_amp = config.training.USE_AMP and config.system.DEVICE == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None
        self.accum_steps = config.training.GRADIENT_ACCUMULATION_STEPS

        self.logger = Logger(config)

        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        print(f"Trainer ready | device={self.device} | AMP={self.use_amp} | "
              f"accum={self.accum_steps} | grad_clip={config.training.GRADIENT_CLIP}")

    def _setup_optimizer(self):
        param_groups = self.model.get_param_groups(self.config)
        name = self.config.training.OPTIMIZER.lower()
        wd = self.config.training.WEIGHT_DECAY

        if name == 'adamw':
            return optim.AdamW(param_groups, weight_decay=wd)
        elif name == 'adam':
            return optim.Adam(param_groups, weight_decay=wd)
        elif name == 'sgd':
            return optim.SGD(param_groups, momentum=0.9, weight_decay=wd)
        else:
            raise ValueError(f"Unknown optimizer: {name}")

    def _setup_scheduler(self):
        total_steps = len(self.train_loader) * self.config.training.NUM_EPOCHS
        warmup_steps = (len(self.train_loader) * self.config.training.WARMUP_EPOCHS
                        if self.config.training.USE_WARMUP else 0)

        sched_type = self.config.training.SCHEDULER

        if sched_type == 'cosine':

            if warmup_steps > 0:
                warmup = LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
                cosine = CosineAnnealingLR(self.optimizer, T_max=total_steps - warmup_steps,
                                           eta_min=self.config.training.SCHEDULER_ETA_MIN)
                return SequentialLR(self.optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
            else:
                return CosineAnnealingLR(self.optimizer, T_max=total_steps,
                                         eta_min=self.config.training.SCHEDULER_ETA_MIN)

        elif sched_type == 'step':
            return StepLR(self.optimizer, step_size=len(self.train_loader) * 30, gamma=0.1)

        return None

    def _flatten_metrics(self, metrics, parent_key='', sep='_'):
        """Flatten nested metric dicts for logging."""
        items = []
        for k, v in metrics.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_metrics(v, new_key, sep=sep).items())
            elif isinstance(v, (int, float)):
                items.append((new_key, v))
        return dict(items)

    def train_epoch(self, epoch):
        """Run one training epoch."""
        self.model.train()
        epoch_metrics = {}

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config.training.NUM_EPOCHS}")

        for batch_idx, batch in enumerate(pbar):
            images = batch['images'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            chexpert_labels = batch['chexpert_labels'].to(self.device)

            if self.use_amp:
                with autocast():
                    loss, metrics = self._train_step(images, input_ids, attention_mask, chexpert_labels)
            else:
                loss, metrics = self._train_step(images, input_ids, attention_mask, chexpert_labels)

            loss = loss / self.accum_steps

            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % self.accum_steps == 0:
                if self.config.training.GRADIENT_CLIP > 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.GRADIENT_CLIP)

                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

                if self.scheduler is not None:
                    self.scheduler.step()

                self.global_step += 1

            flat = self._flatten_metrics(metrics)
            for key, value in flat.items():
                epoch_metrics.setdefault(key, []).append(value)

            pbar.set_postfix({
                'loss': f"{metrics['total_loss']:.4f}",
                'acc': f"{flat.get('contrastive_avg_acc', flat.get('contrastive_acc', 0)):.3f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}",
            })

            if self.global_step % self.config.system.LOG_INTERVAL == 0:
                self.logger.log_metrics(flat, self.global_step, prefix='train')
                self.logger.log_learning_rate(self.optimizer, self.global_step)

        return {k: sum(v) / len(v) for k, v in epoch_metrics.items()}

    def _train_step(self, images, input_ids, attention_mask, chexpert_labels):
        """Single forward + loss computation."""
        image_embeds, text_embeds, chexpert_logits = self.model(images, input_ids, attention_mask)
        return compute_total_loss(
            image_embeds, text_embeds, chexpert_logits, chexpert_labels,
            self.config, self.contrastive_loss_fn, self.chexpert_loss_fn,
        )

    @torch.no_grad()
    def validate(self, epoch):
        """Run validation and return averaged metrics."""
        self.model.eval()
        val_losses = []
        val_metrics = {}

        for batch in tqdm(self.val_loader, desc="Validating"):
            images = batch['images'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            chexpert_labels = batch['chexpert_labels'].to(self.device)

            image_embeds, text_embeds, chexpert_logits = self.model(images, input_ids, attention_mask)

            loss, metrics = compute_total_loss(
                image_embeds, text_embeds, chexpert_logits, chexpert_labels,
                self.config, self.contrastive_loss_fn, self.chexpert_loss_fn,
            )

            val_losses.append(loss.item())
            flat = self._flatten_metrics(metrics)
            for key, value in flat.items():
                val_metrics.setdefault(key, []).append(value)

        avg = {k: sum(v) / len(v) for k, v in val_metrics.items()}
        avg['val_loss'] = sum(val_losses) / len(val_losses)
        return avg

    def fit(self):
        """Complete training loop."""
        print(f"\nStarting training for {self.config.training.NUM_EPOCHS} epochs")

        for epoch in range(1, self.config.training.NUM_EPOCHS + 1):
            self.current_epoch = epoch

            train_metrics = self.train_epoch(epoch)
            print(f"\nEpoch {epoch} - train loss: {train_metrics['total_loss']:.4f} | "
                  f"acc: {train_metrics.get('contrastive_avg_acc', train_metrics.get('contrastive_acc', 0)):.3f}")

            if epoch % self.config.evaluation.EVAL_EVERY_N_EPOCHS == 0:
                val_metrics = self.validate(epoch)
                print(f"  val loss: {val_metrics['val_loss']:.4f}")

                self.logger.log_metrics(val_metrics, epoch, prefix='val')

                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    save_checkpoint(
                        self.model, self.optimizer, self.scheduler, epoch, val_metrics,
                        os.path.join(self.config.paths.CHECKPOINT_DIR, 'best_model.pth'),
                        is_best=True,
                    )

            if epoch % self.config.evaluation.SAVE_EVERY_N_EPOCHS == 0:
                save_checkpoint(
                    self.model, self.optimizer, self.scheduler, epoch, train_metrics,
                    os.path.join(self.config.paths.CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}.pth'),
                )

        print("\nTraining complete.")
        self.logger.close()