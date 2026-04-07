"""Trainer for Medical CLIP.

Bug fixed from original:
  Scheduler T_max was set in mini-batch steps (len(loader) × epochs).
  But scheduler.step() fires every accum_steps mini-batches (once per optimizer
  step). With accum_steps=16, the cosine traversed only 1/16 of its intended
  period — the LR effectively never decayed.

  Fix: compute total_steps = (len(loader) // accum_steps) × epochs.
  T_max and warmup_steps are now in OPTIMIZER STEPS, matching the rate at
  which scheduler.step() is actually called.
"""

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
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.config       = config

        self.device      = torch.device(config.system.DEVICE)
        self.model       = self.model.to(self.device)
        # accum_steps and use_amp must be set BEFORE _setup_scheduler(),
        # which reads self.accum_steps to compute T_max in optimizer steps.
        self.accum_steps = config.training.GRADIENT_ACCUMULATION_STEPS
        self.use_amp     = config.training.USE_AMP and config.system.DEVICE == 'cuda'
        self.scaler      = GradScaler() if self.use_amp else None

        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()

        self.contrastive_loss_fn = CLIPLoss(config)
        self.chexpert_loss_fn    = CheXpertLoss(config)

        self.logger = Logger(config)

        self.current_epoch = 0
        self.global_step   = 0
        self.best_val_loss = float('inf')
        self._first_batch  = True

        eff_batch = config.training.BATCH_SIZE * config.training.GRADIENT_ACCUMULATION_STEPS
        print(f"\n[Trainer] device={self.device} | AMP={self.use_amp} | "
              f"accum={self.accum_steps} | effective_batch={eff_batch} | "
              f"grad_clip={config.training.GRADIENT_CLIP}")

    def _setup_optimizer(self):
        param_groups = self.model.get_param_groups(self.config)
        name = self.config.training.OPTIMIZER.lower()
        wd   = self.config.training.WEIGHT_DECAY

        if name == 'adamw':
            return optim.AdamW(param_groups, weight_decay=wd)
        elif name == 'adam':
            return optim.Adam(param_groups, weight_decay=wd)
        elif name == 'sgd':
            return optim.SGD(param_groups, momentum=0.9, weight_decay=wd)
        raise ValueError(f"Unknown optimizer: {name}")

    def _setup_scheduler(self):
        # total_steps in OPTIMIZER STEPS, not mini-batch steps.
        # scheduler.step() is called once per accum_steps mini-batches.
        # Original bug: T_max = len(loader)*epochs → with accum=16 the scheduler
        # received only 1/16 of the expected calls → cosine never completed.
        steps_per_epoch = len(self.train_loader) // self.accum_steps
        total_steps     = steps_per_epoch * self.config.training.NUM_EPOCHS
        warmup_steps    = (steps_per_epoch * self.config.training.WARMUP_EPOCHS
                           if self.config.training.USE_WARMUP else 0)

        print(f"[Trainer] Scheduler '{self.config.training.SCHEDULER}' | "
              f"optimizer_steps_per_epoch={steps_per_epoch} | "
              f"total_optimizer_steps={total_steps:,} | warmup={warmup_steps:,}")

        if self.config.training.SCHEDULER == 'cosine':
            if warmup_steps > 0:
                warmup = LinearLR(self.optimizer,
                                  start_factor=0.01, end_factor=1.0,
                                  total_iters=warmup_steps)
                cosine = CosineAnnealingLR(self.optimizer,
                                           T_max=total_steps - warmup_steps,
                                           eta_min=self.config.training.SCHEDULER_ETA_MIN)
                return SequentialLR(self.optimizer,
                                    schedulers=[warmup, cosine],
                                    milestones=[warmup_steps])
            return CosineAnnealingLR(self.optimizer, T_max=total_steps,
                                     eta_min=self.config.training.SCHEDULER_ETA_MIN)

        if self.config.training.SCHEDULER == 'step':
            return StepLR(self.optimizer, step_size=steps_per_epoch * 30, gamma=0.1)

        return None

    def _flatten_metrics(self, metrics, parent_key='', sep='_'):
        items = []
        for k, v in metrics.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_metrics(v, new_key, sep=sep).items())
            elif isinstance(v, (int, float)):
                items.append((new_key, v))
        return dict(items)

    def train_epoch(self, epoch):
        self.model.train()
        epoch_metrics = {}
        pbar = tqdm(self.train_loader,
                    desc=f"Epoch {epoch}/{self.config.training.NUM_EPOCHS}")

        for batch_idx, batch in enumerate(pbar):
            images          = batch['images'].to(self.device)
            input_ids       = batch['input_ids'].to(self.device)
            attention_mask  = batch['attention_mask'].to(self.device)
            chexpert_labels = batch['chexpert_labels'].to(self.device)

            # ── DIAGNOSTIC: first batch ───────────────────────────────────────
            if self._first_batch:
                print(f"\n[Trainer] First batch shapes:")
                print(f"  images:          {tuple(images.shape)}")
                print(f"  input_ids:       {tuple(input_ids.shape)}")
                print(f"  attention_mask:  {tuple(attention_mask.shape)}")
                print(f"  chexpert_labels: {tuple(chexpert_labels.shape)}")
                print(f"  images min/max:  {images.min():.3f} / {images.max():.3f}")
                print(f"  real tokens [0]: {int(attention_mask[0].sum())} / {input_ids.shape[1]}")
                self._first_batch = False
            # ─────────────────────────────────────────────────────────────────

            if self.use_amp:
                with autocast():
                    loss, metrics = self._train_step(
                        images, input_ids, attention_mask, chexpert_labels)
            else:
                loss, metrics = self._train_step(
                    images, input_ids, attention_mask, chexpert_labels)

            loss = loss / self.accum_steps

            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % self.accum_steps == 0:
                if self.config.training.GRADIENT_CLIP > 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.training.GRADIENT_CLIP)

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
                'acc':  f"{flat.get('contrastive_avg_acc', flat.get('contrastive_acc', 0)):.3f}",
                'lr':   f"{self.optimizer.param_groups[0]['lr']:.2e}",
            })

            if self.global_step % self.config.system.LOG_INTERVAL == 0:
                self.logger.log_metrics(flat, self.global_step, prefix='train')
                self.logger.log_learning_rate(self.optimizer, self.global_step)

        return {k: sum(v) / len(v) for k, v in epoch_metrics.items()}

    def _train_step(self, images, input_ids, attention_mask, chexpert_labels):
        image_embeds, text_embeds, chexpert_logits = self.model(
            images, input_ids, attention_mask)
        return compute_total_loss(
            image_embeds, text_embeds, chexpert_logits, chexpert_labels,
            self.config, self.contrastive_loss_fn, self.chexpert_loss_fn,
        )

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        val_losses  = []
        val_metrics = {}

        for batch in tqdm(self.val_loader, desc="Validating"):
            images          = batch['images'].to(self.device)
            input_ids       = batch['input_ids'].to(self.device)
            attention_mask  = batch['attention_mask'].to(self.device)
            chexpert_labels = batch['chexpert_labels'].to(self.device)

            image_embeds, text_embeds, chexpert_logits = self.model(
                images, input_ids, attention_mask)
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
        print(f"\n[Trainer] Starting training for {self.config.training.NUM_EPOCHS} epochs")

        for epoch in range(1, self.config.training.NUM_EPOCHS + 1):
            self.current_epoch = epoch
            train_metrics = self.train_epoch(epoch)

            acc_key = 'contrastive_avg_acc' if 'contrastive_avg_acc' in train_metrics \
                      else 'contrastive_acc'
            print(f"\nEpoch {epoch:>3} | "
                  f"train_loss={train_metrics['total_loss']:.4f} | "
                  f"acc={train_metrics.get(acc_key, 0):.3f} | "
                  f"lr={self.optimizer.param_groups[0]['lr']:.2e}")

            if epoch % self.config.evaluation.EVAL_EVERY_N_EPOCHS == 0:
                val_metrics = self.validate(epoch)
                print(f"         val_loss={val_metrics['val_loss']:.4f}")
                self.logger.log_metrics(val_metrics, epoch, prefix='val')

                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    save_checkpoint(
                        self.model, self.optimizer, self.scheduler, epoch, val_metrics,
                        os.path.join(self.config.paths.CHECKPOINT_DIR, 'best_model.pth'),
                        is_best=True,
                    )
                    print(f"         ** new best {self.best_val_loss:.4f} — saved **")

            if epoch % self.config.evaluation.SAVE_EVERY_N_EPOCHS == 0:
                save_checkpoint(
                    self.model, self.optimizer, self.scheduler, epoch, train_metrics,
                    os.path.join(self.config.paths.CHECKPOINT_DIR,
                                 f'checkpoint_epoch_{epoch:04d}.pth'),
                )

        print("\n[Trainer] Training complete.")
        self.logger.close()