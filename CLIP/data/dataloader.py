"""DataLoader factory for Medical CLIP training.

Changes from original:
  - image_size and use_findings_only forwarded from config to MIMICCXRDataset.
  - combine_sections forwarded explicitly (was already there, kept for clarity).
  - drop_last=True on train_loader (keeps batch size fixed for InfoNCE loss).
  - Dataloader summary printed after construction.

Bugs fixed vs previous draft:
  - split='val' corrected to split='validate' (matches CSV split column values).
  - import path corrected: from .transforms (sibling file) not ..transforms.transforms.
"""

from torch.utils.data import DataLoader

from .dataset import MIMICCXRDataset, collate_fn
from .transforms import get_train_transforms, get_val_transforms


def create_dataloaders(config):
    """Build and return (train_loader, val_loader, test_loader)."""
    train_transform = get_train_transforms(config)
    val_transform   = get_val_transforms(config)

    # Shared kwargs passed to all three splits
    dataset_kwargs = dict(
        csv_path=config.paths.DATA_CSV,
        tokenizer_name=config.model.TEXT_ENCODER,
        image_size=config.data.IMAGE_SIZE,
        max_length=config.model.TEXT_MAX_LENGTH,
        use_findings_only=config.data.USE_FINDINGS_ONLY,
        combine_sections=config.data.COMBINE_SECTIONS,
        section_separator=config.data.SECTION_SEPARATOR,
        text_fallback=config.data.TEXT_FALLBACK,
        chexpert_labels=config.data.CHEXPERT_LABELS,
    )

    train_dataset = MIMICCXRDataset(
        split='train', image_transform=train_transform, **dataset_kwargs)
    # 'validate' matches the value in the CSV split column (not 'val')
    val_dataset = MIMICCXRDataset(
        split='validate', image_transform=val_transform, **dataset_kwargs)
    test_dataset = MIMICCXRDataset(
        split='test', image_transform=val_transform, **dataset_kwargs)

    # Shared DataLoader kwargs
    loader_kwargs = dict(
        batch_size=config.training.BATCH_SIZE,
        num_workers=config.data.NUM_WORKERS,
        pin_memory=config.data.PIN_MEMORY,
        collate_fn=collate_fn,
    )

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,   # keeps batch size constant — important for InfoNCE
        prefetch_factor=config.data.PREFETCH_FACTOR if config.data.NUM_WORKERS > 0 else None,
        persistent_workers=config.data.PERSISTENT_WORKERS if config.data.NUM_WORKERS > 0 else False,
        **loader_kwargs,
    )
    val_loader  = DataLoader(val_dataset,  shuffle=False, drop_last=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False, **loader_kwargs)

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    eff_batch = config.training.BATCH_SIZE * config.training.GRADIENT_ACCUMULATION_STEPS
    print(f"\n[DataLoaders] Summary")
    print(f"  batch_size / grad_accum / effective:  "
          f"{config.training.BATCH_SIZE} / {config.training.GRADIENT_ACCUMULATION_STEPS} / {eff_batch}")
    print(f"  train: {len(train_dataset):>7,} samples | {len(train_loader):>5,} batches")
    print(f"  val:   {len(val_dataset):>7,} samples | {len(val_loader):>5,} batches")
    print(f"  test:  {len(test_dataset):>7,} samples | {len(test_loader):>5,} batches")
    print(f"  text mode:   {'FINDINGS only' if config.data.USE_FINDINGS_ONLY else 'FINDINGS+IMPRESSION'}")
    print(f"  image size:  {config.data.IMAGE_SIZE}×{config.data.IMAGE_SIZE}")
    print()

    return train_loader, val_loader, test_loader