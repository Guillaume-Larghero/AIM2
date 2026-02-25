"""DataLoader factory for MIMIC-CXR."""

from torch.utils.data import DataLoader
from .dataset import MIMICCXRDataset, collate_fn
from .transforms import get_train_transforms, get_val_transforms


def create_dataloaders(config):
    """Create train, validation, and test dataloaders from config."""
    train_transform = get_train_transforms(config)
    val_transform = get_val_transforms(config)

    dataset_kwargs = dict(
        csv_path=config.paths.DATA_CSV,
        tokenizer_name=config.model.TEXT_ENCODER,
        max_length=config.model.TEXT_MAX_LENGTH,
        combine_sections=config.data.COMBINE_SECTIONS,
        section_separator=config.data.SECTION_SEPARATOR,
        text_fallback=config.data.TEXT_FALLBACK,
        chexpert_labels=config.data.CHEXPERT_LABELS,
    )

    train_dataset = MIMICCXRDataset(split='train', image_transform=train_transform, **dataset_kwargs)
    val_dataset = MIMICCXRDataset(split='validate', image_transform=val_transform, **dataset_kwargs)
    test_dataset = MIMICCXRDataset(split='test', image_transform=val_transform, **dataset_kwargs)

    print(f"\nTrain: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    loader_kwargs = dict(
        batch_size=config.training.BATCH_SIZE,
        num_workers=config.data.NUM_WORKERS,
        pin_memory=config.data.PIN_MEMORY,
        collate_fn=collate_fn,
    )

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        prefetch_factor=config.data.PREFETCH_FACTOR if config.data.NUM_WORKERS > 0 else None,
        persistent_workers=config.data.PERSISTENT_WORKERS if config.data.NUM_WORKERS > 0 else False,
        **loader_kwargs,
    )

    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False, **loader_kwargs)

    print(f"Batches - Train: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)}")

    return train_loader, val_loader, test_loader