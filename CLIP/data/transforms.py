"""Image transforms for MIMIC-CXR training and validation.

Changes from original:
  val:   Resize(256, BICUBIC) + CenterCrop(224)
         → Resize((IMAGE_SIZE, IMAGE_SIZE), LANCZOS)   [direct, no crop]
  train: interpolation BICUBIC → LANCZOS
         ratio (0.9, 1.1)     → (1.0, 1.0)             [always square crop]
  no-aug fallback: same direct LANCZOS resize, no CenterCrop

Why no CenterCrop:
  The original Resize(256)+CenterCrop(224) produced asymmetric field-of-view.
  A portrait GT image (~2735×2790) lost ~12.5% of height from both ends after
  crop; a square generated image (512×512) lost only ~6.25%. The direct square
  resize treats all inputs identically, which is required for fair embedding
  comparisons between GT scans and FLUX.2 outputs.
"""

from torchvision import transforms


def get_train_transforms(config):
    """Augmented training transforms."""
    if config.data.USE_AUGMENTATION:
        transform_list = [
            transforms.RandomResizedCrop(
                config.data.IMAGE_SIZE,
                scale=(config.data.RANDOM_CROP_SCALE_MIN, config.data.RANDOM_CROP_SCALE_MAX),
                ratio=(config.data.RANDOM_CROP_RATIO_MIN, config.data.RANDOM_CROP_RATIO_MAX),
                interpolation=transforms.InterpolationMode.LANCZOS,
            ),
        ]
        if config.data.RANDOM_ROTATION:
            transform_list.append(
                transforms.RandomRotation(
                    degrees=config.data.RANDOM_ROTATION_DEGREES,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                )
            )
        if config.data.COLOR_JITTER:
            transform_list.append(
                transforms.ColorJitter(
                    brightness=config.data.COLOR_JITTER_BRIGHTNESS,
                    contrast=config.data.COLOR_JITTER_CONTRAST,
                    saturation=config.data.COLOR_JITTER_SATURATION,
                    hue=config.data.COLOR_JITTER_HUE,
                )
            )
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=config.data.IMAGE_MEAN, std=config.data.IMAGE_STD),
        ])
        if config.data.RANDOM_ERASING:
            transform_list.append(
                transforms.RandomErasing(
                    p=config.data.RANDOM_ERASING_P,
                    scale=(config.data.RANDOM_ERASING_SCALE_MIN, config.data.RANDOM_ERASING_SCALE_MAX),
                    ratio=(config.data.RANDOM_ERASING_RATIO_MIN, config.data.RANDOM_ERASING_RATIO_MAX),
                )
            )
    else:
        transform_list = [
            transforms.Resize(
                (config.data.IMAGE_SIZE, config.data.IMAGE_SIZE),
                interpolation=transforms.InterpolationMode.LANCZOS,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.data.IMAGE_MEAN, std=config.data.IMAGE_STD),
        ]

    return transforms.Compose(transform_list)


def get_val_transforms(config):
    """Deterministic validation / inference transforms.

    Direct LANCZOS resize to (IMAGE_SIZE, IMAGE_SIZE). No CenterCrop.
    All inputs — GT scans and FLUX.2 outputs — go through exactly this
    transform before entering MedCLIP, guaranteeing a fair comparison.
    """
    return transforms.Compose([
        transforms.Resize(
            (config.data.IMAGE_SIZE, config.data.IMAGE_SIZE),
            interpolation=transforms.InterpolationMode.LANCZOS,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.data.IMAGE_MEAN, std=config.data.IMAGE_STD),
    ])