"""Image transforms for MIMIC-CXR training and validation."""

from torchvision import transforms


def get_train_transforms(config):
    """Training transforms with augmentation for high-resolution medical images."""
    if config.data.USE_AUGMENTATION:
        transform_list = [
            transforms.RandomResizedCrop(
                config.data.IMAGE_SIZE,
                scale=(config.data.RANDOM_CROP_SCALE_MIN, config.data.RANDOM_CROP_SCALE_MAX),
                ratio=(config.data.RANDOM_CROP_RATIO_MIN, config.data.RANDOM_CROP_RATIO_MAX),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
        ]

        if config.data.RANDOM_ROTATION:
            transform_list.append(
                transforms.RandomRotation(
                    degrees=config.data.RANDOM_ROTATION_DEGREES,
                    interpolation=transforms.InterpolationMode.BICUBIC,
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
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(config.data.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.data.IMAGE_MEAN, std=config.data.IMAGE_STD),
        ]

    return transforms.Compose(transform_list)


def get_val_transforms(config):
    """Deterministic validation transforms."""
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(config.data.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.data.IMAGE_MEAN, std=config.data.IMAGE_STD),
    ])