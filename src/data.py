"""Input pipeline: loads images from folders, applies augmentation to the training split only.

Images are kept as float32 in the raw 0-255 range. Each model has its own
``preprocess`` layer inside the network (see models.py), so the same raw pixels are
used in training, evaluation and the web app - no train/serve mismatch.
"""
from __future__ import annotations

import keras
import tensorflow as tf
from keras import layers

from . import config
from .utils import find_dataset_root

AUTOTUNE = tf.data.AUTOTUNE


def build_augmentation() -> keras.Sequential:
    """Mild, anatomically plausible augmentation.

    Horizontal flips are deliberately NOT used: a flipped chest X-ray puts the heart
    on the right side, which almost never occurs in real data.
    """
    return keras.Sequential(
        [
            layers.RandomRotation(0.05),                # about +/- 18 degrees
            layers.RandomZoom(0.10),
            layers.RandomTranslation(0.05, 0.05),
            layers.RandomContrast(0.15),
        ],
        name="augmentation",
    )


def _read_folder(folder, img_size, batch_size, shuffle, **kwargs):
    return keras.utils.image_dataset_from_directory(
        folder,
        labels="inferred",
        label_mode="binary",
        class_names=config.CLASS_NAMES,
        color_mode="rgb",                  # grayscale X-rays are replicated to 3 channels
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=shuffle,
        **kwargs,
    )


def load_train_val(
    data_dir=config.DATA_DIR,
    img_size: int = config.IMG_SIZE,
    batch_size: int = config.BATCH_SIZE,
    val_split: float = config.VAL_SPLIT,
    seed: int = config.SEED,
    augment: bool = True,
):
    """Return (train_ds, val_ds, dataset_root).

    The official ``val/`` folder of the Kaggle dataset only has 16 images, which makes
    validation metrics very noisy. We therefore carve a proper validation split out of
    ``train/`` and keep ``test/`` completely untouched for the final evaluation.
    """
    root = find_dataset_root(data_dir)
    train_ds, val_ds = _read_folder(
        root / "train",
        img_size,
        batch_size,
        shuffle=True,
        validation_split=val_split,
        subset="both",
        seed=seed,
    )
    if augment:
        augmentation = build_augmentation()
        train_ds = train_ds.map(
            lambda x, y: (augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE
        )
    return train_ds.prefetch(AUTOTUNE), val_ds.prefetch(AUTOTUNE), root


def load_test(
    data_dir=config.DATA_DIR,
    img_size: int = config.IMG_SIZE,
    batch_size: int = config.BATCH_SIZE,
):
    """Return (test_ds, file_paths). ``shuffle=False`` keeps predictions aligned with file_paths."""
    root = find_dataset_root(data_dir)
    ds = _read_folder(root / "test", img_size, batch_size, shuffle=False)
    file_paths = list(getattr(ds, "file_paths", []))
    return ds.prefetch(AUTOTUNE), file_paths
