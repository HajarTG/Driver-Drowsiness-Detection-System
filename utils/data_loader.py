"""
Data loading and augmentation utilities for the drowsiness dataset.

Expected dataset directory layout:
    data/
    ├── train/
    │   ├── Awake/
    │   └── Drowsy/
    ├── val/
    │   ├── Awake/
    │   └── Drowsy/
    └── test/
        ├── Awake/
        └── Drowsy/

Compatible datasets:
    - MRL Eye Dataset  (http://mrl.cs.vsb.cz/eyedataset)
    - Driver Drowsiness Dataset (Kaggle)
    - Custom labelled eye images
"""

import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLASS_NAMES = ["Awake", "Drowsy"]
DEFAULT_IMAGE_SIZE = (64, 64)
DEFAULT_BATCH_SIZE = 32
SEED = 42


# ---------------------------------------------------------------------------
# High-level generator factory
# ---------------------------------------------------------------------------

def create_data_generators(
    data_dir: str,
    image_size: tuple = DEFAULT_IMAGE_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    augment_train: bool = True,
):
    """
    Create train, validation and test data generators from a structured
    directory.

    Args:
        data_dir (str): Root directory containing ``train/``, ``val/``, and
                        ``test/`` sub-directories, each with one sub-folder
                        per class (``Awake/`` and ``Drowsy/``).
        image_size (tuple): Target ``(height, width)`` for resizing images.
        batch_size (int): Number of samples per batch.
        augment_train (bool): Apply random augmentation to training data.

    Returns:
        tuple: ``(train_gen, val_gen, test_gen)`` — Keras
               ``DirectoryIterator`` objects.

    Raises:
        FileNotFoundError: If the expected sub-directories are missing.
    """
    data_dir = Path(data_dir)
    for split in ("train", "val", "test"):
        split_path = data_dir / split
        if not split_path.exists():
            raise FileNotFoundError(
                f"Expected directory not found: {split_path}. "
                "Ensure your dataset follows the structure described in the "
                "module docstring."
            )

    if augment_train:
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.15,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode="nearest",
        )
    else:
        train_datagen = ImageDataGenerator(rescale=1.0 / 255)

    val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        str(data_dir / "train"),
        target_size=image_size,
        batch_size=batch_size,
        class_mode="binary",
        classes=CLASS_NAMES,
        shuffle=True,
        seed=SEED,
    )
    val_gen = val_test_datagen.flow_from_directory(
        str(data_dir / "val"),
        target_size=image_size,
        batch_size=batch_size,
        class_mode="binary",
        classes=CLASS_NAMES,
        shuffle=False,
        seed=SEED,
    )
    test_gen = val_test_datagen.flow_from_directory(
        str(data_dir / "test"),
        target_size=image_size,
        batch_size=batch_size,
        class_mode="binary",
        classes=CLASS_NAMES,
        shuffle=False,
        seed=SEED,
    )

    return train_gen, val_gen, test_gen


# ---------------------------------------------------------------------------
# Dataset class (tf.data API alternative)
# ---------------------------------------------------------------------------

class DrowsinessDataLoader:
    """
    Load the drowsiness dataset using the ``tf.data`` API.

    This loader is more efficient than ``ImageDataGenerator`` for large
    datasets and supports prefetching.

    Args:
        data_dir (str): Root directory (same layout as ``create_data_generators``).
        image_size (tuple): Target ``(height, width)`` for resizing.
        batch_size (int): Samples per batch.
        augment_train (bool): Apply random augmentation to the training split.

    Example:
        >>> loader = DrowsinessDataLoader("data/", image_size=(64, 64))
        >>> train_ds, val_ds, test_ds = loader.get_datasets()
    """

    def __init__(
        self,
        data_dir: str,
        image_size: tuple = DEFAULT_IMAGE_SIZE,
        batch_size: int = DEFAULT_BATCH_SIZE,
        augment_train: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.batch_size = batch_size
        self.augment_train = augment_train

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_datasets(self):
        """
        Build and return ``(train_ds, val_ds, test_ds)`` ``tf.data.Dataset``
        objects ready for model training.
        """
        train_ds = self._load_split("train", augment=self.augment_train)
        val_ds = self._load_split("val", augment=False)
        test_ds = self._load_split("test", augment=False)
        return train_ds, val_ds, test_ds

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_split(self, split: str, augment: bool) -> tf.data.Dataset:
        split_dir = str(self.data_dir / split)
        ds = tf.keras.utils.image_dataset_from_directory(
            split_dir,
            labels="inferred",
            label_mode="binary",
            class_names=CLASS_NAMES,
            image_size=self.image_size,
            batch_size=self.batch_size,
            shuffle=(split == "train"),
            seed=SEED,
        )
        ds = ds.map(self._normalize, num_parallel_calls=tf.data.AUTOTUNE)
        if augment:
            ds = ds.map(self._augment, num_parallel_calls=tf.data.AUTOTUNE)
        return ds.prefetch(tf.data.AUTOTUNE)

    @staticmethod
    def _normalize(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    @staticmethod
    def _augment(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
        return image, label

    @staticmethod
    def get_class_weights(train_dir: str) -> dict:
        """
        Compute class weights to handle class imbalance.

        Args:
            train_dir (str): Path to the ``train/`` directory.

        Returns:
            dict: ``{class_index: weight}`` mapping.
        """
        train_dir = Path(train_dir)
        counts = {}
        for idx, cls in enumerate(CLASS_NAMES):
            cls_path = train_dir / cls
            if cls_path.exists():
                counts[idx] = len(list(cls_path.iterdir()))
            else:
                counts[idx] = 0

        total = sum(counts.values())
        n_classes = len(CLASS_NAMES)
        weights = {
            idx: total / (n_classes * cnt) if cnt > 0 else 1.0
            for idx, cnt in counts.items()
        }
        return weights
