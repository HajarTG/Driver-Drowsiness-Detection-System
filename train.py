"""
Training script for the Driver Drowsiness Detection System.

Trains both the CNN from scratch and the MobileNetV2 transfer learning models
on a labelled eye/face image dataset.

Usage
-----
Train CNN from scratch::

    python train.py --model cnn --data_dir data/ --epochs 50

Train MobileNetV2 (feature extraction, then fine-tuning)::

    python train.py --model mobilenet --data_dir data/ --epochs 30 --fine_tune

Train both sequentially::

    python train.py --model both --data_dir data/ --epochs 30

Dataset layout expected in ``data_dir``::

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
"""

import argparse
import os
from pathlib import Path

import numpy as np
import tensorflow as tf

from models.cnn_model import build_cnn_model, get_cnn_callbacks
from models.mobilenet_model import (
    build_mobilenet_model,
    get_mobilenet_callbacks,
    unfreeze_for_fine_tuning,
)
from utils.data_loader import create_data_generators, DrowsinessDataLoader
from utils.metrics import (
    plot_training_history,
    plot_confusion_matrix,
    print_classification_report,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAVED_MODELS_DIR = Path("saved_models")
RESULTS_DIR = Path("results")


# ---------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------

def train_cnn(args):
    """Train the CNN from-scratch model."""
    print("\n" + "=" * 60)
    print("  Training: CNN from Scratch")
    print("=" * 60)

    image_size = (64, 64)
    train_gen, val_gen, test_gen = create_data_generators(
        args.data_dir,
        image_size=image_size,
        batch_size=args.batch_size,
        augment_train=True,
    )

    model = build_cnn_model(input_shape=(*image_size, 3))
    model.summary()
    print(f"\nTotal parameters: {model.count_params():,}")

    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    callbacks = get_cnn_callbacks(
        checkpoint_path=str(SAVED_MODELS_DIR / "cnn_best.weights.h5")
    )

    # Optional class weights for imbalanced datasets
    class_weights = None
    if args.class_weights:
        class_weights = DrowsinessDataLoader.get_class_weights(
            str(Path(args.data_dir) / "train")
        )
        print(f"Class weights: {class_weights}")

    history = model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    # Plots
    plot_training_history(
        history,
        model_name="CNN from Scratch",
        save_path=str(RESULTS_DIR / "cnn_training_history.png"),
    )

    # Evaluation on test set
    _evaluate_and_save(model, test_gen, "CNN from Scratch", "cnn")

    # Save full model
    model.save(str(SAVED_MODELS_DIR / "cnn_final.keras"))
    print(f"Model saved to {SAVED_MODELS_DIR / 'cnn_final.keras'}")

    return model, history


def train_mobilenet(args):
    """Train the MobileNetV2 transfer learning model."""
    print("\n" + "=" * 60)
    print("  Training: MobileNetV2 Transfer Learning")
    print("=" * 60)

    image_size = (96, 96)  # MobileNetV2 prefers larger inputs
    train_gen, val_gen, test_gen = create_data_generators(
        args.data_dir,
        image_size=image_size,
        batch_size=args.batch_size,
        augment_train=True,
    )

    model, base_model = build_mobilenet_model(input_shape=(*image_size, 3))
    model.summary()
    print(f"\nTotal parameters: {model.count_params():,}")
    print(
        f"Trainable parameters (feature extraction phase): "
        f"{sum(tf.size(w).numpy() for w in model.trainable_weights):,}"
    )

    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 1 — Feature extraction (base frozen)
    print("\n--- Phase 1: Feature Extraction (base frozen) ---")
    callbacks_p1 = get_mobilenet_callbacks(
        checkpoint_path=str(SAVED_MODELS_DIR / "mobilenet_phase1.weights.h5")
    )
    history_p1 = model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=val_gen,
        callbacks=callbacks_p1,
        verbose=1,
    )

    if args.fine_tune:
        # Phase 2 — Fine-tuning (unfreeze top layers)
        print("\n--- Phase 2: Fine-Tuning (top layers unfrozen) ---")
        unfreeze_for_fine_tuning(base_model, model, fine_tune_at=100)

        callbacks_p2 = get_mobilenet_callbacks(
            checkpoint_path=str(SAVED_MODELS_DIR / "mobilenet_best.weights.h5")
        )
        fine_tune_epochs = max(10, args.epochs // 3)
        history_p2 = model.fit(
            train_gen,
            epochs=fine_tune_epochs,
            validation_data=val_gen,
            callbacks=callbacks_p2,
            verbose=1,
        )
        # Combine histories for plotting
        combined = _combine_histories(history_p1, history_p2)
        plot_training_history(
            combined,
            model_name="MobileNetV2 (Phase 1 + Fine-Tuning)",
            save_path=str(RESULTS_DIR / "mobilenet_training_history.png"),
        )
    else:
        plot_training_history(
            history_p1,
            model_name="MobileNetV2 Transfer Learning",
            save_path=str(RESULTS_DIR / "mobilenet_training_history.png"),
        )

    # Evaluation on test set
    _evaluate_and_save(model, test_gen, "MobileNetV2", "mobilenet")

    # Save full model
    model.save(str(SAVED_MODELS_DIR / "mobilenet_final.keras"))
    print(f"Model saved to {SAVED_MODELS_DIR / 'mobilenet_final.keras'}")

    return model, history_p1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _evaluate_and_save(model, test_gen, model_name: str, prefix: str):
    """Run evaluation on the test generator and save confusion matrix."""
    print(f"\nEvaluating {model_name} on test set …")
    test_gen.reset()
    y_pred_proba = model.predict(test_gen, verbose=0).ravel()
    y_true = test_gen.classes

    metrics = print_classification_report(
        y_true, y_pred_proba, model_name=model_name
    )
    plot_confusion_matrix(
        y_true,
        (y_pred_proba >= 0.5).astype(int),
        model_name=model_name,
        save_path=str(RESULTS_DIR / f"{prefix}_confusion_matrix.png"),
    )
    return metrics


class _CombinedHistory:
    """Minimal container to merge two Keras History objects for plotting."""

    def __init__(self, h1, h2):
        self.history = {}
        for key in h1.history:
            self.history[key] = h1.history[key] + h2.history.get(key, [])


def _combine_histories(h1, h2):
    return _CombinedHistory(h1, h2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train drowsiness detection models."
    )
    parser.add_argument(
        "--model",
        choices=["cnn", "mobilenet", "both"],
        default="both",
        help="Which model(s) to train (default: both).",
    )
    parser.add_argument(
        "--data_dir",
        default="data/",
        help="Root directory of the dataset (default: data/).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Maximum training epochs per phase (default: 30).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size (default: 32).",
    )
    parser.add_argument(
        "--fine_tune",
        action="store_true",
        default=True,
        help="Run fine-tuning phase for MobileNetV2 (default: True).",
    )
    parser.add_argument(
        "--no_fine_tune",
        dest="fine_tune",
        action="store_false",
        help="Skip fine-tuning phase for MobileNetV2.",
    )
    parser.add_argument(
        "--class_weights",
        action="store_true",
        default=False,
        help="Apply class weights to handle dataset imbalance.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Reproducibility
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    # GPU memory growth to avoid OOM
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    print(f"TensorFlow version : {tf.__version__}")
    print(f"GPUs available     : {len(tf.config.list_physical_devices('GPU'))}")
    print(f"Dataset directory  : {args.data_dir}")

    if args.model in ("cnn", "both"):
        train_cnn(args)

    if args.model in ("mobilenet", "both"):
        train_mobilenet(args)

    print("\nTraining complete. Results saved to:", RESULTS_DIR)


if __name__ == "__main__":
    main()
