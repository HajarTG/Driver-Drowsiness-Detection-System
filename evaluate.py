"""
Model evaluation and comparison script.

Loads saved model weights, runs inference on the test set, and produces a
comprehensive comparison report with:
- Classification metrics (accuracy, recall, precision, F1, AUC)
- Confusion matrices
- ROC curves
- Side-by-side bar chart

Usage
-----
Evaluate both models::

    python evaluate.py --data_dir data/

Evaluate only CNN::

    python evaluate.py --data_dir data/ --model cnn
"""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from models.cnn_model import build_cnn_model
from models.mobilenet_model import build_mobilenet_model
from utils.data_loader import create_data_generators
from utils.metrics import (
    plot_confusion_matrix,
    plot_model_comparison,
    plot_roc_curves,
    print_classification_report,
)

SAVED_MODELS_DIR = Path("saved_models")
RESULTS_DIR = Path("results")

CNN_WEIGHTS = SAVED_MODELS_DIR / "cnn_best.weights.h5"
MOBILENET_WEIGHTS = SAVED_MODELS_DIR / "mobilenet_best.weights.h5"


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def load_cnn(weights_path: Path) -> tf.keras.Model:
    model = build_cnn_model(input_shape=(64, 64, 3))
    if weights_path.exists():
        model.load_weights(str(weights_path))
        print(f"CNN weights loaded from {weights_path}")
    else:
        print(f"[WARNING] CNN weights not found at {weights_path}. Using random weights.")
    return model


def load_mobilenet(weights_path: Path):
    model, base = build_mobilenet_model(input_shape=(96, 96, 3))
    if weights_path.exists():
        model.load_weights(str(weights_path))
        print(f"MobileNetV2 weights loaded from {weights_path}")
    else:
        print(
            f"[WARNING] MobileNetV2 weights not found at {weights_path}. "
            "Using random weights."
        )
    return model


def evaluate_model(model, test_gen, model_name: str, prefix: str):
    """Run inference and return metrics dict."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    test_gen.reset()
    y_pred_proba = model.predict(test_gen, verbose=1).ravel()
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
    return metrics, y_true, y_pred_proba


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and compare drowsiness detection models."
    )
    parser.add_argument(
        "--model",
        choices=["cnn", "mobilenet", "both"],
        default="both",
        help="Which model(s) to evaluate (default: both).",
    )
    parser.add_argument(
        "--data_dir",
        default="data/",
        help="Root dataset directory (default: data/).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference (default: 32).",
    )
    parser.add_argument(
        "--cnn_weights",
        default=str(CNN_WEIGHTS),
        help=f"Path to CNN weights file (default: {CNN_WEIGHTS}).",
    )
    parser.add_argument(
        "--mobilenet_weights",
        default=str(MOBILENET_WEIGHTS),
        help=f"Path to MobileNetV2 weights file (default: {MOBILENET_WEIGHTS}).",
    )
    args = parser.parse_args()

    results = {}
    roc_data = {}

    if args.model in ("cnn", "both"):
        _, val_gen, test_gen_cnn = create_data_generators(
            args.data_dir,
            image_size=(64, 64),
            batch_size=args.batch_size,
            augment_train=False,
        )
        cnn_model = load_cnn(Path(args.cnn_weights))
        metrics_cnn, y_true_cnn, y_score_cnn = evaluate_model(
            cnn_model, test_gen_cnn, "CNN from Scratch", "cnn"
        )
        results["CNN from Scratch"] = metrics_cnn
        roc_data["CNN from Scratch"] = (y_true_cnn, y_score_cnn)

    if args.model in ("mobilenet", "both"):
        _, val_gen_mn, test_gen_mn = create_data_generators(
            args.data_dir,
            image_size=(96, 96),
            batch_size=args.batch_size,
            augment_train=False,
        )
        mn_model = load_mobilenet(Path(args.mobilenet_weights))
        metrics_mn, y_true_mn, y_score_mn = evaluate_model(
            mn_model, test_gen_mn, "MobileNetV2", "mobilenet"
        )
        results["MobileNetV2"] = metrics_mn
        roc_data["MobileNetV2"] = (y_true_mn, y_score_mn)

    if len(results) > 1:
        plot_model_comparison(
            results,
            save_path=str(RESULTS_DIR / "model_comparison.png"),
        )
        plot_roc_curves(
            roc_data,
            save_path=str(RESULTS_DIR / "roc_curves.png"),
        )
        print("\n📊 Summary")
        print(f"{'Model':<25} {'Accuracy':>10} {'Recall':>10} {'AUC':>10}")
        print("-" * 58)
        for name, m in results.items():
            print(
                f"{name:<25} {m['accuracy']:>10.4f} "
                f"{m['recall_drowsy']:>10.4f} {m['auc']:>10.4f}"
            )

    print(f"\nAll evaluation artefacts saved to: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
