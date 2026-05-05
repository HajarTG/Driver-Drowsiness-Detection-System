"""
Evaluation metrics and visualisation utilities.

Provides helpers for:
- Plotting training/validation curves
- Confusion matrix visualisation
- Full classification report (precision, recall, F1, AUC)
- Model comparison bar charts
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (safe for scripts & CI)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from pathlib import Path


# ---------------------------------------------------------------------------
# Training history
# ---------------------------------------------------------------------------

def plot_training_history(history, model_name: str = "Model", save_path: str = None):
    """
    Plot training and validation accuracy/loss curves.

    Args:
        history: Keras ``History`` object returned by ``model.fit()``.
        model_name (str): Title prefix for the plots.
        save_path (str | None): If provided, saves the figure to this path.
                                Otherwise, displays it interactively.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{model_name} — Training History", fontsize=14, fontweight="bold")

    # Accuracy
    axes[0].plot(history.history["accuracy"], label="Train Accuracy", linewidth=2)
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy", linewidth=2)
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history["loss"], label="Train Loss", linewidth=2)
    axes[1].plot(history.history["val_loss"], label="Val Loss", linewidth=2)
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names=None,
    model_name: str = "Model",
    save_path: str = None,
    normalize: bool = True,
):
    """
    Plot a confusion matrix heatmap.

    Args:
        y_true (array-like): True binary labels.
        y_pred (array-like): Predicted binary labels (after thresholding).
        class_names (list | None): Display names for the classes.
                                   Defaults to ``["Awake", "Drowsy"]``.
        model_name (str): Title prefix.
        save_path (str | None): Save figure path; shows interactively if None.
        normalize (bool): Normalise confusion matrix values to proportions.
    """
    class_names = class_names or ["Awake", "Drowsy"]
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2%"
        vmax = 1.0
    else:
        cm_display = cm
        fmt = "d"
        vmax = cm.max()

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0,
        vmax=vmax,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title(f"{model_name} — Confusion Matrix", fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    _save_or_show(fig, save_path)
    return cm


# ---------------------------------------------------------------------------
# Classification report
# ---------------------------------------------------------------------------

def print_classification_report(
    y_true,
    y_pred_proba,
    threshold: float = 0.5,
    class_names=None,
    model_name: str = "Model",
):
    """
    Print a detailed classification report and compute AUC.

    Args:
        y_true (array-like): True binary labels.
        y_pred_proba (array-like): Predicted probabilities for the positive
                                   (Drowsy) class.
        threshold (float): Decision threshold for converting probabilities to
                           binary predictions. Default 0.5.
        class_names (list | None): Class display names.
        model_name (str): Header label.

    Returns:
        dict: Dictionary with ``accuracy``, ``auc``, ``recall_drowsy``,
              ``precision_drowsy``, and ``f1_drowsy`` keys.
    """
    class_names = class_names or ["Awake", "Drowsy"]
    y_pred = (np.array(y_pred_proba) >= threshold).astype(int)

    auc = roc_auc_score(y_true, y_pred_proba)

    print(f"\n{'='*60}")
    print(f"  {model_name} — Classification Report")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    print(f"  ROC-AUC : {auc:.4f}")
    print(f"{'='*60}\n")

    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )
    return {
        "accuracy": report["accuracy"],
        "auc": auc,
        "recall_drowsy": report["Drowsy"]["recall"],
        "precision_drowsy": report["Drowsy"]["precision"],
        "f1_drowsy": report["Drowsy"]["f1-score"],
    }


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------

def plot_model_comparison(results: dict, save_path: str = None):
    """
    Bar chart comparing key metrics across multiple models.

    Args:
        results (dict): ``{model_name: metrics_dict}`` where each
                        ``metrics_dict`` has at least the keys
                        ``accuracy``, ``recall_drowsy``, and ``auc``.
        save_path (str | None): Save figure path; shows interactively if None.

    Example:
        >>> results = {
        ...     "CNN": {"accuracy": 0.9899, "recall_drowsy": 0.98, "auc": 0.997},
        ...     "MobileNetV2": {"accuracy": 0.9701, "recall_drowsy": 0.94, "auc": 0.991},
        ... }
        >>> plot_model_comparison(results, save_path="comparison.png")
    """
    metrics = ["accuracy", "recall_drowsy", "auc"]
    labels = ["Accuracy", "Recall (Drowsy)", "ROC-AUC"]
    model_names = list(results.keys())
    x = np.arange(len(metrics))
    width = 0.8 / max(len(model_names), 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, name in enumerate(model_names):
        vals = [results[name].get(m, 0.0) for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=name, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Driver Drowsiness Detection", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_roc_curves(models_data: dict, save_path: str = None):
    """
    Plot ROC curves for multiple models on the same axes.

    Args:
        models_data (dict): ``{model_name: (y_true, y_score)}`` mapping.
        save_path (str | None): Save figure path; shows interactively if None.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    for name, (y_true, y_score) in models_data.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Driver Drowsiness Detection", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _save_or_show(fig, save_path):
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()
    plt.close(fig)
