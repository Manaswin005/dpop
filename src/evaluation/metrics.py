"""Classification metrics and confusion-matrix plotting (Stages 6–7)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)


def compute_metrics(y_true, proba, class_names) -> dict:
    """All required metrics from true labels (int codes) and class probabilities."""
    y_pred = np.argmax(proba, axis=1)
    labels = list(range(len(class_names)))
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    # ROC-AUC needs every class present in y_true and valid probabilities
    if len(np.unique(y_true)) == len(class_names) and np.all(np.isfinite(proba)):
        m["roc_auc_ovr_macro"] = roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=labels)
    else:
        m["roc_auc_ovr_macro"] = None
    return {k: (round(float(v), 4) if v is not None else None) for k, v in m.items()}


def report_text(y_true, proba, class_names) -> str:
    return classification_report(y_true, np.argmax(proba, axis=1), labels=range(len(class_names)),
                                 target_names=class_names, digits=3, zero_division=0)


def plot_confusion_grid(results: dict, y_true, class_names, path) -> None:
    """One normalised confusion matrix per model (rows = true class)."""
    n = len(results)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.8 * rows), squeeze=False)
    short = [c.replace("Partially Successful", "Partial") for c in class_names]
    for ax, (name, proba) in zip(axes.flat, results.items()):
        cm = confusion_matrix(y_true, np.argmax(proba, axis=1), labels=range(len(class_names)))
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.55 else "#222")
        ax.set_xticks(range(len(short)), short, fontsize=8)
        ax.set_yticks(range(len(short)), short, fontsize=8)
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("True", fontsize=8)
        ax.set_title(name, fontsize=10, loc="left")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
