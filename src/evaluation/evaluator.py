"""
evaluator.py — Phase 10: Model evaluation utilities.

Generates:
  - Severity: Accuracy, Macro Precision/Recall/F1, Confusion Matrix, per-class recall
  - Risk: ROC-AUC, PR-AUC, Precision, Recall, F1, Confusion Matrix, FN count
  - Comparison table across all model families
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
)
from src.utils.config import LOGS_DIR
from src.utils.logger import get_logger

log = get_logger("evaluator", LOGS_DIR)


# ── Severity evaluation ───────────────────────────────────────────────────────

def evaluate_severity(
    y_true: list, y_pred: list,
    class_names: list[str] | None = None,
    title: str = "Severity Model",
    save_cm: Path | None = None,
) -> dict:
    """Full severity evaluation; optionally saves confusion-matrix plot."""
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True,
                                   zero_division=0, target_names=class_names)
    cm = confusion_matrix(y_true, y_pred)

    macro_prec = report["macro avg"]["precision"]
    macro_rec = report["macro avg"]["recall"]
    macro_f1 = report["macro avg"]["f1-score"]

    # Critical-class recall (class index 3)
    critical_recall = None
    if class_names and "Critical" in class_names:
        idx = class_names.index("Critical")
        critical_recall = report.get(str(idx), report.get("Critical", {})).get("recall")

    log.info(f"[{title}] Accuracy={acc:.4f}, Macro-F1={macro_f1:.4f}, "
             f"Macro-Recall={macro_rec:.4f}, Critical-Recall={critical_recall}")

    if save_cm:
        _plot_confusion_matrix(cm, class_names, title, save_cm)

    return {
        "accuracy": acc,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "critical_recall": critical_recall,
        "class_report": report,
        "confusion_matrix": cm.tolist(),
    }


# ── Risk evaluation ───────────────────────────────────────────────────────────

def evaluate_risk(
    y_true: list, y_pred: list, y_prob: list,
    title: str = "Risk Model",
    save_cm: Path | None = None,
) -> dict:
    """Full risk evaluation with ROC-AUC, PR-AUC, FN analysis."""
    roc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    fn = int(cm[1, 0]) if cm.shape == (2, 2) else -1
    fn_rate = fn / max(cm[1].sum(), 1)

    log.info(f"[{title}] ROC-AUC={roc:.4f}, Recall={rec:.4f}, FN={fn}, FN-Rate={fn_rate:.4f}")

    if save_cm:
        _plot_confusion_matrix(cm, ["No Risk", "Risk"], title, save_cm)

    return {
        "roc_auc": roc,
        "pr_auc": pr_auc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "false_negatives": fn,
        "fn_rate": fn_rate,
        "confusion_matrix": cm.tolist(),
    }


# ── Comparison table ──────────────────────────────────────────────────────────

def build_comparison_table(results: list[dict]) -> pd.DataFrame:
    """
    Build the cross-model comparison table.

    Each result dict must contain at minimum:
      model, task, model_family, feature, accuracy (or roc_auc), macro_f1, recall,
      train_time_s, inf_time_s
    """
    def _get(r: dict, *keys, default=np.nan):
        for k in keys:
            v = r.get(k)
            if v is not None:
                return v
        return default

    rows = []
    for r in results:
        rows.append({
            "Model": r.get("model"),
            "Task": r.get("task"),
            "Model Family": r.get("model_family", r.get("family", "ML")),
            "Feature / Input": r.get("feature"),
            # NaN (not 0 / ROC-AUC) when a metric does not apply to that task
            "Accuracy": round(_get(r, "accuracy"), 4),
            "Macro F1": round(_get(r, "macro_f1", "f1"), 4),
            "Recall": round(_get(r, "macro_recall", "recall"), 4),
            "ROC-AUC": round(_get(r, "roc_auc"), 4),
            "Train Time (s)": round(_get(r, "train_time_s", default=0.0), 1),
            "Inf Time (s)": round(_get(r, "inf_time_s", default=0.0), 4),
        })
    return pd.DataFrame(rows)


# ── Plotting helper ───────────────────────────────────────────────────────────

def _plot_confusion_matrix(
    cm: np.ndarray, labels: list[str] | None, title: str, save_path: Path
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {title}", fontsize=14)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    log.info(f"Confusion matrix saved: {save_path}")
