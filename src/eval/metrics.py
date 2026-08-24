"""
metrics.py — evaluation metrics for binary anomaly detection.

Mirrors the metrics the original MATLAB study reported (accuracy, F-score, MCC,
RMSE) and adds the ones that matter for imbalanced security data and for a
generalization study: balanced accuracy and AUC, plus a helper to compute the
generalization gap between in-domain and cross-domain scores.
Labels are +1 (attack) / -1 (normal).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (balanced_accuracy_score, f1_score,
                             matthews_corrcoef, roc_auc_score)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray,
             scores: np.ndarray | None = None) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    acc = float((y_true == y_pred).mean())
    # F1 / MCC with attack (+1) as the positive class.
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    mcc = float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0
    bal = float(balanced_accuracy_score(y_true, y_pred))
    rmse = float(np.sqrt(((y_true - y_pred) ** 2).mean()))
    out = {"accuracy": acc, "f1": f1, "mcc": mcc,
           "balanced_acc": bal, "rmse": rmse}
    if scores is not None and len(np.unique(y_true)) > 1:
        try:
            out["auc"] = float(roc_auc_score((y_true == 1).astype(int), scores))
        except ValueError:
            out["auc"] = float("nan")
    return out


def generalization_gap(in_domain: dict, cross_domain: dict,
                       key: str = "f1") -> float:
    """How much a metric drops from in-domain to cross-domain (shifted) data.
    The headline number of the study: smaller is better generalization."""
    return in_domain[key] - cross_domain[key]


def format_row(name: str, m: dict) -> str:
    return (f"{name:<22} acc={m['accuracy']:.4f}  f1={m['f1']:.4f}  "
            f"mcc={m['mcc']:.4f}  bal_acc={m['balanced_acc']:.4f}"
            + (f"  auc={m['auc']:.4f}" if "auc" in m else ""))
