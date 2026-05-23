"""Evaluation metrics with explicit dependency boundaries."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def binary_ranking_metrics(labels: list[int] | np.ndarray, scores: list[float] | np.ndarray) -> dict[str, float]:
    """Return AUROC and AUPRC for binary labels, handling degenerate labels."""

    labels_array = np.asarray(labels, dtype=int)
    scores_array = np.asarray(scores, dtype=float)
    if labels_array.shape != scores_array.shape:
        raise ValueError("labels and scores must have the same shape")

    metrics = {"auprc": float(average_precision_score(labels_array, scores_array))}
    if len(np.unique(labels_array)) < 2:
        metrics["auroc"] = float("nan")
    else:
        metrics["auroc"] = float(roc_auc_score(labels_array, scores_array))
    return metrics


def hits_at_k(labels: list[int] | np.ndarray, scores: list[float] | np.ndarray, k: int) -> float:
    """Compute whether a positive appears in the top-k ranked candidates."""

    if k <= 0:
        raise ValueError("k must be positive")
    labels_array = np.asarray(labels, dtype=int)
    scores_array = np.asarray(scores, dtype=float)
    order = np.argsort(-scores_array)
    top = labels_array[order[: min(k, len(order))]]
    return float(np.any(top == 1))


def expected_calibration_error(
    labels: list[int] | np.ndarray,
    probabilities: list[float] | np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute simple equal-width expected calibration error."""

    labels_array = np.asarray(labels, dtype=float)
    probabilities_array = np.asarray(probabilities, dtype=float)
    if labels_array.shape != probabilities_array.shape:
        raise ValueError("labels and probabilities must have the same shape")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")

    clipped = np.clip(probabilities_array, 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for left, right in zip(bins[:-1], bins[1:]):
        in_bin = (clipped >= left) & (clipped < right if right < 1.0 else clipped <= right)
        if not np.any(in_bin):
            continue
        confidence = float(np.mean(clipped[in_bin]))
        accuracy = float(np.mean(labels_array[in_bin]))
        ece += float(np.mean(in_bin)) * abs(accuracy - confidence)
    return ece
