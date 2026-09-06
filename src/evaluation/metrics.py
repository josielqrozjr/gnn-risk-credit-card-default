"""Discrimination and probability-quality metrics for the MVP."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


def select_f1_threshold(targets: np.ndarray, probabilities: np.ndarray) -> float:
    """Select a threshold using validation data only."""
    precision, recall, thresholds = precision_recall_curve(targets, probabilities)
    if thresholds.size == 0:
        return 0.5
    scores = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], np.finfo(float).eps
    )
    return float(thresholds[int(np.nanargmax(scores))])


def expected_calibration_error(
    targets: np.ndarray, probabilities: np.ndarray, bins: int
) -> float:
    """Compute equal-width Expected Calibration Error."""
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    bin_ids = np.minimum(np.digitize(probabilities, boundaries[1:-1]), bins - 1)
    error = 0.0
    for bin_id in range(bins):
        mask = bin_ids == bin_id
        if mask.any():
            error += mask.mean() * abs(targets[mask].mean() - probabilities[mask].mean())
    return float(error)


def amex_score(targets: np.ndarray, probabilities: np.ndarray) -> float:
    """Compute the official AMEX competition metric."""
    targets = np.asarray(targets, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=float)
    if np.unique(targets).size != 2:
        raise ValueError("AMEX score requires both target classes")

    order = np.argsort(-probabilities, kind="stable")
    sorted_targets = targets[order]
    weights = np.where(sorted_targets == 0, 20.0, 1.0)
    cutoff = 0.04 * weights.sum()
    top_mask = np.cumsum(weights) <= cutoff
    top_four_capture = sorted_targets[top_mask].sum() / sorted_targets.sum()

    predicted_gini = _weighted_gini(targets, probabilities)
    perfect_gini = _weighted_gini(targets, targets.astype(float))
    return float(0.5 * (predicted_gini / perfect_gini + top_four_capture))


def evaluate_binary_classifier(
    targets: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    ece_bins: int,
) -> dict[str, float | int]:
    """Evaluate a held-out binary classification partition."""
    predictions = (probabilities >= threshold).astype(np.int8)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        targets, predictions, labels=[0, 1]
    ).ravel()
    return {
        "roc_auc": float(roc_auc_score(targets, probabilities)),
        "pr_auc": float(average_precision_score(targets, probabilities)),
        "f1": float(f1_score(targets, predictions)),
        "brier_score": float(brier_score_loss(targets, probabilities)),
        "ece": expected_calibration_error(targets, probabilities, ece_bins),
        "amex_score": amex_score(targets, probabilities),
        "threshold": float(threshold),
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "true_positive": int(true_positive),
    }


def _weighted_gini(targets: np.ndarray, ranking: np.ndarray) -> float:
    order = np.argsort(-ranking, kind="stable")
    sorted_targets = targets[order]
    weights = np.where(sorted_targets == 0, 20.0, 1.0)
    cumulative_weight = np.cumsum(weights) / weights.sum()
    total_positive = np.sum(sorted_targets * weights)
    lorentz = np.cumsum(sorted_targets * weights) / total_positive
    return float(np.sum((lorentz - cumulative_weight) * weights))