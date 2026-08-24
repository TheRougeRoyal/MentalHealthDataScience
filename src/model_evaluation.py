"""Offline evaluation metrics for risk-model validation.

These metrics are for validation reports only. They do not calibrate or
establish clinical performance without a representative labelled dataset.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Sequence


def _validate_binary(labels: Sequence[int], scores: Sequence[float]) -> None:
    if len(labels) != len(scores) or not labels:
        raise ValueError("labels and scores must be non-empty and the same length")
    if any(label not in (0, 1) for label in labels):
        raise ValueError("labels must contain only 0 or 1")
    if any(score < 0 or score > 1 for score in scores):
        raise ValueError("scores must be probabilities in [0, 1]")


def brier_score(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Return mean squared probability error; lower is better."""
    _validate_binary(labels, scores)
    return sum((score - label) ** 2 for label, score in zip(labels, scores)) / len(labels)


def expected_calibration_error(
    labels: Sequence[int], scores: Sequence[float], bins: int = 10,
) -> float:
    """Return equal-width expected calibration error; lower is better."""
    _validate_binary(labels, scores)
    if bins < 1:
        raise ValueError("bins must be positive")
    buckets: list[list[tuple[int, float]]] = [[] for _ in range(bins)]
    for label, score in zip(labels, scores):
        index = min(int(score * bins), bins - 1)
        buckets[index].append((label, score))
    total = len(labels)
    return sum(
        len(bucket) / total * abs(
            sum(score for _, score in bucket) / len(bucket)
            - sum(label for label, _ in bucket) / len(bucket)
        )
        for bucket in buckets if bucket
    )


def classification_metrics(
    labels: Sequence[int], scores: Sequence[float], threshold: float = 0.5,
) -> dict[str, float]:
    """Return sensitivity, specificity, PPV, and NPV at a threshold."""
    _validate_binary(labels, scores)
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    predicted = [score >= threshold for score in scores]
    tp = sum(label == 1 and prediction for label, prediction in zip(labels, predicted))
    tn = sum(label == 0 and not prediction for label, prediction in zip(labels, predicted))
    fp = sum(label == 0 and prediction for label, prediction in zip(labels, predicted))
    fn = sum(label == 1 and not prediction for label, prediction in zip(labels, predicted))
    return {
        "sensitivity": tp / (tp + fn) if tp + fn else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "ppv": tp / (tp + fp) if tp + fp else 0.0,
        "npv": tn / (tn + fn) if tn + fn else 0.0,
    }


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Return ROC-AUC using the probability-ranking definition."""
    _validate_binary(labels, scores)
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError("ROC-AUC requires both positive and negative labels")
    concordant = sum(
        1.0 if positive_score > negative_score else 0.5 if positive_score == negative_score else 0.0
        for positive_score, positive_label in zip(scores, labels) if positive_label == 1
        for negative_score, negative_label in zip(scores, labels) if negative_label == 0
    )
    return concordant / (positives * negatives)


def subgroup_metrics(
    labels: Sequence[int], scores: Sequence[float], groups: Sequence[str], threshold: float = 0.5,
) -> dict[str, dict[str, float]]:
    """Return classification metrics separately for each supplied subgroup."""
    if len(labels) != len(groups):
        raise ValueError("labels and groups must have the same length")
    grouped: dict[str, list[int]] = defaultdict(list)
    grouped_scores: dict[str, list[float]] = defaultdict(list)
    for label, score, group in zip(labels, scores, groups):
        grouped[group].append(label)
        grouped_scores[group].append(score)
    return {
        group: classification_metrics(grouped[group], grouped_scores[group], threshold)
        for group in sorted(grouped)
    }


def evaluate_predictions(
    labels: Sequence[int], scores: Sequence[float], groups: Iterable[str] | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Build a validation report from labelled predictions."""
    report: dict[str, Any] = {
        "brier_score": brier_score(labels, scores),
        "expected_calibration_error": expected_calibration_error(labels, scores),
        "classification": classification_metrics(labels, scores, threshold),
        "roc_auc": roc_auc(labels, scores),
    }
    if groups is not None:
        report["subgroups"] = subgroup_metrics(labels, scores, list(groups), threshold)
    return report