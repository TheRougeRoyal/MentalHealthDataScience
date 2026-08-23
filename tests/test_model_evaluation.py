"""Tests for offline model-validity metrics."""

import pytest

from src.model_evaluation import (
    brier_score,
    classification_metrics,
    evaluate_predictions,
    expected_calibration_error,
    roc_auc,
    subgroup_metrics,
)


def test_calibration_metrics_are_computed():
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]
    assert brier_score(labels, scores) == pytest.approx(0.025)
    assert expected_calibration_error(labels, scores, bins=2) == pytest.approx(0.15)


def test_classification_and_auc_metrics():
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.8, 0.7, 0.9]
    metrics = classification_metrics(labels, scores)
    assert metrics["sensitivity"] == 1.0
    assert metrics["specificity"] == 0.5
    assert roc_auc(labels, scores) == pytest.approx(0.75)


def test_subgroup_metrics_and_report():
    report = evaluate_predictions(
        [0, 1, 0, 1], [0.1, 0.9, 0.6, 0.8], ["A", "A", "B", "B"]
    )
    assert set(report["subgroups"]) == {"A", "B"}
    assert "roc_auc" in report


def test_auc_requires_both_classes():
    with pytest.raises(ValueError, match="both positive and negative"):
        roc_auc([1, 1], [0.2, 0.8])
