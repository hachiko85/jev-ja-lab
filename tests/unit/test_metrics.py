import pytest

from openjev_ja.eval.metrics import (
    binary_classification_metrics,
    calibration_metrics,
    ordinal_metrics,
    percentile,
    summarize,
)


def test_percentile_interpolates() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5


def test_perfect_calibration_metrics() -> None:
    metrics = calibration_metrics([[1.0, 0.0], [0.0, 1.0]], [0, 1])
    assert metrics["nll"] == pytest.approx(0.0)
    assert metrics["brier_score"] == pytest.approx(0.0)
    assert metrics["ece"] == pytest.approx(0.0)


def test_nli_scores_do_not_produce_calibration_metrics() -> None:
    summary = summarize(
        [{"predicted_index": 0, "gold_index": 0, "latency_ms": 1.0, "probabilities": None}],
        1.0,
    )
    assert summary["accuracy"] == 1.0
    assert "nll" not in summary


def test_noul_metrics() -> None:
    metrics = binary_classification_metrics([1, 1, 0, 0], [1, 0, 1, 0])
    assert metrics == {"precision": 0.5, "recall": 0.5, "f1": 0.5}


def test_perfect_ordinal_metrics() -> None:
    metrics = ordinal_metrics([0, 1, 2, 3], [0, 1, 2, 3])
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["spearman"] == pytest.approx(1.0)
    assert metrics["quadratic_weighted_kappa"] == pytest.approx(1.0)
    assert metrics["normalized_quadratic_weighted_kappa"] == pytest.approx(1.0)
