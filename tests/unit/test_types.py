import pytest

from openjev_ja.common import BenchmarkItem, ScoreResult


def test_benchmark_item_validates_gold_index() -> None:
    with pytest.raises(ValueError, match="gold_index"):
        BenchmarkItem("x", "質問", ["a", "b"], 2)


def test_score_result_validates_probability_length() -> None:
    with pytest.raises(ValueError, match="equal length"):
        ScoreResult([1.0, 2.0], 1, 1.0, probabilities=[1.0])
