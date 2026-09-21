import json

import pytest

from openjev_ja.common import BenchmarkItem
from openjev_ja.eval.runner import run_evaluation
from openjev_ja.eval.scorers import MockScorer


def test_result_serialization(tmp_path) -> None:
    items = [BenchmarkItem("x", "日本語の質問", ["甲", "乙"], 0)]
    output = run_evaluation(
        items, MockScorer(), dataset_name="fixture", output_root=tmp_path, run_id="test-run"
    )
    assert {path.name for path in output.iterdir()} == {
        "metadata.json",
        "predictions.jsonl",
        "summary.json",
    }
    prediction = json.loads((output / "predictions.jsonl").read_text(encoding="utf-8"))
    assert prediction["id"] == "x"
    assert json.loads((output / "summary.json").read_text())["total"] == 1


def test_empty_run_directory_is_reusable(tmp_path) -> None:
    (tmp_path / "test-run").mkdir()
    output = run_evaluation(
        [BenchmarkItem("x", "質問", ["甲", "乙"], 0)],
        MockScorer(),
        dataset_name="fixture",
        output_root=tmp_path,
        run_id="test-run",
    )
    assert (output / "summary.json").is_file()


def test_nonempty_run_directory_is_not_overwritten(tmp_path) -> None:
    output = tmp_path / "test-run"
    output.mkdir()
    (output / "partial.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run_evaluation(
            [BenchmarkItem("x", "質問", ["甲", "乙"], 0)],
            MockScorer(),
            dataset_name="fixture",
            output_root=tmp_path,
            run_id="test-run",
        )
