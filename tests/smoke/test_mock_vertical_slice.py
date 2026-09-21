from openjev_ja.common import BenchmarkItem
from openjev_ja.eval.runner import run_evaluation
from openjev_ja.eval.scorers import MockScorer


def test_five_item_mock_vertical_slice(tmp_path) -> None:
    items = [BenchmarkItem(str(i), f"質問{i}", ["甲", "乙", "丙", "丁"], i % 4) for i in range(5)]
    output = run_evaluation(
        items, MockScorer(), dataset_name="mmmlu_ja", output_root=tmp_path, run_id="smoke"
    )
    assert (output / "summary.json").is_file()
