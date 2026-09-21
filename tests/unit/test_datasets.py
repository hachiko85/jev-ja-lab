import pytest

from openjev_ja.eval.datasets.adapters import convert_row, numeric_options


@pytest.mark.parametrize(
    ("name", "row", "expected_options", "expected_gold"),
    [
        (
            "mmmlu_ja",
            {"Question": "問い", "A": "甲", "B": "乙", "C": "丙", "D": "丁", "Answer": "B"},
            ["甲", "乙", "丙", "丁"],
            1,
        ),
        (
            "jmmlu",
            {"question": "問い", "A": "甲", "B": "乙", "C": "丙", "D": "丁", "answer": "D"},
            ["甲", "乙", "丙", "丁"],
            3,
        ),
        (
            "jcommonsenseqa",
            {
                "q_id": "1",
                "question": "問い",
                **{f"choice{i}": str(i) for i in range(5)},
                "label": 2,
            },
            ["0", "1", "2", "3", "4"],
            2,
        ),
        (
            "xwinograd_ja",
            {"sentence": "_は来た。", "option1": "太郎", "option2": "花子", "answer": "花子"},
            ["太郎", "花子"],
            1,
        ),
    ],
)
def test_schema_conversion(name, row, expected_options, expected_gold) -> None:
    item = convert_row(name, row, 0)
    assert item.options == expected_options
    assert item.gold_index == expected_gold


def test_jgpqa_conversion_keeps_gold_first() -> None:
    row = {
        "Question": "問い",
        "Correct Answer": "正解",
        "Incorrect Answer 1": "誤1",
        "Incorrect Answer 2": "誤2",
        "Incorrect Answer 3": "誤3",
    }
    item = convert_row("jgpqa_diamond", row, 0)
    assert item.options[item.gold_index] == "正解"


def test_jnli_conversion() -> None:
    item = convert_row(
        "jnli", {"sentence1": "猫がいる。", "sentence2": "動物がいる。", "label": "entailment"}, 0
    )
    assert item.gold_index == 0
    assert item.options == ["含意", "矛盾", "中立"]


def test_gsm_distractors_are_deterministic_and_unique() -> None:
    first = numeric_options("42", 10, 7, "item")
    second = numeric_options("42", 10, 7, "item")
    assert first == second
    assert len(set(first[0])) == 10
    assert first[0][first[1]] == "42"


def test_gsm_row_conversion() -> None:
    item = convert_row("gsm8k_ja_mc4", {"question": "計算", "answer_number": 17}, 3)
    assert len(item.options) == 4
    assert item.options[item.gold_index] == "17"
