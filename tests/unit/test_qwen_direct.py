import pytest

from openjev_ja.eval.scorers.qwen_direct import validate_answer_tokens


class FakeTokenizer:
    def __init__(self, mapping):
        self.mapping = mapping

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return self.mapping[text]


def test_validate_answer_tokens() -> None:
    assert validate_answer_tokens(FakeTokenizer({"A": [1], "B": [2]}), ["A", "B"]) == [1, 2]


def test_validate_answer_tokens_rejects_multiple_tokens() -> None:
    with pytest.raises(ValueError, match="exactly one token"):
        validate_answer_tokens(FakeTokenizer({"A": [1, 2]}), ["A"])


def test_validate_answer_tokens_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="distinct"):
        validate_answer_tokens(FakeTokenizer({"A": [1], "B": [1]}), ["A", "B"])
