from types import SimpleNamespace

import pytest

from openjev_ja.methods.lev.scorer import build_questions, parse_answer


def test_lev_questions_match_the_jev_wire_format():
    state, questions = build_questions("noul", "空は青い？", ["いいえ", "はい"])
    assert state == "" and questions["answer"] == {"type": "noul", "instructions": "空は青い？"}
    state, questions = build_questions("choice", "質問", ["a", "b"])
    assert state == "質問"
    assert questions["answer"]["criteria"] == {"option_0": "a", "option_1": "b"}
    state, questions = build_questions("score", "文章", ["低", "高"])
    assert questions["answer"]["criteria"] == ["低", "高"]
    with pytest.raises(ValueError):
        build_questions("rank", "q", ["a", "b"])


def test_lev_answers_are_parsed_per_primitive():
    _, probabilities, predicted = parse_answer("noul", SimpleNamespace(noul=0.8), 2)
    assert probabilities == pytest.approx([0.2, 0.8]) and predicted == 1
    answer = SimpleNamespace(choice="option_1", probabilities={"option_0": 0.1, "option_1": 0.9})
    _, probabilities, predicted = parse_answer("choice", answer, 2)
    assert probabilities == [0.1, 0.9] and predicted == 1
    answer = SimpleNamespace(probabilities={0: 0.1, 1: 0.2, 2: 0.7})
    _, probabilities, predicted = parse_answer("score", answer, 3)
    assert probabilities == [0.1, 0.2, 0.7] and predicted == 2
