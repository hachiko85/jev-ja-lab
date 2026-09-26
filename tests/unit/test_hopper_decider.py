import json
import math

import pytest

from openjev_ja.methods.decider.scorer import build_questions, parse_answer
from openjev_ja.methods.hopper.scorer import (
    INSTRUCTIONS,
    POLICY,
    TEMPERATURES,
    build_request,
    rescale,
)


def _body(primitive, question, options):
    text, shown = build_request(primitive, question, options)
    return json.loads(text), shown


def test_hopper_choice_shows_options_verbatim_with_letters():
    body, shown = _body("choice", "首都は？", ["東京", "大阪"])
    assert shown == ["東京", "大阪"]
    assert body["evidence"] == "首都は？"
    assert body["criterion"] == f"{POLICY}\n\n{INSTRUCTIONS}"
    assert body["options"] == [
        {"letter": "A", "description": "東京"},
        {"letter": "B", "description": "大阪"},
    ]


def test_hopper_score_levels_are_numbered():
    body, shown = _body("score", "文章", ["低い", "高い"])
    assert shown == ["0: 低い", "1: 高い"]
    assert [o["description"] for o in body["options"]] == shown


def test_hopper_noul_uses_true_then_false_and_puts_rubric_in_policy():
    body, shown = _body("noul", "空は青い？", ["いいえ", "はい"])
    assert shown == ["true", "false"]  # letter A is `true`
    assert body["evidence"] == "空は青い？"  # empty document falls back to the question
    assert body["criterion"].startswith(f"{POLICY}\ntrue: yes\nfalse: no\n\n")
    assert body["options"][0] == {"letter": "A", "description": "true"}


def test_hopper_rejects_more_options_than_letters_and_unknown_primitive():
    with pytest.raises(ValueError):
        build_request("choice", "q", [str(i) for i in range(27)])
    with pytest.raises(ValueError):
        build_request("rank", "q", ["a", "b"])


def test_rescale_is_a_normalised_temperature_that_never_reorders():
    probabilities = [0.6, 0.3, 0.1]
    for temperature in TEMPERATURES.values():
        scaled = rescale(probabilities, temperature)
        assert math.isclose(sum(scaled), 1.0)
        assert scaled == sorted(scaled, reverse=True)
        assert scaled[0] > probabilities[0]  # T < 1 sharpens
    assert rescale(probabilities, 1.0) == pytest.approx(probabilities)


def test_decider_questions_match_the_jev_wire_format():
    state, questions = build_questions("choice", "質問", ["a", "b"])
    assert state == "質問"
    assert questions["answer"] == {
        "type": "choice",
        "instructions": INSTRUCTIONS,
        "criteria": {"option_0": "a", "option_1": "b"},
    }
    state, questions = build_questions("noul", "空は青い？", ["いいえ", "はい"])
    assert state == ""
    assert questions["answer"] == {"type": "noul", "instructions": "空は青い？"}
    state, questions = build_questions("score", "文章", ["低", "高"])
    assert questions["answer"]["criteria"] == ["低", "高"]


def test_decider_answers_are_parsed_per_primitive():
    scores, probabilities, predicted = parse_answer("noul", {"noul": 0.8}, 2)
    assert probabilities == pytest.approx([0.2, 0.8]) and predicted == 1
    _, probabilities, predicted = parse_answer(
        "choice",
        {"choice": "option_1", "probabilities": {"option_0": 0.1, "option_1": 0.9}},
        2,
    )
    assert probabilities == [0.1, 0.9] and predicted == 1
    _, probabilities, predicted = parse_answer(
        "score", {"probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}}, 3
    )
    assert probabilities == [0.1, 0.2, 0.7] and predicted == 2


def test_decider_few_shot_examples_go_into_the_state():
    from openjev_ja.methods.decider.scorer import format_few_shot

    examples = [("Q1", ["いいえ", "はい"], 1), ("Q2", ["低", "高"], 0)]
    state = format_few_shot("choice", examples, "本題")
    assert "[例1]\nQ1\n選択肢: 0: いいえ / 1: はい\n正解: はい" in state
    assert "[例2]" in state
    assert state.endswith("[判定対象]\n本題")
    noul = format_few_shot("noul", examples, "")
    assert "選択肢" not in noul and noul.endswith("上と同じ基準で答えてください。")
