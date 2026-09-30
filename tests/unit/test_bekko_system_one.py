import json

import pytest

from openjev_ja.methods.bekko_system_one.scorer import (
    RELEASES,
    build_decision,
    parse_answer,
)


def test_choice_and_score_put_the_item_in_state_with_a_fixed_instruction():
    state, decision = build_decision("choice", "首都は？", ["東京", "大阪"])
    assert state == "首都は？"
    assert decision["type"] == "choice"
    assert [c["id"] for c in decision["criteria"]] == ["option_0", "option_1"]
    assert json.loads(decision["criteria"][0]["description_json"]) == "東京"

    state, decision = build_decision("score", "文章", ["低い", "高い"])
    assert state == "文章"
    assert [c["id"] for c in decision["criteria"]] == ["0", "1"]
    assert [c["value"] for c in decision["criteria"]] == [0, 1]


def test_noul_has_empty_state_and_fixed_true_false_ids():
    state, decision = build_decision("noul", "空は青い？", ["いいえ", "はい"])
    assert state == ""
    assert decision["type"] == "noul"
    assert json.loads(decision["instructions_json"]) == "空は青い？"
    assert [c["id"] for c in decision["criteria"]] == ["false", "true"]


def test_unsupported_primitive_is_rejected():
    with pytest.raises(ValueError):
        build_decision("rank", "q", ["a", "b"])


def test_noul_reads_probability_yes_directly():
    scores, probabilities, predicted = parse_answer(
        "noul", {"probability_yes": 0.8}, 2
    )
    assert probabilities == pytest.approx([0.2, 0.8])
    assert predicted == 1


def test_score_reads_probabilities_by_string_index():
    _, probabilities, predicted = parse_answer(
        "score", {"probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}}, 3
    )
    assert probabilities == [0.1, 0.2, 0.7] and predicted == 2


def test_choice_reads_selected_id_and_option_probabilities():
    answer = {"selected_id": "option_1", "probabilities": {"option_0": 0.1, "option_1": 0.9}}
    _, probabilities, predicted = parse_answer("choice", answer, 2)
    assert probabilities == [0.1, 0.9] and predicted == 1


def test_releases_are_pinned_to_a_full_hex_revision():
    for repo_id, revision in RELEASES.values():
        assert "/" in repo_id
        assert len(revision) == 40 and all(c in "0123456789abcdef" for c in revision)
