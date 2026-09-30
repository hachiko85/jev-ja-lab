import pytest

from openjev_ja.methods.jeff.scorer import (
    INSTRUCTIONS,
    RELEASES,
    build_question,
    parse_probabilities,
)


def test_choice_and_score_put_the_item_in_state_with_a_fixed_instruction():
    state, question = build_question("choice", "首都は？", ["東京", "大阪"])
    assert state == "首都は？"
    assert question == {
        "type": "choice",
        "instructions": INSTRUCTIONS,
        "criteria": {"option_0": "東京", "option_1": "大阪"},
    }
    state, question = build_question("score", "文章", ["低い", "高い"])
    assert state == "文章"
    assert question == {"type": "score", "instructions": INSTRUCTIONS, "criteria": ["低い", "高い"]}


def test_noul_has_empty_state_and_the_question_as_instructions():
    state, question = build_question("noul", "空は青い？", ["いいえ", "はい"])
    assert state == ""
    assert question == {"type": "noul", "instructions": "空は青い？"}


def test_unsupported_primitive_is_rejected():
    with pytest.raises(ValueError):
        build_question("rank", "q", ["a", "b"])


def test_noul_reads_index_1_as_p_yes_regardless_of_option_text():
    scores, probabilities, predicted = parse_probabilities("noul", [0.2, 0.8], 2)
    assert probabilities == pytest.approx([0.2, 0.8])
    assert scores == pytest.approx([0.0, 0.8])
    assert predicted == 1
    _, probabilities, predicted = parse_probabilities("noul", [0.9, 0.1], 2)
    assert predicted == 0


def test_choice_and_score_take_the_argmax_of_the_distribution_directly():
    _, probabilities, predicted = parse_probabilities("choice", [0.1, 0.7, 0.2], 3)
    assert probabilities == [0.1, 0.7, 0.2] and predicted == 1
    _, probabilities, predicted = parse_probabilities("score", [0.6, 0.3, 0.1], 3)
    assert predicted == 0


def test_releases_are_pinned_to_a_full_hex_revision():
    for repo_id, revision in RELEASES.values():
        assert "/" in repo_id
        assert len(revision) == 40 and all(c in "0123456789abcdef" for c in revision)
