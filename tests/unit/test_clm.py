import math

import pytest

from openjev_ja.methods.clm.scorer import build_texts, make_head, softmax


def test_choice_and_score_use_the_question_alone_and_options_verbatim_as_candidates():
    state, candidates = build_texts("choice", " 首都は？ ", ["東京", "大阪"])
    assert state == "首都は？"
    assert candidates == ["東京", "大阪"]
    state, candidates = build_texts("score", "文章", ["低い", "高い"])
    assert state == "文章" and candidates == ["低い", "高い"]


def test_noul_state_is_the_question_and_candidates_are_false_then_true():
    state, candidates = build_texts("noul", "空は青い？", ["いいえ", "はい"])
    assert state == "空は青い？"
    assert candidates == [
        "false: No. This is false: 空は青い？",
        "true: Yes. This is true: 空は青い？",
    ]


def test_unknown_primitive_is_rejected():
    with pytest.raises(ValueError):
        build_texts("rank", "q", ["a", "b"])


def test_softmax_is_a_distribution():
    probabilities = softmax([1.0, 2.0, 3.0])
    assert math.isclose(sum(probabilities), 1.0)
    assert probabilities == sorted(probabilities)


def test_head_matches_the_published_checkpoint_layout():
    torch = pytest.importorskip("torch")
    head = make_head(8, 3, 4, layernorm=True, hidden=16)
    assert sorted(head.state_dict()) == [
        "hidden.0.bias",
        "hidden.0.weight",
        "inp.bias",
        "inp.weight",
        "norms.0.bias",
        "norms.0.weight",
        "out.bias",
        "out.weight",
    ]
    assert head(torch.zeros(2, 16)).shape == (2, 4)
