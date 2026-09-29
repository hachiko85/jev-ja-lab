import json

import pytest

from openjev_ja.train.corpus import CorpusError, load_corpus

ROWS = [
    {
        "id": "n1", "lang": "Japanese", "question_type": "noul",
        "state": "チャットログ: サーバーが落ちた。", "instructions": "これは障害ですか？",
        "options": [
            {"label": "yes", "description": "障害である"},
            {"label": "no", "description": "障害でない"},
        ],
        "expected": "yes",
    },
    {
        "id": "n2", "lang": "Brazilian Portuguese", "question_type": "noul",
        "state": "ok", "instructions": "e um bug?",
        "options": [{"label": "yes", "description": ""}, {"label": "no", "description": ""}],
        "expected": "no",
    },
    {
        "id": "c1", "lang": "Japanese", "question_type": "choice",
        "state": "領収書: 合計 1,200円。", "instructions": "正しい勘定科目を選んでください。",
        "options": [
            {"label": "a", "description": "交通費"},
            {"label": "b", "description": "消耗品費"},
        ],
        "expected": "b",
    },
    {
        "id": "s1", "lang": "Japanese", "question_type": "score",
        "state": "とても良い対応でした。", "instructions": "満足度を評価してください。",
        "options": [
            {"label": "0", "description": "不満"},
            {"label": "1", "description": "普通"},
            {"label": "2", "description": "満足"},
        ],
        "expected": "2",
    },
]


@pytest.fixture
def corpus_file(tmp_path):
    path = tmp_path / "eikos.jsonl"
    lines = "\n".join(json.dumps(row, ensure_ascii=False) for row in ROWS)
    path.write_text(lines, encoding="utf-8")
    return path


def test_noul_keeps_canonical_options_and_folds_the_rubric_into_the_question(corpus_file):
    items = load_corpus(corpus_file, primitive="noul")
    assert len(items) == 1  # the Portuguese row is filtered out
    item = items[0]
    assert item.options == ["いいえ", "はい"]
    assert item.gold_index == 1
    assert "障害ですか？" in item.question and "はい: 障害である" in item.question


def test_choice_keeps_option_order_and_uses_descriptions(corpus_file):
    items = load_corpus(corpus_file, primitive="choice")
    assert len(items) == 1
    assert items[0].options == ["交通費", "消耗品費"]
    assert items[0].gold_index == 1


def test_score_uses_the_full_level_list_and_expected_position(corpus_file):
    items = load_corpus(corpus_file, primitive="score")
    assert items[0].options == ["不満", "普通", "満足"]
    assert items[0].gold_index == 2


def test_lang_filter_can_be_disabled(corpus_file):
    items = load_corpus(corpus_file, primitive="noul", lang=None)
    assert len(items) == 2


def test_limit_stops_early(corpus_file):
    items = load_corpus(corpus_file, primitive="noul", lang=None, limit=1)
    assert len(items) == 1


def test_unsupported_format_and_missing_file_raise(corpus_file, tmp_path):
    with pytest.raises(CorpusError):
        load_corpus(corpus_file, primitive="noul", format="unknown")
    with pytest.raises(CorpusError):
        load_corpus(tmp_path / "missing.jsonl", primitive="noul")


def test_expected_not_among_labels_is_rejected(tmp_path):
    bad = dict(ROWS[0], expected="maybe")
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(bad, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(CorpusError):
        load_corpus(path, primitive="noul")
