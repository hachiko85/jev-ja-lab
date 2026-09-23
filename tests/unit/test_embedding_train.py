"""Embedding-method smoke: frozen encoder + Linear head, forward/backward/overfit.

Requires downloading intfloat/multilingual-e5-small (the smallest of the
configured embedding models) and real CPU forward/backward passes, so this
is marked integration and excluded from the default `pytest` run.
"""

import json
from pathlib import Path

import pytest

from openjev_ja.common import BenchmarkItem
from openjev_ja.methods.embedding.encoder_head import EmbeddingEncoder
from openjev_ja.methods.embedding.scorer import EmbeddingScorer
from openjev_ja.methods.embedding.train import _evaluate, train_head

MODEL = "intfloat/multilingual-e5-small"


def _choice_items() -> list[BenchmarkItem]:
    pairs = [
        ("日本の首都は？", ["東京", "大阪"], 0),
        ("フランスの首都は？", ["パリ", "ロンドン"], 0),
        ("最大の惑星は？", ["木星", "水星"], 0),
        ("光の三原色でないものは？", ["黒", "赤"], 0),
    ]
    return [
        BenchmarkItem(f"choice:{i}", question, options, gold)
        for i, (question, options, gold) in enumerate(pairs * 5)
    ]


def _noul_items() -> list[BenchmarkItem]:
    statements = [
        ("次の文章は有害ですか？\n文章: 死ね", 1),
        ("次の文章は有害ですか？\n文章: おはよう", 0),
        ("次の文章は有害ですか？\n文章: 殺してやる", 1),
        ("次の文章は有害ですか？\n文章: ありがとう", 0),
    ]
    return [
        BenchmarkItem(f"noul:{i}", question, ["いいえ", "はい"], gold)
        for i, (question, gold) in enumerate(statements * 5)
    ]


@pytest.mark.integration
def test_embedding_encoder_extracts_one_state_per_candidate() -> None:
    encoder = EmbeddingEncoder(MODEL, device="cpu", dtype="float32")
    states = encoder.candidate_states("質問", ["甲", "乙", "丙"])
    assert states.shape == (3, encoder.hidden_size)


@pytest.mark.integration
def test_train_head_end_to_end_overfits_tiny_jsonl_noul_set(tmp_path) -> None:
    """Exercises train_head()'s full path (file -> local_data -> encoder -> head)."""
    data_path = tmp_path / "noul_smoke.jsonl"
    rows = [
        {"id": "1", "text": "死ね", "label": "toxic"},
        {"id": "2", "text": "おはよう", "label": "clean"},
        {"id": "3", "text": "殺してやる", "label": "toxic"},
        {"id": "4", "text": "ありがとう", "label": "clean"},
    ] * 6
    data_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8"
    )
    result = train_head(
        model_name=MODEL,
        primitive="noul",
        datasets_root=str(tmp_path),
        output_path=tmp_path / "noul.pt",
        dataset_name="noul_smoke",
        source={
            "format": "jsonl",
            "path": "noul_smoke.jsonl",
            "adapter": "noul",
            "gold_field": "label",
            "positive_values": ["toxic"],
            "question_template": "次の文章は有害ですか？\n文章: {text}",
        },
        epochs=10,
        lr=5e-2,
        val_fraction=0.0,
    )
    assert Path(result["head_path"]).is_file()
    assert result["train_items"] == 24


@pytest.mark.integration
def test_embedding_head_overfits_choice_and_noul_in_memory() -> None:
    import torch

    from openjev_ja.methods.embedding.train import _forward_logits

    encoder = EmbeddingEncoder(MODEL, device="cpu", dtype="float32")

    for primitive, items, loss_fn in (
        ("choice", _choice_items(), torch.nn.CrossEntropyLoss()),
        ("noul", _noul_items(), torch.nn.BCEWithLogitsLoss()),
    ):
        head = torch.nn.Linear(encoder.hidden_size, 1)
        optimizer = torch.optim.Adam(head.parameters(), lr=5e-2)
        for _ in range(15):
            for item in items:
                logits = _forward_logits(encoder, head, primitive, item)
                if primitive == "noul":
                    target = torch.tensor([float(item.gold_index)])
                    loss = loss_fn(logits, target)
                else:
                    target = torch.tensor([item.gold_index])
                    loss = loss_fn(logits.unsqueeze(0), target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        accuracy = _evaluate(encoder, head, primitive, items, torch)
        assert accuracy == 1.0, f"{primitive} head failed to overfit a 2-pattern smoke set"


@pytest.mark.integration
def test_embedding_scorer_round_trips_choice_head(tmp_path) -> None:
    import torch

    encoder = EmbeddingEncoder(MODEL, device="cpu", dtype="float32")
    head = torch.nn.Linear(encoder.hidden_size, 1)
    head_path = tmp_path / "choice.pt"
    torch.save(head.state_dict(), head_path)

    scorer = EmbeddingScorer(
        MODEL, primitive="choice", head_path=head_path, device="cpu", dtype="float32"
    )
    result = scorer.score("質問", ["甲", "乙", "丙"])
    assert len(result.scores) == 3
    assert len(result.probabilities) == 3
    assert 0 <= result.predicted_index < 3
    assert scorer.metadata()["primitive"] == "choice"
