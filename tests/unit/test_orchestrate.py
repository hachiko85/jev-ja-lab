from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
import yaml

from openjev_ja.eval.local_data import load_local_benchmark
from openjev_ja.eval.orchestrate import (
    _assign_datasets,
    _model_reference,
    _write_radar_config,
    load_orchestration_config,
)


def test_load_local_arrow_benchmark(tmp_path) -> None:
    table = pa.Table.from_pylist(
        [{"Question": "question", "A": "a", "B": "b", "C": "c", "D": "d", "Answer": "B"}]
    )
    path = tmp_path / "data.arrow"
    with path.open("wb") as sink, ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    items = load_local_benchmark(
        "mmmlu_ja", {"format": "arrow", "path": "data.arrow"}, datasets_root=tmp_path
    )
    assert len(items) == 1
    assert items[0].gold_index == 1


def test_load_jnli_noul_view(tmp_path) -> None:
    path = tmp_path / "jnli.jsonl"
    path.write_text(
        json.dumps(
            {
                "sentence_pair_id": "1",
                "sentence1": "犬が走る。",
                "sentence2": "動物が走る。",
                "label": "entailment",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    items = load_local_benchmark(
        "jnli_entailment",
        {"format": "jsonl", "path": "jnli.jsonl", "adapter": "jnli_noul", "target": "entailment"},
        datasets_root=tmp_path,
    )
    assert items[0].options == ["いいえ", "はい"]
    assert items[0].gold_index == 1


def test_load_generic_score_view(tmp_path) -> None:
    path = tmp_path / "score.jsonl"
    path.write_text(
        json.dumps({"id": "1", "state": "回答", "gold_score": 2}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    items = load_local_benchmark(
        "synthetic_score",
        {
            "format": "jsonl",
            "path": "score.jsonl",
            "adapter": "score",
            "criteria": ["低い", "中程度", "高い"],
        },
        datasets_root=tmp_path,
    )
    assert items[0].options == ["低い", "中程度", "高い"]
    assert items[0].gold_index == 2


def test_load_wrime_score_view_with_split_and_offset(tmp_path) -> None:
    path = tmp_path / "wrime.tsv"
    path.write_text(
        "Sentence\tTrain/Dev/Test\tAvg. Readers_Sentiment\n"
        "訓練例\ttrain\t0\n"
        "評価例\ttest\t-2\n",
        encoding="utf-8",
    )
    items = load_local_benchmark(
        "wrime_sentiment",
        {
            "format": "tsv",
            "path": "wrime.tsv",
            "adapter": "score",
            "question_field": "Sentence",
            "gold_field": "Avg. Readers_Sentiment",
            "gold_offset": 2,
            "criteria": ["強い負", "負", "中立", "正", "強い正"],
            "where": {"Train/Dev/Test": "test"},
        },
        datasets_root=tmp_path,
    )
    assert len(items) == 1
    assert items[0].gold_index == 0


def test_load_generic_noul_parquet_with_filters(tmp_path) -> None:
    path = tmp_path / "noul.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"id": "1", "text": "対象", "label": "yes", "split": "test"},
                {"id": "2", "text": "除外", "label": "no", "split": "train"},
            ]
        ),
        path,
    )
    items = load_local_benchmark(
        "generic_noul",
        {
            "format": "parquet",
            "path": "noul.parquet",
            "adapter": "noul",
            "gold_field": "label",
            "positive_values": ["yes"],
            "question_template": "判定: {text}",
            "where_in": {"split": ["test"]},
        },
        datasets_root=tmp_path,
    )
    assert len(items) == 1
    assert items[0].gold_index == 1


def test_load_continuous_score_with_thresholds(tmp_path) -> None:
    path = tmp_path / "score.jsonl"
    path.write_text('{"text":"対象","score":0.65}\n', encoding="utf-8")
    items = load_local_benchmark(
        "continuous_score",
        {
            "format": "jsonl",
            "path": "score.jsonl",
            "adapter": "score",
            "question_field": "text",
            "gold_field": "score",
            "gold_thresholds": [0.2, 0.4, 0.6, 0.8],
            "criteria": ["0", "1", "2", "3", "4"],
        },
        datasets_root=tmp_path,
    )
    assert items[0].gold_index == 3


def test_load_hub_dataset_by_repo_id(tmp_path, monkeypatch) -> None:
    calls = []

    def fake_load_dataset(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return [{"text": "対象", "label": "yes"}]

    monkeypatch.setitem(
        sys.modules, "datasets", types.SimpleNamespace(load_dataset=fake_load_dataset)
    )
    items = load_local_benchmark(
        "hub_noul",
        {
            "repo_id": "example/dataset",
            "config": "ja",
            "split": "validation",
            "revision": "abc123",
            "adapter": "noul",
            "gold_field": "label",
            "positive_values": ["yes"],
            "question_template": "判定: {text}",
        },
        datasets_root=tmp_path,
    )
    assert items[0].gold_index == 1
    assert calls[0][0] == "example/dataset"
    assert calls[0][1]["name"] == "ja"
    assert calls[0][1]["split"] == "validation"
    assert calls[0][1]["revision"] == "abc123"


def test_assignment_balances_largest_datasets() -> None:
    datasets = [
        {"id": "large", "expected_items": 10},
        {"id": "medium", "expected_items": 6},
        {"id": "small", "expected_items": 4},
    ]
    assignments = _assign_datasets(datasets, 2)
    assert [item["id"] for item in assignments[0]] == ["large"]
    assert [item["id"] for item in assignments[1]] == ["medium", "small"]


def test_load_orchestration_config(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"version": 1}), encoding="utf-8")
    assert load_orchestration_config(path)["version"] == 1


def test_model_reference_supports_local_and_hub() -> None:
    runtime = {"models_root": "/models"}
    assert _model_reference({"path": "Qwen3-8B"}, runtime) == "/models/Qwen3-8B"
    assert _model_reference({"repo_id": "google/gemma-model"}, runtime) == (
        "google/gemma-model"
    )


def test_run_config_uses_container_paths() -> None:
    path = "configs/eval/orchestration.20260918.yaml"
    text = Path(path).read_text(encoding="utf-8")
    config = yaml.safe_load(text)
    assert config["runtime"]["models_root"] == "/models"
    assert config["runtime"]["datasets_root"] == "/datasets"
    selected = set(config["runtime"]["selected_model_ids"])
    radar = set(config["runtime"]["radar_model_ids"])
    assert radar <= selected
    assert "gemma-4-e2b-it" in selected
    assert "gemma-4-e2b-it" not in radar
    assert json.dumps(config)


def test_radar_labels_include_dataset_feature(tmp_path) -> None:
    model_root = tmp_path / "model" / "dataset"
    model_root.mkdir(parents=True)
    (model_root / "summary.json").write_text('{"accuracy": 1.0}', encoding="utf-8")
    config_path = _write_radar_config(
        tmp_path,
        [{"id": "model", "label": "Model"}],
        [{"id": "dataset", "label": "Dataset", "feature": "特徴"}],
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert payload["axes"][0]["label"] == "Dataset\n(特徴)"
