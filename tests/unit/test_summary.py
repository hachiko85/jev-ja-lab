from __future__ import annotations

import json

import yaml

from openjev_ja.aggregate.summary import build_evaluation_summary


def test_summary_aggregates_dataset_then_primitive(tmp_path) -> None:
    results = tmp_path / "results"
    model_root = results / "run" / "model"
    for dataset, accuracy in (("choice_a", 0.8), ("choice_b", 0.6), ("noul_a", 0.5)):
        path = model_root / dataset
        path.mkdir(parents=True)
        payload = {"accuracy": accuracy, "f1": accuracy, "total": 2}
        (path / "summary.json").write_text(json.dumps(payload), encoding="utf-8")
    config = {
        "version": 1,
        "runtime": {"run_name": "run", "output_root": str(results)},
        "models": [{"id": "model", "label": "Model"}],
        "datasets": [
            {"id": "choice_a", "task": "choice"},
            {"id": "choice_b", "task": "choice"},
            {"id": "noul_a", "task": "noul"},
        ],
        "tasks": {
            "choice": {"datasets": ["choice_a", "choice_b"], "aggregate_metric": "accuracy"},
            "noul": {"datasets": ["noul_a"], "aggregate_metric": "f1"},
            "score": {"datasets": [], "aggregate_metric": "quadratic_weighted_kappa"},
            "summary": {"weights": {"choice": 1, "noul": 1, "score": 1}},
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = build_evaluation_summary(config_path)
    payload = json.loads(output.read_text(encoding="utf-8"))
    model = payload["models"][0]
    assert model["primitives"]["choice"]["score"] == 0.7
    assert model["primitives"]["noul"]["score"] == 0.5
    assert model["primitives"]["score"]["status"] == "unavailable"
    assert model["overall_score"] == 0.6
    assert model["overall_status"] == "partial"
    assert model["primitive_coverage"] == "2/3"


def test_summary_chart_contains_primitive_features(tmp_path) -> None:
    results = tmp_path / "results"
    model_root = results / "run" / "model"
    dataset_metrics = (
        ("noul_a", "f1"),
        ("choice_a", "accuracy"),
        ("score_a", "normalized_quadratic_weighted_kappa"),
    )
    for dataset, metric in dataset_metrics:
        path = model_root / dataset
        path.mkdir(parents=True)
        (path / "summary.json").write_text(
            json.dumps({metric: 0.5, "total": 2}), encoding="utf-8"
        )
    config = {
        "version": 1,
        "runtime": {"run_name": "run", "output_root": str(results)},
        "models": [{"id": "model", "label": "Model", "color": "#123456"}],
        "datasets": [
            {"id": "noul_a", "task": "noul"},
            {"id": "noul_unavailable", "task": "noul"},
            {"id": "choice_a", "task": "choice"},
            {"id": "score_a", "task": "score"},
        ],
        "tasks": {
            "noul": {
                "datasets": ["noul_a", "noul_unavailable"],
                "aggregate_metric": "f1",
            },
            "choice": {"datasets": ["choice_a"], "aggregate_metric": "accuracy"},
            "score": {
                "datasets": ["score_a"],
                "aggregate_metric": "normalized_quadratic_weighted_kappa",
            },
            "summary": {
                "weights": {"noul": 1, "choice": 1, "score": 1},
                "chart": {"enabled": True, "name": "radar-summary"},
            },
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    summary_path = build_evaluation_summary(config_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["models"][0]["overall_status"] == "partial"
    assert summary["models"][0]["primitive_coverage"] == "3/3"
    chart = yaml.safe_load((results / "run" / "radar-summary.yaml").read_text())
    assert all("(" in axis["label"] for axis in chart["axes"])
    assert (results / "run" / "radar-summary.png").is_file()
