from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from openjev_ja.eval.orchestrate import OrchestrationError, load_orchestration_config


def _write_summary_chart(
    run_root: Path, models: list[dict[str, Any]], chart: dict[str, Any]
) -> None:
    import yaml

    completed = [
        model
        for model in models
        if len(model["available_primitives"]) == len(model["primitives"])
    ]
    if not completed:
        return
    configured = {
        str(model["id"]): model for model in chart.pop("configured_models", [])
    }
    axes = (
        ("noul", "Noul\n(Yes/No判定)"),
        ("choice", "Choice\n(選択)"),
        ("score", "Score\n(スコアリング)"),
        ("overall", "Overall\n(等加重総合)"),
    )
    series = []
    for model in completed:
        model_id = str(model["id"])
        metrics = {
            "noul": model["primitives"]["noul"]["score"],
            "choice": model["primitives"]["choice"]["score"],
            "score": model["primitives"]["score"]["score"],
            "overall": model["overall_score"],
        }
        metrics_path = run_root / model_id / "summary.overall.json"
        metrics_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        model_config = configured.get(model_id, {})
        series.append(
            {
                "name": model["label"],
                "color": model_config.get("color"),
                "points": {
                    axis_id: {
                        "path": f"{model_id}/summary.overall.json",
                        "metric": axis_id,
                    }
                    for axis_id, _ in axes
                },
            }
        )
    name = str(chart.get("name", "radar-summary"))
    if Path(name).name != name:
        raise OrchestrationError("summary chart name must be a file name")
    language = str(chart.get("language", "")).lower()
    note = (
        "データセット単位のmacro平均を算出後、Primitive間を設定重みで加重平均。"
        if language == "ja"
        else "Dataset macro average, then configured weighted mean by primitive."
    )
    config: dict[str, Any] = {
        "version": 1,
        "title": str(chart.get("title", "Overall Primitive evaluation")),
        "subtitle": str(
            chart.get("subtitle", "Noul, Choice, Score and overall score")
        ),
        "note": note,
        "output": f"{name}.png",
        "input_scale": "fraction",
        "dpi": 180,
        "figure_size": [12, 12],
        "legend_columns": min(3, len(series)),
        "axes": [{"id": axis_id, "label": label} for axis_id, label in axes],
        "series": series,
    }
    if chart.get("theme"):
        config["theme"] = str(chart["theme"])
    if chart.get("language"):
        config["language"] = str(chart["language"])
    config_path = run_root / f"{name}.yaml"
    config_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    from openjev_ja.visualize.radar import render_radar

    render_radar(config_path)


def _normalized_metric(summary: dict[str, Any], metric: str, dataset: dict[str, Any]) -> float:
    if metric not in summary:
        raise OrchestrationError(
            f"metric {metric!r} missing for dataset {dataset['id']!r}"
        )
    value = float(summary[metric])
    if metric in {"spearman", "quadratic_weighted_kappa"}:
        return max(0.0, min(1.0, (value + 1.0) / 2.0))
    if metric in {"mae", "rmse"}:
        max_error = float(dataset.get("max_error", 0))
        if max_error <= 0:
            raise OrchestrationError(
                f"dataset {dataset['id']!r} requires max_error to normalize {metric}"
            )
        return max(0.0, 1.0 - value / max_error)
    if not 0.0 <= value <= 1.0:
        raise OrchestrationError(f"metric {metric!r} must be in the 0-1 range")
    return value


def build_evaluation_summary(
    config_path: str | Path, *, phase: str = "production"
) -> Path:
    config = load_orchestration_config(config_path)
    if phase not in {"smoke", "production"}:
        raise OrchestrationError("phase must be smoke or production")
    runtime = dict(config["runtime"])
    run_name = str(runtime["run_name"])
    if phase == "smoke":
        run_name = str(runtime.get("smoke_run_name", f"{run_name}-smoke"))
    run_root = Path(runtime["output_root"]) / run_name
    run_root.mkdir(parents=True, exist_ok=True)
    datasets = {str(item["id"]): dict(item) for item in config.get("datasets", [])}
    tasks = config.get("tasks", {})
    primitive_names = ("noul", "choice", "score")
    configured_models = [
        dict(model) for model in config.get("models", []) if model.get("enabled", True)
    ]
    output_models: list[dict[str, Any]] = []
    for model in configured_models:
        model_root = run_root / str(model["id"])
        primitive_results: dict[str, Any] = {}
        for primitive in primitive_names:
            task = dict(tasks.get(primitive, {}))
            dataset_ids = [str(value) for value in task.get("datasets", [])]
            metric = str(task.get("aggregate_metric", "accuracy"))
            rows = []
            for dataset_id in dataset_ids:
                dataset = datasets[dataset_id]
                summary_path = model_root / dataset_id / "summary.json"
                if not summary_path.is_file():
                    continue
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                rows.append(
                    {
                        "dataset": dataset_id,
                        "feature": dataset.get("feature"),
                        "metric": metric,
                        "value": float(summary[metric]),
                        "normalized_score": _normalized_metric(summary, metric, dataset),
                        "total": int(summary["total"]),
                    }
                )
            primitive_results[primitive] = {
                "status": "complete" if len(rows) == len(dataset_ids) and rows else "partial"
                if rows
                else "unavailable",
                "aggregate_metric": metric,
                "score": sum(row["normalized_score"] for row in rows) / len(rows)
                if rows
                else None,
                "completed_datasets": len(rows),
                "configured_datasets": len(dataset_ids),
                "datasets": rows,
            }
        weights = {
            key: float(value)
            for key, value in dict(tasks.get("summary", {}).get("weights", {})).items()
            if key in primitive_names
        }
        available = [
            primitive
            for primitive in primitive_names
            if primitive_results[primitive]["score"] is not None
            and weights.get(primitive, 1.0) > 0
        ]
        denominator = sum(weights.get(primitive, 1.0) for primitive in available)
        overall = (
            sum(
                float(primitive_results[primitive]["score"])
                * weights.get(primitive, 1.0)
                for primitive in available
            )
            / denominator
            if denominator
            else None
        )
        if available:
            output_models.append(
                {
                    "id": model["id"],
                    "label": model.get("label", model["id"]),
                    "primitives": primitive_results,
                    "overall_score": overall,
                    "overall_status": "complete"
                    if len(available) == len(primitive_names)
                    and all(
                        primitive_results[primitive]["status"] == "complete"
                        for primitive in primitive_names
                    )
                    else "partial",
                    "primitive_coverage": f"{len(available)}/{len(primitive_names)}",
                    "available_primitives": available,
                }
            )
    payload = {
        "version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "phase": phase,
        "aggregation": "macro average by dataset, then configured weighted mean by primitive",
        "models": output_models,
    }
    output = run_root / "eval_summary.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    chart = dict(tasks.get("summary", {}).get("chart", {}))
    if chart.get("enabled", False):
        chart["configured_models"] = configured_models
        _write_summary_chart(run_root, output_models, chart)
    return output
