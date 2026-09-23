"""Generate configs/eval/{laya,jevlike}-series.yaml from bert-series.yaml's dataset list.

laya needs no training (pretrained RLCD decision heads): one model entry per
primitive, using the multilingual checkpoint. jevlike is Choice-only and
needs a checkpoint trained by jev-ja-lab-jevlike-train first.
"""

from __future__ import annotations

from pathlib import Path

import yaml

PRIMITIVES = ("noul", "choice", "score")
ROOT = Path(__file__).resolve().parent.parent


def _base_tasks_and_datasets() -> tuple[dict, list[dict]]:
    base = yaml.safe_load((ROOT / "configs/eval/bert-series.yaml").read_text(encoding="utf-8"))
    return base["tasks"], base["datasets"]


def _common_runtime(run_name: str) -> dict:
    return {
        "run_name": run_name,
        "smoke_run_name": f"{run_name}-smoke",
        "resume": True,
        "seed": 42,
        "models_root": "models",
        "datasets_root": "datasets",
        "output_root": "results",
        "devices": ["cuda:0"],
        "parallelism": 1,
        "batch_size": 1,
        "worker_timeout_seconds": 86400,
    }


def write_laya_config() -> None:
    base_tasks, datasets = _base_tasks_and_datasets()
    model_ids = {p: [f"laya-multilingual-{p}"] for p in PRIMITIVES}
    tasks = {}
    for primitive in PRIMITIVES:
        task = dict(base_tasks[primitive])
        task.pop("parallelism", None)
        task["model_ids"] = model_ids[primitive]
        chart = dict(task.get("chart", {}))
        if chart:
            chart["name"] = f"radar-laya-{primitive}"
            task["chart"] = chart
        tasks[primitive] = task
    summary_task = dict(base_tasks["summary"])
    summary_chart = dict(summary_task.get("chart", {}))
    if summary_chart:
        summary_chart["name"] = "radar-laya-summary"
        summary_task["chart"] = summary_chart
    tasks["summary"] = summary_task

    models = [
        {
            "id": f"laya-multilingual-{primitive}",
            "label": f"laya-multilingual ({primitive})",
            "scorer": "laya",
            "primitive": primitive,
            "checkpoint": "multilingual",
            "color": color,
        }
        for primitive, color in zip(PRIMITIVES, ("#4b83ad", "#e15759", "#59a14f"), strict=True)
    ]
    config = {
        "version": 1,
        "runtime": _common_runtime("eval-laya-series"),
        "workflow": {"order": ["noul", "choice", "score", "summary"]},
        "tasks": tasks,
        "models": models,
        "datasets": datasets,
    }
    output_path = ROOT / "configs/eval/laya-series.yaml"
    output_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    print(f"wrote {output_path} ({len(models)} model entries, {len(datasets)} datasets)")


def write_jevlike_config() -> None:
    # jevlike's ChoiceExample is candidate-count agnostic, so Noul (2
    # options) and Score (N ordered-level options) reuse the exact same
    # AttentionHead as Choice once trained on their own primitive's data
    # (see methods.jevlike.train.DEFAULT_SOURCES) — jevlike upstream just
    # never shipped a head for them itself.
    base_tasks, datasets = _base_tasks_and_datasets()
    model_ids = {p: [f"jevlike-ruri130m-{p}"] for p in PRIMITIVES}
    tasks = {}
    for primitive in PRIMITIVES:
        task = dict(base_tasks[primitive])
        task.pop("parallelism", None)
        task["model_ids"] = model_ids[primitive]
        chart = dict(task.get("chart", {}))
        if chart:
            chart["name"] = f"radar-jevlike-{primitive}"
            task["chart"] = chart
        tasks[primitive] = task
    summary_task = dict(base_tasks["summary"])
    summary_chart = dict(summary_task.get("chart", {}))
    if summary_chart:
        summary_chart["name"] = "radar-jevlike-summary"
        summary_task["chart"] = summary_chart
    tasks["summary"] = summary_task

    models = [
        {
            "id": f"jevlike-ruri130m-{primitive}",
            "label": f"jevlike (ruri-v3-130m frozen encoder, {primitive})",
            "scorer": "jevlike",
            "checkpoint_path": f"results/jevlike_heads/ruri-130m/{primitive}.pt",
            "color": color,
        }
        for primitive, color in zip(PRIMITIVES, ("#edc948", "#b07aa1", "#76b7b2"), strict=True)
    ]
    config = {
        "version": 1,
        "runtime": _common_runtime("eval-jevlike-series"),
        "workflow": {"order": ["noul", "choice", "score", "summary"]},
        "tasks": tasks,
        "models": models,
        "datasets": datasets,
    }
    output_path = ROOT / "configs/eval/jevlike-series.yaml"
    output_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    print(f"wrote {output_path} ({len(models)} model entries, {len(datasets)} datasets)")


if __name__ == "__main__":
    write_laya_config()
    write_jevlike_config()
