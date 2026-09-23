"""Generate configs/eval/openjev-series.yaml (AlexWortega/openjev NLI cross-encoder).

No training needed. One model entry per checkpoint size tried; each is
evaluated against all noul/choice/score datasets, same as the other
no-training methods (laya).
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
PRIMITIVES = ("noul", "choice", "score")

CHECKPOINTS = [
    ("openjev-0.8b", "qwen3.5-0.8b-nli-v2s-long", "#4b83ad"),
    ("openjev-4b-v2", "qwen3.5-4b-nli-v2", "#e15759"),
]


def main() -> None:
    base = yaml.safe_load((ROOT / "configs/eval/bert-series.yaml").read_text(encoding="utf-8"))
    tasks = {}
    for primitive in PRIMITIVES:
        task = dict(base["tasks"][primitive])
        task.pop("parallelism", None)
        task["model_ids"] = [short_id for short_id, _, _ in CHECKPOINTS]
        chart = dict(task.get("chart", {}))
        if chart:
            chart["name"] = f"radar-openjev-{primitive}"
            task["chart"] = chart
        tasks[primitive] = task
    summary_task = dict(base["tasks"]["summary"])
    summary_chart = dict(summary_task.get("chart", {}))
    if summary_chart:
        summary_chart["name"] = "radar-openjev-summary"
        summary_task["chart"] = summary_chart
    tasks["summary"] = summary_task

    models = [
        {
            "id": short_id,
            "label": f"AlexWortega/openjev ({subfolder})",
            "repo_id": "AlexWortega/openjev",
            "scorer": "nli-cross-encoder",
            "subfolder": subfolder,
            "trust_remote_code": True,
            "template": "ja",
            "dtype": "bfloat16",
            "batch_size": 1,
            "color": color,
        }
        for short_id, subfolder, color in CHECKPOINTS
    ]
    config = {
        "version": 1,
        "runtime": {
            "run_name": "eval-openjev-series",
            "smoke_run_name": "eval-openjev-series-smoke",
            "resume": True,
            "seed": 42,
            "models_root": "models",
            "datasets_root": "datasets",
            "output_root": "results",
            "devices": ["cuda:0"],
            "parallelism": 1,
            "batch_size": 1,
            "worker_timeout_seconds": 86400,
        },
        "workflow": {"order": ["noul", "choice", "score", "summary"]},
        "tasks": tasks,
        "models": models,
        "datasets": base["datasets"],
    }
    output_path = ROOT / "configs/eval/openjev-series.yaml"
    output_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    print(f"wrote {output_path} ({len(models)} model entries, {len(base['datasets'])} datasets)")


if __name__ == "__main__":
    main()
