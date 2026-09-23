"""Generate configs/eval/semif-logit-series.yaml (SemIf's chat-template +
JSON-structured prompt recipe, applied via methods.semif_logit).

Uses the same base model (Qwen/Qwen3.5-4B, no fine-tuning) already
evaluated under methods.next_token_logit's own prompt, so the two are
directly comparable.
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
MODEL = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def main() -> None:
    base = yaml.safe_load((ROOT / "configs/eval/bert-series.yaml").read_text(encoding="utf-8"))
    tasks = {}
    for primitive in ("noul", "choice", "score"):
        task = dict(base["tasks"][primitive])
        task.pop("parallelism", None)
        task["model_ids"] = ["semif-logit-qwen3.5-4b"]
        chart = dict(task.get("chart", {}))
        if chart:
            chart["name"] = f"radar-semif-logit-{primitive}"
            task["chart"] = chart
        tasks[primitive] = task
    summary_task = dict(base["tasks"]["summary"])
    summary_chart = dict(summary_task.get("chart", {}))
    if summary_chart:
        summary_chart["name"] = "radar-semif-logit-summary"
        summary_task["chart"] = summary_chart
    tasks["summary"] = summary_task

    models = [
        {
            "id": "semif-logit-qwen3.5-4b",
            "label": "Qwen3.5-4B (SemIf chat-template recipe)",
            "repo_id": MODEL,
            "revision": MODEL_REVISION,
            "scorer": "semif-logit",
            "dtype": "bfloat16",
            "batch_size": 1,
            "color": "#8c564b",
        }
    ]
    config = {
        "version": 1,
        "runtime": {
            "run_name": "eval-semif-logit-series",
            "smoke_run_name": "eval-semif-logit-series-smoke",
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
    output_path = ROOT / "configs/eval/semif-logit-series.yaml"
    output_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    print(f"wrote {output_path} ({len(models)} model entries, {len(base['datasets'])} datasets)")


if __name__ == "__main__":
    main()
