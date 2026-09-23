"""Generate configs/eval/semif-logit-fewshot-series.yaml.

Same Qwen3.5-4B base model as semif-logit-series.yaml, `scorer:
semif-logit-fewshot` with `few_shot_count: 2` per primitive — the setting a
300-item check found best (2-shot matched or beat 3/4-shot on every
primitive, at lower cost). Directly comparable to both semif-logit-series
(same model, zero-shot) and next_token_logit's own qwen3.5-4b run (same
model, different prompt recipe).
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
MODEL = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
FEW_SHOT_COUNT = 2


def main() -> None:
    base = yaml.safe_load((ROOT / "configs/eval/bert-series.yaml").read_text(encoding="utf-8"))
    model_id = f"semif-logit-fewshot-qwen3.5-4b-{FEW_SHOT_COUNT}shot"
    tasks = {}
    for primitive in ("noul", "choice", "score"):
        task = dict(base["tasks"][primitive])
        task.pop("parallelism", None)
        task["model_ids"] = [model_id]
        chart = dict(task.get("chart", {}))
        if chart:
            chart["name"] = f"radar-semif-logit-fewshot-{primitive}"
            task["chart"] = chart
        tasks[primitive] = task
    summary_task = dict(base["tasks"]["summary"])
    summary_chart = dict(summary_task.get("chart", {}))
    if summary_chart:
        summary_chart["name"] = "radar-semif-logit-fewshot-summary"
        summary_task["chart"] = summary_chart
    tasks["summary"] = summary_task

    models = [
        {
            "id": model_id,
            "label": f"Qwen3.5-4B (SemIf recipe, {FEW_SHOT_COUNT}-shot)",
            "repo_id": MODEL,
            "revision": MODEL_REVISION,
            "scorer": "semif-logit-fewshot",
            "primitive": primitive,
            "few_shot_count": FEW_SHOT_COUNT,
            "dtype": "bfloat16",
            "batch_size": 1,
            "color": "#e377c2",
        }
        for primitive in ("noul", "choice", "score")
    ]
    # One physical model id is shared across primitives above for labeling,
    # but each primitive needs its own entry (few-shot examples differ per
    # primitive) — give each a distinct id the way embedding/laya/jevlike do.
    models = [
        {**entry, "id": f"{model_id}-{entry['primitive']}"}
        for entry in models
    ]
    for primitive in ("noul", "choice", "score"):
        tasks[primitive]["model_ids"] = [f"{model_id}-{primitive}"]

    config = {
        "version": 1,
        "runtime": {
            "run_name": "eval-semif-logit-fewshot-series",
            "smoke_run_name": "eval-semif-logit-fewshot-series-smoke",
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
    output_path = ROOT / "configs/eval/semif-logit-fewshot-series.yaml"
    output_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    print(f"wrote {output_path} ({len(models)} model entries, {len(base['datasets'])} datasets)")


if __name__ == "__main__":
    main()
