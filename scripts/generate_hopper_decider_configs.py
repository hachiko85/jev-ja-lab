"""Generate configs/eval/hopper-series.yaml and configs/eval/decider-series.yaml.

Both are one model entry per primitive (each primitive builds its own scorer, like
semif-logit-fewshot / embedding / laya), evaluated on the same 30-axis profile as every
other method: tasks and datasets are copied from bert-series.yaml.

- hopper: HopitAI/hopper LoRA on Qwen/Qwen3.5-4B at the revision the adapter was built for.
  Adapter weights are research/demo-use only.
- decider: Mapika/decider-4b at Hub tag `v2` (decider-4b v2). Change `revision` to `main`
  for v2.1 (per-answer-type temperatures) or `v1`.
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
PRIMITIVES = ("noul", "choice", "score")

SERIES = {
    "hopper": {
        "run_name": "eval-hopper-series",
        "model_id": "hopper-qwen3.5-4b-lora",
        "label": "Hopper (LoRA on Qwen3.5-4B)",
        "color": "#8c564b",
        "entry": {
            "repo_id": "Qwen/Qwen3.5-4B",
            "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            "adapter": "HopitAI/hopper",
            "scorer": "hopper",
            "dtype": "bfloat16",
        },
    },
    "decider": {
        "run_name": "eval-decider-series",
        "model_id": "decider-4b-v2",
        "label": "decider-4b v2",
        "color": "#17becf",
        "entry": {
            "repo_id": "Mapika/decider-4b",
            "revision": "v2",
            "scorer": "decider",
            "dtype": "bfloat16",
        },
    },
}


def build(name: str, spec: dict) -> Path:
    base = yaml.safe_load((ROOT / "configs/eval/bert-series.yaml").read_text(encoding="utf-8"))
    ids = {p: f"{spec['model_id']}-{p}" for p in PRIMITIVES}
    tasks = {}
    for primitive in PRIMITIVES:
        task = dict(base["tasks"][primitive])
        task.pop("parallelism", None)
        task["model_ids"] = [ids[primitive]]
        chart = dict(task.get("chart", {}))
        if chart:
            chart["name"] = f"radar-{name}-{primitive}"
            task["chart"] = chart
        tasks[primitive] = task
    summary_task = dict(base["tasks"]["summary"])
    summary_chart = dict(summary_task.get("chart", {}))
    if summary_chart:
        summary_chart["name"] = f"radar-{name}-summary"
        summary_task["chart"] = summary_chart
    tasks["summary"] = summary_task

    models = [
        {
            "id": ids[primitive],
            "label": f"{spec['label']} ({primitive})",
            **spec["entry"],
            "primitive": primitive,
            "batch_size": 1,
            "color": spec["color"],
        }
        for primitive in PRIMITIVES
    ]
    config = {
        "version": 1,
        "runtime": {
            "run_name": spec["run_name"],
            "smoke_run_name": f"{spec['run_name']}-smoke",
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
    output_path = ROOT / f"configs/eval/{name}-series.yaml"
    output_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    print(f"wrote {output_path} ({len(models)} model entries, {len(base['datasets'])} datasets)")
    return output_path


def main() -> None:
    for name, spec in SERIES.items():
        build(name, spec)


if __name__ == "__main__":
    main()
