"""Generate configs/eval/laya-bert-series.yaml (laya's DecisionModel head
trained from scratch on a frozen Japanese BERT encoder).

Needs a checkpoint per primitive from jev-ja-lab-laya-bert-train first.
"""

from __future__ import annotations

from pathlib import Path

import yaml

PRIMITIVES = ("noul", "choice", "score")
ROOT = Path(__file__).resolve().parent.parent
ENCODER = "cl-nagoya/ruri-v3-130m"
SHORT_ID = "laya-bert-ruri130m"


def main() -> None:
    base = yaml.safe_load((ROOT / "configs/eval/bert-series.yaml").read_text(encoding="utf-8"))
    model_ids = {p: [f"{SHORT_ID}-{p}"] for p in PRIMITIVES}
    tasks = {}
    for primitive in PRIMITIVES:
        task = dict(base["tasks"][primitive])
        task.pop("parallelism", None)
        task["model_ids"] = model_ids[primitive]
        chart = dict(task.get("chart", {}))
        if chart:
            chart["name"] = f"radar-laya-bert-{primitive}"
            task["chart"] = chart
        tasks[primitive] = task
    summary_task = dict(base["tasks"]["summary"])
    summary_chart = dict(summary_task.get("chart", {}))
    if summary_chart:
        summary_chart["name"] = "radar-laya-bert-summary"
        summary_task["chart"] = summary_chart
    tasks["summary"] = summary_task

    models = [
        {
            "id": f"{SHORT_ID}-{primitive}",
            "label": f"laya architecture on {ENCODER} ({primitive})",
            "repo_id": ENCODER,
            "scorer": "laya-bert",
            "primitive": primitive,
            "head_path": f"results/laya_bert_heads/ruri-130m/{primitive}.pt",
            "color": color,
        }
        for primitive, color in zip(PRIMITIVES, ("#ff7f0e", "#2ca02c", "#d62728"), strict=True)
    ]
    config = {
        "version": 1,
        "runtime": {
            "run_name": "eval-laya-bert-series",
            "smoke_run_name": "eval-laya-bert-series-smoke",
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
    output_path = ROOT / "configs/eval/laya-bert-series.yaml"
    output_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    print(f"wrote {output_path} ({len(models)} model entries, {len(base['datasets'])} datasets)")


if __name__ == "__main__":
    main()
