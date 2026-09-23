"""Generate configs/eval/embedding-series.yaml from bert-series.yaml's dataset list.

One-off generator script (not part of the package): reuses the project's
already-defined 47 Noul/Choice/Score datasets verbatim and adds 7 embedding
models x 3 primitives = 21 model entries, each pointing at its own trained
head under results/embedding_heads/.
"""

from __future__ import annotations

from pathlib import Path

import yaml

MODELS = [
    ("e5-small", "intfloat/multilingual-e5-small", "multilingual-e5-small", "#4b83ad"),
    ("ruri-v3-310m", "cl-nagoya/ruri-v3-310m", "ruri-v3-310m", "#e15759"),
    ("ruri-v3-130m", "cl-nagoya/ruri-v3-130m", "ruri-v3-130m", "#59a14f"),
    ("ruri-v3-70m", "cl-nagoya/ruri-v3-70m", "ruri-v3-70m", "#edc948"),
    ("ruri-v3-30m", "cl-nagoya/ruri-v3-30m", "ruri-v3-30m", "#b07aa1"),
    ("bekko-v1-a8m", "hotchpotch/bekko-embedding-v1-a8m", "bekko-embedding-v1-a8m", "#76b7b2"),
    ("bekko-v1-a25m", "hotchpotch/bekko-embedding-v1-a25m", "bekko-embedding-v1-a25m", "#ff9da7"),
]
PRIMITIVES = ("noul", "choice", "score")
ROOT = Path(__file__).resolve().parent.parent


def head_path(short_id: str, primitive: str) -> str:
    return f"results/embedding_heads/{short_id}/{primitive}.pt"


def main() -> None:
    base = yaml.safe_load((ROOT / "configs/eval/bert-series.yaml").read_text(encoding="utf-8"))

    models = []
    model_ids_by_primitive: dict[str, list[str]] = {p: [] for p in PRIMITIVES}
    for short_id, repo_id, label, color in MODELS:
        for primitive in PRIMITIVES:
            model_id = f"{short_id}-{primitive}"
            model_ids_by_primitive[primitive].append(model_id)
            models.append(
                {
                    "id": model_id,
                    "label": f"{label} (embedding/{primitive})",
                    "repo_id": repo_id,
                    "scorer": "embedding",
                    "primitive": primitive,
                    "head_path": head_path(short_id, primitive),
                    "dtype": "bfloat16",
                    "batch_size": 32,
                    "color": color,
                }
            )

    tasks = {}
    for primitive in PRIMITIVES:
        source_task = dict(base["tasks"][primitive])
        source_task.pop("parallelism", None)
        source_task["model_ids"] = model_ids_by_primitive[primitive]
        chart = dict(source_task.get("chart", {}))
        if chart:
            chart["name"] = f"radar-embedding-{primitive}"
            source_task["chart"] = chart
        tasks[primitive] = source_task
    summary_task = dict(base["tasks"]["summary"])
    summary_chart = dict(summary_task.get("chart", {}))
    if summary_chart:
        summary_chart["name"] = "radar-embedding-summary"
        summary_task["chart"] = summary_chart
    tasks["summary"] = summary_task

    config = {
        "version": 1,
        "runtime": {
            "run_name": "eval-embedding-series",
            "smoke_run_name": "eval-embedding-series-smoke",
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
    output_path = ROOT / "configs/eval/embedding-series.yaml"
    output_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    print(f"wrote {output_path} ({len(models)} model entries, {len(base['datasets'])} datasets)")


if __name__ == "__main__":
    main()
