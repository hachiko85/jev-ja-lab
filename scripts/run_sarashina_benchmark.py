from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from openjev_ja.eval.orchestrate import run_orchestration
from openjev_ja.eval.summary import build_evaluation_summary

MODELS = [
    {
        "id": "sarashina2.2-1b-instruct-v0.1",
        "label": "Sarashina2.2 1B Instruct v0.1",
        "repo_id": "sbintuitions/sarashina2.2-1b-instruct-v0.1",
        "dtype": "bfloat16",
        "batch_size": 32,
        "metadata_revision": (
            "sha256:c1bdf6f4b9098581fa8e15533c7cef929e83d1cf1144532c4a65870d21cd18e8"
        ),
        "color": "#3aa6a0",
    },
    {
        "id": "sarashina2.2-3b-instruct-v0.1",
        "label": "Sarashina2.2 3B Instruct v0.1",
        "repo_id": "sbintuitions/sarashina2.2-3b-instruct-v0.1",
        "revision": "4f3626fb1b64b3e97c908e67f27b2d627ba2a999",
        "dtype": "bfloat16",
        "batch_size": 32,
        "color": "#59a14f",
    },
]


def build_config(source: Path, destination: Path) -> None:
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["runtime"].update(
        {
            "run_name": "eval-20260920",
            "smoke_run_name": "eval-sarashina-20260920-smoke",
            "models_root": str(Path("models").resolve()),
            "datasets_root": "D:/clean_datasets_root",
            "output_root": "results",
            "devices": ["cuda:0"],
            "parallelism": 1,
            "batch_size": 32,
            "resume": True,
        }
    )
    config["models"] = MODELS
    for task in ("noul", "choice", "score"):
        config["tasks"][task]["parallelism"] = 1
        config["tasks"][task].pop("model_ids", None)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="configs/eval/qwen-series.yaml")
    parser.add_argument(
        "--generated",
        default="results/eval-20260920/_sources/sarashina.generated.yaml",
    )
    args = parser.parse_args()
    generated = Path(args.generated)
    build_config(Path(args.source), generated)
    for task in ("noul", "choice", "score"):
        run_orchestration(generated, task_type=task, phase="production")
    build_evaluation_summary(generated, phase="production")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
