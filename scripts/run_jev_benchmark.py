from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from openjev_ja.aggregate.summary import build_evaluation_summary
from openjev_ja.eval.orchestrate import run_orchestration


def build_config(
    source: Path,
    destination: Path,
    *,
    run_name: str,
    datasets: list[str] | None = None,
    parallelism: int = 4,
) -> None:
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    runtime = config["runtime"]
    runtime.update(
        {
            "run_name": run_name,
            "smoke_run_name": f"{run_name}-smoke",
            "models_root": str(Path("models").resolve()),
            "datasets_root": str(Path("D:/clean_datasets_root")),
            "output_root": "results",
            "devices": [f"api:{index}" for index in range(parallelism)],
            "parallelism": parallelism,
            "batch_size": 1,
            "resume": True,
        }
    )
    config["models"] = [
        {
            "id": "jev-latest",
            "label": "Jev Latest",
            "scorer": "jev",
            "model": "jev-latest",
            "batch_size": 1,
            "timeout": 60,
            "max_retries": 3,
            "color": "#9467bd",
        }
    ]
    for task in ("noul", "choice", "score"):
        config["tasks"][task]["parallelism"] = parallelism
    if datasets:
        selected = set(datasets)
        for task in ("noul", "choice", "score"):
            configured = config["tasks"][task]["datasets"]
            config["tasks"][task]["datasets"] = [item for item in configured if item in selected]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="configs/eval/qwen-series.yaml")
    parser.add_argument("--generated", default="results/eval-jev-latest/run_config.source.yaml")
    parser.add_argument("--run-name", default="eval-jev-latest")
    parser.add_argument("--phase", choices=["smoke", "production"], default="production")
    parser.add_argument("--tasks", nargs="+", choices=["noul", "choice", "score", "summary"])
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--parallelism", type=int, default=4)
    args = parser.parse_args()
    source = Path(args.source)
    generated = Path(args.generated)
    build_config(
        source,
        generated,
        run_name=args.run_name,
        datasets=args.datasets,
        parallelism=args.parallelism,
    )
    tasks = args.tasks or ["noul", "choice", "score", "summary"]
    for task in tasks:
        if task == "summary":
            build_evaluation_summary(generated, phase=args.phase)
        else:
            run_orchestration(generated, task_type=task, phase=args.phase)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
