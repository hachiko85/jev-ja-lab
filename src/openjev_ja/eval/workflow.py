from __future__ import annotations

import argparse

from openjev_ja.aggregate.summary import build_evaluation_summary
from openjev_ja.eval.orchestrate import load_orchestration_config, run_orchestration


def run_workflow(config_path: str, *, phase: str) -> None:
    config = load_orchestration_config(config_path)
    tasks = config.get("tasks", {})
    order = config.get("workflow", {}).get(
        "order", ["noul", "choice", "score", "summary"]
    )
    for task in order:
        if task == "summary":
            if tasks.get("summary", {}).get("enabled", True):
                build_evaluation_summary(config_path, phase=phase)
            continue
        if task not in {"noul", "choice", "score"}:
            raise ValueError(f"unsupported workflow task: {task}")
        run_orchestration(config_path, task_type=task, phase=phase)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="jev-ja-lab-eval-workflow",
        description="Run enabled primitive stages in YAML-defined order.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", choices=["smoke", "production"], default="production")
    args = parser.parse_args(argv)
    run_workflow(args.config, phase=args.phase)
    print(f"workflow: {args.phase}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
