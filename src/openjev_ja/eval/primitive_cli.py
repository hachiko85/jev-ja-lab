from __future__ import annotations

import argparse
from collections.abc import Sequence

from openjev_ja.eval.orchestrate import run_orchestration


def run_task_cli(task_type: str, argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog=f"jev-ja-lab-eval-{task_type}",
        description=f"Run the configured Jev {task_type} evaluation stage.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", choices=["smoke", "production"], default="production")
    args = parser.parse_args(argv)
    output = run_orchestration(args.config, task_type=task_type, phase=args.phase)
    print(f"eval_{task_type}: {output.as_posix()}")
    return 0
