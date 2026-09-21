from __future__ import annotations

import argparse

from openjev_ja.eval.orchestrate import run_orchestration


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="jev-ja-lab-orchestrate",
        description="Run configured models sequentially and datasets in parallel.",
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    output = run_orchestration(args.config)
    print(f"orchestration: {output.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
