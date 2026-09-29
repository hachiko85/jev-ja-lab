from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="jev-ja-lab-train-workflow",
        description="Run a YAML-configured, k-fold cross-validated training job.",
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    from openjev_ja.train.config import run_training_workflow

    result = run_training_workflow(args.config)
    print(f"wrote {result['output_root']}/run_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
