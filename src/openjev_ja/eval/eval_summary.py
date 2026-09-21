from __future__ import annotations

import argparse

from openjev_ja.aggregate.summary import build_evaluation_summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="jev-ja-lab-eval-summary",
        description="Aggregate dataset results into primitive and overall scores.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", choices=["smoke", "production"], default="production")
    args = parser.parse_args(argv)
    output = build_evaluation_summary(args.config, phase=args.phase)
    print(f"eval_summary: {output.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
