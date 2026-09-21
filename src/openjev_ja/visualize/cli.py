from __future__ import annotations

import argparse

from openjev_ja.visualize.radar import render_radar


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jev-ja-lab-radar",
        description="Render a radar chart from benchmark result files declared in YAML.",
    )
    parser.add_argument("--config", required=True, help="YAML configuration file")
    parser.add_argument("--output", help="Override the output path from YAML")
    parser.add_argument("--show", action="store_true", help="Open an interactive chart window")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = render_radar(args.config, output_override=args.output, show=args.show)
    print(f"chart: {output.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
