from __future__ import annotations

import argparse
import json

from openjev_ja.common.testing import MockScorer
from openjev_ja.eval.datasets import DATASET_NAMES, load_benchmark, resolve_dataset_revision
from openjev_ja.eval.runner import run_evaluation
from openjev_ja.methods.next_token_logit import NextTokenLogitScorer
from openjev_ja.methods.nli_cross_encoder import NLICrossEncoderScorer
from openjev_ja.methods.typesafe_jev import JevScorer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jev-ja-lab-eval", description="Evaluate decision scorers on Japanese benchmarks."
    )
    parser.add_argument("--dataset", required=True, choices=[*DATASET_NAMES, "all"])
    parser.add_argument(
        "--scorer", required=True, choices=["mock", "qwen-direct", "jev", "nli-cross-encoder"]
    )
    parser.add_argument("--model")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-revision")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--nli-template", choices=["ja", "en"], default="ja")
    parser.add_argument("--output-dir", default="results")
    return parser


def create_scorer(args: argparse.Namespace):
    if args.scorer == "mock":
        return MockScorer()
    if args.scorer == "qwen-direct":
        return NextTokenLogitScorer(
            args.model or "Qwen/Qwen3.5-4B",
            device=args.device,
            dtype=args.dtype,
        )
    if args.scorer == "jev":
        return JevScorer(model=args.model or "jev-latest")
    device = "cuda" if args.device == "auto" else args.device
    return NLICrossEncoderScorer(
        args.model or "AlexWortega/openjev",
        device=device,
        dtype=args.dtype,
        template=args.nli_template,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scorer = create_scorer(args)
    names = DATASET_NAMES if args.dataset == "all" else (args.dataset,)
    for name in names:
        dataset_revision = resolve_dataset_revision(name, args.dataset_revision)
        items = load_benchmark(name, limit=args.limit, seed=args.seed, revision=dataset_revision)
        output_dir = run_evaluation(
            items,
            scorer,
            dataset_name=name,
            output_root=args.output_dir,
            seed=args.seed,
            dataset_revision=dataset_revision,
        )
        summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
        print(f"{name}: {summary['correct']}/{summary['total']} ({summary['accuracy']:.4f})")
        print(f"results: {output_dir.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
