"""Export a project dataset to jevlike's JSONL format, then train a
checkpoint through jevlike's own `jevlike.train` CLI (its training loop is
upstream code, not reimplemented here).

jevlike's `ChoiceExample(context, options, label)` is candidate-count
agnostic, so Noul (2 options) and Score (N ordered-level options) train and
score through the exact same AttentionHead as Choice — no jevlike code
change needed, only which dataset gets exported. Same representative
per-primitive dataset the embedding method trains its head on (see
methods.embedding.train.DEFAULT_SOURCES).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from openjev_ja.common import BenchmarkItem
from openjev_ja.eval.local_data import load_local_benchmark

DEFAULT_SOURCES: dict[str, dict[str, Any]] = {
    "choice": {
        "name": "jcommonsenseqa",
        "train_source": {
            "format": "arrow",
            "path": (
                "sbintuitions___j_commonsense_qa/default/0.0.0/*/"
                "j_commonsense_qa-train.arrow"
            ),
        },
        "validation_source": {
            "format": "arrow",
            "path": (
                "sbintuitions___j_commonsense_qa/default/0.0.0/*/"
                "j_commonsense_qa-validation.arrow"
            ),
        },
    },
    "noul": {
        "name": "jnli_entailment_train",
        "train_source": {
            "format": "jsonl",
            "path": "JGLUE-v1.1.0/datasets/jnli-v1.1/train-v1.1.json",
            "adapter": "jnli_noul",
            "target": "entailment",
        },
        "validation_source": {
            "format": "jsonl",
            "path": "JGLUE-v1.1.0/datasets/jnli-v1.1/valid-v1.1.json",
            "adapter": "jnli_noul",
            "target": "entailment",
        },
    },
    "score": {
        "name": "wrime_joy_train",
        "train_source": {
            "format": "tsv",
            "path": "wrime-official/wrime-ver2.tsv",
            "adapter": "score",
            "question_field": "Sentence",
            "gold_field": "Avg. Readers_Joy",
            "criteria": ["喜びなし", "弱い喜び", "中程度の喜び", "強い喜び"],
            "prompt": "次の文章から読み手が感じる喜びの強度を評価してください。",
            "where": {"Train/Dev/Test": "train"},
        },
        "validation_source": {
            "format": "tsv",
            "path": "wrime-official/wrime-ver2.tsv",
            "adapter": "score",
            "question_field": "Sentence",
            "gold_field": "Avg. Readers_Joy",
            "criteria": ["喜びなし", "弱い喜び", "中程度の喜び", "強い喜び"],
            "prompt": "次の文章から読み手が感じる喜びの強度を評価してください。",
            "where": {"Train/Dev/Test": "dev"},
        },
    },
}


def _write_jsonl(items: list[BenchmarkItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            row = {"context": item.question, "options": item.options, "label": item.gold_index}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_jsonl(
    datasets_root: str,
    output_dir: str | Path,
    *,
    primitive: str = "choice",
    limit: int | None = None,
    seed: int = 42,
) -> tuple[Path, Path]:
    if primitive not in DEFAULT_SOURCES:
        raise ValueError(f"unsupported primitive: {primitive}")
    default = DEFAULT_SOURCES[primitive]
    output_dir = Path(output_dir)
    train_items = load_local_benchmark(
        default["name"],
        default["train_source"],
        datasets_root=datasets_root,
        seed=seed,
        limit=limit,
    )
    validation_items = load_local_benchmark(
        default["name"],
        default["validation_source"],
        datasets_root=datasets_root,
        seed=seed,
        limit=limit,
    )
    train_path = output_dir / "train.jsonl"
    validation_path = output_dir / "validation.jsonl"
    _write_jsonl(train_items, train_path)
    _write_jsonl(validation_items, validation_path)
    return train_path, validation_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jev-ja-lab-jevlike-train",
        description="Export a project Choice dataset and train a jevlike checkpoint.",
    )
    parser.add_argument("--datasets-root", required=True)
    parser.add_argument("--data-dir", required=True, help="where to write train/validation JSONL")
    parser.add_argument("--output", required=True)
    parser.add_argument("--primitive", choices=list(DEFAULT_SOURCES), default="choice")
    parser.add_argument("--encoder", choices=("tiny", "hf"), default="hf")
    parser.add_argument("--hf-model", default="cl-nagoya/ruri-v3-130m")
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--context-tokens", type=int, default=256)
    parser.add_argument("--option-tokens", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    train_path, validation_path = prepare_jsonl(
        args.datasets_root,
        args.data_dir,
        primitive=args.primitive,
        limit=args.limit,
        seed=args.seed,
    )
    command = [
        sys.executable,
        "-m",
        "jevlike.train",
        str(train_path),
        "--validation",
        str(validation_path),
        "--output",
        args.output,
        "--encoder",
        args.encoder,
        "--hf-model",
        args.hf_model,
        "--rank",
        str(args.rank),
        "--context-tokens",
        str(args.context_tokens),
        "--option-tokens",
        str(args.option_tokens),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
    ]
    subprocess.run(command, check=True)
    print(f"head: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
