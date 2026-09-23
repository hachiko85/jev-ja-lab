"""Train laya's own DecisionModel head on a frozen Japanese BERT encoder.

Supervised cross-entropy over laya's `[MASK]`-marker logits, not laya's own
RLCD pipeline (see methods.laya_bert.model for what is reused). Same
representative per-primitive dataset the embedding method trains its head
on (see methods.embedding.train.DEFAULT_SOURCES).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

from openjev_ja.common import BenchmarkItem
from openjev_ja.eval.local_data import load_local_benchmark
from openjev_ja.methods.laya_bert.model import build_model, encode

DEFAULT_SOURCES: dict[str, dict[str, Any]] = {
    "choice": {
        "name": "jcommonsenseqa",
        "source": {
            "format": "arrow",
            "path": (
                "sbintuitions___j_commonsense_qa/default/0.0.0/*/"
                "j_commonsense_qa-train.arrow"
            ),
        },
    },
    "score": {
        "name": "wrime_joy_train",
        "source": {
            "format": "tsv",
            "path": "wrime-official/wrime-ver2.tsv",
            "adapter": "score",
            "question_field": "Sentence",
            "gold_field": "Avg. Readers_Joy",
            "criteria": ["喜びなし", "弱い喜び", "中程度の喜び", "強い喜び"],
            "prompt": "次の文章から読み手が感じる喜びの強度を評価してください。",
            "where": {"Train/Dev/Test": "train"},
        },
    },
    "noul": {
        "name": "jnli_entailment_train",
        "source": {
            "format": "jsonl",
            "path": "JGLUE-v1.1.0/datasets/jnli-v1.1/train-v1.1.json",
            "adapter": "jnli_noul",
            "target": "entailment",
        },
    },
}


def _split_train_val(
    items: list[BenchmarkItem], val_fraction: float, seed: int
) -> tuple[list[BenchmarkItem], list[BenchmarkItem]]:
    shuffled = items[:]
    random.Random(seed).shuffle(shuffled)
    if val_fraction <= 0.0 or len(shuffled) <= 1:
        return shuffled, []
    val_count = max(1, int(len(shuffled) * val_fraction))
    return shuffled[val_count:], shuffled[:val_count]


def _forward_logits(model: Any, tokenizer: Any, device: str, item: BenchmarkItem, primitive: str):
    from laya.common import collate_items

    encoded = encode(tokenizer, item.question, item.options, primitive)
    if encoded is None:
        return None
    batch = collate_items([[encoded]], tokenizer.pad_token_id)
    logits, _ = model(
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
        batch["marker_pos"].to(device),
        batch["marker_mask"].to(device),
        batch["qtype"].to(device),
        detach_encoder=True,
    )
    return logits[0, : len(item.options)]


def _evaluate(
    model: Any, tokenizer: Any, device: str, items: list[BenchmarkItem], primitive: str, torch: Any
) -> float | None:
    if not items:
        return None
    model.eval()
    correct, counted = 0, 0
    with torch.no_grad():
        for item in items:
            logits = _forward_logits(model, tokenizer, device, item, primitive)
            if logits is None:
                continue
            counted += 1
            correct += int(int(logits.argmax(-1).item()) == item.gold_index)
    model.train()
    return correct / counted if counted else None


def train_head(
    *,
    model_name: str,
    primitive: str,
    datasets_root: str,
    output_path: str | Path,
    dataset_name: str | None = None,
    source: dict[str, Any] | None = None,
    limit: int | None = None,
    val_fraction: float = 0.1,
    epochs: int = 5,
    lr: float = 1e-3,
    seed: int = 42,
    device: str = "cpu",
    log_every: int = 500,
) -> dict[str, Any]:
    import torch

    if primitive not in DEFAULT_SOURCES:
        raise ValueError(f"unsupported primitive: {primitive}")
    default = DEFAULT_SOURCES[primitive]
    name = dataset_name or default["name"]
    resolved_source = source or default["source"]
    items = load_local_benchmark(
        name, resolved_source, datasets_root=datasets_root, seed=seed, limit=limit
    )
    train_items, val_items = _split_train_val(items, val_fraction, seed)

    tokenizer, model = build_model(model_name, device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr)
    ce_loss = torch.nn.CrossEntropyLoss()

    rng = random.Random(seed)
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(epochs):
        order = train_items[:]
        rng.shuffle(order)
        total_loss, counted = 0.0, 0
        for step, item in enumerate(order, start=1):
            logits = _forward_logits(model, tokenizer, device, item, primitive)
            if logits is None:
                continue
            target = torch.tensor([item.gold_index], device=device)
            loss = ce_loss(logits.unsqueeze(0), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            counted += 1
            if log_every and step % log_every == 0:
                print(
                    f"epoch {epoch + 1}/{epochs} step {step}/{len(order)} "
                    f"loss={total_loss / max(1, counted):.4f}"
                )
        sample = train_items[: min(200, len(train_items))]
        record = {
            "epoch": epoch + 1,
            "mean_loss": total_loss / counted if counted else 0.0,
            "train_accuracy_sample": _evaluate(model, tokenizer, device, sample, primitive, torch),
            "val_accuracy": _evaluate(model, tokenizer, device, val_items, primitive, torch),
        }
        history.append(record)
        print(f"[{model_name} / {primitive}] {record}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trainable_state = {
        param_name: parameter.detach().cpu()
        for param_name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    torch.save(trainable_state, output_path)

    result = {
        "model": model_name,
        "primitive": primitive,
        "dataset": name,
        "train_items": len(train_items),
        "val_items": len(val_items),
        "epochs": epochs,
        "lr": lr,
        "elapsed_seconds": time.perf_counter() - started,
        "history": history,
        "head_path": str(output_path),
    }
    output_path.with_suffix(".metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jev-ja-lab-laya-bert-train",
        description="Train laya's DecisionModel head on a frozen Japanese BERT encoder.",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--primitive", required=True, choices=list(DEFAULT_SOURCES))
    parser.add_argument("--datasets-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-name")
    parser.add_argument("--source-json")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = json.loads(args.source_json) if args.source_json else None
    result = train_head(
        model_name=args.model,
        primitive=args.primitive,
        datasets_root=args.datasets_root,
        output_path=args.output,
        dataset_name=args.dataset_name,
        source=source,
        limit=args.limit,
        val_fraction=args.val_fraction,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
    )
    print(f"head: {result['head_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
