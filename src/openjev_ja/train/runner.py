"""Architecture-agnostic k-fold cross-validation training loop.

Given a list of `BenchmarkItem`s for one primitive and an `ArchitectureSpec` (see
`architectures.py`), this trains one fold at a time — a fresh model per fold, supervised
cross-entropy over the architecture's own logits — and scores each fold's held-out split with
the same metrics the evaluation harness uses (`eval.metrics`), so a trained head's numbers read
against the benchmark tables the same way. It knows nothing about laya, embeddings, or any
particular encoder: swapping `architecture` in the YAML config is the only thing that changes.
"""

from __future__ import annotations

import random
import time
from typing import Any

from openjev_ja.common import BenchmarkItem
from openjev_ja.eval.metrics import binary_classification_metrics, ordinal_metrics
from openjev_ja.train.architectures import ArchitectureSpec
from openjev_ja.train.cv import kfold_split


def _accuracy(predicted: list[int], gold: list[int]) -> float:
    correct = sum(p == g for p, g in zip(predicted, gold, strict=True))
    return correct / len(gold) if gold else 0.0


def _fold_metrics(primitive: str, predicted: list[int], gold: list[int]) -> dict[str, float]:
    metrics = {"accuracy": _accuracy(predicted, gold), "count": float(len(gold))}
    if primitive == "noul":
        metrics.update(binary_classification_metrics(predicted, gold))
    elif primitive == "score":
        metrics.update(ordinal_metrics(predicted, gold))
    return metrics


def _predict_all(
    architecture: ArchitectureSpec, model: Any, tokenizer: Any, device: str,
    items: list[BenchmarkItem], primitive: str, torch: Any,
) -> tuple[list[int], list[int]]:
    predicted, gold = [], []
    with torch.no_grad():
        for item in items:
            logits = architecture.forward_logits(
                model, tokenizer, device, item.question, item.options, primitive
            )
            if logits is None:
                continue
            predicted.append(int(logits.argmax(-1).item()))
            gold.append(item.gold_index)
    return predicted, gold


def train_one_fold(
    architecture: ArchitectureSpec,
    *,
    model_name: str,
    primitive: str,
    train_items: list[BenchmarkItem],
    val_items: list[BenchmarkItem],
    epochs: int = 3,
    lr: float = 1e-3,
    seed: int = 42,
    device: str = "cpu",
    log_every: int = 0,
    build_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train one fold from a freshly built model; -> {history, val_metrics, state_dict}."""
    import torch

    tokenizer, model = architecture.build(model_name, device, **(build_kwargs or {}))
    trainable = list(architecture.trainable_parameters(model))
    if not trainable:
        raise ValueError(f"architecture {architecture.name!r} has no trainable parameters")
    optimizer = torch.optim.Adam(trainable, lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    rng = random.Random(seed)
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(epochs):
        order = train_items[:]
        rng.shuffle(order)
        total_loss, counted = 0.0, 0
        for step, item in enumerate(order, start=1):
            logits = architecture.forward_logits(
                model, tokenizer, device, item.question, item.options, primitive
            )
            if logits is None:
                continue
            target = torch.tensor([item.gold_index], device=device)
            loss = loss_fn(logits.unsqueeze(0), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            counted += 1
            if log_every and step % log_every == 0:
                print(
                    f"epoch {epoch + 1}/{epochs} step {step}/{len(order)} "
                    f"loss={total_loss / max(1, counted):.4f}",
                    flush=True,
                )
        model.eval()
        predicted, gold = _predict_all(
            architecture, model, tokenizer, device, val_items, primitive, torch
        )
        model.train()
        history.append({
            "epoch": epoch + 1,
            "mean_loss": total_loss / counted if counted else 0.0,
            "val_metrics": _fold_metrics(primitive, predicted, gold) if val_items else None,
        })

    model.eval()
    predicted, gold = _predict_all(
        architecture, model, tokenizer, device, val_items, primitive, torch
    )
    return {
        "history": history,
        "val_metrics": _fold_metrics(primitive, predicted, gold),
        "elapsed_seconds": time.perf_counter() - started,
        "state_dict": architecture.state_dict(model),
    }


def cross_validate(
    architecture: ArchitectureSpec,
    items: list[BenchmarkItem],
    *,
    model_name: str,
    primitive: str,
    folds: int = 5,
    epochs: int = 3,
    lr: float = 1e-3,
    seed: int = 42,
    device: str = "cpu",
    log_every: int = 0,
    build_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One fold at a time (`train_one_fold`); -> per-fold results plus the mean/std of every
    val_metrics key across folds (the number to report for this model+primitive).

    `folds=1` skips cross-validation and trains a single "production" model on every item
    with nothing held out (no `summary`, since there is no val split to score) — use this once
    cross-validation on the same data/architecture has already shown the recipe generalizes,
    to spend every row on the model that then gets benchmarked externally.
    """
    if folds == 1:
        result = train_one_fold(
            architecture, model_name=model_name, primitive=primitive, train_items=items,
            val_items=[], epochs=epochs, lr=lr, seed=seed, device=device, log_every=log_every,
            build_kwargs=build_kwargs,
        )
        fold_results = [{**result, "fold": 0, "val_items": 0}]
        return {
            "architecture": architecture.name, "model": model_name, "primitive": primitive,
            "folds": 1, "items": len(items), "fold_results": fold_results, "summary": {},
        }
    splits = kfold_split(items, folds, seed=seed)
    fold_results = []
    for fold_index, (train_items, val_items) in enumerate(splits):
        print(f"[{model_name}/{primitive}] fold {fold_index + 1}/{folds} "
              f"(train={len(train_items)}, val={len(val_items)})", flush=True)
        result = train_one_fold(
            architecture,
            model_name=model_name,
            primitive=primitive,
            train_items=train_items,
            val_items=val_items,
            epochs=epochs,
            lr=lr,
            seed=seed + fold_index,
            device=device,
            log_every=log_every,
            build_kwargs=build_kwargs,
        )
        fold_results.append({**result, "fold": fold_index, "val_items": len(val_items)})

    keys = fold_results[0]["val_metrics"].keys() if fold_results else []
    summary = {}
    for key in keys:
        if key == "count":
            continue
        values = [fold["val_metrics"][key] for fold in fold_results]
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        summary[key] = {"mean": mean, "std": variance**0.5, "values": values}
    return {
        "architecture": architecture.name,
        "model": model_name,
        "primitive": primitive,
        "folds": folds,
        "items": len(items),
        "fold_results": fold_results,
        "summary": summary,
    }
