"""K-fold cross-validation splitting, independent of any architecture or data format."""

from __future__ import annotations

import random
from typing import TypeVar

T = TypeVar("T")


def kfold_indices(n: int, folds: int, *, seed: int = 42) -> list[tuple[list[int], list[int]]]:
    """`folds` (train_indices, val_indices) pairs over `range(n)`.

    A shuffled `range(n)` is cut into `folds` contiguous, near-equal chunks (the first
    `n % folds` chunks get one extra item); each chunk is the validation set of one fold and
    every other index is that fold's training set. Every index is a validation index exactly
    once, so the folds' validation sets partition the whole corpus.
    """
    if folds < 2:
        raise ValueError(f"folds must be >= 2, got {folds}")
    if n < folds:
        raise ValueError(f"{n} items cannot be split into {folds} folds")
    order = list(range(n))
    random.Random(seed).shuffle(order)
    base, remainder = divmod(n, folds)
    chunks: list[list[int]] = []
    start = 0
    for fold in range(folds):
        size = base + (1 if fold < remainder else 0)
        chunks.append(order[start : start + size])
        start += size
    return [
        (
            [index for other, chunk in enumerate(chunks) if other != fold for index in chunk],
            chunks[fold],
        )
        for fold in range(folds)
    ]


def kfold_split(items: list[T], folds: int, *, seed: int = 42) -> list[tuple[list[T], list[T]]]:
    """`kfold_indices` applied to `items` directly: `folds` (train_items, val_items) pairs."""
    return [
        ([items[i] for i in train], [items[i] for i in val])
        for train, val in kfold_indices(len(items), folds, seed=seed)
    ]
