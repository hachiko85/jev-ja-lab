from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from openjev_ja.common import BenchmarkItem
from openjev_ja.eval.metrics import summarize
from openjev_ja.eval.scorers.base import Scorer


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _environment() -> dict[str, Any]:
    environment: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": sys.platform,
        "machine": platform.machine(),
    }
    try:
        import torch

        environment.update(
            {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        )
    except ImportError:
        environment["torch"] = None
    return environment


def make_run_id(dataset: str, scorer: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{dataset}-{scorer}"


def run_evaluation(
    items: list[BenchmarkItem],
    scorer: Scorer,
    *,
    dataset_name: str,
    output_root: str | Path = "results",
    run_id: str | None = None,
    seed: int = 42,
    dataset_revision: str | None = None,
    batch_size: int = 1,
    task_type: str = "choice",
) -> Path:
    run_id = run_id or make_run_id(dataset_name, scorer.name)
    output_dir = Path(output_root) / run_id
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        # A failed worker can leave an empty run directory behind. Reuse only
        # that harmless case; never overwrite partial or completed results.
        if any(output_dir.iterdir()):
            raise
    predictions: list[dict[str, Any]] = []
    started = time.perf_counter()
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    score_batch = getattr(scorer, "score_batch", None)
    for offset in range(0, len(items), batch_size):
        batch = items[offset : offset + batch_size]
        if score_batch is not None and batch_size > 1:
            results = score_batch([(item.question, item.options) for item in batch])
        else:
            results = [scorer.score(item.question, item.options) for item in batch]
        for item, result in zip(batch, results, strict=True):
            row = {
                "id": item.id,
                "gold_index": item.gold_index,
                "predicted_index": result.predicted_index,
                "scores": result.scores,
                "probabilities": result.probabilities,
                "latency_ms": result.latency_ms,
                "item_metadata": item.metadata,
                "score_metadata": result.metadata,
            }
            predictions.append(row)
    elapsed = time.perf_counter() - started
    metadata = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": dataset_name,
        "task_type": task_type,
        "dataset_revision": dataset_revision,
        "seed": seed,
        "batch_size": batch_size,
        "git_commit": _git_commit(),
        "scorer": scorer.metadata(),
        "environment": _environment(),
    }
    summary = summarize(predictions, elapsed, task_type=task_type)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output_dir
