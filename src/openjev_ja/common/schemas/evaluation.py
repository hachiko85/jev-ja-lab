"""The one thing every method must agree on: how a result gets reported.

Per JEV_JA_LAB_REFACTOR_GUIDE_v2 section 9, method internals stay
method-specific (a `qwen-direct` prompt has nothing in common with a
`[MASK]` position or an NLI premise/hypothesis pair). What is shared is the
answer to: which dataset, which sample, which method/system, what did it
predict. `EvaluationRecord` is that shared answer, and `iter_evaluation_records`
converts an existing `results/<run_id>/` directory (written by
`openjev_ja.eval.runner.run_evaluation`) into a stream of them without
requiring any change to how that directory is produced or already stored.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    schema_version: int

    run_id: str
    method_id: str
    system_id: str
    model_id: str | None

    dataset_id: str
    dataset_revision: str | None
    sample_id: str

    primitive: str

    gold_index: int | None
    predicted_index: int | None

    scores: list[float] | None
    probabilities: list[float] | None

    latency_ms: float | None

    metadata: dict[str, Any] = field(default_factory=dict)


def iter_evaluation_records(run_dir: str | Path) -> Iterator[EvaluationRecord]:
    """Read one `run_evaluation` output directory as common-schema records.

    Legacy runs (written before `schema_version` existed in metadata.json)
    default to schema_version=1; the row shape has not changed, so no
    conversion beyond that default is needed yet (guide section 30).
    """
    run_dir = Path(run_dir)
    run_metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    scorer_metadata = run_metadata.get("scorer", {})
    method_id = str(scorer_metadata.get("scorer", "unknown"))
    model_id = scorer_metadata.get("model")
    system_id = f"{method_id}:{model_id}" if model_id else method_id
    run_id = str(run_metadata.get("run_id", run_dir.name))
    dataset_id = str(run_metadata.get("dataset", run_dir.name))
    dataset_revision = run_metadata.get("dataset_revision")
    primitive = str(run_metadata.get("task_type", "choice"))
    schema_version = int(run_metadata.get("schema_version", SCHEMA_VERSION))

    predictions_path = run_dir / "predictions.jsonl"
    with predictions_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            yield EvaluationRecord(
                schema_version=schema_version,
                run_id=run_id,
                method_id=method_id,
                system_id=system_id,
                model_id=model_id,
                dataset_id=dataset_id,
                dataset_revision=dataset_revision,
                sample_id=str(row["id"]),
                primitive=primitive,
                gold_index=row.get("gold_index"),
                predicted_index=row.get("predicted_index"),
                scores=row.get("scores"),
                probabilities=row.get("probabilities"),
                latency_ms=row.get("latency_ms"),
                metadata={
                    "item_metadata": row.get("item_metadata", {}),
                    "score_metadata": row.get("score_metadata", {}),
                },
            )
