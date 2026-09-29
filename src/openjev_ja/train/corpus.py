"""Training corpora, loaded into the project's own `BenchmarkItem` shape.

Parallel to `openjev_ja.eval.local_data` (which turns an *evaluation* dataset's raw rows into
`BenchmarkItem`s), this turns a *training* corpus into the same shape, so any architecture
trainer (`openjev_ja.train.architectures`) consumes items the same way every scorer does:
`item.question`, `item.options`, `item.gold_index`.

Only one format is implemented so far: "eikos" — the `{state, question_type, instructions,
options, expected}` JSONL rows produced by the eikos-decisions corpus (one file per system-one
decision model release, e.g. `eikos-4b`). A corpus is one or more JSONL files with that shape;
`load_corpus` filters by `lang` and `question_type` (the project's `primitive`) and converts.

Adding a new corpus format: give it a name and a `_convert_<name>_row` function below, matching
the pattern the eval side uses for dataset adapters.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from openjev_ja.common import BenchmarkItem

NOUL_OPTIONS = ("いいえ", "はい")


class CorpusError(RuntimeError):
    """A corpus file or row could not be converted."""


def _eikos_row_to_item(row: dict[str, Any], primitive: str, index: int) -> BenchmarkItem:
    state = str(row.get("state") or "").strip()
    instructions = str(row.get("instructions") or "").strip()
    raw_options: list[dict[str, Any]] = row["options"]
    labels = [str(option["label"]) for option in raw_options]
    descriptions = [
        str(option.get("description") or label)
        for option, label in zip(raw_options, labels, strict=True)
    ]
    expected = str(row["expected"])
    if expected not in labels:
        raise CorpusError(f"{row.get('id', index)}: expected {expected!r} not among {labels}")

    if primitive == "noul":
        # The project's convention is a fixed ["いいえ", "はい"] pair with the rubric folded
        # into the question text; eikos already writes yes/no rubric text per option, so it is
        # appended rather than dropped.
        by_label = dict(zip(labels, descriptions, strict=True))
        rubric_parts = [f"はい: {by_label[label]}" for label in labels if label == "yes"]
        rubric_parts += [f"いいえ: {by_label[label]}" for label in labels if label == "no"]
        rubric = "\n".join(rubric_parts)
        question = f"{state}\n\n{instructions}" if state else instructions
        if rubric:
            question = f"{question}\n{rubric}"
        gold = NOUL_OPTIONS.index("はい" if expected == "yes" else "いいえ")
        options = list(NOUL_OPTIONS)
    else:
        question = f"{state}\n\n{instructions}" if state else instructions
        options = descriptions
        gold = labels.index(expected)

    return BenchmarkItem(
        str(row.get("id", index)),
        question,
        options,
        gold,
        {
            "dataset": "eikos",
            "family": row.get("family"),
            "topic": row.get("topic"),
            "source": row.get("source"),
            "difficulty": row.get("difficulty"),
            "lang": row.get("lang"),
        },
    )


_ROW_CONVERTERS = {"eikos": _eikos_row_to_item}


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise CorpusError(f"{path}:{line_number}: invalid JSON") from exc


def load_corpus(
    paths: str | Path | list[str | Path],
    *,
    primitive: str,
    format: str = "eikos",  # noqa: A002 - matches the eval side's `source["format"]` naming
    lang: str | None = "Japanese",
    limit: int | None = None,
) -> list[BenchmarkItem]:
    """Every row of `paths` whose `question_type == primitive` (and `lang == lang`, when given),
    converted to `BenchmarkItem`. Rows are read in file order, files in the order given, so the
    result is deterministic; shuffling is the caller's job (`openjev_ja.train.cv`)."""
    converter = _ROW_CONVERTERS.get(format)
    if converter is None:
        raise CorpusError(f"unsupported corpus format: {format!r}")
    files = [paths] if isinstance(paths, str | Path) else list(paths)
    items: list[BenchmarkItem] = []
    for path in files:
        path = Path(path)
        if not path.is_file():
            raise CorpusError(f"corpus file not found: {path}")
        for index, row in enumerate(_read_jsonl(path)):
            if row.get("question_type") != primitive:
                continue
            if lang is not None and row.get("lang") != lang:
                continue
            items.append(converter(row, primitive, index))
            if limit is not None and len(items) >= limit:
                return items
    return items
