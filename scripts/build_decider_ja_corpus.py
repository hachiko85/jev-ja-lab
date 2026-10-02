"""Convert datasets/for-decider-ja/*.parquet into decider's own training-cache format.

decider.train expects `--data <pickle>` to unpickle to (train, evals): a flat list of
decider.data.core.Example(context, qs=[Q(text, options, gold)], task) and a dict of
held-out eval examples by task name (decider.data.core.load_cache / decider.train.main).
This script builds exactly that pickle from the two parquet files the user generated
following decider's own published schema (programmatic-all.parquet: solver-verified
arithmetic/logic decisions; SeeDummy_*.parquet: document-grounded decisions with
teacher-distillation fields decider.train itself does not consume -- only target_index
is used, matching decider's own supervised CE recipe).

Usage:
    python scripts/build_decider_ja_corpus.py --out datasets/for-decider-ja/corpus.pkl
"""
from __future__ import annotations

import argparse
import json
import pickle
import random

import pyarrow.parquet as pq

from decider.data.core import Example, Q

PROGRAMMATIC = "datasets/for-decider-ja/programmatic-all.parquet"
SEEDUMMY = "datasets/for-decider-ja/SeeDummy_jev_native_reasoning_2258rows.parquet"
EVAL_HOLDOUT = 0.05
SEED = 0


def _sorted_labels(criteria: list[dict]) -> list[str]:
    return [c["label"] for c in sorted(criteria, key=lambda c: c["index"])]


def _programmatic_examples() -> list[Example]:
    rows = pq.read_table(PROGRAMMATIC).to_pylist()
    out = []
    for row in rows:
        options = _sorted_labels(row["criteria"])
        if row["primitive"] == "choice":
            gold = row["answer_index"]
        elif row["primitive"] == "bool":
            want = "true" if row["answer_bool"] else "false"
            gold = next(c["index"] for c in row["criteria"] if c["id"] == want)
        else:
            raise ValueError(f"unexpected primitive {row['primitive']!r}")
        task = f"decider_ja_programmatic_{row['primitive']}"
        out.append(Example(row["state_text"], [Q(row["instructions"], options, gold)], task))
    return out


def _seedummy_examples() -> list[Example]:
    rows = pq.read_table(SEEDUMMY).to_pylist()
    out = []
    for row in rows:
        context = f"{row['document_title']}\n\n{row['document_text']}"
        criteria = json.loads(row["criteria_json"])
        options = _sorted_labels(criteria)
        gold = row["target_index"]
        task = f"decider_ja_seedummy_{row['primitive']}"
        out.append(Example(context, [Q(row["instructions"], options, gold)], task))
    return out


def _split(examples: list[Example], rng: random.Random) -> tuple[list[Example], dict[str, list[Example]]]:
    by_task: dict[str, list[Example]] = {}
    for e in examples:
        by_task.setdefault(e.task, []).append(e)
    train: list[Example] = []
    evals: dict[str, list[Example]] = {}
    for task, items in by_task.items():
        items = list(items)
        rng.shuffle(items)
        n_eval = max(1, int(len(items) * EVAL_HOLDOUT))
        evals[task] = items[:n_eval]
        train.extend(items[n_eval:])
    rng.shuffle(train)
    return train, evals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="datasets/for-decider-ja/corpus.pkl")
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    examples = _programmatic_examples() + _seedummy_examples()
    train, evals = _split(examples, rng)

    print(f"[corpus] {len(examples)} examples -> {len(train)} train, "
          f"{sum(len(v) for v in evals.values())} eval across {len(evals)} tasks")
    for task, items in sorted(evals.items()):
        print(f"  {task}: eval={len(items)}")

    pickle.dump((train, evals), open(a.out, "wb"))
    print(f"[corpus] wrote {a.out}")


if __name__ == "__main__":
    main()
