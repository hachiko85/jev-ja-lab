"""Recompute eval_summary.json's choice/overall scores for runs whose
config isn't checked out in this repo (eval-primitives-20260918,
eval-jev-latest, eval-20260920), after a dataset-level fix script updated
their summary.choice.json on disk but build_evaluation_summary() can't be
re-run (no config, or the config only covers a subset of models).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

DATASET = sys.argv[1] if len(sys.argv) > 1 else "xwinograd_ja"

TARGETS = (
    [
        ("eval-primitives-20260918", "qwen3.5-4b"),
        ("eval-jev-latest", "jev-latest"),
    ]
    if DATASET == "xwinograd_ja"
    else [
        ("eval-primitives-20260918", "qwen3.5-4b"),
        ("eval-20260920", "sarashina2.2-1b-instruct-v0.1"),
        ("eval-20260920", "sarashina2.2-3b-instruct-v0.1"),
        ("eval-jev-latest", "jev-latest"),
    ]
)


def main() -> None:
    for run_name, model_id in TARGETS:
        run_root = RESULTS / run_name
        summary_path = run_root / "eval_summary.json"
        choice_path = run_root / model_id / "summary.choice.json"
        if not summary_path.is_file() or not choice_path.is_file():
            print(f"skip (missing files): {run_name}")
            continue

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        choice_rows = json.loads(choice_path.read_text(encoding="utf-8"))
        xwinograd = next(r for r in choice_rows if r["dataset"] == DATASET)

        model_entry = next(m for m in summary["models"] if m["id"] == model_id)
        choice_datasets = model_entry["primitives"]["choice"]["datasets"]
        entry = next(d for d in choice_datasets if d["dataset"] == DATASET)
        old_value = entry["value"]
        entry["value"] = xwinograd["accuracy"]
        entry["normalized_score"] = xwinograd["accuracy"]  # accuracy is already 0-1

        new_choice_score = sum(d["normalized_score"] for d in choice_datasets) / len(
            choice_datasets
        )
        model_entry["primitives"]["choice"]["score"] = new_choice_score

        primitive_names = ("noul", "choice", "score")
        available = [
            p for p in primitive_names if model_entry["primitives"][p]["score"] is not None
        ]
        old_overall = model_entry["overall_score"]
        model_entry["overall_score"] = (
            sum(model_entry["primitives"][p]["score"] for p in available) / len(available)
            if available
            else None
        )

        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"{run_name}/{model_id}: {DATASET} {old_value:.4f}->{xwinograd['accuracy']:.4f}, "
            f"choice.score updated, overall {old_overall:.4f}->{model_entry['overall_score']:.4f}"
        )


if __name__ == "__main__":
    main()
