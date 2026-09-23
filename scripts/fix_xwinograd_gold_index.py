"""Recompute xwinograd_ja's gold_index everywhere it was evaluated, using
the _answer_index off-by-one fix, and re-summarize — without re-running any
model (predicted_index/scores are unaffected, only the gold label was wrong).
"""

from __future__ import annotations

import json
from pathlib import Path

from openjev_ja.aggregate.summary import build_evaluation_summary
from openjev_ja.eval.datasets.adapters import _raw_rows, convert_row
from openjev_ja.eval.metrics import summarize

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# (run_root, model_id) pairs that include an xwinograd_ja evaluation
TARGETS: list[tuple[str, str]] = [
    ("eval-embedding-series", "e5-small-choice"),
    ("eval-embedding-series", "ruri-v3-310m-choice"),
    ("eval-embedding-series", "ruri-v3-130m-choice"),
    ("eval-embedding-series", "ruri-v3-70m-choice"),
    ("eval-embedding-series", "ruri-v3-30m-choice"),
    ("eval-embedding-series", "bekko-v1-a8m-choice"),
    ("eval-embedding-series", "bekko-v1-a25m-choice"),
    ("eval-laya-series", "laya-multilingual-choice"),
    ("eval-jevlike-series", "jevlike-ruri130m-choice"),
    ("eval-laya-bert-series", "laya-bert-ruri130m-choice"),
    ("eval-openjev-series", "openjev-0.8b"),
    ("eval-openjev-series", "openjev-4b-v2"),
    ("eval-semif-logit-series", "semif-logit-qwen3.5-4b"),
    ("eval-semif-logit-fewshot-series", "semif-logit-fewshot-qwen3.5-4b-2shot-choice"),
    ("eval-primitives-20260918", "qwen3.5-4b"),
    ("eval-jev-latest", "jev-latest"),
]

# configs used for each run_root, so eval_choice.json / eval_summary.json can
# be rebuilt afterwards (skip ones whose config isn't in this repo checkout).
RUN_CONFIGS: dict[str, str] = {
    "eval-embedding-series": "configs/eval/embedding-series.yaml",
    "eval-laya-series": "configs/eval/laya-series.yaml",
    "eval-jevlike-series": "configs/eval/jevlike-series.yaml",
    "eval-laya-bert-series": "configs/eval/laya-bert-series.yaml",
    "eval-openjev-series": "configs/eval/openjev-series.yaml",
    "eval-semif-logit-series": "configs/eval/semif-logit-series.yaml",
    "eval-semif-logit-fewshot-series": "configs/eval/semif-logit-fewshot-series.yaml",
}


def correct_gold_map() -> dict[str, int]:
    rows = list(_raw_rows("xwinograd_ja", None))
    mapping = {}
    for index, row in enumerate(rows):
        item = convert_row("xwinograd_ja", row, index)
        mapping[item.id] = item.gold_index
    return mapping


def fix_one(run_root: Path, model_id: str, correct_gold: dict[str, int]) -> None:
    ds_dir = run_root / model_id / "xwinograd_ja"
    pred_path = ds_dir / "predictions.jsonl"
    if not pred_path.is_file():
        print(f"  skip (no predictions.jsonl): {run_root.name}/{model_id}")
        return
    lines = pred_path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    changed = 0
    for row in rows:
        new_gold = correct_gold[row["id"]]
        if new_gold != row["gold_index"]:
            changed += 1
        row["gold_index"] = new_gold
    with pred_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    elapsed = sum(row["latency_ms"] for row in rows) / 1000.0
    new_summary = summarize(rows, elapsed, task_type="choice")
    (ds_dir / "summary.json").write_text(
        json.dumps(new_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    old_accuracy = None
    choice_summary_path = run_root / model_id / "summary.choice.json"
    if choice_summary_path.is_file():
        choice_rows = json.loads(choice_summary_path.read_text(encoding="utf-8"))
        for entry in choice_rows:
            if entry["dataset"] == "xwinograd_ja":
                old_accuracy = entry["accuracy"]
                entry.update(new_summary)
                entry["dataset"] = "xwinograd_ja"
        choice_summary_path.write_text(
            json.dumps(choice_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(
        f"  {run_root.name}/{model_id}: {changed}/{len(rows)} gold flipped, "
        f"accuracy {old_accuracy} -> {new_summary['accuracy']:.4f}"
    )


def main() -> None:
    correct_gold = correct_gold_map()
    print(f"correct gold map built: {len(correct_gold)} items\n")

    touched_runs: set[str] = set()
    for run_name, model_id in TARGETS:
        run_root = RESULTS / run_name
        if not run_root.is_dir():
            print(f"skip (missing run): {run_name}")
            continue
        fix_one(run_root, model_id, correct_gold)
        touched_runs.add(run_name)

    print("\nrebuilding eval_choice.json / eval_summary.json for runs with a known config...")
    for run_name in touched_runs:
        config_rel = RUN_CONFIGS.get(run_name)
        if not config_rel:
            print(f"  skip (no config on file, summary left as-is): {run_name}")
            continue
        output = build_evaluation_summary(ROOT / config_rel, phase="production")
        print(f"  rebuilt {output}")


if __name__ == "__main__":
    main()
