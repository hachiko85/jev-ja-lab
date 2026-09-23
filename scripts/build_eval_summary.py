"""Consolidate this session's method runs into results/eval-summary/,
following the same pattern as results/eval-20260920 (model directories
copied verbatim, per-run root artifacts kept under _sources/<run-name>/,
model-level eval_summary.json entries merged into one).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = RESULTS / "eval-summary"

# (run_name, [model_ids]) - None means "copy every model dir found in the run"
RUNS: list[tuple[str, list[str] | None]] = [
    ("eval-embedding-series", None),
    ("eval-laya-series", None),
    ("eval-jevlike-series", None),
    ("eval-laya-bert-series", None),
    ("eval-openjev-series", None),
    ("eval-semif-logit-series", None),
    ("eval-semif-logit-fewshot-series", None),
    ("eval-primitives-20260918", ["qwen3.5-4b"]),  # pre-existing next_token_logit baseline
]

ROOT_ARTIFACT_GLOBS = [
    "eval_*.json",
    "radar-*.png",
    "radar-*.yaml",
    "run_config*.json",
    "skipped*.json",
]


def model_dirs(run_root: Path, only: list[str] | None) -> list[Path]:
    if only is not None:
        return [run_root / name for name in only if (run_root / name).is_dir()]
    primitives = ("choice", "noul", "score")

    def has_summary(path: Path) -> bool:
        return any((path / f"summary.{primitive}.json").exists() for primitive in primitives)

    return [p for p in run_root.iterdir() if p.is_dir() and has_summary(p)]


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    sources_dir = OUT / "_sources"
    sources_dir.mkdir()

    merged_models: list[dict] = []
    seen_ids: set[str] = set()
    copied_model_ids: list[tuple[str, str]] = []  # (run_name, model_id)

    for run_name, only in RUNS:
        run_root = RESULTS / run_name
        if not run_root.is_dir():
            print(f"skip (missing): {run_name}")
            continue

        # 1. model directories -> OUT/<model_id>/
        for model_dir in model_dirs(run_root, only):
            model_id = model_dir.name
            dest = OUT / model_id
            if dest.exists():
                print(f"  skip (id collision, keeping earlier run's copy): {model_id}")
                continue
            shutil.copytree(model_dir, dest)
            copied_model_ids.append((run_name, model_id))

        # 2. root-level artifacts -> OUT/_sources/<run_name>/
        run_source_dir = sources_dir / run_name
        run_source_dir.mkdir(exist_ok=True)
        for pattern in ROOT_ARTIFACT_GLOBS:
            for path in run_root.glob(pattern):
                shutil.copy2(path, run_source_dir / path.name)

        # 3. merge this run's eval_summary.json models[] entries
        summary_path = run_root / "eval_summary.json"
        if summary_path.is_file():
            try:
                data = json.loads(summary_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                data = json.loads(summary_path.read_text(encoding="cp932"))
            for model in data.get("models", []):
                if only is not None and model.get("id") not in only:
                    continue
                if model["id"] in seen_ids:
                    continue
                seen_ids.add(model["id"])
                merged_models.append(model)

        print(f"{run_name}: {len(copied_model_ids)} total model dirs copied so far")

    merged_models.sort(key=lambda m: -(m.get("overall_score") or 0))
    from datetime import UTC, datetime

    payload = {
        "version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "phase": "production",
        "aggregation": "macro average by dataset, then equal-weight mean by primitive",
        "models": merged_models,
    }
    (OUT / "eval_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"\nwrote {OUT / 'eval_summary.json'} ({len(merged_models)} models)")
    for m in merged_models:
        print(f"  {m['id']:40s} overall={m.get('overall_score')}")


if __name__ == "__main__":
    main()
