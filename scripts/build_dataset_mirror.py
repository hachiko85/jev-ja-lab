"""Build the mirrored subsets (noul / choice / score / all, split `test`) of openjev-ja-eval.

Every mirrored dataset is read through the same adapters the evaluation harness uses
(`load_local_benchmark`), so a mirrored row is exactly the item a model is scored on:

    id, primitive, dataset_id, source_dataset, source_repo, source_url, source_split,
    source_license, question, options, gold_index, metadata

`primitive` is one of "Noul" / "Choice" / "Score"; `source_dataset` is the name of the upstream
dataset the row comes from. `metadata` is a JSON string with the adapter's own fields (the
original record's identifiers etc.), so that no upstream information is lost.

Only datasets that can be redistributed inside a CC BY-SA 4.0 collection are mirrored
(`distribution: "mirror"` in datasets_router/manifest.json). The rest stay routed.

    python scripts/build_dataset_mirror.py --datasets-root ./datasets --out build/mirror
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from openjev_ja.eval.local_data import load_local_benchmark  # noqa: E402

MANIFEST = ROOT / "datasets_router" / "manifest.json"
PRIMITIVE_LABEL = {"noul": "Noul", "choice": "Choice", "score": "Score"}


def profile_sources() -> dict[str, dict]:
    """profile dataset id -> (source config) from the eval configs."""
    sources: dict[str, dict] = {}
    for name in ("bert-series.yaml", "helpsteer-series.yaml"):
        config = yaml.safe_load((ROOT / "configs/eval" / name).read_text(encoding="utf-8"))
        task_of = {
            dataset_id: task
            for task, body in config["tasks"].items()
            if isinstance(body, dict)
            for dataset_id in body.get("datasets", [])
        }
        for dataset in config["datasets"]:
            # helpsteer-series.yaml (benchmark-v1) wins over bert-series.yaml (full 19,958)
            if dataset["id"] in sources and name == "bert-series.yaml":
                continue
            sources[dataset["id"]] = {
                "source": dataset["source"],
                "primitive": dataset.get("task") or task_of[dataset["id"]],
            }
    return sources


def build_rows(datasets_root: Path) -> dict[str, list[dict]]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    sources = profile_sources()
    rows: dict[str, list[dict]] = {"noul": [], "choice": [], "score": []}
    for entry in manifest["datasets"]:
        if entry.get("distribution") != "mirror":
            continue
        origin = entry["source"]
        source_repo = origin.get("repo_id") or origin.get("url") or manifest["repo_id"]
        derived = origin.get("derived_from")
        if derived:
            source_repo = derived["repo_id"]
        for dataset_id in entry["profile_datasets"]:
            spec = sources[dataset_id]
            primitive = spec["primitive"]
            items = load_local_benchmark(
                dataset_id, spec["source"], datasets_root=str(datasets_root), seed=42, limit=None
            )
            for item in items:
                metadata = {k: v for k, v in item.metadata.items() if k != "criteria"}
                item_id = str(item.id)
                row_id = item_id if item_id.startswith(dataset_id) else f"{dataset_id}:{item_id}"
                rows[primitive].append(
                    {
                        "id": row_id,
                        "primitive": PRIMITIVE_LABEL[primitive],
                        "dataset_id": dataset_id,
                        "source_dataset": entry["title"],
                        "source_repo": source_repo,
                        "source_url": entry["url"],
                        "source_split": origin.get("split") or "(none)",
                        "source_license": entry["license"],
                        "question": item.question,
                        "options": list(item.options),
                        "gold_index": int(item.gold_index),
                        "metadata": json.dumps(metadata, ensure_ascii=False, default=str),
                    }
                )
    return rows


def main() -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets-root", default="datasets")
    parser.add_argument("--out", default="build/mirror")
    args = parser.parse_args()
    out = Path(args.out)
    rows = build_rows(Path(args.datasets_root))
    rows["all"] = rows["noul"] + rows["choice"] + rows["score"]
    for subset, subset_rows in rows.items():
        path = out / subset / "test.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(subset_rows), path)
        by_dataset: dict[str, int] = {}
        for row in subset_rows:
            by_dataset[row["dataset_id"]] = by_dataset.get(row["dataset_id"], 0) + 1
        print(f"{subset}: {len(subset_rows)} rows  {by_dataset if subset != 'all' else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
