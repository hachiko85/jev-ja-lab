"""Generate a configs/eval/laya-bert-*.yaml (laya's DecisionModel head trained from scratch on
a frozen Japanese BERT encoder), reusing bert-series.yaml's dataset/task definitions so every
laya-bert variant is measured on the same standard profile as every other method.

Needs a checkpoint per primitive first (`jev-ja-lab-laya-bert-train`, or
`openjev_ja.train` for the eikos-corpus variant).

    python scripts/generate_laya_bert_config.py                              # ruri-v3-130m
    python scripts/generate_laya_bert_config.py --variant modernbert-ja \
        --encoder sbintuitions/modernbert-ja-310m \
        --revision 77675fc96a7e445e982e2ba90246b816efc74ec6 \
        --head-dir results/laya_bert_heads/modernbert-ja-310m
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

PRIMITIVES = ("noul", "choice", "score")
ROOT = Path(__file__).resolve().parent.parent
COLORS = ("#ff7f0e", "#2ca02c", "#d62728")


def build(
    *,
    series_name: str,
    id_prefix: str,
    chart_prefix: str,
    encoder: str,
    revision: str | None,
    head_dir: str,
    label_suffix: str = "",
) -> Path:
    base = yaml.safe_load((ROOT / "configs/eval/bert-series.yaml").read_text(encoding="utf-8"))
    model_ids = {p: [f"{id_prefix}-{p}"] for p in PRIMITIVES}
    tasks = {}
    for primitive in PRIMITIVES:
        task = dict(base["tasks"][primitive])
        task.pop("parallelism", None)
        task["model_ids"] = model_ids[primitive]
        chart = dict(task.get("chart", {}))
        if chart:
            chart["name"] = f"radar-{chart_prefix}-{primitive}"
            task["chart"] = chart
        tasks[primitive] = task
    summary_task = dict(base["tasks"]["summary"])
    summary_chart = dict(summary_task.get("chart", {}))
    if summary_chart:
        summary_chart["name"] = f"radar-{chart_prefix}-summary"
        summary_task["chart"] = summary_chart
    tasks["summary"] = summary_task

    models = [
        {
            "id": f"{id_prefix}-{primitive}",
            "label": f"laya architecture on {encoder}{label_suffix} ({primitive})",
            "repo_id": encoder,
            **({"revision": revision} if revision else {}),
            "scorer": "laya-bert",
            "primitive": primitive,
            "head_path": f"{head_dir}/{primitive}.pt",
            "color": color,
        }
        for primitive, color in zip(PRIMITIVES, COLORS, strict=True)
    ]
    config = {
        "version": 1,
        "runtime": {
            "run_name": f"eval-{series_name}",
            "smoke_run_name": f"eval-{series_name}-smoke",
            "resume": True,
            "seed": 42,
            "models_root": "models",
            "datasets_root": "datasets",
            "output_root": "results",
            "devices": ["cuda:0"],
            "parallelism": 1,
            "batch_size": 1,
            "worker_timeout_seconds": 86400,
        },
        "workflow": {"order": ["noul", "choice", "score", "summary"]},
        "tasks": tasks,
        "models": models,
        "datasets": base["datasets"],
    }
    output_path = ROOT / f"configs/eval/{series_name}.yaml"
    output_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    print(f"wrote {output_path} ({len(models)} model entries, {len(base['datasets'])} datasets)")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant", default="", help="empty reproduces the original ruri-v3-130m config"
    )
    parser.add_argument("--encoder", default="cl-nagoya/ruri-v3-130m")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--head-dir", default="results/laya_bert_heads/ruri-130m")
    parser.add_argument("--label-suffix", default="")
    args = parser.parse_args()
    if args.variant:
        series_name = f"laya-bert-{args.variant}-series"
        id_prefix = chart_prefix = f"laya-bert-{args.variant}"
    else:
        series_name, id_prefix, chart_prefix = "laya-bert-series", "laya-bert-ruri130m", "laya-bert"
    build(
        series_name=series_name,
        id_prefix=id_prefix,
        chart_prefix=chart_prefix,
        encoder=args.encoder,
        revision=args.revision,
        head_dir=args.head_dir,
        label_suffix=args.label_suffix,
    )


if __name__ == "__main__":
    main()
