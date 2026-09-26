"""Generate the HelpSteer2-JA Score evaluation configs.

HelpSteer2-JA (datasets/helpsteer2_ja/benchmark-v1.parquet: 2,500 prompt-deduplicated rows,
five 0-4 axes) is an additional Score indicator. It is evaluated as five score datasets
(correctness / helpfulness / verbosity / complexity / coherence) with the same
normalized-QWK metric as the other Score datasets, but kept in its own run so the existing
noul / choice / score aggregates are not changed.

- configs/eval/helpsteer-series.yaml: local models (semif, semif-ja, laya, AlexWortega
  openjev, BERT, hopper, decider, plus the ruri-v3-310m embedding row of the summary
  table), one GPU, sequential.
- configs/eval/helpsteer-jev.yaml: the Jev API (`jev-latest`), evaluated over the API.

Each dataset entry reuses the criteria/prompt already defined in bert-series.yaml and only
swaps the source file for the benchmark-v1 subset.
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_PATH = "helpsteer2_ja/benchmark-v1.parquet"
BENCHMARK_ROWS = 2500
AXES = ("correctness", "helpfulness", "verbosity", "complexity", "coherence")
QWEN = {"repo_id": "Qwen/Qwen3.5-4B", "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"}

LOCAL_MODELS = [
    {"id": "semif-logit-qwen3.5-4b", "label": "semif 0-shot (Qwen3.5-4B)", **QWEN,
     "scorer": "semif-logit", "dtype": "bfloat16", "batch_size": 1},
    {"id": "semif-logit-fewshot-qwen3.5-4b-2shot-score",
     "label": "semif 2-shot (Qwen3.5-4B)", **QWEN, "scorer": "semif-logit",
     "primitive": "score", "few_shot_count": 2, "dtype": "bfloat16", "batch_size": 1},
    {"id": "qwen3.5-4b", "label": "semif-ja 0-shot (Qwen3.5-4B)", **QWEN,
     "scorer": "qwen-direct", "dtype": "bfloat16", "batch_size": 1},
    {"id": "laya-multilingual-score", "label": "laya multilingual", "scorer": "laya",
     "primitive": "score", "checkpoint": "multilingual"},
    {"id": "openjev-0.8b", "label": "AlexWortega_openjev 0.8B v2",
     "repo_id": "AlexWortega/openjev", "scorer": "nli-cross-encoder",
     "subfolder": "qwen3.5-0.8b-nli-v2s-long", "trust_remote_code": True, "template": "ja",
     "dtype": "bfloat16", "batch_size": 1},
    {"id": "openjev-4b-v2", "label": "AlexWortega_openjev 4B v2",
     "repo_id": "AlexWortega/openjev", "scorer": "nli-cross-encoder",
     "subfolder": "qwen3.5-4b-nli-v2", "trust_remote_code": True, "template": "ja",
     "dtype": "bfloat16", "batch_size": 1},
    {"id": "hopper-qwen3.5-4b-lora-score", "label": "Hopper (LoRA on Qwen3.5-4B)",
     "repo_id": "Qwen/Qwen3.5-4B", "revision": QWEN["revision"], "adapter": "HopitAI/hopper",
     "scorer": "hopper", "primitive": "score", "dtype": "bfloat16", "batch_size": 1},
    {"id": "decider-4b-v2-score", "label": "decider-4b v2", "repo_id": "Mapika/decider-4b",
     "revision": "v2", "scorer": "decider", "primitive": "score", "dtype": "bfloat16",
     "batch_size": 1},
    {"id": "ruri-v3-310m-score", "label": "ruri-v3-310m (embedding)",
     "repo_id": "cl-nagoya/ruri-v3-310m", "scorer": "embedding", "primitive": "score",
     "head_path": "results/embedding_heads/ruri-v3-310m/score.pt", "dtype": "bfloat16",
     "batch_size": 32},
    {"id": "modernbert-ja-310m", "label": "modernbert-ja-310m",
     "repo_id": "sbintuitions/modernbert-ja-310m",
     "revision": "77675fc96a7e445e982e2ba90246b816efc74ec6", "dtype": "bfloat16",
     "batch_size": 32},
]
JEV_MODELS = [
    {"id": "jev-latest", "label": "Jev Latest", "scorer": "jev", "model": "jev-latest",
     "batch_size": 1, "timeout": 60, "max_retries": 3},
]


def build(name: str, models: list[dict], runtime_extra: dict) -> Path:
    base = yaml.safe_load((ROOT / "configs/eval/bert-series.yaml").read_text(encoding="utf-8"))
    by_id = {d["id"]: d for d in base["datasets"]}
    datasets = []
    for axis in AXES:
        dataset = yaml.safe_load(yaml.safe_dump(by_id[f"helpsteer_{axis}"]))
        dataset["expected_items"] = BENCHMARK_ROWS
        dataset["label"] = f"HelpSteer2 {axis.capitalize()} JA (benchmark-v1)"
        dataset["source"]["path"] = BENCHMARK_PATH
        datasets.append(dataset)

    score_task = {
        "enabled": True,
        "model_ids": [m["id"] for m in models],
        "datasets": [f"helpsteer_{axis}" for axis in AXES],
        "aggregate_metric": "normalized_quadratic_weighted_kappa",
        "smoke_limit": 5,
        "parallelism": runtime_extra.get("parallelism", 1),
        "chart": {
            "enabled": True,
            "name": f"radar-{name}",
            "title": "HelpSteer2-JA Score evaluation",
            "subtitle": "Normalized quadratic weighted kappa",
            "metric": "normalized_quadratic_weighted_kappa",
            "theme": "dark",
            "language": "ja",
        },
    }
    disabled = {"enabled": False}
    config = {
        "version": 1,
        "runtime": {
            "run_name": f"eval-{name}",
            "smoke_run_name": f"eval-{name}-smoke",
            "resume": True,
            "seed": 42,
            "models_root": "models",
            "datasets_root": "datasets",
            "output_root": "results",
            "devices": ["cuda:0"],
            "parallelism": 1,
            "batch_size": 1,
            "worker_timeout_seconds": 86400,
            **runtime_extra,
        },
        "workflow": {"order": ["score", "summary"]},
        "tasks": {
            "noul": {**disabled, "datasets": []},
            "choice": {**disabled, "datasets": []},
            "score": score_task,
            "summary": {"enabled": True, "weights": {"score": 1.0}},
        },
        "models": models,
        "datasets": datasets,
    }
    output_path = ROOT / f"configs/eval/{name}.yaml"
    output_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    print(f"wrote {output_path} ({len(models)} models, {len(datasets)} datasets)")
    return output_path


def main() -> None:
    build("helpsteer-series", LOCAL_MODELS, {})
    build(
        "helpsteer-jev",
        JEV_MODELS,
        {"devices": [f"api:{i}" for i in range(4)], "parallelism": 4},
    )


if __name__ == "__main__":
    main()
