"""YAML-configured training workflow: one architecture + corpus per model entry, k-fold
cross-validated, the training-side counterpart to `openjev_ja.eval.orchestrate`.

    jev-ja-lab-train-workflow --config configs/train/eikos-laya-bert.yaml

`plan_training` only reads and validates the config (no torch/laya import, so it is unit-testable
without either); `run_training_workflow` executes the plan and writes results.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Keys of a model entry that name a known TrainingJob field rather than an architecture
# hyperparameter; anything else on the entry is passed through as `build_kwargs`.
_JOB_FIELDS = {
    "id", "architecture", "repo_id", "revision", "primitive", "corpus", "folds", "epochs",
    "lr", "seed", "device", "log_every", "limit",
}


@dataclass(frozen=True)
class TrainingJob:
    model_id: str
    architecture: str
    model_name: str
    revision: str | None
    primitive: str
    corpus_format: str
    corpus_paths: list[str]
    lang: str | None
    limit: int | None
    folds: int
    epochs: int
    lr: float
    seed: int
    device: str
    log_every: int
    build_kwargs: dict[str, Any] = field(default_factory=dict)


class TrainingConfigError(RuntimeError):
    pass


def load_training_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        message = "Install orchestration dependencies: pip install -e '.[orchestrate]'"
        raise RuntimeError(message) from exc
    config_path = Path(path).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise TrainingConfigError("training config version must be 1")
    return payload


def plan_training(config: dict[str, Any]) -> list[TrainingJob]:
    runtime = config.get("runtime", {})
    default_corpus = config.get("corpus", {})
    models = config.get("models")
    if not models:
        raise TrainingConfigError("config has no 'models' entries")

    jobs = []
    for entry in models:
        if "id" not in entry or "architecture" not in entry or "primitive" not in entry:
            raise TrainingConfigError(f"model entry missing id/architecture/primitive: {entry}")
        corpus = {**default_corpus, **entry.get("corpus", {})}
        paths = corpus.get("paths")
        if not paths:
            raise TrainingConfigError(f"model {entry['id']!r}: no corpus paths configured")
        jobs.append(
            TrainingJob(
                model_id=str(entry["id"]),
                architecture=str(entry["architecture"]),
                model_name=str(entry.get("repo_id", entry["id"])),
                revision=entry.get("revision"),
                primitive=str(entry["primitive"]),
                corpus_format=str(corpus.get("format", "eikos")),
                corpus_paths=[str(p) for p in paths],
                lang=corpus.get("lang", "Japanese"),
                limit=entry.get("limit", corpus.get("limit")),
                folds=int(entry.get("folds", runtime.get("folds", 5))),
                epochs=int(entry.get("epochs", runtime.get("epochs", 3))),
                lr=float(entry.get("lr", runtime.get("lr", 1e-3))),
                seed=int(entry.get("seed", runtime.get("seed", 42))),
                device=str(entry.get("device", runtime.get("device", "cpu"))),
                log_every=int(entry.get("log_every", runtime.get("log_every", 0))),
                build_kwargs={k: v for k, v in entry.items() if k not in _JOB_FIELDS},
            )
        )
    return jobs


def run_training_workflow(config_path: str | Path) -> dict[str, Any]:
    from openjev_ja.train.architectures import get_architecture
    from openjev_ja.train.corpus import load_corpus
    from openjev_ja.train.runner import cross_validate

    config = load_training_config(config_path)
    runtime = config.get("runtime", {})
    run_name = str(runtime.get("run_name", "train"))
    output_root = Path(runtime.get("output_root", "results")) / run_name
    output_root.mkdir(parents=True, exist_ok=True)

    jobs = plan_training(config)
    results = []
    for job in jobs:
        print(f"=== {job.model_id} ({job.architecture}, {job.primitive}) ===", flush=True)
        items = load_corpus(
            job.corpus_paths,
            primitive=job.primitive,
            format=job.corpus_format,
            lang=job.lang,
            limit=job.limit,
        )
        if not items:
            raise TrainingConfigError(
                f"model {job.model_id!r}: no items for primitive {job.primitive!r} in "
                f"{job.corpus_paths} (lang={job.lang!r})"
            )
        architecture = get_architecture(job.architecture)
        build_kwargs = (
            {**job.build_kwargs, "revision": job.revision} if job.revision else job.build_kwargs
        )
        started = time.perf_counter()
        result = cross_validate(
            architecture,
            items,
            model_name=job.model_name,
            primitive=job.primitive,
            folds=job.folds,
            epochs=job.epochs,
            lr=job.lr,
            seed=job.seed,
            device=job.device,
            log_every=job.log_every,
            build_kwargs=build_kwargs,
        )
        result["elapsed_seconds"] = time.perf_counter() - started

        model_dir = output_root / job.model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        for fold in result["fold_results"]:
            import torch

            torch.save(fold.pop("state_dict"), model_dir / f"fold-{fold['fold']}.pt")
        (model_dir / "summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"  summary: {json.dumps(result['summary'], ensure_ascii=False)}", flush=True)
        results.append(result)

    overview = [
        {"model": r["model"], "primitive": r["primitive"], "summary": r["summary"]}
        for r in results
    ]
    (output_root / "run_summary.json").write_text(
        json.dumps(overview, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {"run_name": run_name, "output_root": str(output_root), "results": results}
