from __future__ import annotations

import json
import multiprocessing as mp
import os
import queue
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from openjev_ja.common import DatasetUnavailableError
from openjev_ja.eval.local_data import load_local_benchmark
from openjev_ja.eval.runner import run_evaluation
from openjev_ja.methods.bert_masked_lm import MaskedLMScorer
from openjev_ja.methods.decider import DeciderScorer
from openjev_ja.methods.embedding import EmbeddingScorer
from openjev_ja.methods.hopper import HopperScorer
from openjev_ja.methods.hopper.scorer import BASE_REVISION as HOPPER_BASE_REVISION
from openjev_ja.methods.jevlike import JevlikeScorer
from openjev_ja.methods.laya import LayaScorer
from openjev_ja.methods.laya_bert import LayaBertScorer
from openjev_ja.methods.next_token_logit import NextTokenLogitScorer
from openjev_ja.methods.nli_cross_encoder import NLICrossEncoderScorer
from openjev_ja.methods.semif_logit import SemifLogitScorer
from openjev_ja.methods.typesafe_jev import JevScorer

_FETCH_SKIP_EXCEPTIONS = (FileNotFoundError, DatasetUnavailableError, OSError, httpx.HTTPError)


class OrchestrationError(RuntimeError):
    """Raised when a model or required benchmark fails."""


def _model_reference(model: dict[str, Any], runtime: dict[str, Any]) -> str:
    repo_id = model.get("repo_id")
    if repo_id:
        return str(repo_id)
    if not model.get("path"):
        raise OrchestrationError("model requires path or repo_id")
    return str(Path(runtime["models_root"]) / str(model["path"]))


def resolve_scorer_class(model_name: str, revision: str | None = None) -> type:
    """Pick NextTokenLogitScorer or MaskedLMScorer from the model's own config.

    Kept only as the fallback behind an explicit `scorer: auto` (or omitted
    `scorer`) in a model config; per JEV_JA_LAB_REFACTOR_GUIDE_v2 section 29,
    new configs should prefer naming the method explicitly (`scorer:
    qwen-direct` / `scorer: masked-lm` / ...), since the same model can in
    principle support more than one method. Any encoder architecture ending
    in "ForMaskedLM" (BERT, RoBERTa, ModernBERT, ELECTRA, DeBERTa, ALBERT,
    ...) gets MaskedLMScorer; every causal or image-text-to-text
    architecture gets NextTokenLogitScorer.
    """
    try:
        from transformers import AutoConfig
    except ImportError as exc:
        raise RuntimeError("Install evaluation dependencies: pip install -e '.[eval]'") from exc
    kwargs: dict[str, str] = {"revision": revision} if revision else {}
    config = AutoConfig.from_pretrained(model_name, **kwargs)
    architectures = getattr(config, "architectures", None) or []
    if any(str(architecture).endswith("ForMaskedLM") for architecture in architectures):
        return MaskedLMScorer
    return NextTokenLogitScorer


def _create_scorer(model: dict[str, Any], runtime: dict[str, Any], device: str) -> Any:
    scorer_name = model.get("scorer")
    if scorer_name == "jev":
        return JevScorer(
            model=str(model.get("model") or model.get("model_id") or "jev-latest"),
            timeout=float(model.get("timeout", 30.0)),
            max_retries=int(model.get("max_retries", 2)),
        )
    if scorer_name == "embedding":
        if not model.get("primitive") or not model.get("head_path"):
            raise OrchestrationError(
                f"model {model.get('id')!r}: scorer 'embedding' requires 'primitive' and "
                "'head_path' (a head trained with jev-ja-lab-embedding-train)"
            )
        return EmbeddingScorer(
            _model_reference(model, runtime),
            primitive=str(model["primitive"]),
            head_path=str(model["head_path"]),
            device=device,
            dtype=str(model.get("dtype", "float32")),
            revision=model.get("revision"),
            model_id=str(model.get("model_id") or model.get("repo_id") or model.get("path")),
            metadata_revision=model.get("metadata_revision"),
        )
    if scorer_name == "laya":
        if not model.get("primitive"):
            raise OrchestrationError(
                f"model {model.get('id')!r}: scorer 'laya' requires 'primitive'"
            )
        return LayaScorer(
            primitive=str(model["primitive"]),
            checkpoint=str(model.get("checkpoint", "multilingual")),
            device=None if device == "auto" else device,
        )
    if scorer_name == "jevlike":
        if not model.get("checkpoint_path"):
            raise OrchestrationError(
                f"model {model.get('id')!r}: scorer 'jevlike' requires 'checkpoint_path' "
                "(a checkpoint trained with jev-ja-lab-jevlike-train)"
            )
        return JevlikeScorer(str(model["checkpoint_path"]), device=device)
    if scorer_name == "laya-bert":
        if not model.get("primitive") or not model.get("head_path"):
            raise OrchestrationError(
                f"model {model.get('id')!r}: scorer 'laya-bert' requires 'primitive' and "
                "'head_path' (a head trained with jev-ja-lab-laya-bert-train)"
            )
        return LayaBertScorer(
            _model_reference(model, runtime),
            primitive=str(model["primitive"]),
            head_path=str(model["head_path"]),
            device=device,
            model_id=str(model.get("model_id") or model.get("repo_id") or model.get("path")),
        )
    if scorer_name == "nli-cross-encoder":
        return NLICrossEncoderScorer(
            _model_reference(model, runtime),
            subfolder=model.get("subfolder"),
            trust_remote_code=bool(model.get("trust_remote_code", False)),
            template=str(model.get("template", "ja")),
            device=device,
            dtype=str(model.get("dtype", "bfloat16")),
            max_length=int(model.get("max_length", 1024)),
        )
    if scorer_name == "hopper":
        if not model.get("primitive"):
            raise OrchestrationError(
                f"model {model.get('id')!r}: scorer 'hopper' requires 'primitive'"
            )
        return HopperScorer(
            _model_reference(model, runtime),
            primitive=str(model["primitive"]),
            adapter=str(model.get("adapter", "HopitAI/hopper")),
            device=device,
            dtype=str(model.get("dtype", "bfloat16")),
            revision=model.get("revision", HOPPER_BASE_REVISION),
            adapter_revision=model.get("adapter_revision"),
            model_id=str(model.get("model_id") or model.get("adapter") or "HopitAI/hopper"),
            calibrate=bool(model.get("calibrate", True)),
        )
    if scorer_name == "decider":
        if not model.get("primitive"):
            raise OrchestrationError(
                f"model {model.get('id')!r}: scorer 'decider' requires 'primitive'"
            )
        return DeciderScorer(
            _model_reference(model, runtime),
            primitive=str(model["primitive"]),
            revision=str(model.get("revision", "v2")),
            device=device,
            dtype=str(model.get("dtype", "bfloat16")),
            use_graphs=bool(model.get("use_graphs", False)),
            model_id=model.get("model_id"),
        )
    if scorer_name == "semif-logit":
        few_shot_count = int(model.get("few_shot_count", 0))
        few_shot_kwargs: dict[str, Any] = {}
        if few_shot_count:
            if not model.get("primitive"):
                raise OrchestrationError(
                    f"model {model.get('id')!r}: 'few_shot_count' requires 'primitive'"
                )
            few_shot_kwargs = {
                "primitive": str(model["primitive"]),
                "datasets_root": str(runtime["datasets_root"]),
                "few_shot_count": few_shot_count,
            }
        return SemifLogitScorer(
            _model_reference(model, runtime),
            device=device,
            dtype=str(model.get("dtype", "bfloat16")),
            revision=model.get("revision"),
            model_id=str(model.get("model_id") or model.get("repo_id") or model.get("path")),
            metadata_revision=model.get("metadata_revision"),
            **few_shot_kwargs,
        )
    model_path = _model_reference(model, runtime)
    if scorer_name in (None, "auto"):
        # No explicit scorer: pick NextTokenLogitScorer (causal / image-text-to-text)
        # or MaskedLMScorer (BERT-family encoders) from the model's own config,
        # so a repo_id alone is enough for either kind of model.
        scorer_cls = resolve_scorer_class(str(model_path), model.get("revision"))
    elif scorer_name == "qwen-direct":
        scorer_cls = NextTokenLogitScorer
    elif scorer_name == "masked-lm":
        scorer_cls = MaskedLMScorer
    else:
        raise OrchestrationError(f"unsupported scorer: {scorer_name}")
    return scorer_cls(
        str(model_path),
        device=device,
        dtype=str(model.get("dtype", "bfloat16")),
        revision=model.get("revision"),
        model_id=str(model.get("model_id") or model.get("repo_id") or model.get("path")),
        metadata_revision=model.get("metadata_revision"),
    )


def load_orchestration_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        message = "Install orchestration dependencies: pip install -e '.[orchestrate]'"
        raise RuntimeError(message) from exc
    config_path = Path(path).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise OrchestrationError("orchestration config version must be 1")
    return payload


def _assign_datasets(datasets: list[dict[str, Any]], workers: int) -> list[list[dict[str, Any]]]:
    assignments: list[list[dict[str, Any]]] = [[] for _ in range(workers)]
    loads = [0] * workers
    ordered = sorted(
        datasets, key=lambda item: int(item.get("expected_items", 0)), reverse=True
    )
    for dataset in ordered:
        worker = loads.index(min(loads))
        assignments[worker].append(dataset)
        loads[worker] += int(dataset.get("expected_items", 0))
    return assignments


def _worker(
    model: dict[str, Any],
    datasets: list[dict[str, Any]],
    runtime: dict[str, Any],
    device: str,
    output_root: str,
    result_queue: Any,
) -> None:
    os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton")
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/torchinductor")
    try:
        scorer = _create_scorer(model, runtime, device)
        completed: list[dict[str, Any]] = []
        skipped: list[dict[str, str]] = []
        for dataset in datasets:
            name = str(dataset["id"])
            source = dict(dataset["source"])
            if dataset.get("revision") is not None:
                source.setdefault("revision", dataset["revision"])
            try:
                items = load_local_benchmark(
                    name,
                    source,
                    datasets_root=runtime["datasets_root"],
                    seed=int(runtime.get("seed", 42)),
                    limit=dataset.get("limit"),
                )
            except _FETCH_SKIP_EXCEPTIONS as exc:
                skipped.append({"dataset": name, "reason": f"fetch_failed: {exc}"})
                continue
            run_dir = run_evaluation(
                items,
                scorer,
                dataset_name=name,
                output_root=output_root,
                run_id=name,
                seed=int(runtime.get("seed", 42)),
                dataset_revision=dataset.get("revision"),
                batch_size=int(model.get("batch_size", runtime.get("batch_size", 1))),
                task_type=str(dataset.get("task", "choice")),
            )
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            completed.append({"dataset": name, **summary})
        result_queue.put({"device": device, "completed": completed, "skipped": skipped})
    except Exception as exc:
        result_queue.put(
            {
                "device": device,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )


def _available_datasets(
    datasets: list[dict[str, Any]], datasets_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    available: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for dataset in datasets:
        source = dataset["source"]
        if source.get("repo_id") or source.get("adapter_dataset") or source.get("hub_source"):
            available.append(dataset)
            continue
        pattern = str(source["path"])
        if list(datasets_root.glob(pattern)):
            available.append(dataset)
        else:
            skipped.append({"dataset": str(dataset["id"]), "reason": "source_not_found"})
    return available, skipped


def _write_radar_config(
    run_root: Path,
    models: list[dict[str, Any]],
    datasets: list[dict[str, Any]],
    radar_name: str = "radar",
    *,
    title: str = "Japanese benchmark comparison",
    subtitle: str = "Direct option-logit accuracy",
    metric: str = "accuracy",
    theme: str | None = None,
    language: str | None = None,
) -> Path:
    import yaml

    if Path(radar_name).name != radar_name:
        raise OrchestrationError("radar_name must be a file name without a directory")

    completed_models = models
    common = [
        dataset
        for dataset in datasets
        if all(
            (run_root / model["id"] / dataset["id"] / "summary.json").is_file()
            for model in completed_models
        )
    ]
    figure_size = max(11, min(20, 8 + len(common) * 0.45))
    note = (
        "テキスト生成なし。項目ごとに決定論的な単回forward passで比較。"
        if str(language).lower() == "ja"
        else "No text generation; one deterministic forward pass per item."
    )
    config: dict[str, Any] = {
        "version": 1,
        "title": title,
        "subtitle": subtitle,
        "note": note,
        "output": f"{radar_name}.png",
        "input_scale": "fraction",
        "dpi": 180,
        "figure_size": [figure_size, figure_size],
        "legend_columns": min(3, len(completed_models)),
        **({"theme": theme} if theme else {}),
        **({"language": language} if language else {}),
        "axes": [
            {
                "id": dataset["id"],
                "label": (
                    f"{dataset.get('label', dataset['id'])}\n({dataset['feature']})"
                    if dataset.get("feature")
                    else dataset.get("label", dataset["id"])
                ),
            }
            for dataset in common
        ],
        "series": [
            {
                "name": model.get("label", model["id"]),
                "color": model.get("color"),
                "points": {
                    dataset["id"]: {
                        "path": f"{model['id']}/{dataset['id']}/summary.json",
                        "metric": metric,
                    }
                    for dataset in common
                },
            }
            for model in completed_models
        ],
    }
    path = run_root / f"{radar_name}.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def _write_model_summary(
    model_root: Path, rows: list[dict[str, Any]], *, name: str = "summary.json"
) -> None:
    (model_root / name).write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def run_orchestration(
    config_path: str | Path,
    *,
    task_type: str | None = None,
    phase: str = "production",
) -> Path:
    config = load_orchestration_config(config_path)
    runtime = dict(config["runtime"])
    if phase not in {"smoke", "production"}:
        raise OrchestrationError("phase must be smoke or production")
    task_config = dict(config.get("tasks", {}).get(task_type, {})) if task_type else {}
    chart_config = dict(task_config.get("chart", {}))
    render_chart = bool(chart_config.get("enabled", True))
    models = [dict(item) for item in config["models"] if item.get("enabled", True)]
    selected_model_ids = task_config.get("model_ids", runtime.get("selected_model_ids"))
    if selected_model_ids is not None:
        selected = {str(model_id) for model_id in selected_model_ids}
        models = [model for model in models if str(model["id"]) in selected]
        missing_models = selected - {str(model["id"]) for model in models}
        if missing_models:
            raise OrchestrationError(
                f"selected models not found: {', '.join(sorted(missing_models))}"
            )
    radar_name = str(
        chart_config.get(
            "name", task_config.get("radar_name", runtime.get("radar_name", "radar"))
        )
    )
    radar_model_ids = task_config.get("radar_model_ids", runtime.get("radar_model_ids"))
    if radar_model_ids is not None:
        radar_selected = {str(model_id) for model_id in radar_model_ids}
        missing_radar_models = radar_selected - {str(model["id"]) for model in models}
        if missing_radar_models:
            raise OrchestrationError(
                f"radar models not selected for evaluation: "
                f"{', '.join(sorted(missing_radar_models))}"
            )
    else:
        radar_selected = None
    datasets = [dict(item) for item in config["datasets"] if item.get("enabled", True)]
    if task_type:
        configured_ids = [str(value) for value in task_config.get("datasets", [])]
        known_ids = {str(item["id"]) for item in datasets}
        missing_datasets = set(configured_ids) - known_ids
        if missing_datasets:
            raise OrchestrationError(
                f"task datasets not found: {', '.join(sorted(missing_datasets))}"
            )
        datasets = [
            item
            for item in datasets
            if str(item.get("task", "choice")) == task_type
            and (not configured_ids or str(item["id"]) in configured_ids)
        ]
        if phase == "smoke":
            smoke_limit = int(task_config.get("smoke_limit", 5))
            for dataset in datasets:
                configured_limit = dataset.get("limit")
                dataset["limit"] = (
                    min(int(configured_limit), smoke_limit)
                    if configured_limit is not None
                    else smoke_limit
                )
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_name = str(runtime.get("run_name", timestamp))
    if phase == "smoke":
        run_name = str(runtime.get("smoke_run_name", f"{run_name}-smoke"))
    run_root = Path(runtime["output_root"]) / run_name
    resume = bool(runtime.get("resume", False))
    run_root.mkdir(parents=True, exist_ok=resume)
    status_name = f"eval_{task_type}.json" if task_type else "eval.json"
    if task_type and not task_config.get("enabled", True):
        (run_root / status_name).write_text(
            json.dumps(
                {"task": task_type, "phase": phase, "status": "disabled"},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return run_root
    if not datasets:
        (run_root / status_name).write_text(
            json.dumps(
                {"task": task_type, "phase": phase, "status": "no_datasets"},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return run_root
    available, skipped = _available_datasets(datasets, Path(runtime["datasets_root"]))
    suffix = f".{task_type}.{phase}" if task_type else ""
    (run_root / f"run_config{suffix}.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (run_root / f"skipped{suffix}.json").write_text(
        json.dumps(skipped, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not available:
        raise OrchestrationError("no configured datasets are available")

    devices = [str(device) for device in runtime["devices"]]
    parallelism = int(task_config.get("parallelism", runtime.get("parallelism", len(devices))))
    worker_count = min(parallelism, len(devices), len(available))
    context = mp.get_context("spawn")
    completed_models: list[dict[str, Any]] = []
    for model in models:
        model_root = run_root / str(model["id"])
        model_root.mkdir(exist_ok=resume)
        pending = [
            dataset
            for dataset in available
            if not (model_root / dataset["id"] / "summary.json").is_file()
        ]
        if not pending:
            completed_models.append(model)
            chart_models = [
                item
                for item in completed_models
                if radar_selected is None or str(item["id"]) in radar_selected
            ]
            if chart_models and render_chart and len(available) >= 3:
                radar_config = _write_radar_config(
                    run_root,
                    chart_models,
                    available,
                    radar_name=radar_name,
                    title=str(chart_config.get("title", "Japanese benchmark comparison")),
                    subtitle=str(
                        chart_config.get("subtitle", "Direct option-logit accuracy")
                    ),
                    metric=str(chart_config.get("metric", "accuracy")),
                    theme=chart_config.get("theme"),
                    language=chart_config.get("language"),
                )
                from openjev_ja.visualize.radar import render_radar

                render_radar(radar_config)
            continue
        model_worker_count = min(worker_count, len(pending))
        assignments = _assign_datasets(pending, model_worker_count)
        result_queue = context.Queue()
        processes = [
            context.Process(
                target=_worker,
                args=(model, assignment, runtime, devices[index], str(model_root), result_queue),
            )
            for index, assignment in enumerate(assignments)
        ]
        for process in processes:
            process.start()
        messages = []
        for _ in processes:
            try:
                timeout = float(runtime.get("worker_timeout_seconds", 7200))
                messages.append(result_queue.get(timeout=timeout))
            except queue.Empty as exc:
                for process in processes:
                    process.terminate()
                raise OrchestrationError(f"worker timeout for model {model['id']}") from exc
        for process in processes:
            process.join(timeout=30)
        errors = [message for message in messages if "error" in message]
        if errors:
            (model_root / "errors.json").write_text(
                json.dumps(errors, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            raise OrchestrationError(f"model {model['id']} failed: {errors[0]['error']}")
        for message in messages:
            for entry in message.get("skipped", []):
                if entry["dataset"] not in {item["dataset"] for item in skipped}:
                    skipped.append(entry)
        (run_root / f"skipped{suffix}.json").write_text(
            json.dumps(skipped, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        rows = []
        for dataset in available:
            summary_path = model_root / dataset["id"] / "summary.json"
            if summary_path.is_file():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                rows.append({"dataset": dataset["id"], **summary})
        rows.sort(key=lambda row: row["dataset"])
        summary_name = f"summary.{task_type}.json" if task_type else "summary.json"
        _write_model_summary(model_root, rows, name=summary_name)
        completed_models.append(model)
        chart_models = [
            item
            for item in completed_models
            if radar_selected is None or str(item["id"]) in radar_selected
        ]
        if chart_models and render_chart and len(available) >= 3:
            radar_config = _write_radar_config(
                run_root,
                chart_models,
                available,
                radar_name=radar_name,
                title=str(chart_config.get("title", "Japanese benchmark comparison")),
                subtitle=str(chart_config.get("subtitle", "Direct option-logit accuracy")),
                metric=str(chart_config.get("metric", "accuracy")),
                theme=chart_config.get("theme"),
                language=chart_config.get("language"),
            )
            from openjev_ja.visualize.radar import render_radar

            render_radar(radar_config)
    if task_type:
        (run_root / status_name).write_text(
            json.dumps(
                {
                    "task": task_type,
                    "phase": phase,
                    "status": "partial" if skipped else "complete",
                    "models": [str(model["id"]) for model in completed_models],
                    "datasets": [str(dataset["id"]) for dataset in available],
                    "skipped": skipped,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return run_root
