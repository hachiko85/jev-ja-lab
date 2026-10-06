"""Generate configs/eval/hopper-series.yaml and configs/eval/decider-series.yaml.

Both are one model entry per primitive (each primitive builds its own scorer, like
semif-logit-fewshot / embedding / laya), evaluated on the same 30-axis profile as every
other method: tasks and datasets are copied from bert-series.yaml.

- hopper: HopitAI/hopper LoRA on Qwen/Qwen3.5-4B at the revision the adapter was built for.
  Adapter weights are research/demo-use only.
- decider: Mapika/decider-4b at Hub tag `v2` (decider-4b v2). Change `revision` to `main`
  for v2.1 (per-answer-type temperatures) or `v1`.
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
PRIMITIVES = ("noul", "choice", "score")

SERIES = {
    "hopper": {
        "run_name": "eval-hopper-series",
        "model_id": "hopper-qwen3.5-4b-lora",
        "label": "Hopper (LoRA on Qwen3.5-4B)",
        "color": "#8c564b",
        "entry": {
            "repo_id": "Qwen/Qwen3.5-4B",
            "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            "adapter": "HopitAI/hopper",
            "scorer": "hopper",
            "dtype": "bfloat16",
        },
    },
    "decider": {
        "run_name": "eval-decider-series",
        "model_id": "decider-4b-v2",
        "label": "decider-4b v2",
        "color": "#17becf",
        "entry": {
            "repo_id": "Mapika/decider-4b",
            "revision": "v2",
            "scorer": "decider",
            "dtype": "bfloat16",
        },
    },
    "jeff-qwen3.5-0.8b": {
        "run_name": "eval-jeff-qwen3.5-0.8b-series",
        "model_id": "jeff-qwen3.5-0.8b",
        "label": "Jeff (Qwen3.5-0.8B)",
        "color": "#aec7e8",
        "entry": {"scorer": "jeff", "model_name": "qwen3.5-0.8b", "dtype": "bfloat16"},
    },
    "jeff-qwen3.5-2b": {
        "run_name": "eval-jeff-qwen3.5-2b-series",
        "model_id": "jeff-qwen3.5-2b",
        "label": "Jeff (Qwen3.5-2B)",
        "color": "#ffbb78",
        "entry": {"scorer": "jeff", "model_name": "qwen3.5-2b", "dtype": "bfloat16"},
    },
    "jeff-gemma4-e2b": {
        "run_name": "eval-jeff-gemma4-e2b-series",
        "model_id": "jeff-gemma4-e2b",
        "label": "Jeff (Gemma4-E2B)",
        "color": "#98df8a",
        "entry": {"scorer": "jeff", "model_name": "gemma4-e2b", "dtype": "bfloat16"},
    },
    "bekko-system-one-400m": {
        "run_name": "eval-bekko-system-one-400m-series",
        "model_id": "bekko-system-one-400m",
        "label": "Bekko System One v0 (400M)",
        "color": "#9edae5",
        "entry": {"scorer": "bekko-system-one", "model_name": "400m", "dtype": "bfloat16"},
    },
    "lev": {
        "run_name": "eval-lev-series",
        "model_id": "lev-4b",
        "label": "Lev (LoRA on Qwen3.5-4B)",
        "color": "#7f7f7f",
        "entry": {
            "repo_id": "interfaze-ai/lev",
            "revision": "f8ef71157ec06a7d3b6435bc0756f9d735c33748",
            "scorer": "lev",
            "dtype": "bfloat16",
        },
    },
    "clm": {
        "run_name": "eval-clm-series",
        "model_id": "clm-v0.1-8b",
        "label": "CLM v0.1 (Qwen3-8B)",
        "color": "#e377c2",
        "entry": {
            "repo_id": "Qwen/Qwen3-8B",
            "revision": "b968826d9c46dd6066d109eabc6255188de91218",
            "scorer": "clm",
            "heads_repo": "Contrastive-LM/CLM-v0.1-8B",
            "heads_revision": "e939398d4556fcd9400c76fa8c5a513202f42b0a",
            "dtype": "bfloat16",
        },
    },
    "clef-flash": {
        "run_name": "eval-clef-flash-series",
        "model_id": "clef-flash",
        "label": "Clef-Flash (Qwen3.5-9B)",
        "color": "#f6821f",
        "entry": {"scorer": "clef", "model_name": "flash", "quantization": "8bit", "dtype": "bfloat16"},
    },
    "clef-flash-gguf-q4km": {
        "run_name": "eval-clef-flash-gguf-q4km-series",
        "model_id": "clef-flash-gguf-q4km",
        "label": "Clef-Flash GGUF Q4_K_M (llama.cpp)",
        "color": "#fbad41",
        "entry": {"scorer": "clef-gguf", "model_name": "q4_k_m"},
    },
    "decider-ja-main-baseline": {
        "run_name": "eval-decider-ja-main-baseline-series",
        "model_id": "decider-4b-main-baseline",
        "label": "decider-4b main (zero-shot, pre-QLoRA)",
        "color": "#c49c94",
        "entry": {
            # the exact commit `Mapika/decider-4b` (no revision pin) resolved to when
            # scripts/train_decider_ja_qlora.py trained from it -- the correct zero-shot
            # baseline for that run (not decider-4b-v2, a different tagged commit).
            "repo_id": "Mapika/decider-4b",
            "revision": "eb5fbdfc9448473ec25e399882912863afbdb70e",
            "scorer": "decider",
            "dtype": "bfloat16",
        },
    },
    "decider-ja-qlora": {
        "run_name": "eval-decider-ja-qlora-series",
        "model_id": "decider-4b-ja-qlora",
        "label": "decider-4b + QLoRA (datasets/for-decider-ja)",
        "color": "#d62728",
        "entry": {
            # scripts/merge_decider_ja_qlora.py's merged output, under models_root.
            "path": "decider-ja-qlora",
            "scorer": "decider",
            "dtype": "bfloat16",
        },
    },
    "decider-fewshot": {
        "run_name": "eval-decider-fewshot-series",
        "model_id": "decider-4b-v2-2shot",
        "label": "decider-4b v2 2-shot",
        "color": "#bcbd22",
        "entry": {
            "repo_id": "Mapika/decider-4b",
            "revision": "v2",
            "scorer": "decider",
            "few_shot_count": 2,
            "dtype": "bfloat16",
        },
    },
}


def build(name: str, spec: dict) -> Path:
    base = yaml.safe_load((ROOT / "configs/eval/bert-series.yaml").read_text(encoding="utf-8"))
    ids = {p: f"{spec['model_id']}-{p}" for p in PRIMITIVES}
    tasks = {}
    for primitive in PRIMITIVES:
        task = dict(base["tasks"][primitive])
        task.pop("parallelism", None)
        task["model_ids"] = [ids[primitive]]
        chart = dict(task.get("chart", {}))
        if chart:
            chart["name"] = f"radar-{name}-{primitive}"
            task["chart"] = chart
        tasks[primitive] = task
    summary_task = dict(base["tasks"]["summary"])
    summary_chart = dict(summary_task.get("chart", {}))
    if summary_chart:
        summary_chart["name"] = f"radar-{name}-summary"
        summary_task["chart"] = summary_chart
    tasks["summary"] = summary_task

    models = [
        {
            "id": ids[primitive],
            "label": f"{spec['label']} ({primitive})",
            **spec["entry"],
            "primitive": primitive,
            "batch_size": 1,
            "color": spec["color"],
        }
        for primitive in PRIMITIVES
    ]
    config = {
        "version": 1,
        "runtime": {
            "run_name": spec["run_name"],
            "smoke_run_name": f"{spec['run_name']}-smoke",
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
    output_path = ROOT / f"configs/eval/{name}-series.yaml"
    output_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    print(f"wrote {output_path} ({len(models)} model entries, {len(base['datasets'])} datasets)")
    return output_path


def main() -> None:
    for name, spec in SERIES.items():
        build(name, spec)


if __name__ == "__main__":
    main()
