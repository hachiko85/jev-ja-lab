"""Few-shot example sources.

Same representative per-primitive train dataset every from-scratch method
in this project trains its head on (see
methods.embedding.train.DEFAULT_SOURCES) — reused here as in-context
examples instead of training data, so no new dataset dependency is added.
"""

from __future__ import annotations

from typing import Any

from openjev_ja.eval.local_data import load_local_benchmark

FewShotExample = tuple[str, list[str], int]

DEFAULT_SOURCES: dict[str, dict[str, Any]] = {
    "choice": {
        "name": "jcommonsenseqa",
        "source": {
            "format": "arrow",
            "path": (
                "sbintuitions___j_commonsense_qa/default/0.0.0/*/"
                "j_commonsense_qa-train.arrow"
            ),
        },
    },
    "noul": {
        "name": "jnli_entailment_train",
        "source": {
            "format": "jsonl",
            "path": "JGLUE-v1.1.0/datasets/jnli-v1.1/train-v1.1.json",
            "adapter": "jnli_noul",
            "target": "entailment",
        },
    },
    "score": {
        "name": "wrime_joy_train",
        "source": {
            "format": "tsv",
            "path": "wrime-official/wrime-ver2.tsv",
            "adapter": "score",
            "question_field": "Sentence",
            "gold_field": "Avg. Readers_Joy",
            "criteria": ["喜びなし", "弱い喜び", "中程度の喜び", "強い喜び"],
            "prompt": "次の文章から読み手が感じる喜びの強度を評価してください。",
            "where": {"Train/Dev/Test": "train"},
        },
    },
}


def load_few_shot_examples(
    primitive: str, datasets_root: str, count: int = 2, *, seed: int = 42
) -> list[FewShotExample]:
    if primitive not in DEFAULT_SOURCES:
        raise ValueError(f"unsupported primitive: {primitive}")
    if count <= 0:
        return []
    default = DEFAULT_SOURCES[primitive]
    items = load_local_benchmark(
        default["name"], default["source"], datasets_root=datasets_root, seed=seed, limit=count
    )
    return [(item.question, item.options, item.gold_index) for item in items]
