from __future__ import annotations

import json
import time
from typing import Any

from openjev_ja.common.types import ScoreResult

PRIMITIVES = ("noul", "choice", "score")
INSTRUCTIONS = "正しい答えを一つ選んでください。"

# variant name -> (Hub repo, revision). The v0 family also ships 17M and 68M sizes with the
# same custom inference class and request schema; add them here (with a pinned revision) if
# they are evaluated too.
RELEASES: dict[str, tuple[str, str]] = {
    "400m": ("hotchpotch/bekko-system-one-v0-400m", "4aeb85b9d4042d75d8b8adf6ff7ba9e4629510ba"),
}


def build_decision(primitive: str, question: str, options: list[str]) -> tuple[str, dict[str, Any]]:
    """(state text, Bekko decision dict) — the same split as JevScorer/DeciderScorer/LevScorer/
    JeffScorer: choice and score put the item text in the state and a fixed instruction on the
    decision; noul has an empty state and the yes/no question as the instruction.

    Score criteria carry a numeric `value` (their index), matching this project's convention
    that a Score primitive's gold/predicted index doubles as its ordinal level.
    """
    if primitive == "noul":
        return "", {
            "id": "answer",
            "kind": "judgment",
            "type": "noul",
            "instructions_json": json.dumps(question, ensure_ascii=False),
            "system_prompt": "",
            "criteria": [
                {
                    "id": "false",
                    "description_json": json.dumps("No.", ensure_ascii=False),
                    "value": None,
                },
                {
                    "id": "true",
                    "description_json": json.dumps("Yes.", ensure_ascii=False),
                    "value": None,
                },
            ],
            "documents": [],
            "scoring": None,
        }
    if primitive == "score":
        criteria = [
            {
                "id": str(index),
                "description_json": json.dumps(text, ensure_ascii=False),
                "value": index,
            }
            for index, text in enumerate(options)
        ]
        return question, {
            "id": "answer",
            "kind": "judgment",
            "type": "score",
            "instructions_json": json.dumps(INSTRUCTIONS, ensure_ascii=False),
            "system_prompt": "",
            "criteria": criteria,
            "documents": [],
            "scoring": None,
        }
    if primitive == "choice":
        criteria = [
            {
                "id": f"option_{index}",
                "description_json": json.dumps(text, ensure_ascii=False),
                "value": None,
            }
            for index, text in enumerate(options)
        ]
        return question, {
            "id": "answer",
            "kind": "judgment",
            "type": "choice",
            "instructions_json": json.dumps(INSTRUCTIONS, ensure_ascii=False),
            "system_prompt": "",
            "criteria": criteria,
            "documents": [],
            "scoring": None,
        }
    raise ValueError(f"unsupported primitive: {primitive}")


def parse_answer(
    primitive: str, answer: dict[str, Any], count: int
) -> tuple[list[float], list[float], int]:
    """(scores, probabilities, predicted_index) from one `answer` entry of `model.predict()`."""
    if primitive == "noul":
        p_true = float(answer["probability_yes"])
        return [0.0, p_true], [1.0 - p_true, p_true], 1 if p_true >= 0.5 else 0
    if primitive == "score":
        probabilities = [float(answer["probabilities"][str(index)]) for index in range(count)]
        return probabilities, probabilities, max(range(count), key=probabilities.__getitem__)
    keys = [f"option_{index}" for index in range(count)]
    probabilities = [float(answer["probabilities"][key]) for key in keys]
    return probabilities, probabilities, keys.index(answer["selected_id"])


class BekkoSystemOneScorer:
    """hotchpotch/bekko-system-one-v0: a compact ModernBERT/Ettin-reranker encoder fine-tuned
    for typed Choice/Noul/Score decisions — the same category as Jev, evaluating every
    candidate in parallel from a shared-prefix encoding rather than generating text. The
    publisher itself flags "v0" as limited generalization (strong on in-distribution task
    families, weaker on held-out ones); see the model card before reading too much into a
    single checkpoint's numbers.

    Loaded through `transformers.dynamic_module_utils.get_class_from_dynamic_module`, the
    model's own recommended loading path (`trust_remote_code`), so the published
    `BekkoSentenceTransformer.predict()` behavior is exactly what runs here.
    """

    name = "bekko-system-one"

    def __init__(
        self,
        model_name: str = "400m",
        *,
        primitive: str,
        device: str = "cuda",
        model_id: str | None = None,
    ) -> None:
        if primitive not in PRIMITIVES:
            raise ValueError(f"unsupported primitive: {primitive}")
        if model_name not in RELEASES:
            available = sorted(RELEASES)
            raise ValueError(
                f"unknown bekko-system-one variant {model_name!r}; choose one of {available}"
            )
        try:
            from transformers.dynamic_module_utils import get_class_from_dynamic_module
        except ImportError as exc:
            raise RuntimeError(
                "Install the bekko extra: pip install -e '.[bekko]'"
            ) from exc
        self.model_name = model_name
        self.repo_id, self.revision = RELEASES[model_name]
        self.primitive = primitive
        self.device = device
        self.model_id = model_id or f"bekko-system-one-{model_name}"
        cls = get_class_from_dynamic_module(
            "inference_v0.BekkoSentenceTransformer", self.repo_id, revision=self.revision
        )
        self.model = cls(
            self.repo_id, revision=self.revision, trust_remote_code=True, device=device
        )

    def score(self, question: str, options: list[str]) -> ScoreResult:
        import torch

        state, decision = build_decision(self.primitive, question, options)
        request = {"state_json": json.dumps(state, ensure_ascii=False), "decisions": [decision]}
        is_cuda = self.device.startswith("cuda")
        if is_cuda:
            torch.cuda.synchronize()
        started = time.perf_counter()
        result = self.model.predict(request, show_progress_bar=False)
        if is_cuda:
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000
        answer = result["answer"]
        scores, probabilities, predicted = parse_answer(self.primitive, answer, len(options))
        return ScoreResult(
            scores=scores,
            probabilities=probabilities,
            predicted_index=predicted,
            latency_ms=latency_ms,
            metadata={"primitive": self.primitive, "raw_answer": answer},
        )

    def metadata(self) -> dict[str, object]:
        return {
            "scorer": self.name,
            "model": self.model_id,
            "release": f"{self.repo_id}@{self.revision}",
            "base_model": "cross-encoder/ettin-reranker-400m-v1",
            "primitive": self.primitive,
            "device": self.device,
            "generation": False,
            "generalization_note": (
                "publisher-flagged v0: limited generalization beyond training task families"
            ),
        }
