from __future__ import annotations

import time
from typing import Any

from openjev_ja.common.types import ScoreResult

PRIMITIVES = ("choice", "score", "noul")
_INSTRUCTIONS = "正しい答えを一つ選んでください。"


class LayaScorer:
    """Wraps `laya.Router` — a pretrained System-1 decision engine, not a
    method this project trains. `primitive` picks which of laya's native
    question types (`choice` / `score` / `noul`) a `BenchmarkItem`'s
    (question, options, gold_index) triple is translated into; laya's own
    response shape already matches the Jev decision API's.
    """

    name = "laya"

    def __init__(
        self,
        *,
        primitive: str,
        checkpoint: str = "multilingual",
        device: str | None = None,
    ) -> None:
        if primitive not in PRIMITIVES:
            raise ValueError(f"unsupported primitive: {primitive}")
        try:
            from laya import Router
        except ImportError as exc:
            raise RuntimeError("Install the laya extra: pip install -e '.[laya]'") from exc
        kwargs: dict[str, Any] = {"preload": True}
        if device:
            kwargs["device"] = device
        self.router = Router(**kwargs)
        self.primitive = primitive
        self.checkpoint = checkpoint

    def score(self, question: str, options: list[str]) -> ScoreResult:
        started = time.perf_counter()
        if self.primitive == "noul":
            state = ""
            questions = {"answer": {"type": "noul", "instructions": question}}
        elif self.primitive == "score":
            state = question
            questions = {
                "answer": {"type": "score", "instructions": _INSTRUCTIONS, "criteria": options}
            }
        else:
            state = question
            keys = [f"option_{index}" for index in range(len(options))]
            questions = {
                "answer": {
                    "type": "choice",
                    "instructions": _INSTRUCTIONS,
                    "criteria": dict(zip(keys, options, strict=True)),
                }
            }
        result = self.router.predict(state, questions, model=self.checkpoint)
        answer = result["answers"]["answer"]
        latency_ms = (time.perf_counter() - started) * 1000

        if self.primitive == "noul":
            p_true = float(answer["noul"])
            scores = [0.0, p_true]
            probabilities = [1.0 - p_true, p_true]
            predicted_index = 1 if p_true >= 0.5 else 0
        elif self.primitive == "score":
            probabilities = [float(answer["probabilities"][str(i)]) for i in range(len(options))]
            scores = probabilities
            predicted_index = max(range(len(options)), key=probabilities.__getitem__)
        else:
            keys = [f"option_{index}" for index in range(len(options))]
            probabilities = [float(answer["probabilities"][key]) for key in keys]
            scores = probabilities
            predicted_index = keys.index(answer["choice"])

        return ScoreResult(
            scores=scores,
            probabilities=probabilities,
            predicted_index=predicted_index,
            latency_ms=latency_ms,
            metadata={
                "primitive": self.primitive,
                "confidence": answer.get("confidence"),
                "routing": result.get("routing"),
            },
        )

    def metadata(self) -> dict[str, object]:
        return {
            "scorer": self.name,
            "model": f"laya-{self.checkpoint}",
            "primitive": self.primitive,
            "checkpoint": self.checkpoint,
            "generation": False,
            "frozen_encoder": True,
            "pretrained_decision_head": True,
        }
