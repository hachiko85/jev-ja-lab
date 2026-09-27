from __future__ import annotations

import time
from typing import Any

from openjev_ja.common.types import ScoreResult

PRIMITIVES = ("noul", "choice", "score")
# Same wire-level instruction JevScorer/DeciderScorer send for choice and score items.
INSTRUCTIONS = "正しい答えを一つ選んでください。"
REPO = "interfaze-ai/lev"
REVISION = "f8ef71157ec06a7d3b6435bc0756f9d735c33748"


def build_questions(primitive: str, question: str, options: list[str]) -> tuple[Any, dict]:
    """(state, questions) in TypeSafe's /v1/systemone wire format — the exact shape
    JevScorer sends, which `DecisionEngine.system_one` accepts unchanged."""
    if primitive == "noul":
        return "", {"answer": {"type": "noul", "instructions": question}}
    if primitive == "score":
        return question, {
            "answer": {"type": "score", "instructions": INSTRUCTIONS, "criteria": options}
        }
    if primitive == "choice":
        keys = [f"option_{index}" for index in range(len(options))]
        return question, {
            "answer": {
                "type": "choice",
                "instructions": INSTRUCTIONS,
                "criteria": dict(zip(keys, options, strict=True)),
            }
        }
    raise ValueError(f"unsupported primitive: {primitive}")


def parse_answer(primitive: str, answer: Any, count: int) -> tuple[list[float], list[float], int]:
    """(scores, probabilities, predicted_index) from one lev answer object."""
    if primitive == "noul":
        p_true = float(answer.noul)
        return [0.0, p_true], [1.0 - p_true, p_true], 1 if p_true >= 0.5 else 0
    if primitive == "score":
        probabilities = [float(answer.probabilities[index]) for index in range(count)]
        return probabilities, probabilities, max(range(count), key=probabilities.__getitem__)
    keys = [f"option_{index}" for index in range(count)]
    probabilities = [float(answer.probabilities[key]) for key in keys]
    return probabilities, probabilities, keys.index(answer.choice)


class LevScorer:
    """interfaze-ai/lev (Qwen3.5-4B + LoRA, Apache-2.0). `lev.load` reads the release manifest
    (base model, chat prompt style, label-token readout, calibration and candidate-path head)
    from the downloaded checkpoint, so nothing is configured here beyond which revision to
    fetch. The model is English-trained; Japanese input relies on the multilingual Qwen3.5 base.
    """

    name = "lev"

    def __init__(
        self,
        model_name: str = REPO,
        *,
        primitive: str,
        revision: str | None = REVISION,
        device: str = "cuda",
        dtype: str = "bfloat16",
        model_id: str | None = None,
    ) -> None:
        if primitive not in PRIMITIVES:
            raise ValueError(f"unsupported primitive: {primitive}")
        try:
            import lev
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError("Install the lev extra: pip install -e '.[lev]'") from exc
        self.model_name = model_name
        self.revision = revision
        self.primitive = primitive
        self.device = device
        self.dtype = dtype
        self.model_id = model_id or f"{model_name}@{revision}"
        folder = snapshot_download(model_name, revision=revision)
        self.engine = lev.load(folder)  # bf16 on the (single) GPU, adapter + head + calibration
        self.base_model = self.engine.config.model_id

    def score(self, question: str, options: list[str]):
        import torch

        state, questions = build_questions(self.primitive, question, options)
        torch.cuda.synchronize()
        started = time.perf_counter()
        response = self.engine.system_one(state, questions)
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000
        answer = response.answers["answer"]
        scores, probabilities, predicted = parse_answer(self.primitive, answer, len(options))
        return ScoreResult(
            scores=scores,
            probabilities=probabilities,
            predicted_index=predicted,
            latency_ms=latency_ms,
            metadata={
                "primitive": self.primitive,
                "confidence": answer.confidence,
                "usage": response.usage.model_dump(),
            },
        )

    def metadata(self) -> dict[str, object]:
        return {
            "scorer": self.name,
            "model": self.model_id,
            "adapter": f"{self.model_name}@{self.revision}",
            "base_model": self.base_model,
            "primitive": self.primitive,
            "dtype": self.dtype,
            "device": self.device,
            "generation": False,
        }
