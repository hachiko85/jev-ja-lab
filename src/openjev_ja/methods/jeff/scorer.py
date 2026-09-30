from __future__ import annotations

import time
from typing import Any

from openjev_ja.common.types import ScoreResult

PRIMITIVES = ("noul", "choice", "score")
# Same wire-level instruction JevScorer/DeciderScorer/LevScorer send for choice and score items.
INSTRUCTIONS = "正しい答えを一つ選んでください。"

# name -> (Hub repo, revision). All three are full-weight fine-tunes (no base model to merge),
# Apache-2.0, trained for up to 26 options (`decision_config.json`'s "max_options").
RELEASES: dict[str, tuple[str, str]] = {
    "qwen3.5-0.8b": ("mstrasser/Jeff-Qwen3.5-0.8B", "d66458d54426fcf52046b896261df8909bbc8b05"),
    "qwen3.5-2b": ("mstrasser/Jeff-Qwen3.5-2B", "30824caa5f255df0086fecba5bdfa63374c4f758"),
    "gemma4-e2b": ("mstrasser/Jeff-Gemma4-E2B", "afcb75ae269494582aab3ec3cea0d278685a8cb2"),
}


def build_question(primitive: str, question: str, options: list[str]) -> tuple[str, dict[str, Any]]:
    """(state, jeff Question dict) — the same split as JevScorer/DeciderScorer/LevScorer: choice
    and score put the item text in `state` and a fixed instruction after it; noul has an empty
    state and the yes/no question itself as the instruction."""
    if primitive == "noul":
        return "", {"type": "noul", "instructions": question}
    if primitive == "score":
        return question, {"type": "score", "instructions": INSTRUCTIONS, "criteria": list(options)}
    if primitive == "choice":
        keys = [f"option_{index}" for index in range(len(options))]
        return question, {
            "type": "choice",
            "instructions": INSTRUCTIONS,
            "criteria": dict(zip(keys, options, strict=True)),
        }
    raise ValueError(f"unsupported primitive: {primitive}")


def parse_probabilities(
    primitive: str, probabilities: list[float], count: int
) -> tuple[list[float], list[float], int]:
    """(scores, probabilities, predicted_index) from `DecisionModel.predict`'s one distribution.

    `jeff.model.options()` fixes the noul option order to `["false", "true"]`, so index 1 is
    always P(yes) regardless of what text the caller's `options` argument holds (noul's
    `criteria` is optional and not sent here; see `build_question`).
    """
    if primitive == "noul":
        p_true = probabilities[1]
        return [0.0, p_true], [1.0 - p_true, p_true], 1 if p_true >= 0.5 else 0
    return probabilities, probabilities, max(range(count), key=probabilities.__getitem__)


class JeffScorer:
    """firelex/jeff: full-weight fine-tunes of Qwen3.5 (0.8B / 2B) and Gemma 4 (E2B) for typed,
    calibrated zero-shot decisions in one forward pass — the same `/v1/systemone`-style request
    shape as Jev, Decider and Lev. Loaded through the package's own
    `jeff.models.load_decision_model`, which picks the checkpoint's own architecture class
    (`jeff.model.DecisionModel` for the Qwen releases, `jeff.decoder.GenericDecoderDecisionModel`
    for Gemma) from `decision_config.json`, so the publisher's prompt layout, answer-code readout
    and calibration temperature apply unchanged. `model_name` selects the release:
    "qwen3.5-0.8b" (default), "qwen3.5-2b" or "gemma4-e2b" (see `RELEASES`).
    """

    name = "jeff"

    def __init__(
        self,
        model_name: str = "qwen3.5-0.8b",
        *,
        primitive: str,
        device: str = "cuda",
        cache_dir: str | None = None,
        model_id: str | None = None,
    ) -> None:
        if primitive not in PRIMITIVES:
            raise ValueError(f"unsupported primitive: {primitive}")
        if model_name not in RELEASES:
            available = sorted(RELEASES)
            raise ValueError(f"unknown jeff release {model_name!r}; choose one of {available}")
        try:
            from huggingface_hub import snapshot_download
            from jeff.models import load_decision_model
        except ImportError as exc:
            raise RuntimeError("Install the jeff extra: pip install -e '.[jeff]'") from exc
        self.model_name = model_name
        self.repo_id, self.revision = RELEASES[model_name]
        self.primitive = primitive
        self.device = device
        self.model_id = model_id or model_name
        folder = snapshot_download(self.repo_id, revision=self.revision, cache_dir=cache_dir)
        self.model = load_decision_model(checkpoint=folder, device=device)
        self.base_model = self.model.base_model

    def score(self, question: str, options: list[str]) -> ScoreResult:
        import torch

        state, wire_question = build_question(self.primitive, question, options)
        row = {"state": state, "question": wire_question}
        is_cuda = self.device.startswith("cuda")
        if is_cuda:
            torch.cuda.synchronize()
        started = time.perf_counter()
        [probabilities] = self.model.predict([row], batch_size=1)
        if is_cuda:
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000
        scores, result_probabilities, predicted = parse_probabilities(
            self.primitive, probabilities, len(options)
        )
        return ScoreResult(
            scores=scores,
            probabilities=result_probabilities,
            predicted_index=predicted,
            latency_ms=latency_ms,
            metadata={"primitive": self.primitive, "raw_probabilities": probabilities},
        )

    def metadata(self) -> dict[str, object]:
        return {
            "scorer": self.name,
            "model": self.model_id,
            "release": f"{self.repo_id}@{self.revision}",
            "base_model": self.base_model,
            "primitive": self.primitive,
            "device": self.device,
            "generation": False,
        }
