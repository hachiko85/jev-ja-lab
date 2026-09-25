from __future__ import annotations

import json
import math
import time
from typing import Any

from openjev_ja.common.types import ScoreResult

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
PRIMITIVES = ("noul", "choice", "score")

BASE_MODEL = "Qwen/Qwen3.5-4B"
BASE_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
ADAPTER = "HopitAI/hopper"

SYSTEM = (
    "You make decisions about a document under a policy. Read only what is written in the "
    "document. Reply with the letter of the correct option and nothing else."
)
POLICY = "Decide the case using only what the document states. Exactly one option is correct."
# Same wire-level instruction JevScorer/LayaScorer send for choice and score items.
INSTRUCTIONS = "正しい答えを一つ選んでください。"
# hopper.json (calibration map "per_kind", package 1.1.0+): one temperature per answer type.
TEMPERATURES = {
    "choice": 0.7898505975532796,
    "noul": 0.7531284680253558,
    "score": 0.8997033695617845,
}


def build_request(primitive: str, question: str, options: list[str]) -> tuple[str, list[str]]:
    """The user-turn JSON and the option texts shown, following hopper_decisions/request.py
    + prompt.py. `question` is the benchmark item's own text (for choice/score it plays the
    role of the document/state; for noul it is the yes/no question itself, with an empty
    document, exactly as a Jev-style noul request has no state).
    """
    if primitive == "noul":
        # The rubric has no option to sit on, so it goes in the policy; options are
        # true/false in that order (letter A = true).
        policy = f"{POLICY}\ntrue: yes\nfalse: no"
        document, instructions = "", question
        shown = ["true", "false"]
    elif primitive == "score":
        policy, document, instructions = POLICY, question, INSTRUCTIONS
        shown = [f"{index}: {text}" for index, text in enumerate(options)]
    elif primitive == "choice":
        policy, document, instructions = POLICY, question, INSTRUCTIONS
        shown = list(options)
    else:
        raise ValueError(f"unsupported primitive: {primitive}")
    if len(shown) > len(LETTERS):
        raise ValueError(f"{len(shown)} options, more than {len(LETTERS)} letters")
    criterion = f"{policy}\n\n{instructions}" if policy else instructions
    body = {
        "evidence": document or instructions,
        "criterion": criterion,
        "options": [
            {"letter": LETTERS[index], "description": text} for index, text in enumerate(shown)
        ],
    }
    return json.dumps(body, ensure_ascii=False), shown


def rescale(probabilities: list[float], temperature: float) -> list[float]:
    """hopper_decisions.calibration.rescale: softmax(log p / T). Never reorders options."""
    logs = [math.log(max(p, 1e-12)) / temperature for p in probabilities]
    top = max(logs)
    weights = [math.exp(value - top) for value in logs]
    total = sum(weights)
    return [weight / total for weight in weights]


class HopperScorer:
    """HopitAI/hopper: a rank-16 LoRA on Qwen3.5-4B (pinned revision) that reads a
    SemIf-layout JSON decision prompt and answers with the option letter's next-token
    logit — one forward pass, no generation — then applies its per-answer-type
    calibration temperature. The adapter is merged into the bf16 weights at load.

    Licence note: the adapter weights are published for research and demo use only.
    """

    name = "hopper"

    def __init__(
        self,
        model_name: str = BASE_MODEL,
        *,
        primitive: str,
        adapter: str = ADAPTER,
        device: str = "auto",
        dtype: str = "bfloat16",
        revision: str | None = BASE_REVISION,
        adapter_revision: str | None = None,
        model_id: str | None = None,
        max_tokens: int = 4096,
        calibrate: bool = True,
    ) -> None:
        if primitive not in PRIMITIVES:
            raise ValueError(f"unsupported primitive: {primitive}")
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install the hopper extra: pip install -e '.[hopper]'") from exc
        self._torch = torch
        self.model_name = model_name
        self.adapter = adapter
        self.adapter_revision = adapter_revision
        self.model_id = model_id or adapter
        self.primitive = primitive
        self.device = device
        self.dtype = dtype
        self.revision = revision
        self.max_tokens = max_tokens
        self.calibrate = calibrate
        kwargs: dict[str, Any] = {"revision": revision} if revision else {}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        model_kwargs: dict[str, Any] = {**kwargs, "dtype": getattr(torch, dtype)}
        if device == "auto":
            model_kwargs["device_map"] = "auto"
        base = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if device != "auto":
            base = base.to(device)
        peft_kwargs = {"revision": adapter_revision} if adapter_revision else {}
        adapted = PeftModel.from_pretrained(base, adapter, **peft_kwargs)
        self.model = adapted.merge_and_unload().eval()
        self._letter_ids = self._single_token_letters()

    def _single_token_letters(self) -> list[int]:
        ids = [self.tokenizer.encode(letter, add_special_tokens=False) for letter in LETTERS]
        if any(len(encoded) != 1 for encoded in ids):
            raise ValueError("every option letter must be a single token for this tokenizer")
        return [encoded[0] for encoded in ids]

    def _encode(self, question: str, options: list[str]) -> tuple[list[int], int]:
        body, shown = build_request(self.primitive, question, options)
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": body},
        ]
        encoded = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=False
        )
        ids = list(encoded if isinstance(encoded, list) else encoded["input_ids"])
        if not ids or len(ids) > self.max_tokens:
            raise ValueError(f"{len(ids)} input tokens exceed limit {self.max_tokens}")
        return ids, len(shown)

    def score(self, question: str, options: list[str]) -> ScoreResult:
        ids, count = self._encode(question, options)
        model_device = next(self.model.parameters()).device
        input_ids = self._torch.tensor([ids], dtype=self._torch.long, device=model_device)
        if model_device.type == "cuda":
            self._torch.cuda.synchronize(model_device)
        started = time.perf_counter()
        with self._torch.inference_mode():
            logits = self.model(input_ids=input_ids, use_cache=False).logits[0, -1, :].float()
        if model_device.type == "cuda":
            self._torch.cuda.synchronize(model_device)
        latency_ms = (time.perf_counter() - started) * 1000
        selected = logits[self._letter_ids[:count]]
        raw = self._torch.softmax(selected, dim=-1).tolist()
        shown_probabilities = (
            rescale(raw, TEMPERATURES[self.primitive]) if self.calibrate else raw
        )
        if self.primitive == "noul":
            # Letter A is `true`: P(yes) is the first entry; report [P(no), P(yes)].
            p_true = shown_probabilities[0]
            probabilities = [1.0 - p_true, p_true]
            scores = [0.0, p_true]
            predicted = 1 if p_true >= 0.5 else 0
        else:
            probabilities = shown_probabilities
            scores = selected.detach().cpu().tolist()
            predicted = max(range(count), key=probabilities.__getitem__)
        return ScoreResult(
            scores=scores,
            probabilities=probabilities,
            predicted_index=predicted,
            latency_ms=latency_ms,
            metadata={"primitive": self.primitive, "raw_probabilities": raw},
        )

    def metadata(self) -> dict[str, object]:
        return {
            "scorer": self.name,
            "model": self.model_id,
            "base_model": self.model_name,
            "base_revision": self.revision,
            "adapter": self.adapter,
            "adapter_revision": self.adapter_revision,
            "primitive": self.primitive,
            "dtype": self.dtype,
            "device": self.device,
            "calibration": TEMPERATURES[self.primitive] if self.calibrate else None,
            "prompt_style": "chat-template + json-structured (hopper_decisions recipe)",
            "system_prompt": SYSTEM,
            "generation": False,
            "license_note": "adapter weights: research and demo use only",
        }
