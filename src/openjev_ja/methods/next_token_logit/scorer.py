from __future__ import annotations

import hashlib
import inspect
import time
from pathlib import Path
from typing import Any

from openjev_ja.common.candidate_labels import (
    ANSWER_LABELS,
    format_choices,
    validate_answer_tokens,
)
from openjev_ja.common.revisions import local_git_revision
from openjev_ja.common.types import ScoreResult

PROMPT_TEMPLATE = "問題:\n{question}\n\n選択肢:\n{choices}\n\n答え:"


class NextTokenLogitScorer:
    name = "qwen-direct"

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-4B",
        *,
        device: str = "auto",
        dtype: str = "bfloat16",
        revision: str | None = None,
        model_id: str | None = None,
        metadata_revision: str | None = None,
    ) -> None:
        try:
            import torch
            from transformers import (
                AutoConfig,
                AutoModelForCausalLM,
                AutoModelForImageTextToText,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise RuntimeError("Install evaluation dependencies: pip install -e '.[eval]'") from exc
        self._torch = torch
        self.model_name = model_name
        self.model_id = model_id or (
            Path(model_name).name if Path(model_name).exists() else model_name
        )
        self.device = device
        self.dtype = dtype
        self.revision = revision
        self.metadata_revision = metadata_revision
        kwargs: dict[str, Any] = {"revision": revision} if revision else {}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        torch_dtype = getattr(torch, dtype)
        model_kwargs: dict[str, Any] = {**kwargs, "dtype": torch_dtype}
        if device == "auto":
            model_kwargs["device_map"] = "auto"
        config = AutoConfig.from_pretrained(model_name, **kwargs)
        loader = (
            AutoModelForImageTextToText
            if config.model_type in {"qwen3_5", "gemma4"}
            else AutoModelForCausalLM
        )
        self.model = loader.from_pretrained(model_name, **model_kwargs).eval()
        if device != "auto":
            self.model.to(device)
        self.prompt_hash = hashlib.sha256(PROMPT_TEMPLATE.encode()).hexdigest()
        self._supports_logits_to_keep = (
            "logits_to_keep" in inspect.signature(self.model.forward).parameters
        )
        self._validated: dict[int, tuple[list[str], list[int]]] = {}

    def _labels(self, count: int) -> tuple[list[str], list[int]]:
        if count > len(ANSWER_LABELS):
            raise ValueError(f"at most {len(ANSWER_LABELS)} choices are supported")
        if count not in self._validated:
            labels = list(ANSWER_LABELS[:count])
            self._validated[count] = (labels, validate_answer_tokens(self.tokenizer, labels))
        return self._validated[count]

    def _prompt(self, question: str, options: list[str], labels: list[str]) -> str:
        choices = format_choices(options, labels)
        return PROMPT_TEMPLATE.format(question=question, choices=choices)

    def score(self, question: str, options: list[str]) -> ScoreResult:
        return self.score_batch([(question, options)])[0]

    def score_batch(self, inputs: list[tuple[str, list[str]]]) -> list[ScoreResult]:
        if not inputs:
            return []
        option_counts = {len(options) for _, options in inputs}
        if len(option_counts) != 1:
            raise ValueError("all items in a batch must have the same number of choices")
        count = option_counts.pop()
        labels, token_ids = self._labels(count)
        prompts = [self._prompt(question, options, labels) for question, options in inputs]
        encoded = self.tokenizer(prompts, return_tensors="pt", padding=True)
        model_device = next(self.model.parameters()).device
        encoded = {key: value.to(model_device) for key, value in encoded.items()}
        if model_device.type == "cuda":
            self._torch.cuda.synchronize(model_device)
        started = time.perf_counter()
        with self._torch.inference_mode():
            model_inputs = dict(encoded)
            if self._supports_logits_to_keep:
                model_inputs["logits_to_keep"] = 1
            output = self.model(**model_inputs)
            logits = output.logits[:, -1, token_ids].float()
            probabilities_tensor = self._torch.softmax(logits, dim=-1)
        if model_device.type == "cuda":
            self._torch.cuda.synchronize(model_device)
        per_item_latency = (time.perf_counter() - started) * 1000 / len(inputs)
        scores = logits.detach().cpu().tolist()
        probabilities = probabilities_tensor.detach().cpu().tolist()
        predicted = probabilities_tensor.argmax(dim=-1).detach().cpu().tolist()
        return [
            ScoreResult(
                scores=item_scores,
                probabilities=item_probabilities,
                predicted_index=int(item_predicted),
                latency_ms=per_item_latency,
                metadata={"answer_labels": labels, "batch_size": len(inputs)},
            )
            for item_scores, item_probabilities, item_predicted in zip(
                scores, probabilities, predicted, strict=True
            )
        ]

    def metadata(self) -> dict[str, object]:
        config = getattr(self.model, "config", None)
        return {
            "scorer": self.name,
            "model": self.model_id,
            "model_revision": (
                self.metadata_revision
                or self.revision
                or getattr(config, "_commit_hash", None)
                or local_git_revision(self.model_name)
            ),
            "dtype": self.dtype,
            "device": self.device,
            "prompt_template": PROMPT_TEMPLATE,
            "prompt_hash": self.prompt_hash,
            "generation": False,
        }


# Compatibility alias: the class was named after the model family this
# project happened to use first (Qwen), even though the method applies to
# any causal / image-text-to-text LM. See guide section 17.
QwenDirectScorer = NextTokenLogitScorer
