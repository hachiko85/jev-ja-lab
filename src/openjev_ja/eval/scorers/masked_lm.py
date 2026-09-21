from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

from openjev_ja.common import ScoreResult
from openjev_ja.common.revisions import local_git_revision
from openjev_ja.eval.scorers._prompting import (
    ANSWER_LABELS,
    format_choices,
    validate_answer_tokens,
)

PROMPT_TEMPLATE = "問題:\n{question}\n\n選択肢:\n{choices}\n\n答え: {mask}"


class MaskedLMScorer:
    """Direct-logit scorer for encoder (masked-LM) models such as BERT/ModernBERT.

    Mirrors QwenDirectScorer's approach (single deterministic forward pass, no
    text generation, compare logits for single-token letter labels) but reads
    the answer at the prompt's `[MASK]` position instead of the next token
    after a causal prompt, since encoder models have no "next token".
    """

    name = "masked-lm"

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "auto",
        dtype: str = "bfloat16",
        revision: str | None = None,
        model_id: str | None = None,
        metadata_revision: str | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForMaskedLM, AutoTokenizer
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
        if self.tokenizer.mask_token_id is None:
            raise ValueError(f"{model_name} tokenizer has no mask token; not a masked-LM model")
        # Unlike QwenDirectScorer's causal prompts, the mask token always sits at
        # the very end of ours ("答え: {mask}"); truncating from the left (instead
        # of the tokenizer's default right-truncation) drops excess context while
        # never cutting off the mask position itself. Some encoders (e.g. classic
        # BERT with absolute position embeddings) hard-error past their max
        # length instead of degrading gracefully, so truncation is mandatory here.
        self.tokenizer.truncation_side = "left"
        torch_dtype = getattr(torch, dtype)
        model_kwargs: dict[str, Any] = {**kwargs, "dtype": torch_dtype}
        if device == "auto":
            model_kwargs["device_map"] = "auto"
        self.model = AutoModelForMaskedLM.from_pretrained(model_name, **model_kwargs).eval()
        if device != "auto":
            self.model.to(device)
        max_length = self.tokenizer.model_max_length
        config_max = getattr(self.model.config, "max_position_embeddings", None)
        if not isinstance(max_length, int) or max_length > 1_000_000:
            max_length = config_max or 512
        elif config_max is not None:
            max_length = min(max_length, config_max)
        self._max_length = int(max_length)
        self.prompt_hash = hashlib.sha256(PROMPT_TEMPLATE.encode()).hexdigest()
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
        return PROMPT_TEMPLATE.format(
            question=question, choices=choices, mask=self.tokenizer.mask_token
        )

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
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_length,
        )
        model_device = next(self.model.parameters()).device
        encoded = {key: value.to(model_device) for key, value in encoded.items()}
        mask_rows, mask_cols = (encoded["input_ids"] == self.tokenizer.mask_token_id).nonzero(
            as_tuple=True
        )
        if mask_rows.numel() != len(inputs):
            raise ValueError(
                f"expected exactly one mask token per prompt, found {mask_rows.numel()} "
                f"across {len(inputs)} prompts"
            )
        if model_device.type == "cuda":
            self._torch.cuda.synchronize(model_device)
        started = time.perf_counter()
        with self._torch.inference_mode():
            output = self.model(**encoded)
            mask_logits = output.logits[mask_rows, mask_cols]
            logits = mask_logits[:, token_ids].float()
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
