from __future__ import annotations

import hashlib
import inspect
import json
import time
from pathlib import Path
from typing import Any

from openjev_ja.common.revisions import local_git_revision
from openjev_ja.common.types import ScoreResult

LETTERS = "ABCDEFGHIJKLMNOP"
DIRECT_SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)


FewShotExample = tuple[str, list[str], int]


def _payload(question: str, options: list[str]) -> dict[str, Any]:
    return {
        "evidence": question,
        "criterion": "Select the correct option.",
        "options": [
            {"letter": LETTERS[index], "description": option}
            for index, option in enumerate(options)
        ],
    }


def _messages(
    question: str, options: list[str], few_shot: list[FewShotExample] | None = None
) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": DIRECT_SYSTEM}]
    for example_question, example_options, gold_index in few_shot or []:
        example_payload = json.dumps(
            _payload(example_question, example_options), ensure_ascii=False
        )
        messages.append({"role": "user", "content": example_payload})
        messages.append({"role": "assistant", "content": LETTERS[gold_index]})
    final_payload = json.dumps(_payload(question, options), ensure_ascii=False)
    messages.append({"role": "user", "content": final_payload})
    return messages


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class SemifLogitScorer:
    """`next_token_logit`'s decision principle, SemIf's prompt/verification recipe.

    See package docstring for exactly what differs from
    `methods.next_token_logit.NextTokenLogitScorer`.
    """

    name = "semif-logit"

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "auto",
        dtype: str = "bfloat16",
        revision: str | None = None,
        model_id: str | None = None,
        metadata_revision: str | None = None,
        max_tokens: int = 4096,
        few_shot: list[FewShotExample] | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
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
        self.max_tokens = max_tokens
        self.few_shot: list[FewShotExample] = list(few_shot or [])
        kwargs: dict[str, Any] = {"revision": revision} if revision else {}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        torch_dtype = getattr(torch, dtype)
        model_kwargs: dict[str, Any] = {**kwargs, "dtype": torch_dtype}
        if device == "auto":
            model_kwargs["device_map"] = "auto"
        config = AutoConfig.from_pretrained(model_name, **kwargs)
        # Mirrors SemIf's own loader: Qwen3.5 ships a native causal-LM class,
        # unlike next_token_logit's generic image-text-to-text wrapper.
        loader = AutoModelForCausalLM
        if config.model_type in {"qwen3_5", "qwen3_5_text"}:
            import transformers as _transformers

            loader = getattr(_transformers, "Qwen3_5ForCausalLM", None)
            if loader is None:
                raise RuntimeError("Installed transformers lacks the native Qwen3.5 model class")
            config = config.get_text_config()
            model_kwargs["config"] = config
        self.model = loader.from_pretrained(model_name, **model_kwargs).eval()
        if device != "auto":
            self.model.to(device)
        self._supports_logits_to_keep = (
            "logits_to_keep" in inspect.signature(self.model.forward).parameters
        )
        self._slot_cache: dict[int, list[int]] = {}

    def _slot_ids(self, count: int) -> list[int]:
        if count not in self._slot_cache:
            ids: list[int] = []
            for letter in LETTERS[:count]:
                encoded = self.tokenizer.encode(letter, add_special_tokens=False)
                if len(encoded) != 1 or self.tokenizer.decode(encoded) != letter:
                    raise ValueError(f"answer slot {letter!r} is not one exact round-trip token")
                ids.append(encoded[0])
            if len(set(ids)) != len(ids):
                raise ValueError("answer-slot tokens collide")
            self._slot_cache[count] = ids
        return self._slot_cache[count]

    def _encode(self, question: str, options: list[str]) -> tuple[list[int], list[int], str]:
        prompt = self.tokenizer.apply_chat_template(
            _messages(question, options, self.few_shot),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if not ids or len(ids) > self.max_tokens:
            raise ValueError(f"{len(ids)} input tokens exceed limit {self.max_tokens}")
        slots = self._slot_ids(len(options))
        for letter, token in zip(LETTERS, slots, strict=False):
            probe = self.tokenizer.encode(prompt + letter, add_special_tokens=False)
            if probe != ids + [token]:
                raise ValueError(f"answer boundary changes tokenization for slot {letter}")
        return ids, slots, _digest(prompt)

    def score(self, question: str, options: list[str]) -> ScoreResult:
        ids, slots, prompt_hash = self._encode(question, options)
        model_device = next(self.model.parameters()).device
        input_ids = self._torch.tensor([ids], dtype=self._torch.long, device=model_device)
        attention_mask = self._torch.ones(
            (1, len(ids)), dtype=self._torch.long, device=model_device
        )
        if model_device.type == "cuda":
            self._torch.cuda.synchronize(model_device)
        started = time.perf_counter()
        with self._torch.inference_mode():
            model_inputs: dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "use_cache": False,
                "return_dict": True,
            }
            if self._supports_logits_to_keep:
                model_inputs["logits_to_keep"] = 1
            vocabulary = self.model(**model_inputs).logits[0, -1, :].float()
        if model_device.type == "cuda":
            self._torch.cuda.synchronize(model_device)
        latency_ms = (time.perf_counter() - started) * 1000
        selected = vocabulary[slots]
        probabilities_tensor = self._torch.softmax(selected, dim=-1)
        return ScoreResult(
            scores=selected.detach().cpu().tolist(),
            probabilities=probabilities_tensor.detach().cpu().tolist(),
            predicted_index=int(probabilities_tensor.argmax(-1).item()),
            latency_ms=latency_ms,
            metadata={
                "prompt_sha256": prompt_hash,
                "answer_labels": list(LETTERS[: len(options)]),
            },
        )

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
            "prompt_style": "chat-template + json-structured (SemIf recipe)",
            "system_prompt": DIRECT_SYSTEM,
            "few_shot_count": len(self.few_shot),
            "generation": False,
        }
