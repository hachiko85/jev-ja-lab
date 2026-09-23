"""Frozen encoder + representative-token extraction shared by scorer and train.

Choice/Score: question and candidates are joined into one sequence
("A. option\\nB. option\\n..."); the representative hidden state for each
candidate is the token at its answer-label character ("A", "B", ...),
located via the tokenizer's offset mapping so it works regardless of how a
given tokenizer splits that label into subwords.

Noul: the sequence's first token (whatever a given tokenizer's leading
special token is: <s>, <cls>, <bos>, ...) stands in for the whole
context+statement.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from openjev_ja.common.candidate_labels import ANSWER_LABELS

PRIMITIVES = ("choice", "score", "noul")


def build_candidate_prompt(question: str, options: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """Join question + labeled candidates into one string; return each label's char span."""
    labels = ANSWER_LABELS[: len(options)]
    text = f"{question}\n"
    spans: list[tuple[int, int]] = []
    for label, option in zip(labels, options, strict=True):
        start = len(text)
        spans.append((start, start + len(label)))
        text += f"{label}. {option}\n"
    return text, spans


def _token_index_for_char(offset_mapping: list[tuple[int, int]], char_pos: int) -> int | None:
    for token_index, (start, end) in enumerate(offset_mapping):
        if start == end:
            continue  # special token with an empty offset (CLS, BOS, padding, ...)
        if start <= char_pos < end:
            return token_index
    return None


def _last_real_token_index(offset_mapping: list[tuple[int, int]]) -> int:
    """Index of the last non-empty-offset token (skips right-padding)."""
    for token_index in range(len(offset_mapping) - 1, -1, -1):
        start, end = offset_mapping[token_index]
        if start != end:
            return token_index
    return 0


def _candidate_token_indices(
    offset_mapping: list[tuple[int, int]], spans: list[tuple[int, int]]
) -> list[int]:
    """Map each candidate's label span to a token index.

    A candidate whose label got truncated away (the model's max_length is a
    hard limit — e.g. multilingual-e5-small tops out at 512 tokens, and a
    four-option question can exceed that on its own) falls back to the
    sequence's last real token: an arbitrary state rather than a crash, and
    one every other truncated candidate shares, so it is not favored by the
    softmax over real, in-range candidates.
    """
    fallback = None
    indices = []
    for start, _ in spans:
        index = _token_index_for_char(offset_mapping, start)
        if index is None:
            if fallback is None:
                fallback = _last_real_token_index(offset_mapping)
            index = fallback
        indices.append(index)
    return indices


class EmbeddingEncoder:
    """A frozen embedding model used as a plain Transformer encoder."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cpu",
        dtype: str = "float32",
        revision: str | None = None,
        max_length: int = 1024,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install evaluation dependencies: pip install -e '.[eval]'") from exc
        self._torch = torch
        kwargs: dict[str, Any] = {"revision": revision} if revision else {}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        if not self.tokenizer.is_fast:
            raise ValueError(f"{model_name} has no fast tokenizer; offset mapping is required")
        # Right padding: batched statement_states_batch() reads the first
        # token (index 0) as each sequence's representative state, which is
        # only correct when padding is appended after the real tokens.
        self.tokenizer.padding_side = "right"
        # Truncate from the left: the candidate-label spans this module reads
        # (build_candidate_prompt) always sit at the END of the sequence, so
        # trimming a too-long question from the front keeps every label
        # position intact instead of raising in _token_index_for_char().
        self.tokenizer.truncation_side = "left"
        self.model = (
            AutoModel.from_pretrained(model_name, dtype=getattr(torch, dtype), **kwargs)
            .to(device)
            .eval()
        )
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.device = device
        self.dtype = dtype
        self.max_length = min(int(max_length), int(self.tokenizer.model_max_length or max_length))
        self.hidden_size = int(self.model.config.hidden_size)

    def _forward(self, text: str) -> tuple[Any, list[tuple[int, int]]]:
        encoded = self.tokenizer(
            text,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        # no_grad (not inference_mode): the encoder itself is frozen, but the
        # resulting hidden states still need to support autograd downstream,
        # through the trainable head — an inference-mode tensor cannot be.
        with self._torch.no_grad():
            # Cast up-front: the encoder may run in bfloat16 for speed, but the
            # trainable head (always float32) needs a matching dtype, the same
            # way other methods cast logits to float before comparing them.
            hidden_state = self.model(**encoded).last_hidden_state[0].float()
        return hidden_state, offsets

    def candidate_states(self, question: str, options: list[str]) -> Any:
        """Return one hidden-state vector per candidate: shape (len(options), hidden_size)."""
        text, spans = build_candidate_prompt(question, options)
        hidden_state, offsets = self._forward(text)
        token_indices = _candidate_token_indices(offsets, spans)
        return hidden_state[token_indices]

    def statement_state(self, text: str) -> Any:
        """Return the sequence's first-token hidden state: shape (hidden_size,)."""
        hidden_state, _ = self._forward(text)
        return hidden_state[0]

    def _forward_batch(self, texts: list[str]) -> tuple[Any, list[list[tuple[int, int]]]]:
        encoded = self.tokenizer(
            texts,
            return_offsets_mapping=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        offsets = encoded.pop("offset_mapping").tolist()
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self._torch.no_grad():
            hidden_states = self.model(**encoded).last_hidden_state.float()
        return hidden_states, offsets

    def candidate_states_batch(self, items: list[tuple[str, list[str]]]) -> list[Any]:
        """Batched `candidate_states`: one (len(options), hidden_size) tensor per item.

        Items may have different candidate counts; only the tokenizer padding
        is shared across the batch, so nothing about per-item softmax/loss
        computation downstream needs to change.
        """
        texts: list[str] = []
        spans_per_item: list[list[tuple[int, int]]] = []
        for question, options in items:
            text, spans = build_candidate_prompt(question, options)
            texts.append(text)
            spans_per_item.append(spans)
        hidden_states, offsets_per_item = self._forward_batch(texts)
        results = []
        rows = zip(hidden_states, spans_per_item, offsets_per_item, strict=True)
        for row, spans, offsets in rows:
            token_indices = _candidate_token_indices(offsets, spans)
            results.append(row[token_indices])
        return results

    def statement_states_batch(self, texts: list[str]) -> Any:
        """Batched `statement_state`: shape (len(texts), hidden_size)."""
        hidden_states, _ = self._forward_batch(texts)
        return hidden_states[:, 0, :]


def head_state_path(root: str | Path, model_id: str, primitive: str) -> Path:
    if primitive not in PRIMITIVES:
        raise ValueError(f"unsupported primitive: {primitive}")
    return Path(root) / model_id / f"{primitive}.pt"
