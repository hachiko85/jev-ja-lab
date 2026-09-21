"""Shared letter-choice prompt construction used by direct-logit methods.

Both the `next_token_logit` method (causal LM: reads the next-token logit
after the prompt) and the `bert_masked_lm` method (encoder LM: reads the
logit at a `[MASK]` placed in the prompt) frame every primitive — Noul,
Choice, Score — the same way: list the candidate answers as `A. ...`,
`B. ...`, ... and compare logits for the single-token letter labels. Only
where the answer position sits (end of a causal prompt vs a mask token
inside a masked-LM prompt) differs between the two methods.

This is a formatting/tokenization utility shared by convention, not a
decision-inference interface: each method still owns its own prompt
template, forward pass, and score interpretation.
"""

from __future__ import annotations

from typing import Any

ANSWER_LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def validate_answer_tokens(tokenizer: Any, labels: list[str]) -> list[int]:
    ids: list[int] = []
    for label in labels:
        token_ids = tokenizer.encode(label, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(
                f"answer label {label!r} must map to exactly one token; got {token_ids}"
            )
        ids.append(int(token_ids[0]))
    if len(set(ids)) != len(ids):
        raise ValueError("answer labels must map to distinct tokens")
    return ids


def format_choices(options: list[str], labels: list[str]) -> str:
    return "\n".join(
        f"{label}. {option}" for label, option in zip(labels, options, strict=True)
    )
