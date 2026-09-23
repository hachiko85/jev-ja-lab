"""Shared model construction and prompt conversion for train.py and scorer.py."""

from __future__ import annotations

from typing import Any

MAX_LEN = 512
HEAD_MAX_LEN = 192


def to_laya_question(question: str, options: list[str], primitive: str) -> dict[str, Any]:
    """(question, options) -> laya's internal question dict (see laya.agent._to_internal).

    `state` is left empty and the whole question goes into `ins`
    (instructions): every other method in this project already folds
    context and question into one string, so this keeps that convention
    instead of splitting it laya's way.
    """
    if primitive == "noul":
        return {"t": "noul", "ins": question, "crit": {"false": None, "true": None}}
    if primitive == "score":
        return {"t": "score", "ins": question, "crit": list(options)}
    if primitive == "choice":
        return {
            "t": "choice",
            "ins": question,
            "crit": {f"opt{i}": option for i, option in enumerate(options)},
        }
    raise ValueError(f"unsupported primitive: {primitive}")


def build_model(hf_model: str, device: str, *, head_layers: int = 2):
    """A frozen HF encoder + laya's own DecisionModel head, from scratch."""
    from laya.common import DecisionModel
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    if tokenizer.mask_token_id is None:
        raise ValueError(f"{hf_model} has no mask token; laya's sequence format requires one")
    encoder = AutoModel.from_pretrained(hf_model)
    model = DecisionModel(encoder, head_layers=head_layers, n_act=2).to(device)
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    model.encoder.eval()
    return tokenizer, model


def encode(tokenizer, question: str, options: list[str], primitive: str) -> dict[str, Any] | None:
    from laya.common import QTYPES, build_sequence

    q = to_laya_question(question, options, primitive)
    ids, markers = build_sequence(tokenizer, "", q, MAX_LEN, HEAD_MAX_LEN)
    if len(markers) != len(q["crit"]):
        return None  # an option's [MASK] marker got truncated away
    return {"ids": ids, "markers": markers, "qtype": QTYPES[primitive]}
