"""Registry of trainable architectures, one entry per `train.<architecture>` config value —
the training-side counterpart to `eval.orchestrate`'s `scorer` dispatch.

An architecture is four functions the generic cross-validation runner (`openjev_ja.train.runner`)
calls without knowing anything else about it: build the (tokenizer, model) pair, score one
(question, options) pair, list the trainable parameters, and read back a state dict to save.
Adding a new one (e.g. `methods.embedding`'s frozen-encoder head) means adding an
`ArchitectureSpec` here with the same four functions; nothing else in the training pipeline
changes.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol


class Loaded(Protocol):
    """Whatever `build` returns; passed back into every other function unchanged."""


@dataclass(frozen=True)
class ArchitectureSpec:
    name: str
    # (model_name, device, **hyperparameters) -> (tokenizer, model)
    build: Any
    # (model, tokenizer, device, question, options, primitive) -> logits tensor, or None to
    # skip the item (e.g. an option's marker did not fit the head's max length)
    forward_logits: Any
    # model -> the parameters actually trained (frozen-encoder architectures return only the
    # head's parameters)
    trainable_parameters: Any
    # model -> {name: tensor} to torch.save as the checkpoint
    state_dict: Any


def _laya_bert_build(
    model_name: str, device: str, *, head_layers: int = 2, revision: str | None = None, **_: Any
):
    from openjev_ja.methods.laya_bert.model import build_model

    return build_model(model_name, device, head_layers=head_layers, revision=revision)


def _laya_bert_forward(model, tokenizer, device, question, options, primitive):
    from openjev_ja.methods.laya_bert.model import forward_logits

    return forward_logits(model, tokenizer, device, question, options, primitive)


def _laya_bert_trainable(model) -> Iterable[Any]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def _laya_bert_state_dict(model) -> dict[str, Any]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


ARCHITECTURES: dict[str, ArchitectureSpec] = {
    "laya-bert": ArchitectureSpec(
        name="laya-bert",
        build=_laya_bert_build,
        forward_logits=_laya_bert_forward,
        trainable_parameters=_laya_bert_trainable,
        state_dict=_laya_bert_state_dict,
    ),
}


def get_architecture(name: str) -> ArchitectureSpec:
    try:
        return ARCHITECTURES[name]
    except KeyError as exc:
        available = ", ".join(sorted(ARCHITECTURES))
        raise ValueError(f"unknown architecture {name!r}; available: {available}") from exc
