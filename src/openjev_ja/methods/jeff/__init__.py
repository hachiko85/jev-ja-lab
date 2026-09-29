"""Jeff (firelex/jeff): full-weight fine-tunes of Qwen3.5 and Gemma 4 for typed, calibrated
zero-shot decisions, one forward pass, same request shape as Jev.

Unlike Hopper/Lev (LoRA adapters on top of a base checkpoint), a Jeff release is a full model:
the backbone weights plus a small linear readout over 255 single-token answer codes, published
together as one checkpoint directory (`decision_config.json` names which architecture loaded it
— `jeff.model.DecisionModel` for the Qwen releases, `jeff.decoder.GenericDecoderDecisionModel`
for Gemma). `jeff.models.load_decision_model` picks the right one; this module just downloads a
release and calls its `.predict()`.
"""

from .scorer import JeffScorer

__all__ = ["JeffScorer"]
