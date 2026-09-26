"""HopitAI/hopper: a LoRA on Qwen3.5-4B for one-pass calibrated decisions.

Same decision principle as `methods.semif_logit` (one forward pass, no generation, the
option letters' next-token logits) and the same SemIf-layout JSON user turn, but with
Hopper's own system prompt, policy text, true/false noul options, "i: description" score
levels and per-answer-type calibration temperature, all taken from the published serving
code (github.com/hopit-ai/hopper, `hopper_decisions/`). The LoRA is merged into the base
weights at load. hopper_decisions itself is not a dependency (not on PyPI); its prompt and
calibration arithmetic are reproduced here.

The adapter weights are published for research and demo use only.
"""

from .scorer import HopperScorer

__all__ = ["HopperScorer"]
