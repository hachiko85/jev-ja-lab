"""interfaze-ai/lev: a LoRA adapter on Qwen3.5-4B that answers typed questions (noul / choice /
score) about a state in one forward pass, reading calibrated probabilities from the logits.

Called through the package's own `DecisionEngine.system_one` (`lev` on PyPI-style install from
GitHub), with the same request shape `JevScorer` sends to the hosted API and `DeciderScorer`
sends to decider-4b, so it is a like-for-like local counterpart: the publisher's prompt format,
label-token readout, candidate-path head and calibration apply unchanged.
"""

from .scorer import LevScorer

__all__ = ["LevScorer"]
