"""NLI cross-encoder scoring: independent entailment probability per option.

question + option -> "正しい答えは「option」である。" hypothesis (or the
model's own premise/hypothesis pair template) -> NLI cross-encoder ->
entailment-class probability, scored independently per option (not a joint
softmax across options). The default model, `AlexWortega/openjev`, is the
`alex_openjev` system referenced in JEV_JA_LAB_REFACTOR_GUIDE_v2 section 3.
"""

from .scorer import NLICrossEncoderScorer

__all__ = ["NLICrossEncoderScorer"]
