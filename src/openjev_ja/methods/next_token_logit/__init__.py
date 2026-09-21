"""Next-token Candidate Label Scoring.

question + options -> prompt -> causal/image-text-to-text LM -> next-token
logits -> candidate answer-label tokens (A/B/C/...) -> softmax -> argmax.

See JEV_JA_LAB_REFACTOR_GUIDE_v2 section 2 for why this method is named
`next_token_logit` rather than after any one model (e.g. Qwen) or the
unrelated `OpenJev` project.
"""

from .scorer import NextTokenLogitScorer, QwenDirectScorer

__all__ = ["NextTokenLogitScorer", "QwenDirectScorer"]
