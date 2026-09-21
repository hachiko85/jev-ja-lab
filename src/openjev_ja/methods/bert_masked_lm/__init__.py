"""Masked-LM candidate label scoring for encoder models (BERT/ModernBERT).

question + options -> prompt with a trailing `[MASK]` -> encoder forward
pass -> mask-position logits for candidate answer-label tokens -> softmax
-> argmax. Mirrors `next_token_logit`'s framing but reads the answer at the
prompt's mask position instead of the next token after a causal prompt,
since encoder models have no "next token".
"""

from .scorer import MaskedLMScorer

__all__ = ["MaskedLMScorer"]
