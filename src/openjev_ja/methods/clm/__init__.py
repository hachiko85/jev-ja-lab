"""Contrastive-LM/CLM-v0.1-8B: a System One model made of a frozen Qwen3-8B encoder and two
small projection heads, trained with a contrastive (InfoNCE) objective between states and
actions. Answers are the softmax over `scale * cosine(state_head(state), action_head(option))`.

The `contrastive-lm` package serves the encoder through vLLM (not available on Windows and it
pins a heavy dependency set), so the encoder + heads pipeline is reimplemented here on top of
`transformers`; the request assembly follows the package's `schema.build_pairs`.
"""

from .scorer import ClmScorer

__all__ = ["ClmScorer"]
