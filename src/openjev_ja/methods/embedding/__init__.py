"""Embedding-encoder decision method.

Not embedding similarity / cosine scoring. A small multilingual embedding
model (e.g. `intfloat/multilingual-e5-small`) is used as a frozen Transformer
encoder; a tiny per-primitive `Linear(hidden_size, 1)` head reads decision
logits directly off its `last_hidden_state`, at the representative token for
each Choice/Score candidate (its answer-label position) or at the sequence's
first token for Noul.

See EMBEDDING_METHOD_GUIDE_CONCISE.md for the design this implements.
"""

from .scorer import EmbeddingScorer

__all__ = ["EmbeddingScorer"]
