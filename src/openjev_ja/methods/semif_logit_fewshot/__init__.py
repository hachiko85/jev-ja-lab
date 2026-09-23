"""SemIf's chat-template prompt recipe with a few in-context examples.

Separate from `methods.semif_logit` (which stays zero-shot): see
`SemifLogitFewShotScorer` docstring for why. `primitive` selects which of
the project's own train-split datasets (see `examples.DEFAULT_SOURCES`)
supplies the in-context examples.
"""

from .scorer import SemifLogitFewShotScorer

__all__ = ["SemifLogitFewShotScorer"]
