"""jevlike: an independent Jev-like starter (option-attention over a frozen
or from-scratch encoder — github.com/vinnylarouge/jevlike, not vendored,
see manifests/upstream.yaml). Choice-only: no Noul or Score head exists in
upstream, so only the choice primitive is evaluated against it.
"""

from .scorer import JevlikeScorer

__all__ = ["JevlikeScorer"]
