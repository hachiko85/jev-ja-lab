"""TypeSafe Jev: the hosted Jev decision API, its own schema and API.

question + options -> a `{"type": "choice", ...}` request against the
TypeSafe Jev API -> a chosen key plus per-option probabilities from the
service's own response. Independent HTTP API, not a local model.
"""

from .scorer import JevScorer

__all__ = ["JevScorer"]
