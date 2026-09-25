"""Mapika/decider-4b (decider-ai): typed decisions with calibrated probabilities in one pass.

An open reproduction of the "System One" model class TypeSafe's Jev belongs to, called
through the package's own `Decider.system_one` with the same request shape `JevScorer`
sends to the hosted API — so it is a like-for-like local counterpart to Jev, and the
publisher's prompt layout, Score-level isolation and temperature apply unchanged.

`decider-ai` is installed without its dependencies (`uv pip install --no-deps decider-ai`):
its metadata pins numpy==1.26.4, which would downgrade this project's numpy, while the
inference modules used here (`decider.infer`) do not import numpy.
"""

from .scorer import DeciderScorer

__all__ = ["DeciderScorer"]
