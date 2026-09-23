"""Laya: multilingual, non-autoregressive System-1 decision engine.

External OSS (github.com/NandhaKishorM/laya, PyPI `laya`), not vendored —
see manifests/upstream.yaml. Wraps `laya.Router` behind this project's
common Scorer contract. Ships pretrained, RLCD-trained decision heads over
a ModernBERT/mmBERT encoder, so unlike `methods/embedding`, nothing needs
training here: it is a ready-to-score system, closer in spirit to
`methods/typesafe_jev` (same request/response shape as the Jev decision
API) than to the from-scratch methods in this project.
"""

from .scorer import LayaScorer

__all__ = ["LayaScorer"]
