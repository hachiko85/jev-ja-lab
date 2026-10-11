"""LiquidAI/d1-3B: a 3B decision model on LFM2.5-VL-3B that answers typed Choice / Noul / Score
questions in one forward pass. Evaluated as Liquid AI's GGUF export through llama.cpp's
`llama-server` `/v1/systemone` endpoint (the same route as the Clef-Flash GGUF)."""

from .gguf import D1GgufScorer

__all__ = ["D1GgufScorer"]
