"""Shared types and utilities.

Per JEV_JA_LAB_REFACTOR_GUIDE_v2, this package holds what is common across
Jev / Jev-like methods (result schema, dataset identity primitives) — not a
shared inference interface. See `openjev_ja.methods` for method-specific
implementations and `openjev_ja.common.schemas` for the cross-method result
schema.
"""

from .errors import DatasetUnavailableError
from .protocols import Scorer
from .schemas import SCHEMA_VERSION, EvaluationRecord, iter_evaluation_records
from .types import BenchmarkItem, ScoreResult

__all__ = [
    "SCHEMA_VERSION",
    "BenchmarkItem",
    "DatasetUnavailableError",
    "EvaluationRecord",
    "ScoreResult",
    "Scorer",
    "iter_evaluation_records",
]
