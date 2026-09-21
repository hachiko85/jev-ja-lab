from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class BenchmarkItem:
    id: str
    question: str
    options: list[str]
    gold_index: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError("question must not be empty")
        if len(self.options) < 2:
            raise ValueError("at least two options are required")
        if not 0 <= self.gold_index < len(self.options):
            raise ValueError("gold_index is outside options")


@dataclass(slots=True)
class ScoreResult:
    scores: list[float]
    predicted_index: int
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)
    probabilities: list[float] | None = None

    def __post_init__(self) -> None:
        if not self.scores:
            raise ValueError("scores must not be empty")
        if not 0 <= self.predicted_index < len(self.scores):
            raise ValueError("predicted_index is outside scores")
        if self.probabilities is not None and len(self.probabilities) != len(self.scores):
            raise ValueError("probabilities and scores must have equal length")
