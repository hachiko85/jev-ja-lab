"""Deterministic scorer double for smoke tests and vertical-slice checks.

Not a Jev-like decision method — kept out of `methods/` so it is never
mistaken for one of the systems under comparison.
"""

from __future__ import annotations

import hashlib
import time

from openjev_ja.common.types import ScoreResult


class MockScorer:
    name = "mock"

    def score(self, question: str, options: list[str]) -> ScoreResult:
        started = time.perf_counter()
        digest = hashlib.sha256(question.encode()).digest()
        scores = [float(digest[i]) + 1.0 for i in range(len(options))]
        total = sum(scores)
        probabilities = [value / total for value in scores]
        prediction = max(range(len(scores)), key=scores.__getitem__)
        return ScoreResult(
            scores=scores,
            probabilities=probabilities,
            predicted_index=prediction,
            latency_ms=(time.perf_counter() - started) * 1000,
            metadata={"deterministic": True},
        )

    def metadata(self) -> dict[str, object]:
        return {"scorer": self.name}
