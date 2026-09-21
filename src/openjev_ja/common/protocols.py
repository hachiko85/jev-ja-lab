from __future__ import annotations

from typing import Protocol

from openjev_ja.common.types import ScoreResult


class Scorer(Protocol):
    """Structural type for what `eval.runner` needs to drive any method.

    This is a duck-typed hint for the generic runner/orchestrator, not a
    base class methods must subclass. Each method's scorer implements
    `score` (and, optionally, `score_batch`) and `metadata` however its own
    inference principle requires; nothing about its internal input/output
    representation is constrained beyond this.
    """

    name: str

    def score(self, question: str, options: list[str]) -> ScoreResult: ...

    def metadata(self) -> dict[str, object]: ...
