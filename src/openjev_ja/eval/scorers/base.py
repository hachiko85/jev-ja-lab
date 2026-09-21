from __future__ import annotations

from typing import Protocol

from openjev_ja.common import ScoreResult


class Scorer(Protocol):
    name: str

    def score(self, question: str, options: list[str]) -> ScoreResult: ...

    def metadata(self) -> dict[str, object]: ...
