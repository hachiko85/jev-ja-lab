from __future__ import annotations

import os
import time
from typing import Any

from openjev_ja.common.types import ScoreResult


class JevScorer:
    name = "jev"
    instructions = "正しい答えを一つ選んでください。"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_url: str | None = None,
        model: str = "jev-latest",
        timeout: float = 30.0,
        max_retries: int = 2,
        client: Any | None = None,
    ) -> None:
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("Install evaluation dependencies: pip install -e '.[eval]'") from exc
        self.api_key = api_key or os.getenv("TYPESAFE_API_KEY")
        self.api_url = api_url or os.getenv("TYPESAFE_API_URL")
        if not self.api_key:
            raise RuntimeError("TYPESAFE_API_KEY is required for the Jev scorer")
        if not self.api_url:
            raise RuntimeError("TYPESAFE_API_URL is required for the Jev scorer")
        self.model = model
        self.max_retries = max_retries
        self.client = client or httpx.Client(timeout=timeout)

    def score(self, question: str, options: list[str]) -> ScoreResult:
        keys = [f"option_{index}" for index in range(len(options))]
        payload = {
            "state": question,
            "model": self.model,
            "questions": {
                "answer": {
                    "type": "choice",
                    "instructions": self.instructions,
                    "criteria": dict(zip(keys, options, strict=True)),
                }
            },
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        started = time.perf_counter()
        response = None
        for attempt in range(self.max_retries + 1):
            response = self.client.post(self.api_url, headers=headers, json=payload)
            if response.status_code not in {429, 500, 502, 503, 504} or attempt == self.max_retries:
                break
            retry_after = response.headers.get("retry-after")
            delay = min(float(retry_after), 5.0) if retry_after else min(0.5 * 2**attempt, 2.0)
            time.sleep(delay)
        assert response is not None
        response.raise_for_status()
        latency = (time.perf_counter() - started) * 1000
        body = response.json()
        answer = body["answers"]["answer"]
        probabilities = [float(answer["probabilities"][key]) for key in keys]
        predicted = keys.index(answer["choice"])
        return ScoreResult(
            scores=probabilities.copy(),
            probabilities=probabilities,
            predicted_index=predicted,
            latency_ms=latency,
            metadata={
                "confidence": answer.get("confidence"),
                "response_model": body.get("model"),
                "usage": body.get("usage"),
                "request_id": response.headers.get("x-request-id"),
                "status_code": response.status_code,
            },
        )

    def metadata(self) -> dict[str, object]:
        return {
            "scorer": self.name,
            "model": self.model,
            "max_retries": self.max_retries,
            "prompt_template": self.instructions,
        }
