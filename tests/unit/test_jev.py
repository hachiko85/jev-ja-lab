import httpx

from openjev_ja.eval.metrics import summarize
from openjev_ja.methods.typesafe_jev import JevScorer


def test_jev_http_schema_and_response_mapping() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = request.read().decode()
        assert '"type":"choice"' in body.replace(" ", "")
        assert request.headers["Authorization"] == "Bearer test-key"
        return httpx.Response(
            200,
            headers={"x-request-id": "request-test"},
            json={
                "model": "jev-test",
                "answers": {
                    "answer": {
                        "type": "choice",
                        "choice": "option_1",
                        "probabilities": {"option_0": 0.2, "option_1": 0.8},
                        "confidence": 0.7,
                    }
                },
                "usage": {"input_tokens": 12, "output_tokens": 2},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    scorer = JevScorer(
        api_key="test-key", api_url="https://example.invalid/systemone", client=client
    )
    result = scorer.score("質問", ["甲", "乙"])
    assert result.predicted_index == 1
    assert result.probabilities == [0.2, 0.8]
    assert result.metadata["request_id"] == "request-test"


def test_summary_aggregates_api_usage() -> None:
    predictions = [
        {
            "gold_index": 1,
            "predicted_index": 1,
            "probabilities": [0.2, 0.8],
            "latency_ms": 10.0,
            "score_metadata": {"usage": {"input_tokens": 12, "output_tokens": 2}},
        },
        {
            "gold_index": 0,
            "predicted_index": 1,
            "probabilities": [0.3, 0.7],
            "latency_ms": 20.0,
            "score_metadata": {"usage": {"input_tokens": 8, "output_tokens": 1}},
        },
    ]
    summary = summarize(predictions, 0.1, task_type="noul")
    assert summary["input_tokens"] == 20
    assert summary["output_tokens"] == 3
    assert summary["total_tokens"] == 23
    assert summary["mean_tokens_per_item"] == 11.5
