from __future__ import annotations

import time
from typing import Any

from openjev_ja.common.types import ScoreResult

PRIMITIVES = ("noul", "choice", "score")
# Same wire-level instruction JevScorer/LayaScorer send for choice and score items.
INSTRUCTIONS = "正しい答えを一つ選んでください。"


def build_questions(primitive: str, question: str, options: list[str]) -> tuple[Any, dict]:
    """(state, questions) in TypeSafe's /v1/systemone wire format — the exact shape
    JevScorer sends, which decider.infer.Decider.system_one accepts unchanged."""
    if primitive == "noul":
        return "", {"answer": {"type": "noul", "instructions": question}}
    if primitive == "score":
        return question, {
            "answer": {"type": "score", "instructions": INSTRUCTIONS, "criteria": options}
        }
    if primitive == "choice":
        keys = [f"option_{index}" for index in range(len(options))]
        return question, {
            "answer": {
                "type": "choice",
                "instructions": INSTRUCTIONS,
                "criteria": dict(zip(keys, options, strict=True)),
            }
        }
    raise ValueError(f"unsupported primitive: {primitive}")


FewShotExample = tuple[str, list[str], int]


def format_few_shot(primitive: str, examples: list[FewShotExample], state: Any) -> str:
    """Worked examples placed in front of the case, inside the /v1/systemone `state`.

    decider has no few-shot mode of its own (it is trained on a bare state + typed
    questions), so in-context examples are supplied the only way the wire format allows:
    as more state text. Each example shows its question and the gold answer's text;
    for noul the instruction itself is the question, so the case marker only points at it.
    """
    blocks = ["以下は同じ形式の判定の解答例です。"]
    for number, (example_question, example_options, gold) in enumerate(examples, 1):
        block = [f"[例{number}]", example_question]
        if primitive != "noul":
            listed = " / ".join(f"{i}: {option}" for i, option in enumerate(example_options))
            block.append(f"選択肢: {listed}")
        block.append(f"正解: {example_options[gold]}")
        blocks.append("\n".join(block))
    target = str(state) if state else "以下の質問に、上と同じ基準で答えてください。"
    blocks.append(f"[判定対象]\n{target}")
    return "\n\n".join(blocks)


def parse_answer(primitive: str, answer: dict, count: int) -> tuple[list[float], list[float], int]:
    """(scores, probabilities, predicted_index) from one system_one answer."""
    if primitive == "noul":
        p_true = float(answer["noul"])
        return [0.0, p_true], [1.0 - p_true, p_true], 1 if p_true >= 0.5 else 0
    if primitive == "score":
        probabilities = [float(answer["probabilities"][str(index)]) for index in range(count)]
        return probabilities, probabilities, max(range(count), key=probabilities.__getitem__)
    keys = [f"option_{index}" for index in range(count)]
    probabilities = [float(answer["probabilities"][key]) for key in keys]
    return probabilities, probabilities, keys.index(answer["choice"])


class DeciderScorer:
    """Mapika/decider-4b (decider-ai): a typed-decision model — Qwen3.5-4B-Base fine-tuned
    to return calibrated option probabilities from one forward pass, an open
    reproduction of the "System One" class TypeSafe's Jev belongs to. Called through
    the package's own `Decider.system_one`, so prompt layout, Score-level isolation and
    temperature are the publisher's, not ours.

    `revision` selects the Hub tag: "v2" (decider-4b v2, one global temperature 1.935),
    "main" (v2.1, per-type temperatures), or "v1". The package loads a local folder, so
    the revision is snapshot-downloaded first.

    `few_shot_count` > 0 (with `datasets_root`) prepends that many worked examples to the
    state — an experiment, not a supported mode of the model; see `format_few_shot`.
    """

    name = "decider"

    def __init__(
        self,
        model_name: str = "Mapika/decider-4b",
        *,
        primitive: str,
        revision: str = "v2",
        device: str = "cuda",
        dtype: str = "bfloat16",
        use_graphs: bool = False,
        model_id: str | None = None,
        few_shot_count: int = 0,
        datasets_root: str | None = None,
        few_shot_seed: int = 42,
    ) -> None:
        if primitive not in PRIMITIVES:
            raise ValueError(f"unsupported primitive: {primitive}")
        try:
            import torch
            from decider.infer import Decider
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError("Install the decider extra: pip install -e '.[decider]'") from exc
        self.model_name = model_name
        self.revision = revision
        self.primitive = primitive
        self.device = device
        self.dtype = dtype
        self.model_id = model_id or f"{model_name}@{revision}"
        self.use_graphs = use_graphs
        self.few_shot: list[FewShotExample] = []
        if few_shot_count:
            if datasets_root is None:
                raise ValueError("few_shot_count requires datasets_root")
            from openjev_ja.methods.semif_logit.examples import load_few_shot_examples

            # Same per-primitive train examples semif's few-shot uses, so 2-shot is comparable.
            self.few_shot = load_few_shot_examples(
                primitive, datasets_root, few_shot_count, seed=few_shot_seed
            )
        folder = snapshot_download(model_name, revision=revision)
        self.decider = Decider(
            folder, device=device, dtype=getattr(torch, dtype), use_graphs=use_graphs
        )
        self.reported_model = getattr(self.decider, "name", None)

    def score(self, question: str, options: list[str]) -> ScoreResult:
        state, questions = build_questions(self.primitive, question, options)
        if self.few_shot:
            state = format_few_shot(self.primitive, self.few_shot, state)
        started = time.perf_counter()
        response = self.decider.system_one(state, questions)
        latency_ms = (time.perf_counter() - started) * 1000
        answer = response["answers"]["answer"]
        scores, probabilities, predicted = parse_answer(self.primitive, answer, len(options))
        return ScoreResult(
            scores=scores,
            probabilities=probabilities,
            predicted_index=predicted,
            latency_ms=latency_ms,
            metadata={
                "primitive": self.primitive,
                "confidence": answer.get("confidence"),
                "response_model": response.get("model"),
                "usage": response.get("usage"),
            },
        )

    def metadata(self) -> dict[str, object]:
        return {
            "scorer": self.name,
            "model": self.model_id,
            "model_revision": self.revision,
            "reported_model": self.reported_model,
            "primitive": self.primitive,
            "dtype": self.dtype,
            "device": self.device,
            "use_graphs": self.use_graphs,
            "few_shot_count": len(self.few_shot),
            "generation": False,
        }
