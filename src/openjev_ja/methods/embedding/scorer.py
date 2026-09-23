from __future__ import annotations

import time
from pathlib import Path

from openjev_ja.common.revisions import local_git_revision
from openjev_ja.common.types import ScoreResult
from openjev_ja.methods.embedding.encoder_head import PRIMITIVES, EmbeddingEncoder


class EmbeddingScorer:
    """Frozen embedding encoder + a trained per-primitive Linear(hidden, 1) head.

    Choice/Score share one code path: each candidate's representative hidden
    state goes through the same head, and the resulting per-candidate logits
    are softmax-ed exactly like `next_token_logit`'s candidate-label logits.
    Noul is a single logit off the sequence's first token, read through a
    sigmoid; `options` for Noul items is always `["いいえ", "はい"]`
    (index 1 = True), so it is reported as `scores=[1-p, p]` for schema
    compatibility with the other methods.
    """

    name = "embedding"

    def __init__(
        self,
        model_name: str,
        *,
        primitive: str,
        head_path: str | Path,
        device: str = "cpu",
        dtype: str = "float32",
        revision: str | None = None,
        model_id: str | None = None,
        metadata_revision: str | None = None,
    ) -> None:
        if primitive not in PRIMITIVES:
            raise ValueError(f"unsupported primitive: {primitive}")
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Install evaluation dependencies: pip install -e '.[eval]'") from exc
        self._torch = torch
        self.model_name = model_name
        self.model_id = model_id or (
            Path(model_name).name if Path(model_name).exists() else model_name
        )
        self.primitive = primitive
        self.device = device
        self.dtype = dtype
        self.revision = revision
        self.metadata_revision = metadata_revision
        self.head_path = Path(head_path)
        self.encoder = EmbeddingEncoder(
            model_name, device=device, dtype=dtype, revision=revision
        )
        self.head = torch.nn.Linear(self.encoder.hidden_size, 1)
        state_dict = torch.load(self.head_path, map_location=device, weights_only=True)
        self.head.load_state_dict(state_dict)
        self.head.to(device).eval()
        for parameter in self.head.parameters():
            parameter.requires_grad_(False)

    def score(self, question: str, options: list[str]) -> ScoreResult:
        started = time.perf_counter()
        with self._torch.no_grad():
            if self.primitive == "noul":
                state = self.encoder.statement_state(question)
                logit = float(self.head(state).squeeze(-1).item())
                p_true = float(self._torch.sigmoid(self._torch.tensor(logit)).item())
                # softmax([0.0, logit]) == [1 - sigmoid(logit), sigmoid(logit)], so
                # scores stays consistent with probabilities the same way every
                # other method's raw logits do.
                scores = [0.0, logit]
                probabilities = [1.0 - p_true, p_true]
                predicted_index = 1 if p_true >= 0.5 else 0
            else:
                states = self.encoder.candidate_states(question, options)
                logits = self.head(states).squeeze(-1)
                probabilities_tensor = self._torch.softmax(logits, dim=-1)
                scores = logits.detach().cpu().tolist()
                probabilities = probabilities_tensor.detach().cpu().tolist()
                predicted_index = int(probabilities_tensor.argmax(dim=-1).item())
        return ScoreResult(
            scores=scores,
            probabilities=probabilities,
            predicted_index=predicted_index,
            latency_ms=(time.perf_counter() - started) * 1000,
            metadata={"primitive": self.primitive, "head": str(self.head_path)},
        )

    def score_batch(self, inputs: list[tuple[str, list[str]]]) -> list[ScoreResult]:
        if not inputs:
            return []
        started = time.perf_counter()
        results: list[ScoreResult] = []
        with self._torch.no_grad():
            if self.primitive == "noul":
                questions = [question for question, _ in inputs]
                states = self.encoder.statement_states_batch(questions)
                logits = self.head(states).squeeze(-1)
                probabilities_tensor = self._torch.sigmoid(logits)
                per_item_latency = (time.perf_counter() - started) * 1000 / len(inputs)
                for logit, p_true in zip(
                    logits.detach().cpu().tolist(),
                    probabilities_tensor.detach().cpu().tolist(),
                    strict=True,
                ):
                    results.append(
                        ScoreResult(
                            scores=[0.0, logit],
                            probabilities=[1.0 - p_true, p_true],
                            predicted_index=1 if p_true >= 0.5 else 0,
                            latency_ms=per_item_latency,
                            metadata={"primitive": self.primitive, "head": str(self.head_path)},
                        )
                    )
            else:
                states_list = self.encoder.candidate_states_batch(inputs)
                per_item_latency = (time.perf_counter() - started) * 1000 / len(inputs)
                for states in states_list:
                    logits = self.head(states).squeeze(-1)
                    probabilities_tensor = self._torch.softmax(logits, dim=-1)
                    results.append(
                        ScoreResult(
                            scores=logits.detach().cpu().tolist(),
                            probabilities=probabilities_tensor.detach().cpu().tolist(),
                            predicted_index=int(probabilities_tensor.argmax(dim=-1).item()),
                            latency_ms=per_item_latency,
                            metadata={"primitive": self.primitive, "head": str(self.head_path)},
                        )
                    )
        return results

    def metadata(self) -> dict[str, object]:
        return {
            "scorer": self.name,
            "model": self.model_id,
            "model_revision": (
                self.metadata_revision or self.revision or local_git_revision(self.model_name)
            ),
            "primitive": self.primitive,
            "head_path": str(self.head_path),
            "dtype": self.dtype,
            "device": self.device,
            "generation": False,
            "frozen_encoder": True,
        }
