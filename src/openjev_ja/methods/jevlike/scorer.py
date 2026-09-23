from __future__ import annotations

import time
from pathlib import Path

from openjev_ja.common.types import ScoreResult


class JevlikeScorer:
    """Loads a jevlike checkpoint (trained by `methods.jevlike.train`) and
    scores (question, options) through its option-attention head.

    Choice only — jevlike has no Noul or Score head upstream.
    """

    name = "jevlike"

    def __init__(self, checkpoint_path: str | Path, *, device: str = "auto") -> None:
        try:
            from jevlike.model import load_checkpoint, select_device
        except ImportError as exc:
            raise RuntimeError(
                "Install the jevlike extra: pip install -e '.[jevlike]'"
            ) from exc
        self._torch_device = select_device(device)
        self.checkpoint_path = Path(checkpoint_path)
        self.model, self.collator, self.config = load_checkpoint(
            self.checkpoint_path, self._torch_device
        )
        self.model.eval()
        self.device = device

    def score(self, question: str, options: list[str]) -> ScoreResult:
        import torch
        from jevlike.data import ChoiceExample

        started = time.perf_counter()
        batch = self.collator([ChoiceExample(question, tuple(options), 0)])
        batch = {name: tensor.to(self._torch_device) for name, tensor in batch.items()}
        with torch.no_grad():
            logits = self.model(batch)[0, : len(options)]
            probabilities_tensor = logits.softmax(-1)
        latency_ms = (time.perf_counter() - started) * 1000
        return ScoreResult(
            scores=logits.detach().cpu().tolist(),
            probabilities=probabilities_tensor.detach().cpu().tolist(),
            predicted_index=int(probabilities_tensor.argmax(-1).item()),
            latency_ms=latency_ms,
            metadata={"checkpoint": str(self.checkpoint_path)},
        )

    def metadata(self) -> dict[str, object]:
        return {
            "scorer": self.name,
            "model": self.config.get("hf_model") if self.config.get("encoder") == "hf" else "tiny",
            "encoder": self.config.get("encoder"),
            "rank": self.config.get("rank"),
            "checkpoint": str(self.checkpoint_path),
            "generation": False,
        }
