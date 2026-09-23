from __future__ import annotations

import time
from pathlib import Path

from openjev_ja.common.types import ScoreResult
from openjev_ja.methods.laya_bert.model import build_model, encode

PRIMITIVES = ("noul", "choice", "score")


class LayaBertScorer:
    """laya's own DecisionModel head (type embedding + TransformerEncoder
    scorer over `[MASK]` markers, see laya.common), trained from scratch on
    a frozen Japanese BERT encoder — methods.laya_bert.train does the
    training, laya.Agent itself only loads convaiinnovations' own
    checkpoints.
    """

    name = "laya-bert"

    def __init__(
        self,
        model_name: str,
        *,
        primitive: str,
        head_path: str | Path,
        device: str = "cpu",
        model_id: str | None = None,
    ) -> None:
        if primitive not in PRIMITIVES:
            raise ValueError(f"unsupported primitive: {primitive}")
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Install evaluation dependencies: pip install -e '.[eval]'") from exc
        self._torch = torch
        self.model_name = model_name
        self.model_id = model_id or model_name
        self.primitive = primitive
        self.device = device
        self.head_path = Path(head_path)
        self.tokenizer, self.model = build_model(model_name, device)
        state_dict = torch.load(self.head_path, map_location=device, weights_only=True)
        _, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if unexpected:
            raise ValueError(f"unexpected checkpoint keys: {unexpected}")
        self.model.eval()

    def score(self, question: str, options: list[str]) -> ScoreResult:
        from laya.common import collate_items

        started = time.perf_counter()
        encoded = encode(self.tokenizer, question, options, self.primitive)
        if encoded is None:
            raise ValueError("options exceed laya's head_max_len; cannot score this item")
        batch = collate_items([[encoded]], self.tokenizer.pad_token_id)
        with self._torch.no_grad():
            logits, _ = self.model(
                batch["input_ids"].to(self.device),
                batch["attention_mask"].to(self.device),
                batch["marker_pos"].to(self.device),
                batch["marker_mask"].to(self.device),
                batch["qtype"].to(self.device),
            )
            probabilities_tensor = self._torch.softmax(logits[0, : len(options)], dim=-1)
        latency_ms = (time.perf_counter() - started) * 1000
        return ScoreResult(
            scores=logits[0, : len(options)].detach().cpu().tolist(),
            probabilities=probabilities_tensor.detach().cpu().tolist(),
            predicted_index=int(probabilities_tensor.argmax(-1).item()),
            latency_ms=latency_ms,
            metadata={"primitive": self.primitive, "head": str(self.head_path)},
        )

    def metadata(self) -> dict[str, object]:
        return {
            "scorer": self.name,
            "model": self.model_id,
            "primitive": self.primitive,
            "head_path": str(self.head_path),
            "generation": False,
            "frozen_encoder": True,
            "architecture": "laya.common.DecisionModel",
        }
