from __future__ import annotations

import time
from pathlib import Path

from openjev_ja.common import ScoreResult
from openjev_ja.common.revisions import local_git_revision

TEMPLATES = {
    "ja": "正しい答えは「{option}」である。",
    "en": "The correct answer is: {option}",
}


class NLICrossEncoderScorer:
    name = "nli-cross-encoder"

    def __init__(
        self,
        model_name: str = "AlexWortega/openjev",
        *,
        template: str = "ja",
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_length: int = 1024,
    ) -> None:
        try:
            import torch
            from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install evaluation dependencies: pip install -e '.[eval]'") from exc
        if template not in TEMPLATES:
            raise ValueError(f"unknown NLI template: {template}")
        self._torch = torch
        self.model_name = model_name
        self.model_id = Path(model_name).name if Path(model_name).exists() else model_name
        self.template_name = template
        self.template = TEMPLATES[template]
        self.device = device
        self.dtype = dtype
        self.max_length = max_length
        self.config = AutoConfig.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                model_name, dtype=getattr(torch, dtype)
            )
            .to(device)
            .eval()
        )
        labels = {str(value).lower(): int(key) for key, value in self.config.id2label.items()}
        try:
            self.entailment_index = labels["entailment"]
        except KeyError as exc:
            raise ValueError("model config must define an entailment label") from exc
        self.pair_template = getattr(self.config, "nli_template", None)

    def score(self, question: str, options: list[str]) -> ScoreResult:
        hypotheses = [self.template.format(option=option) for option in options]
        if self.pair_template:
            texts = [
                self.pair_template.format(premise=question.strip(), hypothesis=hypothesis.strip())
                for hypothesis in hypotheses
            ]
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        else:
            encoded = self.tokenizer(
                [question] * len(options),
                hypotheses,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        if self.device.startswith("cuda"):
            self._torch.cuda.synchronize()
        started = time.perf_counter()
        with self._torch.inference_mode():
            logits = self.model(**encoded).logits.float()
            entailment = self._torch.softmax(logits, dim=-1)[:, self.entailment_index]
        if self.device.startswith("cuda"):
            self._torch.cuda.synchronize()
        scores = entailment.cpu().tolist()
        return ScoreResult(
            scores=scores,
            probabilities=None,
            predicted_index=max(range(len(scores)), key=scores.__getitem__),
            latency_ms=(time.perf_counter() - started) * 1000,
            metadata={"independent_entailment_scores": True},
        )

    def metadata(self) -> dict[str, object]:
        return {
            "scorer": self.name,
            "model": self.model_id,
            "template": self.template,
            "template_language": self.template_name,
            "dtype": self.dtype,
            "device": self.device,
            "model_revision": getattr(self.config, "_commit_hash", None)
            or local_git_revision(self.model_name),
        }
