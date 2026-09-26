from __future__ import annotations

import json
import math
import time
from collections import OrderedDict
from typing import Any

from openjev_ja.common.types import ScoreResult

PRIMITIVES = ("noul", "choice", "score")

HEADS_REPO = "Contrastive-LM/CLM-v0.1-8B"
HEADS_FILE = "CLM_v0.1-8B.pt"
HEADS_REVISION = "e939398d4556fcd9400c76fa8c5a513202f42b0a"
ENCODER = "Qwen/Qwen3-8B"
ENCODER_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
MAX_TOKENS = 2048
NOUL_KEYS = ("false", "true")


def to_text(x: Any) -> str:
    """CLM's `schema.to_text` for the shapes this project sends (strings; a dict/list is
    rendered as prose fields rather than JSON, since the heads are trained on prose)."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, (int, float)):
        return str(x)
    return json.dumps(x, ensure_ascii=False)


def build_texts(primitive: str, question: str, options: list[str]) -> tuple[str, list[str]]:
    """(state text, candidate texts) as CLM's `build_pairs` assembles them when the state is
    empty and the whole item text is the question (`instructions`): the state head sees the
    question text alone. The item text already carries its own task instruction and rubric, so
    no fixed instruction is appended. Appending a Japanese one lowered accuracy on a held-out
    train sample (choice 0.40 -> 0.32, score 0.44 -> 0.08), since the heads are trained on
    plain question text.

    - choice: every option verbatim is a candidate (the action head sees plain answer text)
    - score: every level description is a candidate, in order
    - noul: candidates are `false: No. This is false: <q>` / `true: Yes. This is true: <q>`
    """
    if primitive == "noul":
        instructions = question.strip()
        texts = [
            f"false: No. This is false: {instructions}",
            f"true: Yes. This is true: {instructions}",
        ]
        return instructions, texts
    if primitive in ("choice", "score"):
        return question.strip(), [to_text(option) for option in options]
    raise ValueError(f"unsupported primitive: {primitive}")


def softmax(logits: list[float]) -> list[float]:
    top = max(logits)
    weights = [math.exp(value - top) for value in logits]
    total = sum(weights)
    return [weight / total for weight in weights]


def make_head(
    width: int,
    depth: int,
    proj: int,
    *,
    activation: str = "gelu",
    layernorm: bool = False,
    residual: bool = False,
    hidden: int = 4096,
):
    """`hidden -> width -> ... -> proj` MLP, the same module layout as CLM's `heads.make_head`
    (so the published state dicts load unchanged)."""
    import torch.nn as nn

    act = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}[activation]

    class Head(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inp = nn.Linear(hidden, width)
            self.hidden = nn.ModuleList(nn.Linear(width, width) for _ in range(depth - 2))
            self.norms = nn.ModuleList(
                (nn.LayerNorm(width) if layernorm else nn.Identity()) for _ in range(depth - 2)
            )
            self.out = nn.Linear(width, proj)
            self.act = act()
            self.residual = residual

        def forward(self, x):
            x = self.act(self.inp(x))
            for linear, norm in zip(self.hidden, self.norms, strict=True):
                block = self.act(norm(linear(x)))
                x = x + block if self.residual else block
            return self.out(x)

    return Head()


class ClmScorer:
    """Contrastive-LM/CLM-v0.1-8B: a frozen Qwen3-8B encoder (last-token pooled, L2-normalised
    embeddings) with two small projection heads. A candidate's score is
    `exp(logit_scale) * cos(state_head(state), action_head(candidate))` and a softmax over the
    candidates of one question is the answer — no generation.

    The reference implementation serves the encoder through vLLM; this reimplements the
    pipeline with `transformers` so it runs locally (vLLM does not run on Windows). To fit a
    16GB GPU, `embed_tokens` (1.2GB, only a lookup table) stays on the CPU and the unused
    LM head is not loaded, which leaves ~14GB of bf16 layers on the GPU.

    The model is English-trained ("language: en" on the Hub); Japanese input is handled only by
    the multilingual Qwen3 encoder.
    """

    name = "clm"

    def __init__(
        self,
        model_name: str = ENCODER,
        *,
        primitive: str,
        heads_repo: str = HEADS_REPO,
        heads_file: str = HEADS_FILE,
        heads_revision: str | None = HEADS_REVISION,
        revision: str | None = ENCODER_REVISION,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        max_tokens: int = MAX_TOKENS,
        cache_size: int = 50_000,
        model_id: str | None = None,
    ) -> None:
        if primitive not in PRIMITIVES:
            raise ValueError(f"unsupported primitive: {primitive}")
        try:
            import torch
            from huggingface_hub import hf_hub_download
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install the eval extra: pip install -e '.[eval]'") from exc
        self._torch = torch
        self.model_name = model_name
        self.revision = revision
        self.primitive = primitive
        self.device = device
        self.dtype = dtype
        self.max_tokens = max_tokens
        self.heads_repo = heads_repo
        self.heads_revision = heads_revision
        self.model_id = model_id or f"{heads_repo}@{heads_revision}"
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._cache_size = cache_size

        kwargs: dict[str, Any] = {"revision": revision} if revision else {}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        self.tokenizer.padding_side = "left"      # last token sits at position -1
        self.tokenizer.truncation_side = "left"   # vLLM truncate_prompt_tokens keeps the tail
        encoder = AutoModel.from_pretrained(model_name, dtype=getattr(torch, dtype), **kwargs)
        # The 1.2GB token-embedding table is only a lookup: keep it on the CPU and feed the
        # layers `inputs_embeds`, so the 36 layers + norm (~13GB in bf16) fit a 16GB GPU.
        self._embed_tokens = encoder.embed_tokens
        encoder.embed_tokens = torch.nn.Identity()  # unused when inputs_embeds is passed
        self.encoder = encoder.to(device).eval()

        checkpoint = torch.load(
            hf_hub_download(heads_repo, heads_file, revision=heads_revision),
            map_location="cpu",
            weights_only=True,
        )
        cfg = checkpoint["cfg"]
        options = {
            "activation": cfg.get("activation", "gelu"),
            "layernorm": cfg.get("layernorm", False),
            "residual": cfg.get("residual", False),
            "hidden": cfg.get("hidden_size", checkpoint.get("hidden_size", 4096)),
        }
        proj = checkpoint.get("projection_dim", cfg.get("projection_dim", 512))
        self.state_head = make_head(cfg["width"], cfg["depth"], proj, **options)
        self.action_head = make_head(cfg["width"], cfg["depth"], proj, **options)
        self.state_head.load_state_dict(checkpoint["state_head"])
        self.action_head.load_state_dict(checkpoint["action_head"])
        self.state_head.eval().to(device)
        self.action_head.eval().to(device)
        logit_scale = torch.as_tensor(checkpoint["logit_scale"]).float()
        self.scale = float(logit_scale.exp().clamp(max=100.0))
        self.head_config = dict(cfg)

    def _encode(self, texts: list[str]):
        """[n, hidden] L2-normalised float32 last-token embeddings of the Qwen3 encoder."""
        torch = self._torch
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_tokens,
        )
        with torch.inference_mode():
            embeds = self._embed_tokens(batch["input_ids"]).to(self.device)
            hidden = self.encoder(
                inputs_embeds=embeds,
                attention_mask=batch["attention_mask"].to(self.device),
                use_cache=False,
            ).last_hidden_state
        last = hidden[:, -1, :].float()
        return torch.nn.functional.normalize(last, dim=-1)

    def _embed(self, texts: list[str]):
        """Embeddings for `texts`, encoding only the ones not cached (options repeat a lot)."""
        torch = self._torch
        missing = [text for text in dict.fromkeys(texts) if text not in self._cache]
        if missing:
            vectors = self._encode(missing)
            for text, vector in zip(missing, vectors, strict=True):
                self._cache[text] = vector.detach().to(self.device)
        for text in texts:
            self._cache.move_to_end(text)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return torch.stack([self._cache[text] for text in texts])

    def score(self, question: str, options: list[str]) -> ScoreResult:
        torch = self._torch
        state_text, candidate_texts = build_texts(self.primitive, question, options)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            state = self._embed([state_text])
            candidates = self._embed(candidate_texts)
            zs = torch.nn.functional.normalize(self.state_head(state), dim=-1)
            za = torch.nn.functional.normalize(self.action_head(candidates), dim=-1)
            logits = (self.scale * (za @ zs[0])).float().cpu().tolist()
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000
        probabilities = softmax(logits)
        if self.primitive == "noul":
            # candidates are ("false", "true"): report [P(no), P(yes)]
            p_true = probabilities[1]
            scores = [0.0, p_true]
            predicted = 1 if p_true >= 0.5 else 0
        else:
            scores = probabilities
            predicted = max(range(len(probabilities)), key=probabilities.__getitem__)
        return ScoreResult(
            scores=scores,
            probabilities=probabilities,
            predicted_index=predicted,
            latency_ms=latency_ms,
            metadata={"primitive": self.primitive, "logits": logits},
        )

    def metadata(self) -> dict[str, object]:
        return {
            "scorer": self.name,
            "model": self.model_id,
            "encoder": self.model_name,
            "encoder_revision": self.revision,
            "heads": f"{self.heads_repo}@{self.heads_revision}",
            "head_config": self.head_config,
            "logit_scale": self.scale,
            "primitive": self.primitive,
            "dtype": self.dtype,
            "device": self.device,
            "pooling": "last token, L2-normalised",
            "max_tokens": self.max_tokens,
            "generation": False,
        }
