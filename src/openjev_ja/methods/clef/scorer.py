from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

from openjev_ja.common.types import ScoreResult
from openjev_ja.methods.decider.scorer import build_questions, parse_answer

PRIMITIVES = ("noul", "choice", "score")
QUANTIZATIONS = ("none", "8bit", "4bit")

# variant name -> (Hub repo, revision)
RELEASES: dict[str, tuple[str, str]] = {
    "flash": ("Cloudflare/clef-flash", "17f0b0ad64efb65d273590632833508766b2aae6"),
}


def _import_release_module(folder: Path):
    spec = importlib.util.spec_from_file_location("clef_joint_schema_model", folder / "joint_schema_model.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ClefScorer:
    """Cloudflare/clef-flash (Qwen3.5-9B + joint schema head).

    The wire format is the one JevScorer/DeciderScorer/LevScorer already send
    (`build_questions` / `parse_answer` are shared with DeciderScorer), passed to the
    release's own `systemone(model, processor, request)`.

    The bf16 backbone is ~19GB, more than a 16GB GPU holds, so `quantization` picks how it
    fits. Default "8bit" (bitsandbytes; lm_head and the vision tower stay bf16 -- the joint
    head reads the output-embedding matrix directly): on this machine ~200ms/item, 11.8GB
    peak, option probabilities within ~0.002 of bf16. "none" keeps bf16 numerics and offloads
    overflow decoder layers to CPU RAM (~370ms/item, 15.3GB peak -- little headroom for long
    inputs); "4bit" is the smallest.
    """

    name = "clef"

    def __init__(
        self,
        model_name: str = "flash",
        *,
        primitive: str,
        device: str = "cuda",
        quantization: str = "8bit",
        gpu_memory_gib: float = 14.0,
        cpu_memory_gib: float = 48.0,
        model_id: str | None = None,
    ) -> None:
        if primitive not in PRIMITIVES:
            raise ValueError(f"unsupported primitive: {primitive}")
        if model_name not in RELEASES:
            raise ValueError(f"unknown clef variant {model_name!r}; choose one of {sorted(RELEASES)}")
        if quantization not in QUANTIZATIONS:
            raise ValueError(f"quantization must be one of {QUANTIZATIONS}, got {quantization!r}")
        try:
            import torch
            from huggingface_hub import snapshot_download
            from safetensors.torch import load_file
            from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError("Install the clef extra: pip install -e '.[clef]'") from exc
        self.model_name = model_name
        self.repo_id, self.revision = RELEASES[model_name]
        self.primitive = primitive
        self.device = device
        self.quantization = quantization
        self.model_id = model_id or f"clef-{model_name}"
        folder = Path(snapshot_download(self.repo_id, revision=self.revision))
        self._release = _import_release_module(folder)

        kwargs: dict = {"dtype": torch.bfloat16}
        if quantization == "none":
            kwargs["device_map"] = self._offload_device_map(folder, gpu_memory_gib, cpu_memory_gib)
        else:
            kwargs["device_map"] = {"": device}
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=quantization == "8bit",
                load_in_4bit=quantization == "4bit",
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                llm_int8_skip_modules=["lm_head", "visual"],
            )
        backbone = Qwen3_5ForConditionalGeneration.from_pretrained(folder, **kwargs)
        backbone.config.use_cache = False
        head_config = json.loads((folder / "joint_head_config.json").read_text())
        head = self._release.JointSchemaHead(**head_config)
        head.load_state_dict(load_file(folder / "joint_head.safetensors"), strict=True)
        head = head.to(device=device, dtype=torch.bfloat16)
        self.processor = AutoProcessor.from_pretrained(folder)
        self.model = self._release.ClefModel(backbone, head).eval()

    @staticmethod
    def _offload_device_map(folder: Path, gpu_memory_gib: float, cpu_memory_gib: float) -> dict:
        """Decoder layers overflow to CPU RAM; embeddings, norm, lm_head and the vision tower
        stay on GPU (the joint head reads the output-embedding matrix directly, which must
        not be an offloaded/meta tensor)."""
        import torch
        from accelerate import infer_auto_device_map, init_empty_weights
        from transformers import AutoConfig, Qwen3_5ForConditionalGeneration

        with init_empty_weights():
            skeleton = Qwen3_5ForConditionalGeneration._from_config(
                AutoConfig.from_pretrained(folder), dtype=torch.bfloat16
            )
        device_map = infer_auto_device_map(
            skeleton,
            max_memory={0: f"{gpu_memory_gib}GiB", "cpu": f"{cpu_memory_gib}GiB"},
            no_split_module_classes=["Qwen3_5DecoderLayer", "Qwen3_5VisionBlock"],
        )
        return {
            name: (0 if ".layers." not in name and target != 0 else target)
            for name, target in device_map.items()
        }

    def score(self, question: str, options: list[str]) -> ScoreResult:
        import torch

        state, questions = build_questions(self.primitive, question, options)
        request = {"model": self.model_id, "state": state, "questions": questions}
        is_cuda = self.device.startswith("cuda")
        if is_cuda:
            torch.cuda.synchronize()
        started = time.perf_counter()
        response = self._release.systemone(self.model, self.processor, request)
        if is_cuda:
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000
        answer = response["answers"]["answer"]
        scores, probabilities, predicted = parse_answer(self.primitive, answer, len(options))
        return ScoreResult(
            scores=scores,
            probabilities=probabilities,
            predicted_index=predicted,
            latency_ms=latency_ms,
            metadata={"primitive": self.primitive, "usage": response.get("usage", {})},
        )

    def metadata(self) -> dict[str, object]:
        return {
            "scorer": self.name,
            "model": self.model_id,
            "release": f"{self.repo_id}@{self.revision}",
            "base_model": "Qwen/Qwen3.5-9B",
            "primitive": self.primitive,
            "device": self.device,
            "quantization": self.quantization,
            "generation": False,
        }
