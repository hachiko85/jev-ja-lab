from .base import Scorer
from .jev import JevScorer
from .masked_lm import MaskedLMScorer
from .mock import MockScorer
from .nli import NLICrossEncoderScorer
from .qwen_direct import QwenDirectScorer, validate_answer_tokens

SCORER_CLASSES: dict[str, type] = {
    "qwen_direct": QwenDirectScorer,
    "masked_lm": MaskedLMScorer,
}


def resolve_scorer_class(model_name: str, revision: str | None = None) -> type:
    """Pick QwenDirectScorer or MaskedLMScorer from the model's own config.

    Any encoder architecture ending in "ForMaskedLM" (BERT, RoBERTa,
    ModernBERT, ELECTRA, DeBERTa, ALBERT, ...) gets MaskedLMScorer; every
    causal or image-text-to-text architecture keeps using QwenDirectScorer,
    which already auto-selects between those two internally.
    """
    try:
        from transformers import AutoConfig
    except ImportError as exc:
        raise RuntimeError("Install evaluation dependencies: pip install -e '.[eval]'") from exc
    kwargs: dict[str, str] = {"revision": revision} if revision else {}
    config = AutoConfig.from_pretrained(model_name, **kwargs)
    architectures = getattr(config, "architectures", None) or []
    if any(str(architecture).endswith("ForMaskedLM") for architecture in architectures):
        return MaskedLMScorer
    return QwenDirectScorer


__all__ = [
    "SCORER_CLASSES",
    "JevScorer",
    "MaskedLMScorer",
    "MockScorer",
    "NLICrossEncoderScorer",
    "QwenDirectScorer",
    "Scorer",
    "resolve_scorer_class",
    "validate_answer_tokens",
]
