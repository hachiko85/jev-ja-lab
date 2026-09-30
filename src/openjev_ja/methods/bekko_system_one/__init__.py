"""hotchpotch/bekko-system-one-v0: compact ModernBERT/Ettin-reranker encoders fine-tuned as
typed decision models (Choice / Noul / Score), the same category as Jev. Loaded through the
model's own `custom_code` inference class (`inference_v0.BekkoSentenceTransformer`), fetched
from the Hub via `transformers.dynamic_module_utils.get_class_from_dynamic_module` — the method
the model card itself recommends, so nothing about the published inference path is reimplemented.

Runs in this project's ordinary `[eval]` environment (its own README pins
`torch>=2.10,<2.11`, but this project's `torch==2.11.0` loads and predicts fine); the `bekko`
extra adds only `sentence-transformers` and `scikit-learn` (installed `--no-deps`, since
`sentence-transformers` would otherwise pull a newer torch and fight this project's CUDA build).
"""

from .scorer import BekkoSystemOneScorer

__all__ = ["BekkoSystemOneScorer"]
