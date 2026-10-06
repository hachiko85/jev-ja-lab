"""Cloudflare/clef-flash: a 9B Qwen3.5 backbone with a joint schema head that returns a
probability for every option of every typed question (Choice / Noul / Score) in one forward
pass -- the same /v1/systemone category as Jev, Decider, Lev and Jeff.

`ClefScorer` loads the release through its own `joint_schema_model.py` (`encode_record`,
`collate_records`, `JointSchemaHead`, `ClefModel`) from the Hub snapshot, so the published
record encoding and head run unmodified. `ClefGgufScorer` runs ggml-org's GGUF conversion
through llama.cpp's `llama-server` `/v1/systemone` endpoint instead.
"""

from .gguf import ClefGgufScorer
from .scorer import ClefScorer

__all__ = ["ClefGgufScorer", "ClefScorer"]
