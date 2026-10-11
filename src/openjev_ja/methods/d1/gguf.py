from __future__ import annotations

from openjev_ja.methods.clef.gguf import ClefGgufScorer

# variant -> (Hub repo, revision, file): Liquid AI's own GGUF export of d1-3B (a decision model on
# LFM2.5-VL-3B). The vision projector (mmproj) is not needed for text-only decisions.
RELEASES: dict[str, tuple[str, str, str]] = {
    "q8_0": ("LiquidAI/d1-3B-GGUF", "bb1e436ea78eb96a3f1acb6da865f70c2fbeb563", "d1-3B-Q8_0.gguf"),
    "q4_k_m": ("LiquidAI/d1-3B-GGUF", "bb1e436ea78eb96a3f1acb6da865f70c2fbeb563", "d1-3B-Q4_K_M.gguf"),
    "bf16": ("LiquidAI/d1-3B-GGUF", "bb1e436ea78eb96a3f1acb6da865f70c2fbeb563", "d1-3B-BF16.gguf"),
    # d1-omni-600M: a 0.6B text/image/audio sibling (only Q8_0/BF16/F16 are published).
    "omni-600m-q8_0": ("LiquidAI/d1-omni-600M-GGUF", "04397145ed8381350403aa556db0ec6a49dd8c07", "d1-omni-600M-Q8_0.gguf"),
}
# `lfm2-d1` decision models need a llama.cpp build newer than the one clef-flash runs on (b11430
# lacks the `lfm2-d1` type); b11485 loads d1-3B, and d1-omni-600M needs b11541 (b11485 fails with
# `tensor 'blk.16.ffn_gate.weight' not found`).
DEFAULT_SERVER = "models/llama.cpp/b11541/llama-server.exe"


class D1GgufScorer(ClefGgufScorer):
    """LiquidAI/d1-3B as a GGUF, served by llama.cpp's `llama-server` through `/v1/systemone`
    -- the same wire format and server lifecycle as ClefGgufScorer (shared implementation)."""

    name = "d1-gguf"
    releases = RELEASES
    base_model = "LiquidAI/d1-3B"
    model_id_prefix = "d1-gguf"

    def __init__(self, model_name: str = "q8_0", *, server_path: str = DEFAULT_SERVER, **kwargs) -> None:
        super().__init__(model_name, server_path=server_path, **kwargs)
        self.base_model = self.repo_id.removesuffix("-GGUF")
