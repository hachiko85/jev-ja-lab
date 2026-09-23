"""laya's own decision-head architecture, run on a Japanese BERT encoder.

`laya.Agent` only loads convaiinnovations' own RLCD-trained checkpoints
(a `rl_agent_config.json` + `model.safetensors` pair), so it cannot load an
arbitrary encoder. Its underlying `laya.common.DecisionModel` — a shared
`[MASK]`-marker scorer with a type embedding for noul/choice/score and a
small TransformerEncoder head on top of any HF `AutoModel` encoder — has no
such restriction. This method reuses that architecture and laya's own
`build_sequence` prompt format verbatim, swaps in a Japanese encoder (kept
frozen, same as `methods.embedding`), and trains the head with plain
supervised cross-entropy instead of laya's RLCD pipeline.
"""

from .scorer import LayaBertScorer

__all__ = ["LayaBertScorer"]
