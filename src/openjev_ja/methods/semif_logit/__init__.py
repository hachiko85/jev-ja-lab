"""Candidate-label next-token logit scoring, SemIf's prompt recipe.

Same decision principle as `methods.next_token_logit` (one forward pass,
no generation, compare next-token logits restricted to the candidate
answer-label tokens) but with github.com/TheoLeeCJ/SemIf's prompt
construction and verification instead of this project's own:

- a chat-templated system+user message pair (JSON-structured
  evidence/criterion/options) rather than a natural-language prompt
- a stricter per-slot check: `prompt + letter` must tokenize as
  `prompt-tokens + [that letter's token]`, not just "the letter alone is
  one token"
- native `Qwen3_5ForCausalLM` loading (SemIf's own loader) rather than the
  generic image-text-to-text wrapper `next_token_logit` uses
- a `prompt_sha256` recorded per score, for audit/reproducibility

`methods.next_token_logit` (independent Japanese-natural-language prompt,
reported separately as "semif-ja" alongside this method) is left
unchanged; this is a separate method, not a replacement. SemIf itself is
not vendored as a dependency (it pins `torch==2.10.0`, which would
conflict with this project's CUDA torch build) — its recipe is reproduced
in code instead. See manifests/upstream.yaml.

Zero-shot and few-shot are both this same method (`SemifLogitScorer`),
selected via the `few_shot_count` constructor argument — see its
docstring. They used to be two separate classes/packages
(`methods.semif_logit` / `methods.semif_logit_fewshot`); merged here since
few-shot is just an optional mode of the same prompt recipe, not a
different decision principle.
"""

from .scorer import SemifLogitScorer

__all__ = ["SemifLogitScorer"]
