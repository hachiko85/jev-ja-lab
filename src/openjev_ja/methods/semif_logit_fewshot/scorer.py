from __future__ import annotations

from openjev_ja.methods.semif_logit.scorer import SemifLogitScorer
from openjev_ja.methods.semif_logit_fewshot.examples import load_few_shot_examples


class SemifLogitFewShotScorer(SemifLogitScorer):
    """`SemifLogitScorer` with in-context examples prepended to every prompt.

    A separate method (and separate `name`) from `methods.semif_logit`, not
    a mode switch on it: adding examples changes results (measured +5 to
    +10 points depending on primitive on a 300-item check), so the
    zero-shot baseline stays reproducible under its own unchanged name
    while this one is free to vary with `few_shot_count`.
    """

    name = "semif-logit-fewshot"

    def __init__(
        self,
        model_name: str,
        *,
        primitive: str,
        datasets_root: str,
        few_shot_count: int = 2,
        seed: int = 42,
        **kwargs: object,
    ) -> None:
        if primitive not in ("noul", "choice", "score"):
            raise ValueError(f"unsupported primitive: {primitive}")
        examples = load_few_shot_examples(primitive, datasets_root, few_shot_count, seed=seed)
        super().__init__(model_name, few_shot=examples, **kwargs)
        self.primitive = primitive
        self.few_shot_count = few_shot_count

    def metadata(self) -> dict[str, object]:
        meta = super().metadata()
        meta["primitive"] = self.primitive
        return meta
