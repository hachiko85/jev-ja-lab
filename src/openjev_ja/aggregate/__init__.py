"""Method-agnostic cross-model / cross-dataset aggregation.

Per JEV_JA_LAB_REFACTOR_GUIDE_v2 section 24: this package reads already
written `summary.json` files (and, going forward, `common.schemas.EvaluationRecord`
streams) — it never imports a method's scorer, tokenizer, or model code.
"""

from .summary import build_evaluation_summary

__all__ = ["build_evaluation_summary"]
