"""Merge scripts/train_decider_ja_qlora.py's LoRA adapter into the base decider checkpoint,
producing an ordinary HF checkpoint directory that DeciderScorer (decider.infer.Decider)
loads like any other local model -- see configs/eval/decider-ja-qlora-series.yaml's
`path: decider-ja-qlora` entry.

Loads the base in full bf16 (not 4-bit: merge_and_unload needs real-valued weights to add
the LoRA delta into), applies the adapter, merges, and saves. Must load the SAME base
commit the adapter was trained against (default: the exact `Mapika/decider-4b` commit
scripts/train_decider_ja_qlora.py resolved when it ran with no revision pinned -- "main" at
training time, since moved on the Hub) or the merged weights are nonsense.

Usage:
    python scripts/merge_decider_ja_qlora.py \
        --adapter results/train-decider-ja-qlora/adapter --out models/decider-ja-qlora
"""
from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Mapika/decider-4b"
BASE_REVISION = "eb5fbdfc9448473ec25e399882912863afbdb70e"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--base-revision", default=BASE_REVISION)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    print(f"[merge] loading base {a.base_model}@{a.base_revision}")
    base = AutoModelForCausalLM.from_pretrained(a.base_model, revision=a.base_revision, dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(a.base_model, revision=a.base_revision)

    print(f"[merge] applying adapter {a.adapter}")
    merged = PeftModel.from_pretrained(base, a.adapter).merge_and_unload()

    print(f"[merge] saving to {a.out}")
    merged.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print("[merge] done")


if __name__ == "__main__":
    main()
