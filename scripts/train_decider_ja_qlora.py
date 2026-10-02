"""Continue a decider checkpoint on datasets/for-decider-ja/corpus.pkl via QLoRA.

decider's own published recipe (scripts/train.sh, README "Train your own") is a full-
parameter fine-tune: `python -m decider.train --model ... --data ... --out ...`, reusing
decider.model.DecisionModel (backbone -> last_hidden_state at each answer slot -> restricted
logits over the option-letter rows of the tied lm_head, decider.model.slot_logits) and
decider.train's make_items/batches_by_tokens/loss_fn (proper-scoring-rule CE, optionally
+ Brier). A full bf16 AdamW fine-tune of Mapika/decider-4b (Qwen3.5-4B-Base, 4B params)
needs >32GB (weights + grad + Adam states, all bf16); this machine has one 16GB GPU.

This script keeps decider's own data pipeline, loss, batching and LR schedule verbatim
(imported from the installed `decider` package, not reimplemented) and only swaps the
model construction and optimizer: the backbone loads 4-bit (bitsandbytes NF4, double
quant, bf16 compute dtype), lm_head stays bf16 and frozen (decider's slot_logits reads
its raw weight rows directly as the task head -- it cannot be a quantized/wrapped layer),
and LoRA adapters (peft) are trained on every Linear projection in both layer kinds
Qwen3.5's hybrid architecture mixes in (full_attention: self_attn.{q,k,v,o}_proj; the more
common linear_attention blocks: linear_attn.{in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,
out_proj}; both kinds: mlp.{gate,up,down}_proj). This matches the user's own instinct that
decider's smaller continuation runs are themselves LoRA (docs/CHANGELOG.md: decider-12b v2
merges "a LoRA (rank 32) trained on 6,000 generated state-tracking decisions" on top of a
stock checkpoint) -- same shape of job as this one, scaled down further (QLoRA) to fit the
available GPU.

Usage:
    python scripts/train_decider_ja_qlora.py --model Mapika/decider-4b \
        --data datasets/for-decider-ja/corpus.pkl --out results/train-decider-ja-qlora
"""
from __future__ import annotations

import os

# Must be set before torch/CUDA initializes. A run of this script spiked from 11.8GB to
# 15.5GB (16GB GPU) right after the first periodic eval and never released it -- CUDA
# allocator fragmentation from eval's larger batches outliving their tensors; this is the
# same fix used elsewhere in this project for the identical symptom (methods/clm/scorer.py).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import math
import pickle
import random
import time

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from decider import data as D
from decider.evaluate import aggregate, run_eval
from decider.model import DecisionModel
from decider.prompt import letter_ids
from decider.train import batches_by_tokens, loss_fn, make_items

TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",             # full_attention layers
    "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",  # linear_attention layers
    "gate_proj", "up_proj", "down_proj",                 # mlp, every layer
]


class QLoraDecisionModel(DecisionModel):
    """Same forward path as decider.model.DecisionModel (slot_logits/cap/forward,
    inherited unmodified); only __init__ differs, building a 4-bit-quantized backbone
    with LoRA adapters instead of a plain bf16 full-fine-tune model."""

    def __init__(self, name, *, r=16, alpha=32, dropout=0.05, resume_adapter=None):
        nn.Module.__init__(self)
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(False)
        self.tok = AutoTokenizer.from_pretrained(name)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["lm_head"],
        )
        base = AutoModelForCausalLM.from_pretrained(
            name, quantization_config=bnb_config, dtype=torch.bfloat16
        )
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
        if resume_adapter:
            self.peft_model = PeftModel.from_pretrained(base, resume_adapter, is_trainable=True)
        else:
            lora_config = LoraConfig(
                r=r, lora_alpha=alpha, lora_dropout=dropout,
                target_modules=TARGET_MODULES, bias="none", task_type="CAUSAL_LM",
            )
            self.peft_model = get_peft_model(base, lora_config)
        self.peft_model.print_trainable_parameters()
        self.lm = self.peft_model.base_model.model      # same object: .model / .lm_head intact for slot_logits
        for p in self.lm.lm_head.parameters():           # decider's task head: raw lm_head rows, frozen and bf16
            p.requires_grad_(False)
        self.register_buffer("letters", torch.tensor(letter_ids(self.tok)), persistent=False)

    def save(self, out_dir):
        self.peft_model.save_pretrained(out_dir)
        self.tok.save_pretrained(out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Mapika/decider-4b")
    ap.add_argument("--data", default="datasets/for-decider-ja/corpus.pkl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)        # LoRA tolerates a higher LR than full fine-tune (8e-6 delta)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--max_tokens", type=int, default=4096, help="tokens per micro-batch (B*T)")
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max_ctx", type=int, default=2048)
    ap.add_argument("--none_prob", type=float, default=0.0)   # our criteria lists are fixed meaningful sets; skip "none of the above" aug
    ap.add_argument("--schema_first_prob", type=float, default=0.5)
    ap.add_argument("--max_options", type=int, default=10)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--eval_limit", type=int, default=150)
    ap.add_argument("--eval_max_tokens", type=int, default=4096, help="cap on run_eval's own batch-token budget (default 24576 spiked memory)")
    ap.add_argument("--eval_bs", type=int, default=8)
    ap.add_argument("--save_every", type=int, default=50, help="checkpoint cadence, for --resume")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    logf = open(f"{a.out}/train.log", "a")

    def log(*s):
        msg = " ".join(str(x) for x in s)
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    log("[args]", json.dumps(vars(a)))
    torch.manual_seed(a.seed)

    ckpt_dir = f"{a.out}/checkpoint"
    state_path = f"{ckpt_dir}/trainer_state.pkl"
    resuming = os.path.isdir(ckpt_dir) and os.path.exists(state_path)

    train, evals = D.load_cache(a.data)
    evals_small = {k: v[: a.eval_limit] for k, v in evals.items()}
    model = QLoraDecisionModel(
        a.model, r=a.lora_r, alpha=a.lora_alpha, resume_adapter=ckpt_dir if resuming else None
    ).cuda()
    tok = model.tok
    log(f"[data] train examples {len(train)}; tokenizing...")
    t0 = time.time()
    # Deterministic from (train, seed) alone -- always rebuilt the same way, resume or not;
    # the mid-training shuffle/sampling stream (`rng` below) is what actually gets restored.
    items = make_items(train, tok, random.Random(a.seed), a.max_ctx, a.none_prob, a.max_options, a.schema_first_prob)
    ntok = sum(len(it["ids"]) for it in items)
    nq = sum(len(it["slots"]) for it in items)
    log(f"[data] {len(items)} items, {nq} questions, {ntok/1e6:.1f}M tokens, tokenized in {time.time()-t0:.0f}s")

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=a.lr, betas=(0.9, 0.95))
    steps_per_epoch = math.ceil(len(batches_by_tokens(items, a.max_tokens, random.Random(0))) / a.accum)
    total = int(steps_per_epoch * a.epochs)
    log(f"[sched] {steps_per_epoch} optimizer steps/epoch, {total} total")

    def lr_at(s):
        if s < a.warmup:
            return a.lr * s / a.warmup
        return a.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, (s - a.warmup) / max(1, total - a.warmup))))

    step, micro, ep = 0, 0, 0
    hist = []
    rng = random.Random(a.seed)
    if resuming:
        state = pickle.load(open(state_path, "rb"))
        step, ep, hist = state["step"], state["ep"], state["hist"]
        rng.setstate(state["rng_state"])
        torch.set_rng_state(state["torch_rng_state"])
        torch.cuda.set_rng_state_all(state["cuda_rng_state"])
        opt.load_state_dict(torch.load(f"{ckpt_dir}/optimizer.pt"))
        log(f"[resume] from {ckpt_dir} at step {step}/{total}")
    if step >= total:
        log("[done] already complete at", step)
        return

    def checkpoint():
        os.makedirs(ckpt_dir, exist_ok=True)
        model.save(ckpt_dir)
        torch.save(opt.state_dict(), f"{ckpt_dir}/optimizer.pt")
        pickle.dump(
            dict(step=step, ep=ep, hist=hist, rng_state=rng.getstate(),
                 torch_rng_state=torch.get_rng_state(), cuda_rng_state=torch.cuda.get_rng_state_all()),
            open(state_path, "wb"),
        )
        log(f"[checkpoint] step {step}/{total} -> {ckpt_dir}")

    model.train()
    t0 = time.time()
    ce_acc, n_acc, t_last, tok_acc = 0.0, 0, t0, 0
    while step < total:
        for bidx in batches_by_tokens(items, a.max_tokens, rng):
            if step >= total:
                break
            from decider.model import collate
            b = collate([items[i] for i in bidx], tok.pad_token_id)
            b = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in b.items()}
            logits = model(b)
            loss, ce = loss_fn(logits, b["golds"], b["nopts"])
            (loss / a.accum).backward()
            ce_acc += ce.item(); n_acc += 1; micro += 1; tok_acc += b["input_ids"].numel()
            if micro % a.accum == 0:
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step(); opt.zero_grad(set_to_none=True); step += 1
                if step % 10 == 0:
                    el = time.time() - t0; now = time.time(); tps = tok_acc / max(1e-6, now - t_last)
                    log(f"[train] step {step}/{total} ep {ep} ce {ce_acc/n_acc:.4f} gn {gn:.2f} lr {lr_at(step):.2e} "
                        f"{el/60:.1f}min eta {(total-step)*(now-t_last)/10/60:.0f}min {tps:.0f}tok/s "
                        f"mem {torch.cuda.max_memory_allocated()/1e9:.1f}GB")
                    ce_acc, n_acc = 0.0, 0; t_last = now; tok_acc = 0
                if step % a.eval_every == 0 or step == total:
                    res, _ = run_eval(model, evals_small, log=log, max_tokens=a.eval_max_tokens, bs=a.eval_bs)
                    agg = aggregate(res)
                    log(f"[eval-agg] step {step} " + json.dumps(agg))
                    hist.append(dict(step=step, agg=agg, results=res))
                    json.dump(hist, open(f"{a.out}/hist.json", "w"), indent=1)
                    model.train()
                if step % a.save_every == 0 or step == total:
                    checkpoint()
        ep += 1
    model.save(f"{a.out}/adapter")
    log("[done] saved", f"{a.out}/adapter")


if __name__ == "__main__":
    main()
