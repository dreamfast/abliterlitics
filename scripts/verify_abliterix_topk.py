#!/usr/bin/env python3
"""Empirically confirm why Abliterix's card reports ~0.0006 while we measure 0.0536.

Loads base + abliterix, forwards harmless_alpaca, and computes KL THREE ways on the
same first-token logits:
  full       Heretic/ours: log_softmax over the FULL 262k vocab, batchmean.
  top100     Abliterix's vLLM method replicated: keep only top-100 logprobs, fill the
             rest with -30.0, log_softmax, batchmean. (vllm_backend.py lines 1045-1070)
  top20      vLLM V1 default cap (20) for completeness.
"""
import sys, math, logging
import torch
import torch.nn.functional as F
sys.path.insert(0, "/app")
from src.kl.kl_divergence import SYSTEM_PROMPT
from src.kl.kl_experiment import build_chat_text, load_model_and_tokenizer
from datasets import load_dataset

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def collect(path, tok_path, prompts):
    model, tok, dev = load_model_and_tokenizer(path, tok_path)
    lps = []
    for p in prompts:
        chat = build_chat_text(tok, p, "", enable_thinking=True)
        inp = tok(chat, return_tensors="pt", return_token_type_ids=False).to(dev)
        with torch.inference_mode():
            lg = model(**inp).logits[0, -1, :].float()
        lps.append(lg.cpu())
    del model
    import gc; gc.collect(); torch.cuda.empty_cache()
    return torch.stack(lps)  # [N, vocab]

def kl_full(base_lp, var_lp):
    b = F.log_softmax(base_lp, dim=-1); v = F.log_softmax(var_lp, dim=-1)
    return (b.exp() * (b - v)).sum(-1).mean().item()

def kl_topk(base_lp, var_lp, k, fill=-30.0):
    # Replicate Abliterix: each model keeps its own top-k, rest filled, then log_softmax.
    def trunc(lp):
        vals, idx = lp.topk(k, dim=-1)
        dense = lp.new_full(lp.shape, fill)
        dense.scatter_(-1, idx, vals)
        return F.log_softmax(dense, dim=-1)
    b = trunc(base_lp); v = trunc(var_lp)
    return (b.exp() * (b - v)).sum(-1).mean().item()

if __name__ == "__main__":
    base_path, var_path, tok_path = sys.argv[1], sys.argv[2], sys.argv[3]
    ds = load_dataset("mlabonne/harmless_alpaca", split="test")
    prompts = [(ds[i].get("text") or "").strip() for i in range(100)]
    prompts = [p for p in prompts if p][:100]
    log.info("Base forward...")
    base = collect(base_path, tok_path, prompts)
    log.info("Variant forward...")
    var = collect(var_path, tok_path, prompts)
    print(f"\n=== abliterix vs base, harmless_alpaca test[:100], 1-token ===")
    print(f"  full-vocab (Heretic/ours):   {kl_full(base, var):.6f}")
    print(f"  top-100 (Abliterix vLLM):    {kl_topk(base, var, 100):.6f}")
    print(f"  top-20  (vLLM V1 default):   {kl_topk(base, var, 20):.6f}")
