#!/usr/bin/env python3
"""Try to reproduce Abliterix's card KL=0.0006 on good_1000[400:500].

Measures abliterix-vs-base on Abliterix's held-out eval slice four ways, to isolate
whether the low number comes from the dataset or the method:
  full-1t     full-vocab 1-token (our Heretic method) on good_1000
  top100-1t   Abliterix top-100 truncation, 1-token, on good_1000
  full-3t     full-vocab 3-token teacher-forced (base greedy cont) on good_1000
  top100-3t   Abliterix top-100 + 3-token teacher-forced (closest to their pipeline)
"""
import sys, math, logging
import torch
import torch.nn.functional as F
sys.path.insert(0, "/app")
from src.kl.kl_divergence import SYSTEM_PROMPT
from src.kl.kl_experiment import build_chat_text, load_model_and_tokenizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)
K = 100  # Abliterix top-k


def load_good1000_eval():
    from datasets import load_dataset
    ds = load_dataset("wangzhang/abliterix-datasets", "good_1000", split="train")
    return [(ds[i].get("prompt") or "").strip() for i in range(400, 500)]


def forward_logits(model, tok, dev, text, n_last=1):
    inp = tok(text, return_tensors="pt", return_token_type_ids=False).to(dev)
    with torch.inference_mode():
        lg = model(**inp).logits[0].float().cpu()
    return lg[-n_last:, :]  # [n_last, vocab]


def gen_continuation(model, tok, dev, text, n=3):
    inp = tok(text, return_tensors="pt", return_token_type_ids=False).to(dev)
    with torch.inference_mode():
        out = model.generate(**inp, max_new_tokens=n, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    return out[0, inp["input_ids"].shape[1]:]  # the n generated ids


def kl_full(b, v):
    b = F.log_softmax(b, dim=-1); v = F.log_softmax(v, dim=-1)
    return (b.exp() * (b - v)).sum(-1)


def kl_topk(b, v, k=K, fill=-30.0):
    def trunc(lp):
        vals, idx = lp.topk(k, dim=-1)
        d = lp.new_full(lp.shape, fill); d.scatter_(-1, idx, vals)
        return F.log_softmax(d, dim=-1)
    b = trunc(b); v = trunc(v)
    return (b.exp() * (b - v)).sum(-1)


def main():
    base_path, var_path, tok_path = sys.argv[1], sys.argv[2], sys.argv[3]
    prompts = load_good1000_eval()
    log.info("Loaded %d good_1000[400:500] prompts", len(prompts))

    # Base: last-1 logits + 3-token continuation + 3-pos logits on prompt+cont
    model, tok, dev = load_model_and_tokenizer(base_path, tok_path)
    base_last1, base_3pos, conts = [], [], []
    for p in prompts:
        chat = build_chat_text(tok, p, "", enable_thinking=True)
        base_last1.append(forward_logits(model, tok, dev, chat, 1)[0])
        c = gen_continuation(model, tok, dev, chat, 3)
        conts.append(c)
        ext = tok.decode(tok(chat, return_tensors="pt")["input_ids"][0].tolist() + c.tolist())
        base_3pos.append(forward_logits(model, tok, dev, ext, 3))
    del model; import gc; gc.collect(); torch.cuda.empty_cache()

    # Variant: same positions
    model, tok, dev = load_model_and_tokenizer(var_path, tok_path)
    f1 = t1 = f3 = t3 = []
    acc = {"full-1t": [], "top100-1t": [], "full-3t": [], "top100-3t": []}
    for i, p in enumerate(prompts):
        chat = build_chat_text(tok, p, "", enable_thinking=True)
        v1 = forward_logits(model, tok, dev, chat, 1)[0]
        ext = tok.decode(tok(chat, return_tensors="pt")["input_ids"][0].tolist() + conts[i].tolist())
        v3 = forward_logits(model, tok, dev, ext, 3)
        b1, b3 = base_last1[i], base_3pos[i]
        acc["full-1t"].append(kl_full(b1, v1).item())
        acc["top100-1t"].append(kl_topk(b1, v1).item())
        acc["full-3t"].append(kl_full(b3, v3).mean().item())
        acc["top100-3t"].append(kl_topk(b3, v3).mean().item())

    print("\n=== abliterix vs base, good_1000[400:500] (Abliterix's eval slice) ===")
    for k in acc:
        import statistics as st
        vals = acc[k]
        print(f"  {k:10}: mean={st.mean(vals):.6f}  median={st.median(vals):.6f}  (card claims 0.0006)")


if __name__ == "__main__":
    main()
