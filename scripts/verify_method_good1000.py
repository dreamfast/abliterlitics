#!/usr/bin/env python3
"""Isolate output_scores (generate) vs forward-pass logits on good_1000 for abliterix.

The proven harness uses generate(output_scores=True); a raw forward pass reads the
last-position logits directly. These agree on harmless_alpaca but the question is
whether they diverge on good_1000 (which would explain the card vs ours gap).
"""

import gc  # noqa: E402  (import order mirrors the container sys.path setup below)
import logging
import statistics as st
import sys
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812  (standard torch convention)

sys.path.insert(0, "/app")
from datasets import load_dataset  # noqa: E402  (needs the /app path insert above)

from src.kl.kl_experiment import build_chat_text, load_model_and_tokenizer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def lp_forward(model: Any, tok: Any, dev: Any, chat: str) -> Any:
    inp = tok(chat, return_tensors="pt", return_token_type_ids=False).to(dev)
    with torch.inference_mode():
        return model(**inp).logits[0, -1, :].float().cpu()


def lp_scores(model: Any, tok: Any, dev: Any, chat: str) -> Any:
    inp = tok(chat, return_tensors="pt", return_token_type_ids=False).to(dev)
    with torch.inference_mode():
        out = model.generate(
            **inp,
            max_new_tokens=1,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tok.pad_token_id,
        )
    return out.scores[0][0].float().cpu()


def kl(b: Any, v: Any) -> Any:
    b = F.log_softmax(b, -1)
    v = F.log_softmax(v, -1)
    return (b.exp() * (b - v)).sum(-1)


base_p, var_p, tok_p = sys.argv[1], sys.argv[2], sys.argv[3]
ds = load_dataset("wangzhang/abliterix-datasets", "good_1000", split="train")
prompts = [(ds[400 + i].get("prompt") or "").strip() for i in range(100)]

m, tok, dev = load_model_and_tokenizer(base_p, tok_p)
b_fwd = [lp_forward(m, tok, dev, build_chat_text(tok, p, "", True)) for p in prompts]
b_sco = [lp_scores(m, tok, dev, build_chat_text(tok, p, "", True)) for p in prompts]
del m

gc.collect()
torch.cuda.empty_cache()
m, tok, dev = load_model_and_tokenizer(var_p, tok_p)
v_fwd = [lp_forward(m, tok, dev, build_chat_text(tok, p, "", True)) for p in prompts]
v_sco = [lp_scores(m, tok, dev, build_chat_text(tok, p, "", True)) for p in prompts]

print("\n=== abliterix vs base, good_1000[400:500], 1-token, NO response prefix ===")
print(f"  forward-pass logits:  mean={st.mean([kl(b, v).item() for b, v in zip(b_fwd, v_fwd, strict=False)]):.6f}")
print(f"  generate output_scores: mean={st.mean([kl(b, v).item() for b, v in zip(b_sco, v_sco, strict=False)]):.6f}")
print("  (card claims 0.0006; production harmless_alpaca is 0.0536)")
# sanity: do the two methods agree per-prompt on a couple prompts?
print("\n  per-prompt forward vs scores (first 3):")
for i in range(3):
    print(f"    p{i}: fwd={kl(b_fwd[i], v_fwd[i]).item():.6f}  sco={kl(b_sco[i], v_sco[i]).item():.6f}")
