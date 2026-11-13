#!/usr/bin/env python3
"""
KL measurement METHOD comparison.

Computes KL under three tool methodologies, all from a single forward pass per
model on `mlabonne/harmless_alpaca` test[:100] (the production dataset), and
compares each method's variant ranking against production Heretic v1.2.0.

Methods (all forward-pass based, no `generate`):
  heretic_v14  Heretic v1.4.0/master: raw last-position logits, batchmean KL.
               (v1.2.0 production uses generate(output_scores=True); raw logits
               differ only by logits-processor -inf artefacts. v1.4.0 = current.)
  apostate     heterodoxin/apostate kl_harmless: raw last-8-position logits,
               mean-over-positions KL. Scores the prompt's own tail tokens, no
               generation. (evaluate.py _kl_forward_inputs -> slice(-K, None).)
  abliterix    wuwangzhang1216/abliterix: last-position logits, clamped via
               nan_to_num (nan=-30, posinf=0, neginf=-30), mean-over-dists KL.
               (scorer.py _safe_kl_divergence; 1-token default per gemma4 config.)

All three are forward KL(base || variant), matching production. Production
Heretic v1.2.0 numbers are read from the existing results/kl/*.json as the
reference ranking.

Usage:
    python3 kl_methods.py collect-base --model /models/base --tokenizer /tokenizer \
        --output /results/methods_base.pt
    python3 kl_methods.py collect-variant --model /models/variant --tokenizer /tokenizer \
        --base-data /results/methods_base.pt --variant-label heretic \
        --output /results/methods_heretic.json
    python3 kl_methods.py report --experiment-dir /results \
        --existing-kl-dir /work/results/kl --output /results/methods_comparison.json
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Reuse the proven model-loading + chat-template helpers from the experiment harness.
from src.kl.kl_divergence import SYSTEM_PROMPT
from src.kl.kl_experiment import build_chat_text, load_model_and_tokenizer

log = logging.getLogger(__name__)

HARMLESS_DATASET = "mlabonne/harmless_alpaca"
DEFAULT_SPLIT = "test"
APOSTATE_POSITIONS = 8            # Apostate default kl_positions
NUM_PROMPTS = 100


def load_harmless_alpaca(num_prompts: int) -> list[str]:
    """Load harmless_alpaca test prompts (production dataset), column 'text'."""
    from datasets import load_dataset

    ds = load_dataset(HARMLESS_DATASET, split=DEFAULT_SPLIT)
    prompts = [(ds[i].get("text") or "").strip() for i in range(min(num_prompts, len(ds)))]
    prompts = [p for p in prompts if p]
    return prompts[:num_prompts]


def _kl_per_position(base_lp: torch.Tensor, var_lp: torch.Tensor) -> torch.Tensor:
    """Forward KL(base||variant) per position. Inputs are [P, vocab] log-probs. Returns [P]."""
    return (base_lp.exp() * (base_lp - var_lp)).sum(dim=-1)


def cmd_collect_base(args):
    model, tokenizer, device = load_model_and_tokenizer(args.model, args.tokenizer)
    prompts = load_harmless_alpaca(args.num_prompts)
    log.info("Loaded %d harmless_alpaca prompts", len(prompts))

    last1_list, last8_list = [], []
    for ptext in prompts:
        chat = build_chat_text(tokenizer, ptext, response_prefix="", enable_thinking=True)
        inputs = tokenizer(chat, return_tensors="pt", return_token_type_ids=False).to(device)
        with torch.inference_mode():
            logits = model(**inputs).logits[0].float()  # [L, vocab]
        last1_list.append(F.log_softmax(logits[-1, :], dim=-1).cpu())
        last8_list.append(F.log_softmax(logits[-APOSTATE_POSITIONS:, :], dim=-1).cpu())

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "prompts": prompts,
            "base_last1_logprobs": torch.stack(last1_list),      # [N, vocab]
            "base_last8_logprobs": torch.stack(last8_list),      # [N, 8, vocab]
            "system_prompt": SYSTEM_PROMPT,
            "dataset": HARMLESS_DATASET,
            "num_prompts": len(prompts),
        },
        out,
    )
    log.info("Saved base method logits: %s", out)


def cmd_collect_variant(args):
    base = torch.load(args.base_data, map_location="cpu")
    base_last1 = base["base_last1_logprobs"]      # [N, vocab]
    base_last8 = base["base_last8_logprobs"]      # [N, 8, vocab]
    prompts = base["prompts"]

    model, tokenizer, device = load_model_and_tokenizer(args.model, args.tokenizer)

    # Per-method accumulators (per-prompt KL, then we mean).
    h14, apo, abl = [], [], []
    for i, ptext in enumerate(prompts):
        chat = build_chat_text(tokenizer, ptext, response_prefix="", enable_thinking=True)
        inputs = tokenizer(chat, return_tensors="pt", return_token_type_ids=False).to(device)
        with torch.inference_mode():
            logits = model(**inputs).logits[0].float()
        var_last1 = F.log_softmax(logits[-1, :], dim=-1).cpu()
        var_last8 = F.log_softmax(logits[-APOSTATE_POSITIONS:, :], dim=-1).cpu()

        b1 = base_last1[i]; v1 = var_last1
        b8 = base_last8[i]; v8 = var_last8

        # heretic_v14: 1-token batchmean KL(base||variant)
        h14.append(_kl_per_position(b1.unsqueeze(0), v1.unsqueeze(0)).item())
        # apostate: mean-over-8-positions KL(base||variant)
        apo.append(_kl_per_position(b8, v8).mean().item())
        # abliterix: clamp (nan_to_num) then mean-over-dists KL. On clean logits == h14.
        b1c = F.log_softmax(
            torch.nan_to_num(b1.exp(), nan=math.exp(-30.0), posinf=1.0, neginf=math.exp(-30.0)).log(),
            dim=-1,
        )
        v1c = F.log_softmax(
            torch.nan_to_num(v1.exp(), nan=math.exp(-30.0), posinf=1.0, neginf=math.exp(-30.0)).log(),
            dim=-1,
        )
        abl.append(_kl_per_position(b1c.unsqueeze(0), v1c.unsqueeze(0)).item())

    result = {
        "variant": args.variant_label,
        "dataset": HARMLESS_DATASET,
        "methods": {
            "heretic_v14": {"kl": sum(h14) / len(h14), "per_prompt": h14},
            "apostate_8pos": {"kl": sum(apo) / len(apo), "per_prompt": apo},
            "abliterix_clamped": {"kl": sum(abl) / len(abl), "per_prompt": abl},
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    log.info("Saved %s: h14=%.4f apo=%.4f abl=%.4f",
             args.variant_label, result["methods"]["heretic_v14"]["kl"],
             result["methods"]["apostate_8pos"]["kl"],
             result["methods"]["abliterix_clamped"]["kl"])


def _spearman(a: list[float], b: list[float]) -> float:
    n = len(a)
    ra = {v: i + 1 for i, v in enumerate(sorted(range(n), key=lambda k: a[k]))}
    rb = {v: i + 1 for i, v in enumerate(sorted(range(n), key=lambda k: b[k]))}
    d2 = sum((ra[i] - rb[i]) ** 2 for i in range(n))
    return 1 - 6 * d2 / (n * (n * n - 1))


def cmd_report(args):
    exp_dir = Path(args.experiment_dir)
    # Production Heretic v1.2.0 reference (kl_<variant>.json -> kl_divergence_batchmean).
    prod = {}
    for f in Path(args.existing_kl_dir).glob("kl_*.json"):
        try:
            d = json.loads(f.read_text())
            label = d.get("variant") or f.stem.replace("kl_", "")
            val = d.get("kl_divergence_batchmean") or d.get("kl_batchmean") or d.get("kl")
            if val is not None:
                prod[label] = float(val)
        except Exception:
            continue

    # Variant method results.
    rows = []
    for f in sorted(exp_dir.glob("methods_*.json")):
        d = json.loads(f.read_text())
        label = d["variant"]
        if label not in prod:
            continue
        rows.append({
            "variant": label,
            "production_v12": prod[label],
            **{m: d["methods"][m]["kl"] for m in d["methods"]},
        })

    rows.sort(key=lambda r: r["production_v12"])
    methods = ["heretic_v14", "apostate_8pos", "abliterix_clamped"]
    rhos = {f"production_v1.2.0_vs_{m}": _spearman(
        [r["production_v12"] for r in rows], [r[m] for r in rows]) for m in methods}
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            rhos[f"{methods[i]}_vs_{methods[j]}"] = _spearman(
                [r[methods[i]] for r in rows], [r[methods[j]] for r in rows])

    out = {
        "dataset": HARMLESS_DATASET,
        "reference": "production Heretic v1.2.0 (output_scores, batchmean)",
        "variants": rows,
        "rank_correlations": {"n": len(rows), **rhos},
    }
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(json.dumps(out["rank_correlations"], indent=2))
    print("\nPer-variant (sorted by production v1.2.0):")
    print(f"  {'variant':14}{'prod_v12':>10}{'heretic_v14':>13}{'apostate_8p':>13}{'abliterix':>12}")
    for r in rows:
        print(f"  {r['variant']:14}{r['production_v12']:>10.4f}{r['heretic_v14']:>13.4f}"
              f"{r['apostate_8pos']:>13.4f}{r['abliterix_clamped']:>12.4f}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="KL measurement METHOD comparison")
    sub = parser.add_subparsers(dest="command", required=True)
    pb = sub.add_parser("collect-base")
    pb.add_argument("--model", required=True); pb.add_argument("--tokenizer", required=True)
    pb.add_argument("--output", required=True); pb.add_argument("--num-prompts", type=int, default=NUM_PROMPTS)
    pv = sub.add_parser("collect-variant")
    pv.add_argument("--model", required=True); pv.add_argument("--tokenizer", required=True)
    pv.add_argument("--base-data", required=True); pv.add_argument("--variant-label", required=True)
    pv.add_argument("--output", required=True)
    pr = sub.add_parser("report")
    pr.add_argument("--experiment-dir", required=True); pr.add_argument("--existing-kl-dir", required=True)
    pr.add_argument("--output", required=True)
    args = parser.parse_args()
    {"collect-base": cmd_collect_base, "collect-variant": cmd_collect_variant,
     "report": cmd_report}[args.command](args)


if __name__ == "__main__":
    main()
