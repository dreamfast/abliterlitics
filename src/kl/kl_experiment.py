#!/usr/bin/env python3
"""
KL divergence robustness experiment.

PROOF / EXPERIMENT — not part of the production pipeline.
Production KL (src/kl/kl_divergence.py) faithfully matches Heretic v1.2.0:
harmless_alpaca test[:100], 1 token, F.kl_div(batchmean, log_target=True).

This script tests whether the production methodology is robust to:
  1. DATASET choice — replace harmless_alpaca with databricks/databricks-dolly-15k
     (human-written, independent of the Alpaca lineage) to check for heretic-favoritism.
  2. TOKEN COUNT — measure over 32 teacher-forced positions instead of just the
     first token.

Motivation: Heretic optimizes its abliteration against the KL scorer on exactly
harmless_alpaca test[:100]. If heretic-made variants cluster at the low end of
our KL ranking only because they were tuned on those exact prompts, a dataset
swap should shift their rank. If the rank is stable, the methodology is robust
and the low KL is a true result (the abliteration genuinely preserved behavior).

Methodology for the 32-token measurement (teacher forcing):
  - Base model greedily generates a 32-token continuation from each prompt.
  - Both base and variant are then run in a FORWARD PASS on that SAME shared
    continuation (prompt + base_continuation), and we read raw logits at the 32
    prediction positions. This is the only valid way to compare beyond token 1:
    once two models' greedy paths diverge, position-wise comparison is ill-defined
    unless both are scored on an identical prefix.
  - KL is averaged over the 32 positions and N prompts.

Three numbers are produced per variant:
  - alpaca_1t  : existing production number (harmless_alpaca, 1 token, scores) — read
                 from results/kl/kl_<variant>.json, not recomputed here.
  - dolly_1t   : Dolly-15k, 1 token, output_scores  (production method, new dataset
                 — isolates the dataset effect).
  - dolly_32t  : Dolly-15k, 32 tokens, raw-logits forward pass, teacher-forced.

Usage (driven from host via runners/run_kl_experiment.sh):

    # Phase 1: collect base data (continuations + base logprobs)
    python3 kl_experiment.py collect-base \\
        --model /model --tokenizer /tokenizer \\
        --output /results/exp_base.pt --num-prompts 100 --num-tokens 32

    # Phase 2: per variant — compute dolly_1t + dolly_32t vs base
    python3 kl_experiment.py collect-variant \\
        --model /model --tokenizer /tokenizer \\
        --base-data /results/exp_base.pt \\
        --variant-label heretic \\
        --output /results/exp_heretic.json

    # Phase 3: merge + compare against existing production numbers
    python3 kl_experiment.py report \\
        --experiment-dir /results \\
        --existing-kl-dir /work/results/kl \\
        --output /results/comparison.json
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

# Reuse the proven helpers from the production KL module (model loading,
# device placement, response-prefix detection, system prompt).
from src.kl.kl_divergence import (
    SYSTEM_PROMPT,
    _get_model_auto_class,
    detect_response_prefix,
    get_input_device,
    get_loading_config,
    get_model_size_gb,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DOLLY_DATASET = "databricks/databricks-dolly-15k"
DOLLY_SEED = 42  # deterministic shuffle for reproducible prompt selection
GPQA_DATASET = "Idavidrein/gpqa"
GPQA_CONFIG = "gpqa_diamond"
NO_ROBOTS_DATASET = "HuggingFaceH4/no_robots"
GSM8K_DATASET = "openai/gsm8k"
TRUTHFULQA_DATASET = "truthfulqa/truthful_qa"  # old repo id; matches offline cache
MMLU_PRO_DATASET = "TIGER-Lab/MMLU-Pro"
GOOD1000_DATASET = "wangzhang/abliterix-datasets"
GOOD1000_CONFIG = "good_1000"

# Map prompt-source tag -> human-readable dataset id, for metadata + logging.
DATASET_NAMES = {
    "dolly": DOLLY_DATASET,
    "gpqa": f"{GPQA_DATASET} ({GPQA_CONFIG})",
    "no_robots": NO_ROBOTS_DATASET,
    "gsm8k": GSM8K_DATASET,
    "truthfulqa": TRUTHFULQA_DATASET,
    "mmlu_pro": MMLU_PRO_DATASET,
    "good1000": f"{GOOD1000_DATASET} ({GOOD1000_CONFIG}, held-out 400:500)",
}


def load_dolly_prompts(num_prompts: int) -> list[str]:
    """Load harmless instruction prompts from Dolly-15k (Alpaca-independent).

    Dolly-15k was written by Databricks staff — it does NOT descend from the
    Alpaca / self-instruct lineage that harmless_alpaca comes from. Uses the
    ``instruction`` column, prepending ``context`` when present.
    """
    from datasets import load_dataset

    ds = load_dataset(DOLLY_DATASET, split="train").shuffle(seed=DOLLY_SEED)
    selected = ds.select(range(min(num_prompts, len(ds))))
    prompts: list[str] = []
    for row in selected:
        instr = (row.get("instruction") or "").strip()
        ctx = (row.get("context") or "").strip()
        if ctx:
            text = f"{ctx}\n\n{instr}"
        else:
            text = instr
        if text:
            prompts.append(text)
    if len(prompts) < num_prompts:
        log.warning("Requested %d prompts, only %d non-empty available", num_prompts, len(prompts))
    return prompts[:num_prompts]


def load_gpqa_prompts(num_prompts: int) -> list[str]:
    """Load GPQA Diamond prompts (PhD-level science multiple-choice questions).

    Maximally independent of the harmless-instruction distribution: these are
    hard exam questions, not instructions. Answer choices are shuffled with a
    deterministic per-row seed (order is irrelevant for KL since both models see
    the identical prompt). Loaded offline from the lm-eval datasets cache.
    """
    import random

    from datasets import load_dataset

    ds = load_dataset(GPQA_DATASET, GPQA_CONFIG, split="train")
    n = min(num_prompts, len(ds))
    prompts: list[str] = []
    labels = ["(A)", "(B)", "(C)", "(D)"]
    for i in range(n):
        row = ds[i]
        q = (row.get("Question") or "").strip()
        if not q:
            continue
        opts = [
            row.get("Correct Answer", ""),
            row.get("Incorrect Answer 1", ""),
            row.get("Incorrect Answer 2", ""),
            row.get("Incorrect Answer 3", ""),
        ]
        opts = [o.strip() for o in opts if o and o.strip()]
        random.Random(i).shuffle(opts)  # deterministic per-row
        choices = "\n".join(f"{labels[j]} {opts[j]}" for j in range(len(opts)))
        prompts.append(f"Question: {q}\n\n{choices}\n\nAnswer:")
    return prompts[:n]


def load_no_robots_prompts(num_prompts: int) -> list[str]:
    """Load human-written prompts from HuggingFaceH4/no_robots.

    no_robots is a creative-commons licensed set of ~10k prompts authored by
    paid human raters across 10 categories. Deliberately non-templated, it sits
    far from the Alpaca/self-instruct lineage. Uses the ``prompt`` column.
    """
    from datasets import load_dataset

    ds = load_dataset(NO_ROBOTS_DATASET, split="train").shuffle(seed=DOLLY_SEED)
    selected = ds.select(range(min(num_prompts, len(ds))))
    prompts = [(row.get("prompt") or "").strip() for row in selected]
    prompts = [p for p in prompts if p]
    if len(prompts) < num_prompts:
        log.warning("Requested %d prompts, only %d non-empty available", num_prompts, len(prompts))
    return prompts[:num_prompts]


def load_gsm8k_prompts(num_prompts: int) -> list[str]:
    """Load grade-school math word problems from openai/gsm8k (test split).

    A clean harmless capability probe that is distributionally unlike any
    instruction set: short numeric word problems. Uses the ``question`` column.
    """
    from datasets import load_dataset

    ds = load_dataset(GSM8K_DATASET, "main", split="test")
    n = min(num_prompts, len(ds))
    prompts = [(ds[i].get("question") or "").strip() for i in range(n)]
    prompts = [p for p in prompts if p]
    return prompts[:n]


def load_truthfulqa_prompts(num_prompts: int) -> list[str]:
    """Load questions from truthful_qa/truthful_qa (multiple_choice, validation).

    Factual questions designed to expose common misconceptions. A harmless
    knowledge probe independent of the instruction distribution. Uses the
    ``question`` column only (choices omitted; the question alone is the prompt).
    """
    from datasets import load_dataset

    ds = load_dataset(TRUTHFULQA_DATASET, "multiple_choice", split="validation")
    n = min(num_prompts, len(ds))
    prompts = [(ds[i].get("question") or "").strip() for i in range(n)]
    prompts = [p for p in prompts if p]
    return prompts[:n]


def load_mmlu_pro_prompts(num_prompts: int) -> list[str]:
    """Load multiple-choice questions from TIGER-Lab/MMLU-Pro (test split).

    14-discipline graduate-level knowledge MC. Formatted identically to the
    GPQA loader so both models see the same A/B/C/D/... framing. Answer position
    is irrelevant for KL. Deterministic per-row shuffle of the options.
    """
    import random

    from datasets import load_dataset

    ds = load_dataset(MMLU_PRO_DATASET, split="test")
    n = min(num_prompts, len(ds))
    prompts: list[str] = []
    for i in range(n):
        row = ds[i]
        q = (row.get("question") or "").strip()
        if not q:
            continue
        opts = [o for o in (row.get("options") or []) if o and str(o).strip()]
        opts = [str(o).strip() for o in opts]
        if not opts:
            continue
        random.Random(i).shuffle(opts)  # deterministic per-row
        labels = [f"({chr(65 + j)})" for j in range(len(opts))]
        choices = "\n".join(f"{labels[j]} {opts[j]}" for j in range(len(opts)))
        prompts.append(f"Question: {q}\n\n{choices}\n\nAnswer:")
    return prompts[:n]


def load_good1000_prompts(num_prompts: int) -> list[str]:
    """Load Abliterix's curated bilingual benign eval set (held-out slice 400:500).

    good_1000 is Abliterix's own harmless prompt set, bilingual EN+ZH, designed as
    matched benign counterexamples. Abliterix optimises on train[:400] and evaluates
    KL on the held-out train[400:500]. Replicating that slice here.
    """
    from datasets import load_dataset

    ds = load_dataset(GOOD1000_DATASET, GOOD1000_CONFIG, split="train")
    n = min(num_prompts, 100)
    prompts = [(ds[400 + i].get("prompt") or "").strip() for i in range(n)]
    prompts = [p for p in prompts if p]
    return prompts[:n]


def load_prompts(source: str, num_prompts: int) -> list[str]:
    if source == "gpqa":
        return load_gpqa_prompts(num_prompts)
    if source == "no_robots":
        return load_no_robots_prompts(num_prompts)
    if source == "gsm8k":
        return load_gsm8k_prompts(num_prompts)
    if source == "truthfulqa":
        return load_truthfulqa_prompts(num_prompts)
    if source == "mmlu_pro":
        return load_mmlu_pro_prompts(num_prompts)
    if source == "good1000":
        return load_good1000_prompts(num_prompts)
    return load_dolly_prompts(num_prompts)


# ---------------------------------------------------------------------------
# Shared model loader
# ---------------------------------------------------------------------------


def load_model_and_tokenizer(model_path: str, tokenizer_path: str):
    """Load model + tokenizer with the same strategy as production KL."""
    from transformers import AutoTokenizer

    model_size = get_model_size_gb(model_path)
    load_cfg = get_loading_config(model_path, model_size)
    load_cfg["torch_dtype"] = torch.bfloat16

    log.info("Loading tokenizer: %s", tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    AutoModelClass = _get_model_auto_class(model_path)
    log.info("Loading model: %s (%.1f GB)", model_path, model_size)
    model = AutoModelClass.from_pretrained(model_path, trust_remote_code=True, **load_cfg)
    model.eval()

    # Sanity: no meta params
    meta = sum(1 for _, p in model.named_parameters() if p.device.type == "meta")
    if meta:
        log.error("ABORTING: %d params still on meta device — model did not load", meta)
        raise SystemExit(1)

    input_device = get_input_device(model)
    log.info("Input device: %s", input_device)
    return model, tokenizer, input_device


def build_chat_text(tokenizer, prompt_text: str, response_prefix: str, enable_thinking: bool = True) -> str:
    """Apply chat template (system + user) and append the response prefix."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt_text},
    ]
    chat_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, enable_thinking=enable_thinking
    )
    if response_prefix:
        chat_text += response_prefix
    return chat_text


# ---------------------------------------------------------------------------
# Phase 1: collect base
# ---------------------------------------------------------------------------


def cmd_collect_base(args):
    model, tokenizer, input_device = load_model_and_tokenizer(args.model, args.tokenizer)

    enable_thinking = (args.thinking == "enabled")
    log.info("Thinking: %s", "enabled" if enable_thinking else "DISABLED")

    prompts = load_prompts(args.prompt_source, args.num_prompts)
    log.info("Loaded %d prompts from %s", len(prompts), args.prompt_source)

    # Auto-detect response prefix on the base model (same as production).
    # With thinking disabled, responses are direct answers (diverse) -> no common prefix.
    response_prefix = ""
    if args.response_prefix == "auto":
        if enable_thinking:
            response_prefix = detect_response_prefix(
                model, tokenizer, prompts, SYSTEM_PROMPT, input_device
            )
        else:
            log.info("  Thinking disabled -> skipping prefix detection (response_prefix='')")
    elif args.response_prefix not in ("none", ""):
        response_prefix = args.response_prefix
    log.info("Response prefix: %r", response_prefix)

    N = len(prompts)
    prompt_ids_list: list[torch.Tensor] = []
    # Variable-length continuations: greedy generation may stop early at EOS,
    # which is the model's NATURAL path — we score that rather than forcing a
    # fixed 32 tokens. Each entry has length t_i in [1, num_tokens].
    continuation_ids_list: list[torch.Tensor] = []
    cont_lengths: list[int] = []
    # 1-token logprobs via output_scores (production-faithful method).
    base_1t = []
    # per-prompt raw-logits forward pass (teacher-forced scoring). [t_i, vocab] each.
    base_32t = []

    for i, ptext in enumerate(tqdm(prompts, desc="Collecting base")):
        chat_text = build_chat_text(tokenizer, ptext, response_prefix, enable_thinking=enable_thinking)
        inputs = tokenizer(chat_text, return_tensors="pt", return_token_type_ids=False).to(
            input_device
        )
        prompt_ids = inputs["input_ids"][0]
        L = prompt_ids.shape[0]
        prompt_ids_list.append(prompt_ids.cpu())

        # --- 1-token: production method (output_scores) ---
        with torch.no_grad():
            out1 = model.generate(
                **inputs,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
                max_new_tokens=1,
                output_scores=True,
                return_dict_in_generate=True,
            )
        base_1t.append(F.log_softmax(out1.scores[0].float(), dim=-1)[0].cpu())

        # --- greedy continuation (token IDs only); may stop early at EOS ---
        with torch.no_grad():
            outg = model.generate(
                **inputs,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
                max_new_tokens=args.num_tokens,
            )
        cont = outg[0, L : L + args.num_tokens]  # on input_device; length t_i <= num_tokens
        t_i = cont.shape[0]
        continuation_ids_list.append(cont.cpu())
        cont_lengths.append(t_i)

        # --- raw-logits forward pass on prompt + continuation (teacher-forced) ---
        full = torch.cat([prompt_ids, cont], dim=0).unsqueeze(0)
        attn = torch.ones_like(full)
        with torch.no_grad():
            logits = model(input_ids=full, attention_mask=attn).logits
        cont_logits = logits[0, L - 1 : L - 1 + t_i, :].float().cpu()  # [t_i, vocab]
        base_32t.append(F.log_softmax(cont_logits, dim=-1))

        if i == 0:
            fin = torch.isfinite(base_32t[0]).float().mean().item() * 100
            log.info("Prompt 0 sanity: t_0=%d, finite=%.1f%%, vocab=%d", t_i, fin, cont_logits.shape[-1])
            log.info("First continuation: %r", tokenizer.decode(cont, skip_special_tokens=False)[:80])

        del out1, outg, inputs, full, attn, logits, cont_logits
        if (i + 1) % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    base_1t_t = torch.stack(base_1t, dim=0)  # [N, vocab]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "prompts": prompts,
            "prompt_ids": prompt_ids_list,
            "continuation_ids": continuation_ids_list,
            "cont_lengths": cont_lengths,
            "base_1t_logprobs": base_1t_t,
            "base_32t_logprobs": base_32t,
            "response_prefix": response_prefix,
            "enable_thinking": enable_thinking,
            "system_prompt": SYSTEM_PROMPT,
            "num_prompts": N,
            "num_tokens_max": args.num_tokens,
            "prompt_source": args.prompt_source,
            "dataset": DATASET_NAMES.get(args.prompt_source, args.prompt_source),
            "seed": DOLLY_SEED,
        },
        out,
    )
    log.info(
        "Saved base data: %s (1t=%s, 32t=list of %d, avg_t=%.1f, min_t=%d, max_t=%d)",
        out,
        tuple(base_1t_t.shape),
        len(base_32t),
        sum(cont_lengths) / N,
        min(cont_lengths),
        max(cont_lengths),
    )


# ---------------------------------------------------------------------------
# Phase 2: collect variant (compute dolly_1t + dolly_32t)
# ---------------------------------------------------------------------------


def cmd_collect_variant(args):
    model, tokenizer, input_device = load_model_and_tokenizer(args.model, args.tokenizer)

    base = torch.load(args.base_data, map_location="cpu", weights_only=False)
    prompts = base["prompts"]
    prompt_ids_list = base["prompt_ids"]
    continuation_ids_list = base["continuation_ids"]  # list of [t_i]
    cont_lengths = base["cont_lengths"]
    base_1t_lp = base["base_1t_logprobs"]  # [N, vocab]
    base_32t_lp = base["base_32t_logprobs"]  # list of [t_i, vocab]
    response_prefix = base["response_prefix"]
    enable_thinking = base.get("enable_thinking", True)
    base_dataset = base.get("dataset", DOLLY_DATASET)
    N = base["num_prompts"]
    t_max = base["num_tokens_max"]

    log.info("Base data: dataset=%s, thinking=%s, N=%d, t_max=%d, avg_t=%.1f, prefix=%r",
             base_dataset, "enabled" if enable_thinking else "DISABLED", N, t_max,
             sum(cont_lengths) / N, response_prefix)

    var_1t = []   # per-prompt 1-token KL (scores method)
    var_32t = []  # per-prompt mean-over-positions KL (raw forward pass)
    # Per-position KL trajectory accumulators (for the 1/2/4/8/16/32 divergence curve).
    t_cap = int(base.get("num_tokens_max", 32))
    pos_kl_sum = torch.zeros(t_cap)
    pos_count = torch.zeros(t_cap, dtype=torch.long)

    for i, ptext in enumerate(tqdm(prompts, desc=f"Variant {args.variant_label}")):
        # Re-derive prompt_ids from the SAME chat text (deterministic with same tokenizer).
        chat_text = build_chat_text(tokenizer, ptext, response_prefix, enable_thinking=enable_thinking)
        inputs = tokenizer(chat_text, return_tensors="pt", return_token_type_ids=False).to(
            input_device
        )
        prompt_ids = inputs["input_ids"][0]
        L = prompt_ids.shape[0]

        # --- 1-token (scores method, production-faithful) ---
        with torch.no_grad():
            out1 = model.generate(
                **inputs,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
                max_new_tokens=1,
                output_scores=True,
                return_dict_in_generate=True,
            )
        v1t_lp = F.log_softmax(out1.scores[0].float(), dim=-1)[0].cpu()  # [vocab]
        kl_1t_i = (
            F.kl_div(
                v1t_lp.unsqueeze(0),
                base_1t_lp[i].unsqueeze(0),
                reduction="batchmean",
                log_target=True,
            ).item()
        )
        var_1t.append(kl_1t_i)

        # --- variable-length raw-logits forward pass on prompt + base continuation ---
        cont = continuation_ids_list[i].to(input_device)  # [t_i]
        t_i = cont.shape[0]
        full = torch.cat([prompt_ids, cont], dim=0).unsqueeze(0).to(input_device)
        attn = torch.ones_like(full)
        with torch.no_grad():
            logits = model(input_ids=full, attention_mask=attn).logits
        cont_logits = logits[0, L - 1 : L - 1 + t_i, :].float().cpu()  # [t_i, vocab]
        v32t_lp = F.log_softmax(cont_logits, dim=-1)  # [t_i, vocab]
        base_i = base_32t_lp[i].float()  # [t_i, vocab]
        # per-position KL, averaged over t_i valid positions
        kl_per_pos = (
            F.kl_div(v32t_lp, base_i, reduction="none", log_target=True)
            .sum(dim=-1)  # KL per position (sum over vocab) -> [t_i]
        )
        var_32t.append(kl_per_pos.mean().item())
        t_i_k = int(kl_per_pos.shape[0])
        pos_kl_sum[:t_i_k] += kl_per_pos
        pos_count[:t_i_k] += 1

        del out1, inputs, full, attn, logits, cont_logits, v32t_lp
        if (i + 1) % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    kl_1t_batchmean = sum(var_1t) / N
    kl_32t_mean = sum(var_32t) / N
    kl_1t_std = (sum((x - kl_1t_batchmean) ** 2 for x in var_1t) / N) ** 0.5
    kl_32t_std = (sum((x - kl_32t_mean) ** 2 for x in var_32t) / N) ** 0.5

    # Per-position KL trajectory: mean KL at each position across prompts,
    # and cumulative mean over the first N positions (N = 1,2,4,8,16,32).
    pointwise = (pos_kl_sum / pos_count.clamp(min=1)).tolist()
    traj_ns = [n for n in (1, 2, 4, 8, 16, 32) if n <= t_cap]
    cumulative = {str(n): sum(pointwise[:n]) / n for n in traj_ns}

    summary = {
        "variant_label": args.variant_label,
        "dataset": base_dataset,
        "num_prompts": N,
        "avg_continuation_tokens": sum(cont_lengths) / N,
        "dolly_1t_kl": kl_1t_batchmean,
        "dolly_1t_kl_std": kl_1t_std,
        "dolly_32t_kl": kl_32t_mean,
        "dolly_32t_kl_std": kl_32t_std,
        "dolly_1t_per_prompt": var_1t,
        "dolly_32t_per_prompt": var_32t,
        "dolly_32t_per_position": pointwise,
        "dolly_32t_cumulative": cumulative,
        "methodology_dolly_1t": (
            f"F.kl_div(variant, base, batchmean, log_target=True) on {base_dataset} prompts, "
            "1 token via model.generate(output_scores=True) log_softmax — production method."
        ),
        "methodology_dolly_32t": (
            "Mean over the model's natural greedy continuation (t_i tokens, t_i<=32, stops "
            "at EOS) of per-position KL. Base greedily generates the continuation; both "
            "models forward-passed on that shared sequence; raw logits log_softmax at each "
            "valid prediction position; F.kl_div(reduction=none, log_target=True).sum(dim=-1) "
            "per position, averaged over t_i."
        ),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    log.info("=== %s ===", args.variant_label)
    log.info("  dolly_1t  KL: %.5f (std %.5f)", kl_1t_batchmean, kl_1t_std)
    log.info("  dolly_32t KL: %.5f (std %.5f)", kl_32t_mean, kl_32t_std)


# ---------------------------------------------------------------------------
# Phase 3: report
# ---------------------------------------------------------------------------


def _spearman(a: list[float], b: list[float]) -> float:
    """Spearman rank correlation between two equal-length lists."""
    n = len(a)
    if n < 2:
        return float("nan")

    def ranks(xs):
        order = sorted(range(n), key=lambda i: xs[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0  # 1-based average rank
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    ra, rb = ranks(a), ranks(b)
    # Pearson on ranks
    ma = sum(ra) / n
    mb = sum(rb) / n
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    da = (sum((ra[i] - ma) ** 2 for i in range(n))) ** 0.5
    db = (sum((rb[i] - mb) ** 2 for i in range(n))) ** 0.5
    if da == 0 or db == 0:
        return float("nan")
    return num / (da * db)


def cmd_report(args):
    exp_dir = Path(args.experiment_dir)
    existing_dir = Path(args.existing_kl_dir)

    rows = []
    src_label = "src"
    for vf in sorted(exp_dir.glob("exp_*.json")):
        d = json.loads(vf.read_text())
        label = d["variant_label"]
        if src_label == "src":
            ds = d.get("dataset", "")
            if "gpqa" in ds.lower():
                src_label = "gpqa"
            elif "dolly" in ds.lower():
                src_label = "dolly"
            else:
                src_label = ds.split("/")[-1][:12] if ds else "src"
        # existing production (harmless_alpaca, 1 token) number
        alpaca_1t = None
        prod = existing_dir / f"kl_{label}.json"
        if prod.exists():
            alpaca_1t = json.loads(prod.read_text()).get("kl_divergence_batchmean")
        rows.append(
            {
                "variant": label,
                "alpaca_1t": alpaca_1t,
                f"{src_label}_1t": d["dolly_1t_kl"],
                f"{src_label}_32t": d["dolly_32t_kl"],
            }
        )

    # Rank correlations over variants that have all three numbers
    complete = [r for r in rows if r["alpaca_1t"] is not None]
    alpaca = [r["alpaca_1t"] for r in complete]
    d1 = [r[f"{src_label}_1t"] for r in complete]
    d32 = [r[f"{src_label}_32t"] for r in complete]

    report = {
        "prompt_source": src_label,
        "variants": rows,
        "rank_correlations": {
            "n": len(complete),
            f"alpaca_1t_vs_{src_label}_1t": _spearman(alpaca, d1),
            f"alpaca_1t_vs_{src_label}_32t": _spearman(alpaca, d32),
            f"{src_label}_1t_vs_{src_label}_32t": _spearman(d1, d32),
        },
        "interpretation": (
            "Spearman rho near 1.0 => ranking is stable across dataset/token-count changes "
            "=> methodology robust. Low rho => the harmless_alpaca 1-token metric is not "
            "representative and the feedback was justified."
        ),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    log.info("\n=== COMPARISON (sorted by alpaca_1t) ===")
    log.info("%-22s %10s %12s %12s", "variant", "alpaca_1t", f"{src_label}_1t", f"{src_label}_32t")
    for r in sorted(rows, key=lambda x: (x["alpaca_1t"] is None, x["alpaca_1t"] or 0)):
        log.info(
            "%-22s %10s %12.5f %12.5f",
            r["variant"],
            f"{r['alpaca_1t']:.5f}" if r["alpaca_1t"] is not None else "—",
            r[f"{src_label}_1t"],
            r[f"{src_label}_32t"],
        )
    rc = report["rank_correlations"]
    log.info("\nSpearman rank correlations (n=%d):", rc["n"])
    log.info("  alpaca_1t vs %s_1t : %.4f", src_label, rc[f"alpaca_1t_vs_{src_label}_1t"])
    log.info("  alpaca_1t vs %s_32t: %.4f", src_label, rc[f"alpaca_1t_vs_{src_label}_32t"])
    log.info("  %s_1t vs %s_32t   : %.4f", src_label, src_label, rc[f"{src_label}_1t_vs_{src_label}_32t"])
    log.info("Saved: %s", out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="KL robustness experiment (Dolly + 32-token)")
    sub = parser.add_subparsers(dest="command", required=True)

    pb = sub.add_parser("collect-base", help="Collect base model data")
    pb.add_argument("--model", required=True)
    pb.add_argument("--tokenizer", required=True)
    pb.add_argument("--output", required=True)
    pb.add_argument("--num-prompts", type=int, default=100)
    pb.add_argument("--num-tokens", type=int, default=32)
    pb.add_argument("--response-prefix", default="auto")
    pb.add_argument("--prompt-source", default="dolly",
                    choices=["dolly", "gpqa", "no_robots", "gsm8k", "truthfulqa", "mmlu_pro", "good1000"],
                    help="Prompt dataset for the robustness probe")
    pb.add_argument("--thinking", default="enabled", choices=["enabled", "disabled"],
                    help="Enable or disable the model's thinking/reasoning mode (Gemma 4 enable_thinking)")

    pv = sub.add_parser("collect-variant", help="Collect variant + compute KL vs base")
    pv.add_argument("--model", required=True)
    pv.add_argument("--tokenizer", required=True)
    pv.add_argument("--base-data", required=True)
    pv.add_argument("--variant-label", required=True)
    pv.add_argument("--output", required=True)

    pr = sub.add_parser("report", help="Merge results + compare to production numbers")
    pr.add_argument("--experiment-dir", required=True)
    pr.add_argument("--existing-kl-dir", required=True)
    pr.add_argument("--output", required=True)

    args = parser.parse_args()
    if args.command == "collect-base":
        cmd_collect_base(args)
    elif args.command == "collect-variant":
        cmd_collect_variant(args)
    elif args.command == "report":
        cmd_report(args)


if __name__ == "__main__":
    main()
