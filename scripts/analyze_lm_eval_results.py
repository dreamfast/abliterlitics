#!/usr/bin/env python3
"""Analyze lm-eval results for the gemma4-12b comparison.
Generates the full forensic report: per-model scores, deltas vs base,
fingerprint classification, GSM8K three-number report, resilience checks.
"""
import json
import os
import sys
import glob
from pathlib import Path

COMPARISON_DIR = Path("comparisons/gemma4-12b")
RESULTS_DIR = COMPARISON_DIR / "results" / "lm_eval"

# Task -> primary metric mapping (the "headline" metric for each task)
TASK_METRICS = {
    "leaderboard_mmlu_pro": ("acc,none", "MMLU-Pro"),
    "leaderboard_gpqa_diamond": ("acc_norm,none", "GPQA Diamond"),
    "leaderboard_bbh": ("acc_norm,none", "BBH"),
    "leaderboard_musr": ("acc_norm,none", "MuSR"),
    "leaderboard_ifeval": ("prompt_level_strict_acc,none", "IFEval"),
    "hellaswag": ("acc_norm,none", "HellaSwag"),
    "winogrande": ("acc,none", "Winogrande"),
    "arc_challenge": ("acc_norm,none", "ARC-Challenge"),
    "piqa": ("acc_norm,none", "PIQA"),
    "lambada_openai": ("perplexity,none", "LAMBADA"),
    "humaneval": ("pass@1,create_test", "HumanEval"),
    "truthfulqa_gen": ("bleu_acc,none", "TruthfulQA-Gen"),
    "gsm8k": ("exact_match,flexible-extract", "GSM8K"),
}

# BBH subtasks of forensic interest
BBH_FORENSIC_SUBS = ["snarks", "temporal_sequences", "date_understanding",
                     "logical_deduction_three_objects", "causal_judgement"]

# Variant metadata from weights forensics (for fingerprint prediction)
VARIANT_INFO = {
    "huihui": "Most surgical (12 tensors, 6 layers, 100% rank-1) — ARA-style",
    "heretic": "Surgical (16 o_proj only, no down_proj) — Heretic-family",
    "coder3101": "86 tensors, 59% rank-1 — Standard Heretic-family",
    "abliterix": "58 tensors, smallest mean edit (1.14%) — Very clean",
    "prithiv": "56 tensors, textbook Heretic — CLEARED of clone suspicion",
    "apostate": "60 tensors, targets mlp.gate/up_proj — Different surface",
    "obliteratus": "144 tensors, 0% rank-1, full-rank noise — Brute-force 'Obliterated'",
    "jwest33": "329 tensors (49%), 73.9% rank-1 — Null-space method TEST",
    "openyourmind": "620/666 tensors, no rank-1 — DISTILLATION hypothesis",
}

def load_results():
    """Load all lm_eval result files."""
    models = {}
    for f in sorted(RESULTS_DIR.glob("lm_eval_*.json")):
        # Extract model name from filename
        name = f.stem.replace("lm_eval_", "")
        with open(f) as fh:
            data = json.load(fh)
        models[name] = data.get("results", {})
    return models

def get_score(results, task, metric):
    """Extract a score from results dict."""
    if task not in results:
        return None
    return results[task].get(metric)

def classify_fingerprint(base_scores, variant_scores):
    """Classify the abliteration fingerprint per AGENTS.md #21."""
    gsm8k_base = get_score(base_scores, "gsm8k", "exact_match,flexible-extract")
    gsm8k_var = get_score(variant_scores, "gsm8k", "exact_match,flexible-extract")
    tqa_base = get_score(base_scores, "truthfulqa_gen", "bleu_acc,none")
    tqa_var = get_score(variant_scores, "truthfulqa_gen", "bleu_acc,none")

    if gsm8k_base is None or gsm8k_var is None or tqa_base is None or tqa_var is None:
        return "INCOMPLETE", {}

    gsm8k_delta = gsm8k_var - gsm8k_base
    tqa_delta = tqa_var - tqa_base

    info = {
        "gsm8k_delta": gsm8k_delta,
        "tqa_delta": tqa_delta,
    }

    # Classification per AGENTS.md #21
    if abs(gsm8k_delta) <= 0.02 and tqa_delta <= -0.03:
        return "ABLITERATION (GSM8K preserved, TQA down)", info
    elif gsm8k_delta <= -0.04 and abs(tqa_delta) <= 0.03:
        return "DISTILLATION (GSM8K damaged, TQA preserved)", info
    elif gsm8k_delta <= -0.04 and tqa_delta <= -0.03:
        return "HEAVY ABLITERATION (both damaged)", info
    elif abs(gsm8k_delta) <= 0.02 and abs(tqa_delta) <= 0.03:
        return "MINIMAL IMPACT (both preserved)", info
    else:
        return f"MIXED (GSM8K {gsm8k_delta:+.3f}, TQA {tqa_delta:+.3f})", info

def analyze_gsm8k_samples(model_name):
    """Three-number GSM8K report (lesson #20)."""
    # Find samples file
    pattern = str(RESULTS_DIR / "__model" / f"*gsm8k*.jsonl")
    # Also check renamed files
    pattern2 = str(RESULTS_DIR / "__model" / f"{model_name}__*gsm8k*.jsonl")
    files = glob.glob(pattern) + glob.glob(pattern2)

    if not files:
        return None

    # Use the file matching this model if possible
    model_files = [f for f in files if model_name in f]
    f = sorted(model_files[-1:] or files)[-1]

    import re
    total = 0
    empty = 0
    seen_targets = set()

    with open(f) as fh:
        for line in fh:
            d = json.loads(line)
            target = d.get("target", "")
            if target in seen_targets:
                continue  # deduplicate (strict/flexible duplicates)
            seen_targets.add(target)
            total += 1
            resp = d.get("resps", d.get("filtered_resps", []))
            if resp and len(resp) > 0:
                text = resp[0][0] if isinstance(resp[0], list) else resp[0]
            else:
                text = ""
            if not text or text.strip() in ("", "None", "null"):
                empty += 1

    return {"total": total, "empty": empty, "empty_pct": 100*empty/total if total else 0}

def main():
    models = load_results()
    if "gemma-4-12B-it" not in models:
        print("ERROR: base model results not found")
        sys.exit(1)

    base = models["gemma-4-12B-it"]
    variants = {k: v for k, v in models.items() if k != "gemma-4-12B-it"}

    print("=" * 100)
    print("LM-EVAL FORENSIC ANALYSIS: gemma4-12b comparison")
    print("=" * 100)

    # Per-model score table
    print(f"\n{'Task':<20}", end="")
    print(f"{'BASE':>10}", end="")
    for name in sorted(variants):
        print(f"{name[:12]:>12}", end="")
    print()
    print("-" * (20 + 10 + 12 * len(variants)))

    for task, (metric, label) in TASK_METRICS.items():
        print(f"{label:<20}", end="")
        base_val = get_score(base, task, metric)
        print(f"{base_val:>10.4f}" if base_val is not None else f"{'MISS':>10}", end="")
        for name in sorted(variants):
            val = get_score(variants[name], task, metric)
            if val is not None and base_val is not None:
                delta = val - base_val
                print(f"{delta:>+11.3f}*" if abs(delta) > 0.02 else f"{val:>12.4f}", end="")
            elif val is not None:
                print(f"{val:>12.4f}", end="")
            else:
                print(f"{'---':>12}", end="")
        print()

    # Fingerprint classification
    print("\n" + "=" * 100)
    print("FORENSIC FINGERPRINT CLASSIFICATION (AGENTS.md #21, #22)")
    print("=" * 100)
    for name in sorted(variants):
        fingerprint, info = classify_fingerprint(base, variants[name])
        print(f"\n{name} ({VARIANT_INFO.get(name, 'unknown')}):")
        print(f"  Fingerprint: {fingerprint}")
        if info:
            print(f"  GSM8K delta: {info.get('gsm8k_delta', 0):+.4f}")
            print(f"  TQA delta:   {info.get('tqa_delta', 0):+.4f}")

    # GSM8K three-number report
    print("\n" + "=" * 100)
    print("GSM8K THREE-NUMBER REPORT (lesson #20)")
    print("=" * 100)
    for name in ["gemma-4-12B-it"] + sorted(variants):
        score = get_score(models.get(name, {}), "gsm8k", "exact_match,flexible-extract")
        gsm8k_info = analyze_gsm8k_samples(name)
        if gsm8k_info:
            print(f"  {name:<20} headline={score:.4f}  empty={gsm8k_info['empty']}/{gsm8k_info['total']} ({gsm8k_info['empty_pct']:.1f}%)")
        else:
            print(f"  {name:<20} headline={score:.4f}  samples=N/A")

    # GPQA / ARC resilience check
    print("\n" + "=" * 100)
    print("GPQA / ARC-CHALLENGE RESILIENCE CHECK (AGENTS.md #22)")
    print("=" * 100)
    gpqa_base = get_score(base, "leaderboard_gpqa_diamond", "acc_norm,none")
    arc_base = get_score(base, "arc_challenge", "acc_norm,none")
    print(f"  Base: GPQA={gpqa_base:.4f}  ARC={arc_base:.4f}")
    for name in sorted(variants):
        gpqa = get_score(variants[name], "leaderboard_gpqa_diamond", "acc_norm,none")
        arc = get_score(variants[name], "arc_challenge", "acc_norm,none")
        if gpqa is not None and arc is not None:
            gpqa_d = gpqa - gpqa_base if gpqa_base else 0
            arc_d = arc - arc_base if arc_base else 0
            ok = "✓" if abs(gpqa_d) <= 0.04 and abs(arc_d) <= 0.04 else "⚠"
            print(f"  {name:<20} GPQA={gpqa:.4f} ({gpqa_d:+.3f})  ARC={arc:.4f} ({arc_d:+.3f})  {ok}")

    # LAMBADA / HumanEval anomaly check
    print("\n" + "=" * 100)
    print("LAMBADA / HUMANEVAL RAW-COMPLETION ANOMALY (journal #72)")
    print("=" * 100)
    for name in ["gemma-4-12B-it"] + sorted(variants):
        lambada = get_score(models.get(name, {}), "lambada_openai", "perplexity,none")
        humaneval = get_score(models.get(name, {}), "humaneval", "pass@1,create_test")
        l_str = f"{lambada:.0f}" if lambada and lambada < 1e6 else f"{lambada:.2e}" if lambada else "---"
        h_str = f"{humaneval:.4f}" if humaneval is not None else "---"
        print(f"  {name:<20} LAMBADA_ppl={l_str:<15}  HumanEval_pass@1={h_str}")

    # BBH subtask breakdown
    print("\n" + "=" * 100)
    print("BBH SUBTASK BREAKDOWN (snarks = abliteration collateral signal)")
    print("=" * 100)
    for sub in BBH_FORENSIC_SUBS:
        task = f"leaderboard_bbh_{sub}"
        metric = "acc_norm,none"
        base_val = get_score(base, task, metric)
        if base_val is None:
            continue
        print(f"  {sub:<30} base={base_val:.4f}", end="")
        for name in sorted(variants):
            val = get_score(variants[name], task, metric)
            if val is not None:
                delta = val - base_val
                marker = "!" if abs(delta) > 0.05 else ""
                print(f"  {name[:8]}={val:.3f}({delta:+.2f}){marker}", end="")
        print()

    # Summary
    print("\n" + "=" * 100)
    print("COMPLETION SUMMARY")
    print("=" * 100)
    completed = [n for n in models if n != "gemma-4-12B-it"]
    print(f"  Base: COMPLETED")
    print(f"  Variants completed: {len(completed)}/9")
    for name in sorted(completed):
        print(f"    ✓ {name}")
    pending = [n for n in VARIANT_INFO if n not in completed]
    if pending:
        print(f"  Pending: {len(pending)}/9")
        for name in sorted(pending):
            print(f"    - {name}")

if __name__ == "__main__":
    main()
