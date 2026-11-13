#!/usr/bin/env python3
"""Generate every KL-post table directly from the result JSON, with winner bolding.

Reads:
  results/kl/kl_*.json                         -> production 24-variant table
  results/kl_experiment*/comparison.json       -> dataset cross-table + rho tables
  results/kl_experiment*/exp_*.json            -> trajectory (dolly_32t_cumulative)
  results/kl_methods/methods_comparison.json   -> tool-method tables

Lower KL is better; higher rho is better. Winners are **bolded**.
"""
import json, glob, os
from pathlib import Path

BASE = Path("comparisons/gemma4-e4b/results")
# Representative 8, in production-KL order.
EIGHT = ["heretic", "nullpo", "trevorjs", "huihui", "abliterix", "treadon-combo", "bendernina", "obliteratus"]


def b(v, winner, fmt="{:.4f}"):
    s = fmt.format(v)
    return f"**{s}**" if winner else s


def production_table():
    rows = []
    for f in sorted((BASE / "kl").glob("kl_*.json")):
        d = json.loads(f.read_text())
        # interpretation is the authoritative rating (production KL pipeline).
        rows.append((d["variant_label"], d["kl_divergence_batchmean"], d.get("interpretation", "")))
    rows.sort(key=lambda r: r[1])
    lowest = min(r[1] for r in rows)
    print("### Production (24 variants)")
    print("| variant | KL | rating |")
    print("|---|---:|---|")
    for label, val, interp in rows:
        print(f"| {label} | {b(val, val == lowest)} | {interp} |")
    print()


def dataset_table():
    # alpaca (production) baseline per variant, then each dataset's 1t column.
    def prod(label):
        f = BASE / "kl" / f"kl_{label}.json"
        return json.loads(f.read_text())["kl_divergence_batchmean"]
    cols = [
        ("kl_experiment", "dolly_1t", "Dolly"),
        ("kl_experiment_gpqa", "gpqa_1t", "GPQA"),
        ("kl_experiment_gsm8k", "gsm8k_1t", "GSM8K"),
        ("kl_experiment_no_robots", "no_robots_1t", "no_robots"),
        ("kl_experiment_truthfulqa", "truthful_qa_1t", "TruthfulQA"),
        ("kl_experiment_mmlu_pro", "mmlu_pro_1t", "MMLU-Pro"),
        ("kl_experiment_good1000", "good_1000_1t", "good_1000"),
    ]
    # Resolve the actual 1t key per dataset (report uses canonical name).
    data = {}
    for dirn, _key, label in cols:
        d = json.loads((BASE / dirn / "comparison.json").read_text())
        rows = d["variants"]
        k = next(c for c in rows[0] if c.endswith("_1t") and c != "alpaca_1t")
        data[label] = {r["variant"]: r[k] for r in rows}
    alpaca = {label: prod(label) for label in EIGHT}
    # Winner per column = lowest KL. alpaca + each dataset is its own column.
    col_min = {"alpaca": min(alpaca.values())}
    for _, _, label in cols:
        col_min[label] = min(data[label].values())
    print("### Dataset cross-table (1-token KL, **best per column**)")
    hdr = "| variant | alpaca | " + " | ".join(c[2] for c in cols) + " |"
    print(hdr)
    print("|---|---:" + "|---:" * len(cols) + "|")
    for v in EIGHT:
        cells = [v, b(alpaca[v], alpaca[v] == col_min["alpaca"])]
        for _, _, label in cols:
            cells.append(b(data[label][v], data[label][v] == col_min[label]))
        print("| " + " | ".join(cells) + " |")
    print()


def rho_tables():
    specs = [
        ("Dolly-15k", "kl_experiment", "dolly"),
        ("GPQA Diamond", "kl_experiment_gpqa", "gpqa"),
        ("GSM8K", "kl_experiment_gsm8k", "gsm8k"),
        ("no_robots", "kl_experiment_no_robots", "no_robots"),
        ("TruthfulQA", "kl_experiment_truthfulqa", "truthfulqa"),
        ("MMLU-Pro", "kl_experiment_mmlu_pro", "mmlu_pro"),
        ("good_1000", "kl_experiment_good1000", "good1000"),
    ]
    rows = []
    for label, dirn, key in specs:
        rc = json.loads((BASE / dirn / "comparison.json").read_text())["rank_correlations"]
        v1 = next(v for k, v in rc.items() if k.startswith("alpaca_1t_vs") and k.endswith("_1t"))
        v32 = next(v for k, v in rc.items() if k.startswith("alpaca_1t_vs") and k.endswith("_32t"))
        rows.append((label, v1, v32))
    import statistics as st
    m1, m32 = st.mean(r[1] for r in rows), st.mean(r[2] for r in rows)
    hi1, hi32 = max(r[1] for r in rows), max(r[2] for r in rows)
    print("### rho vs production")
    print("| dataset | rho at 1 tok | rho at 32 tok |")
    print("|---|---:|---:|")
    for label, v1, v32 in rows:
        print(f"| {label} | {b(v1, v1 == hi1, '{:.3f}')} | {b(v32, v32 == hi32, '{:.3f}')} |")
    print(f"| **mean** | **{m1:.3f}** | **{m32:.3f}** |")
    print(f"| **range** | **{min(r[1] for r in rows):.3f} to {hi1:.3f}** | "
          f"**{min(r[2] for r in rows):.3f} to {hi32:.3f}** |")
    print()


def trajectory_table(title, dirn, order=None):
    rows = []
    for v in EIGHT:
        f = BASE / dirn / f"exp_{v}.json"
        d = json.loads(f.read_text())
        cum = d["dolly_32t_cumulative"]
        rows.append((v, [cum["1"], cum["2"], cum["4"], cum["8"], cum["16"], cum["32"]]))
    if order is None:
        order = [r[0] for r in sorted(rows, key=lambda r: r[1][-1])]  # by this table's 32-tok
    byvar = {r[0]: r[1] for r in rows}
    print(f"### {title}")
    print("| variant | 1 | 2 | 4 | 8 | 16 | 32 |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for v in order:
        vals = byvar[v]
        print("| " + v + " | " + " | ".join(f"{x:.4f}" for x in vals) + " |")
    print()


def entropy_table():
    """Why absolute KL level varies by dataset: it tracks the base model's first-token
    entropy. Low-entropy (confident) datasets cap the KL; high-entropy (open-ended)
    datasets amplify it. Sorted by entropy descending."""
    import statistics as st
    import torch
    # harmless_alpaca base logprobs live in the methods run; others in each exp_base.pt
    ha = torch.load(BASE / "kl_methods" / "methods_base.pt", map_location="cpu", weights_only=False)
    rows = [("harmless_alpaca", st.mean((-(ha["base_last1_logprobs"].float().exp()
                 * ha["base_last1_logprobs"].float()).sum(-1)).tolist()))]
    specs = [
        ("Dolly", "kl_experiment"), ("GPQA", "kl_experiment_gpqa"),
        ("GSM8K", "kl_experiment_gsm8k"), ("no_robots", "kl_experiment_no_robots"),
        ("TruthfulQA", "kl_experiment_truthfulqa"), ("MMLU-Pro", "kl_experiment_mmlu_pro"),
        ("good_1000", "kl_experiment_good1000"),
    ]
    for label, dirn in specs:
        b = torch.load(BASE / dirn / "exp_base.pt", map_location="cpu", weights_only=False)
        lp = b["base_1t_logprobs"].float()
        rows.append((label, st.mean((-(lp.exp() * lp).sum(-1)).tolist())))
    # mean KL per dataset (over 8 variants, 1-token)
    def mean_kl(dirn):
        d = json.loads((BASE / dirn / "comparison.json").read_text())["variants"]
        k = next(c for c in d[0] if c.endswith("_1t") and c != "alpaca_1t")
        return st.mean(r[k] for r in d)
    kl = {"harmless_alpaca": st.mean(
        json.loads((BASE / "kl" / f"kl_{v}.json").read_text())["kl_divergence_batchmean"]
        for v in EIGHT)}
    dirmap = {l: d for l, d in specs}
    for l, _ in rows[1:]:
        kl[l] = mean_kl(dirmap[l])
    rows.sort(key=lambda r: -r[1])
    print("### Base first-token entropy vs mean KL (8 variants)")
    print("| dataset | base entropy (nats) | mean KL |")
    print("|---|---:|---:|")
    for label, ent in rows:
        print(f"| {label} | {ent:.3f} | {kl[label]:.3f} |")
    print()


def cross_dataset_trajectory():
    import statistics as st
    Ns = ["1", "2", "4", "8", "16", "32"]
    specs = [
        ("Dolly", "kl_experiment"), ("GPQA", "kl_experiment_gpqa"),
        ("GSM8K", "kl_experiment_gsm8k"), ("no_robots", "kl_experiment_no_robots"),
        ("TruthfulQA", "kl_experiment_truthfulqa"), ("MMLU-Pro", "kl_experiment_mmlu_pro"),
        ("good_1000", "kl_experiment_good1000"),
    ]
    print("### Trajectory by dataset (mean over 8 variants, **dip bottom** bold)")
    print("| dataset | 1 | 2 | 4 | 8 | 16 | 32 |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for label, dirn in specs:
        means = []
        for n in Ns:
            vals = []
            for f in sorted((BASE / dirn).glob("exp_*.json")):
                cum = json.loads(f.read_text())["dolly_32t_cumulative"]
                vals.append(cum[n])
            means.append(st.mean(vals))
        mn = min(means)
        cells = " | ".join(b(m, m == mn) for m in means)
        print(f"| {label} | {cells} |")
    print()


def appendix_trajectories():
    """Full per-variant trajectory (N=1..32) for every dataset, sorted by production
    KL so rows line up across all tables. For the appendix."""
    # production order
    order = []
    for v in EIGHT:
        f = BASE / "kl" / f"kl_{v}.json"
        order.append((v, json.loads(f.read_text())["kl_divergence_batchmean"]))
    order = [v for v, _ in sorted(order, key=lambda r: r[1])]
    specs = [
        ("Dolly-15k", "kl_experiment"), ("GPQA Diamond", "kl_experiment_gpqa"),
        ("GSM8K", "kl_experiment_gsm8k"), ("no_robots", "kl_experiment_no_robots"),
        ("TruthfulQA", "kl_experiment_truthfulqa"), ("MMLU-Pro", "kl_experiment_mmlu_pro"),
        ("good_1000", "kl_experiment_good1000"),
    ]
    print("### Appendix: full per-variant trajectories (thinking on, by dataset)")
    print("Rows sorted by production KL across all tables.\n")
    for label, dirn in specs:
        print(f"**{label}**")
        print("| variant | 1 | 2 | 4 | 8 | 16 | 32 |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        for v in order:
            cum = json.loads((BASE / dirn / f"exp_{v}.json").read_text())["dolly_32t_cumulative"]
            vals = [cum["1"], cum["2"], cum["4"], cum["8"], cum["16"], cum["32"]]
            print("| " + v + " | " + " | ".join(f"{x:.4f}" for x in vals) + " |")
        print()


def methods_table():
    d = json.loads((BASE / "kl_methods" / "methods_comparison.json").read_text())
    rows = sorted(d["variants"], key=lambda r: r["production_v12"])
    rc = d["rank_correlations"]
    # NOTE: Abliterix's deployed method (vLLM top-100 + good_1000 + 3-token) is NOT
    # reproduced here, so it is omitted. The abliterix_clamped field in the data is the
    # clamp reduction on full-vocab HF logits, which equals Heretic v1.4.0 and is redundant.
    print("### Tool methods (per variant, harmless_alpaca, full-vocab HF logits)")
    print("| variant | production v1.2.0 | Heretic v1.4.0 | Apostate |")
    print("|---|---:|---:|---:|")
    for r in rows:
        print(f"| {r['variant']} | {r['production_v12']:.4f} | {r['heretic_v14']:.4f} | "
              f"{r['apostate_8pos']:.4f} |")
    r14 = rc["production_v1.2.0_vs_heretic_v14"]
    rap = rc["production_v1.2.0_vs_apostate_8pos"]
    print("| **rho vs production** | | " + b(r14, True, "{:.3f}") + " | "
          + b(rap, False, "{:.3f}") + " |")
    print()


if __name__ == "__main__":
    production_table()
    dataset_table()
    rho_tables()
    entropy_table()
    # Both trajectory tables share the thinking-ON 32-tok order so rows line up.
    on_order = []
    for v in EIGHT:
        cum = json.loads((BASE / "kl_experiment_gpqa" / f"exp_{v}.json").read_text())["dolly_32t_cumulative"]
        on_order.append((v, cum["32"]))
    on_order = [v for v, _ in sorted(on_order, key=lambda r: r[1])]
    trajectory_table("Trajectory (thinking ON, GPQA)", "kl_experiment_gpqa", on_order)
    cross_dataset_trajectory()
    trajectory_table("Trajectory (thinking OFF, GPQA)", "kl_experiment_gpqa_nothink", on_order)
    methods_table()
    appendix_trajectories()
