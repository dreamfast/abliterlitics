#!/usr/bin/env python3
"""Generate the KL-post graphs from the result JSON. Run inside the forensics container
(mount the repo at /app, cwd /app). Outputs PNGs to comparisons/gemma4-e4b/results/kl_post_graphs/.

  dip.png            - KL vs N (1..32), 8 variants, thinking ON (log-log). The boilerplate dip.
  on_vs_off.png      - two panels, thinking ON vs OFF (GPQA). Dip disappears.
  entropy_vs_kl.png  - base first-token entropy vs mean KL, one dot per dataset.
"""
import json, sys, statistics as st
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

BASE = Path("comparisons/gemma4-e4b/results")
OUT = BASE / "kl_post_graphs"
OUT.mkdir(parents=True, exist_ok=True)
EIGHT = ["heretic", "nullpo", "trevorjs", "huihui", "abliterix", "treadon-combo", "bendernina", "obliteratus"]
NS = [1, 2, 4, 8, 16, 32]
# colour by 32-tok rank (cool=surgical, warm=heavy) using a single colormap
CMAP = plt.cm.coolwarm


def load_traj(dirn):
    out = {}
    for v in EIGHT:
        cum = json.loads((BASE / dirn / f"exp_{v}.json").read_text())["dolly_32t_cumulative"]
        out[v] = [cum[str(n)] for n in NS]
    return out


def variant_colors(traj):
    order = sorted(EIGHT, key=lambda v: traj[v][-1])  # by 32-tok
    return {v: CMAP(i / (len(order) - 1)) for i, v in enumerate(order)}


def graph_dip():
    traj = load_traj("kl_experiment_gpqa")
    colors = variant_colors(traj)
    fig, ax = plt.subplots(figsize=(8, 5))
    for v in sorted(EIGHT, key=lambda v: traj[v][-1]):
        ax.plot(NS, traj[v], marker="o", ms=4, color=colors[v], label=v, linewidth=2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(NS); ax.set_xticklabels(NS)
    ax.set_xlabel("generated tokens (N)"); ax.set_ylabel("cumulative mean KL (nats)")
    ax.set_title("The boilerplate dip — KL dips at N=2–4 then climbs (Gemma 4 E4B, thinking on, GPQA)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2, framealpha=0.9)
    fig.tight_layout(); fig.savefig(OUT / "dip.svg"); plt.close(fig)
    print("wrote dip.svg")


def graph_on_vs_off():
    on = load_traj("kl_experiment_gpqa")
    off = load_traj("kl_experiment_gpqa_nothink")
    colors = variant_colors(on)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, traj, title in [(a1, on, "thinking ON"), (a2, off, "thinking OFF")]:
        for v in sorted(EIGHT, key=lambda v: on[v][-1]):
            ax.plot(NS, traj[v], marker="o", ms=4, color=colors[v], label=v, linewidth=2)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks(NS); ax.set_xticklabels(NS)
        ax.set_xlabel("generated tokens (N)"); ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
    a1.set_ylabel("cumulative mean KL (nats)")
    a2.legend(loc="best", fontsize=8, ncol=2, framealpha=0.9)
    fig.suptitle("Thinking on vs off — the dip disappears, the heavy end rises (GPQA)", y=1.02)
    fig.tight_layout(); fig.savefig(OUT / "on_vs_off.svg", bbox_inches="tight"); plt.close(fig)
    print("wrote on_vs_off.svg")


def graph_entropy_vs_kl():
    ha = torch.load(BASE / "kl_methods" / "methods_base.pt", map_location="cpu", weights_only=False)
    pts = [("harmless_alpaca",
            st.mean((-(ha["base_last1_logprobs"].float().exp() * ha["base_last1_logprobs"].float()).sum(-1)).tolist()))]
    specs = [("Dolly", "kl_experiment"), ("GPQA", "kl_experiment_gpqa"), ("GSM8K", "kl_experiment_gsm8k"),
             ("no_robots", "kl_experiment_no_robots"), ("TruthfulQA", "kl_experiment_truthfulqa"),
             ("MMLU-Pro", "kl_experiment_mmlu_pro"), ("good_1000", "kl_experiment_good1000")]
    for label, dirn in specs:
        b = torch.load(BASE / dirn / "exp_base.pt", map_location="cpu", weights_only=False)
        lp = b["base_1t_logprobs"].float()
        pts.append((label, st.mean((-(lp.exp() * lp).sum(-1)).tolist())))
    dirmap = {l: d for l, d in specs}

    def mean_kl(label):
        if label == "harmless_alpaca":
            return st.mean(json.loads((BASE / "kl" / f"kl_{v}.json").read_text())["kl_divergence_batchmean"]
                           for v in EIGHT)
        d = json.loads((BASE / dirmap[label] / "comparison.json").read_text())["variants"]
        k = next(c for c in d[0] if c.endswith("_1t") and c != "alpaca_1t")
        return st.mean(r[k] for r in d)

    fig, ax = plt.subplots(figsize=(8, 5))
    xs = [p[1] for p in pts]; ys = [mean_kl(p[0]) for p in pts]
    ax.scatter(xs, ys, s=70, c="steelblue", edgecolor="black", zorder=3)
    for (label, ent), y in zip(pts, ys):
        ax.annotate(label, (ent, y), textcoords="offset points", xytext=(7, 5), fontsize=9)
    # trend line
    z = np.polyfit(xs, ys, 1); xx = np.linspace(min(xs), max(xs), 50)
    ax.plot(xx, np.polyval(z, xx), "--", color="grey", alpha=0.6, label="linear fit")
    ax.set_xlabel("base model first-token entropy (nats)")
    ax.set_ylabel("mean KL over 8 variants (nats)")
    ax.set_title("Why good_1000 reads low everywhere — absolute KL tracks base entropy")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "entropy_vs_kl.svg"); plt.close(fig)
    print("wrote entropy_vs_kl.svg")


if __name__ == "__main__":
    graph_dip()
    graph_on_vs_off()
    graph_entropy_vs_kl()
    print("done ->", OUT)
