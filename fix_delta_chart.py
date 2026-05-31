#!/usr/bin/env python3
"""Regenerate qwen36_27b_benchmark_delta.svg with correct variant mapping."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

RESULTS_DIR = Path(__file__).parent / "results" / "lm_eval"
OUT_PATH = str(Path(__file__).parent / "qwen36_27b_benchmark_delta_fixed.svg")

VARIANTS = [
    ("heretic",   "Heretic",   "#3498db"),
    ("hauhau",    "HauhauCS",  "#e74c3c"),
    ("huihui",    "Huihui",    "#2ecc71"),
    ("aeon",      "AEON",      "#9b59b6"),
    ("abliterix", "Abliterix", "#1abc9c"),
]

TASKS = [
    ("MMLU",        "results", "mmlu",             "acc,none"),
    ("HellaSwag",   "results", "hellaswag",        "acc_norm,none"),
    ("ARC",         "results", "arc_challenge",    "acc,none"),
    ("WinoGrande",  "results", "winogrande",       "acc,none"),
    ("TruthfulQA",  "results", "truthfulqa_mc2",   "acc,none"),
    ("PiQA",        "results", "piqa",             "acc,none"),
    ("GSM8K",       "results", "gsm8k",            "exact_match,flexible-extract"),
]


def load_results(name):
    path = RESULTS_DIR / f"lm_eval_{name}.json"
    with open(path) as f:
        return json.load(f)


def get_score(data, section, task_key, metric):
    try:
        return data[section][task_key][metric] * 100
    except (KeyError, TypeError):
        return None


base = load_results("base")
base_scores = []
for tname, section, task_key, metric in TASKS:
    base_scores.append(get_score(base, section, task_key, metric))

n_tasks = len(TASKS)
n_vars = len(VARIANTS)
x = np.arange(n_tasks)
width = 0.15
offsets = np.arange(n_vars) - (n_vars - 1) / 2

fig, ax = plt.subplots(figsize=(16, 7))

for i, (vname, vlabel, color) in enumerate(VARIANTS):
    vdata = load_results(vname)
    deltas = []
    for j, (tname, section, task_key, metric) in enumerate(TASKS):
        vs = get_score(vdata, section, task_key, metric)
        d = vs - base_scores[j] if vs is not None and base_scores[j] is not None else 0
        deltas.append(d)
    print(f"{vlabel:10s} deltas: {[f'{d:+.1f}' for d in deltas]}")
    ax.bar(x + offsets[i] * width, deltas, width,
           label=vlabel, color=color, alpha=0.85, edgecolor="white")

ax.set_xticks(x)
ax.set_xticklabels([t[0] for t in TASKS], fontsize=11)
ax.set_ylabel("Delta vs Base (pp)", fontsize=12)
ax.axhline(y=0, color="black", linewidth=0.8)
ax.legend(loc="upper left", fontsize=10)
ax.set_title("Qwen3.6-27B Benchmark Delta vs Base", fontsize=14, fontweight="bold")

fig.savefig(OUT_PATH, format="svg", bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"\nSaved: {OUT_PATH}")
