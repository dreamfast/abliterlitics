#!/usr/bin/env python3
"""Generate Qwen3.6-27B GSM8K reasoning efficiency SVGs.

Two charts:
1. Raw vs Adjusted GSM8K scores with invalid response impact
2. Modification breadth vs invalid response rate (reasoning efficiency)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

MODELS = ["Heretic", "HauhauCS", "Huihui", "AEON", "Abliterix"]
COLORS = {
    "Base":     "#95a5a6",
    "Heretic":  "#3498db",
    "HauhauCS": "#e74c3c",
    "Huihui":   "#2ecc71",
    "AEON":     "#f39c12",
    "Abliterix":"#9b59b6",
}

RAW   = {"Base": 34.4, "Heretic": 27.5, "HauhauCS": 51.0, "Huihui": 75.1, "AEON": 51.2, "Abliterix": 37.6}
ADJ   = {"Base": 96.2, "Heretic": 93.8, "HauhauCS": 96.6, "Huihui": 96.0, "AEON": 95.8, "Abliterix": 95.6}
INV   = {"Base": 68.2, "Heretic": 74.5, "HauhauCS": 49.3, "Huihui": 23.0, "AEON": 69.2, "Abliterix": 62.1}

ORDER = ["Base", "Heretic", "HauhauCS", "Huihui", "AEON", "Abliterix"]

def gen_efficiency_chart(out_path):
    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(ORDER))
    width = 0.32

    raw_vals = [RAW[m] for m in ORDER]
    adj_vals = [ADJ[m] for m in ORDER]
    inv_vals = [INV[m] for m in ORDER]

    bars_raw = ax.bar(x - width/2, raw_vals, width, label="Raw GSM8K",
                      color=[COLORS[m] for m in ORDER], alpha=0.55, edgecolor="white")
    bars_adj = ax.bar(x + width/2, adj_vals, width, label="Adjusted (excl. invalid)",
                      color=[COLORS[m] for m in ORDER], alpha=0.90, edgecolor="white")

    for i, m in enumerate(ORDER):
        ax.text(x[i] - width/2, raw_vals[i] + 1.2, f"{raw_vals[i]:.1f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold", alpha=0.7)
        ax.text(x[i] + width/2, adj_vals[i] + 1.2, f"{adj_vals[i]:.1f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    for i, m in enumerate(ORDER):
        mid_y = (raw_vals[i] + adj_vals[i]) / 2
        ax.annotate(f"{inv_vals[i]:.0f}%\ninvalid",
                    xy=(x[i], mid_y), fontsize=8, ha="center", va="center",
                    color="#555", style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(ORDER, fontsize=11)
    ax.set_ylabel("GSM8K Score (%)", fontsize=12)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=11, loc="upper left")
    ax.set_title("Qwen3.6-27B GSM8K: Raw vs Adjusted (Excluding Thinking-Budget Exhaustion)",
                 fontsize=13, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[OK] {out_path}")


def gen_empty_rate_correlation(out_path):
    fig, ax = plt.subplots(figsize=(10, 7))

    for m in ORDER:
        ax.scatter(INV[m], ADJ[m], s=200, color=COLORS[m], zorder=5,
                   edgecolors="white", linewidths=1.5, label=m)
        offset_x = 1.5
        offset_y = 0.3
        if m == "Huihui":
            offset_x = -8
            offset_y = -1.2
        elif m == "HauhauCS":
            offset_x = 1.5
            offset_y = -1.2
        elif m == "Base":
            offset_x = 1.5
            offset_y = 0.8
        elif m == "AEON":
            offset_x = -6
            offset_y = 0.8
        ax.annotate(f"{m}\n({INV[m]:.0f}% invalid, {ADJ[m]:.1f}% adj)",
                    xy=(INV[m], ADJ[m]),
                    xytext=(INV[m] + offset_x, ADJ[m] + offset_y),
                    fontsize=9, ha="left", va="bottom")

    ax.set_xlabel("Invalid Response Rate (%)", fontsize=12)
    ax.set_ylabel("Adjusted GSM8K (excl. invalid) (%)", fontsize=12)
    ax.set_xlim(15, 82)
    ax.set_ylim(92, 98)
    ax.set_title("Qwen3.6-27B: Invalid Rate vs Actual Reasoning Ability",
                 fontsize=13, fontweight="bold")

    ax.axhspan(93, 97, alpha=0.06, color="green")
    ax.text(78, 97.3, "All models 93.8-96.6% adjusted", fontsize=9,
            ha="right", color="green", fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_path, format="svg", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[OK] {out_path}")


if __name__ == "__main__":
    import sys
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    from pathlib import Path
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gen_efficiency_chart(out_dir / "qwen36_27b_gsm8k_efficiency.svg")
    gen_empty_rate_correlation(out_dir / "qwen36_27b_empty_rate_correlation.svg")
    print(f"\nDone. Files in {out_dir}")
