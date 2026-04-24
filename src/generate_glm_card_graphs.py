#!/usr/bin/env python3
"""
GLM-4.7-Flash Model Card Graph Generator

Generates all SVG visualisations for the GLM-4.7-Flash HauhauCS model card
from benchmark results, weight analysis data, and HarmBench scores.

Reads from data directories (mounted read-only inside Docker):
  FORENSICS_DIR/    Forensics results (panel, SVD, edit vectors, etc.)
  WEIGHT_DIR/       Weight analysis results (all variants including abliterix)
  LM_EVAL_DIR/      lm-evaluation-harness benchmark results
  HARMBENCH_DIR/    HarmBench response and score files

Usage:
    docker run --rm --entrypoint python3 \
        -v "$(pwd)/results/glm_47:/data/forensics:ro" \
        -v "$(pwd)/results/glm47-flash/weight:/data/weight:ro" \
        -v "$(pwd)/results/lm_eval:/data/lm_eval:ro" \
        -v "$(pwd)/results/harmbench:/data/harmbench:ro" \
        -v "$(pwd)/graphs/glm47_card:/output" \
        abliterlitics-forensics:1.0.0 /generate_glm_card_graphs.py
"""

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns

try:
    from matplotlib_venn import venn3, venn3_circles
    HAS_VENN = True
except ImportError:
    HAS_VENN = False

# ---------- Palette ----------

COLORS = {
    "base":     "#95a5a6",
    "heretic":  "#3498db",
    "hauhau":   "#e74c3c",
    "huihui":   "#2ecc71",
    "abliterix": "#9b59b6",
}

VARIANT_ORDER = ["heretic", "hauhau", "huihui", "abliterix"]
VARIANT_LABELS = {
    "heretic": "Heretic",
    "hauhau": "HauhauCS",
    "huihui": "Huihui",
    "abliterix": "Abliterix",
}

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

# ---------- Helpers ----------

def load_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"  [WARN] Could not load {path}: {e}")
        return None


def save_fig(fig, out_dir, name):
    path = out_dir / name
    fig.savefig(path, format="svg", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [OK] {name}")


# ---------- Graph 1: Benchmark grouped bar ----------

def gen_benchmark_bars(lm_eval_dir, out_dir):
    """Grouped bar chart: all 5 models across 7 loglikelihood tasks."""
    variants = ["base", "heretic", "hauhau", "huihui", "abliterix"]
    files = {v: f"lm_eval_{v}.json" for v in variants}

    data = {}
    for v, f in files.items():
        d = load_json(lm_eval_dir / f)
        if d:
            data[v] = d

    if len(data) < 3:
        print("  [SKIP] benchmark_bars: not enough lm_eval data")
        return

    tasks = [
        ("MMLU",           "groups",  "mmlu",           "acc,none"),
        ("HellaSwag",      "results", "hellaswag",      "acc_norm,none"),
        ("ARC-Challenge",  "results", "arc_challenge",  "acc_norm,none"),
        ("WinoGrande",     "results", "winogrande",     "acc,none"),
        ("TruthfulQA",     "results", "truthfulqa_mc2", "acc,none"),
        ("PiQA",           "results", "piqa",           "acc_norm,none"),
    ]

    task_names = []
    scores = {v: [] for v in variants}
    for tname, section, task_key, metric in tasks:
        task_names.append(tname)
        for v in variants:
            if v in data:
                try:
                    val = data[v][section][task_key][metric]
                    scores[v].append(val * 100)
                except (KeyError, TypeError):
                    scores[v].append(0)
            else:
                scores[v].append(0)

    # Lambada (perplexity, lower is better, invert for display)
    task_names.append("Lambada")
    for v in variants:
        if v in data:
            try:
                ppl = data[v]["results"]["lambada_openai"]["perplexity,none"]
                scores[v].append(ppl)
            except (KeyError, TypeError):
                scores[v].append(0)
        else:
            scores[v].append(0)

    n_tasks = len(task_names)
    n_vars = len(variants)
    x = np.arange(n_tasks)
    width = 0.15
    offsets = np.arange(n_vars) - (n_vars - 1) / 2

    fig, ax = plt.subplots(figsize=(16, 7))
    for i, v in enumerate(variants):
        bars = ax.bar(x + offsets[i] * width, scores[v], width,
                      label=VARIANT_LABELS.get(v, v.title()),
                      color=COLORS.get(v, "#888"), alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(task_names, fontsize=11)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", fontsize=10)

    # Annotate Lambada as perplexity
    ax.annotate("(ppl, lower is better)", xy=(n_tasks - 1, 0),
                xytext=(n_tasks - 1, -8), ha="center", fontsize=8, color="gray")

    ax.set_title("GLM-4.7-Flash Benchmark Comparison (7 loglikelihood tasks)", fontsize=14, fontweight="bold")
    save_fig(fig, out_dir, "benchmark_comparison.svg")


# ---------- Graph 2: GSM8K reasoning efficiency ----------

def gen_gsm8k_efficiency(out_dir):
    """Grouped bar: GSM8K raw vs adjusted with empty rate."""
    models = ["Base", "Heretic", "HauhauCS", "Huihui", "Abliterix"]
    raw    = [88.40, 89.16, 81.65, 87.57, 47.38]
    adj    = [93.45, 93.75, 92.57, 92.47, 93.30]
    empty  = [5.4,   4.9,   11.8,  5.3,   49.2]
    colors = [COLORS["base"], COLORS["heretic"], COLORS["hauhau"],
              COLORS["huihui"], COLORS["abliterix"]]

    x = np.arange(len(models))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(12, 7))

    bars1 = ax1.bar(x - width/2, raw, width, label="GSM8K Raw",
                    color=[c for c in colors], alpha=0.6, edgecolor="white")
    bars2 = ax1.bar(x + width/2, adj, width, label="GSM8K Adjusted",
                    color=[c for c in colors], alpha=0.9, edgecolor="white")

    # Add empty rate labels on raw bars
    for bar, e in zip(bars1, empty):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{e:.1f}% empty", ha="center", va="bottom", fontsize=8, color="#e74c3c")

    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=12)
    ax1.set_ylabel("GSM8K Score (%)", fontsize=12)
    ax1.set_ylim(0, 105)
    ax1.legend(loc="upper left", fontsize=10)
    ax1.set_title("GLM-4.7-Flash GSM8K: Raw vs Adjusted (Empty Response Impact)",
                  fontsize=14, fontweight="bold")

    save_fig(fig, out_dir, "gsm8k_efficiency.svg")


# ---------- Graph 3: Aggressiveness bar ----------

def gen_aggressiveness(forensics_dir, weight_dir, out_dir):
    """Bar chart: tensors changed per technique."""
    counts = {}

    # 3 variants from forensics panel
    panel = load_json(forensics_dir / "multi_model_panel.json")
    if panel:
        for v in ["heretic", "hauhau", "huihui"]:
            key = f"base->{v}"
            if key in panel.get("pairwise_changed_counts", {}):
                counts[v] = panel["pairwise_changed_counts"][key]

    # Abliterix from fingerprint
    fp = load_json(weight_dir / "fingerprint_abliterix.json")
    if fp:
        counts["abliterix"] = fp["scope"]["changed_tensors"]

    if not counts:
        print("  [SKIP] aggressiveness: no data")
        return

    variants = [v for v in VARIANT_ORDER if v in counts]
    labels = [VARIANT_LABELS[v] for v in variants]
    values = [counts[v] for v in variants]
    colors = [COLORS[v] for v in variants]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color=colors, alpha=0.9, edgecolor="white")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.02,
                f"{val:,}", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylabel("Tensors Changed", fontsize=12)
    ax.set_title("GLM-4.7-Flash Abliteration Aggressiveness", fontsize=14, fontweight="bold")

    # Note HauhauCS effective count
    if "hauhau" in counts and counts["hauhau"] > 5000:
        ax.annotate("Effective: 2,029\n(7,181 near-zero artifacts excluded)",
                    xy=(variants.index("hauhau"), counts["hauhau"]),
                    xytext=(variants.index("hauhau") + 0.5, counts["hauhau"] * 0.85),
                    fontsize=9, color=COLORS["hauhau"],
                    arrowprops=dict(arrowstyle="->", color=COLORS["hauhau"]))

    save_fig(fig, out_dir, "aggressiveness.svg")


# ---------- Graph 4: Tensor type breakdown ----------

def gen_tensor_type_breakdown(forensics_dir, weight_dir, out_dir):
    """Grouped bar: which tensor types each technique modifies."""
    # Hard-coded from the README data for accuracy
    types = ["Expert\ndown_proj", "Attention\n(o_proj)", "Router\n(gate)",
             "Shared\nexpert", "Expert\ngate_proj", "Expert\nup_proj"]

    heretic   = [1792, 34, 0, 0, 0, 0]
    hauhau    = [1984, 45, 0, 0, 1984, 1984]
    huihui    = [3008, 48, 47, 47, 0, 0]
    abliterix = [966,  32, 46, 44, 0, 0]

    n = len(types)
    x = np.arange(n)
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x - 1.5*width, heretic,   width, label="Heretic",   color=COLORS["heretic"],   alpha=0.85)
    ax.bar(x - 0.5*width, hauhau,    width, label="HauhauCS",  color=COLORS["hauhau"],    alpha=0.85)
    ax.bar(x + 0.5*width, huihui,    width, label="Huihui",    color=COLORS["huihui"],    alpha=0.85)
    ax.bar(x + 1.5*width, abliterix, width, label="Abliterix", color=COLORS["abliterix"], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(types, fontsize=10)
    ax.set_ylabel("Tensors Modified", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_title("GLM-4.7-Flash Tensor Type Targeting by Technique",
                 fontsize=14, fontweight="bold")

    save_fig(fig, out_dir, "tensor_type_breakdown.svg")


# ---------- Graph 5: Layer edit norm comparison ----------

def gen_layer_comparison(forensics_dir, weight_dir, out_dir):
    """Overlay line plot: mean edit norm by layer for all 4 variants."""
    layer_data = {}

    for v in ["heretic", "hauhau", "huihui"]:
        d = load_json(forensics_dir / f"layer_analysis_{v}.json")
        if d and "layer_progression" in d:
            layer_data[v] = d["layer_progression"]

    d = load_json(weight_dir / "layer_abliterix.json")
    if d and "layer_progression" in d:
        layer_data["abliterix"] = d["layer_progression"]

    if not layer_data:
        print("  [SKIP] layer_comparison: no data")
        return

    fig, axes = plt.subplots(2, 1, figsize=(16, 12), sharex=True)

    for v in VARIANT_ORDER:
        if v not in layer_data:
            continue
        lp = layer_data[v]
        layers = sorted(lp.keys(), key=lambda x: int(x))
        edit_norms = [lp[l].get("mean_edit_norm", 0) for l in layers]
        rel_edits  = [lp[l].get("mean_relative_edit", 0) * 100 for l in layers]

        axes[0].plot(range(len(layers)), edit_norms,
                     label=VARIANT_LABELS[v], color=COLORS[v], alpha=0.8, linewidth=1.5)
        axes[1].plot(range(len(layers)), rel_edits,
                     label=VARIANT_LABELS[v], color=COLORS[v], alpha=0.8, linewidth=1.5)

    axes[0].set_ylabel("Mean Edit Norm", fontsize=12)
    axes[0].set_title("GLM-4.7-Flash Layer-wise Edit Magnitude", fontsize=14, fontweight="bold")
    axes[0].legend(fontsize=10)

    axes[1].set_ylabel("Mean Relative Edit (%)", fontsize=12)
    axes[1].set_xlabel("Layer", fontsize=12)
    axes[1].set_title("Layer-wise Relative Edit Magnitude", fontsize=14, fontweight="bold")
    axes[1].legend(fontsize=10)

    tick_step = max(1, len(layers) // 16)
    axes[1].set_xticks(range(0, len(layers), tick_step))
    axes[1].set_xticklabels([layers[i] for i in range(0, len(layers), tick_step)])

    fig.tight_layout()
    save_fig(fig, out_dir, "layer_comparison.svg")


# ---------- Graph 6: Expert heatmaps ----------

def gen_expert_heatmaps(forensics_dir, weight_dir, out_dir):
    """Expert x Layer heatmap for each variant."""
    # Load 3-variant data from forensics
    expert_data = load_json(forensics_dir / "expert_analysis.json")
    # Load abliterix data from weight dir
    abliterix_expert = load_json(weight_dir / "expert_analysis_abliterix.json")

    all_per_expert = {}

    if expert_data and "per_expert_details" in expert_data:
        for v, details in expert_data["per_expert_details"].items():
            if details:
                all_per_expert[v] = details

    if abliterix_expert and "per_expert_details" in abliterix_expert:
        for v, details in abliterix_expert["per_expert_details"].items():
            if details:
                all_per_expert[v] = details

    if not all_per_expert:
        print("  [SKIP] expert_heatmaps: no data")
        return

    for variant, details in all_per_expert.items():
        if not details:
            continue

        layers = set()
        experts = set()
        for key, info in details.items():
            layers.add(info["layer"])
            experts.add(info["expert_id"])

        layers = sorted(layers)
        experts = sorted(experts)

        grid = np.zeros((len(experts), len(layers)))
        for key, info in details.items():
            ei = experts.index(info["expert_id"])
            li = layers.index(info["layer"])
            grid[ei, li] = info["total_edit_norm"]

        fig, ax = plt.subplots(figsize=(max(12, len(layers) * 0.3), max(8, len(experts) * 0.15)))

        # Subsample expert labels if too many
        ytick_step = max(1, len(experts) // 16)
        yticklabels = [f"E{e}" if i % ytick_step == 0 else "" for i, e in enumerate(experts)]

        sns.heatmap(grid, ax=ax, cmap="YlOrRd",
                    xticklabels=[f"L{l}" for l in layers],
                    yticklabels=yticklabels,
                    linewidths=0, cbar_kws={"label": "Edit Norm"})
        ax.set_title(f"GLM-4.7-Flash Expert Edit Heatmap ({VARIANT_LABELS.get(variant, variant)})",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Expert ID")

        # Thin out x labels
        xtick_step = max(1, len(layers) // 16)
        for i, label in enumerate(ax.get_xticklabels()):
            if i % xtick_step != 0:
                label.set_visible(False)

        save_fig(fig, out_dir, f"expert_heatmap_{variant}.svg")


# ---------- Graph 7: Cross-technique cosine heatmap ----------

def gen_cosine_heatmap(forensics_dir, weight_dir, out_dir):
    """4x4 heatmap of cross-technique cosine similarities."""
    # Build the matrix from correlation data
    variants = ["heretic", "hauhau", "huihui", "abliterix"]
    labels = [VARIANT_LABELS[v] for v in variants]

    # Source data: pairwise cosines from various files
    cosine_map = {}

    # 3-way from forensics
    tc = load_json(forensics_dir / "technique_correlation.json")
    if tc and "pairwise_cosines" in tc:
        for pair_key, pdata in tc["pairwise_cosines"].items():
            parts = pair_key.split("_vs_")
            if len(parts) == 2:
                cosine_map[(parts[0], parts[1])] = pdata["mean"]
                cosine_map[(parts[1], parts[0])] = pdata["mean"]

    # Abliterix pairs from weight dir
    for other in ["heretic", "hauhau", "huihui"]:
        d = load_json(weight_dir / f"correlation_{other}_vs_abliterix.json")
        if d and "pairwise_cosines" in d:
            for pair_key, pdata in d["pairwise_cosines"].items():
                parts = pair_key.split("_vs_")
                if len(parts) == 2:
                    cosine_map[(parts[0], parts[1])] = pdata["mean"]
                    cosine_map[(parts[1], parts[0])] = pdata["mean"]

    n = len(variants)
    matrix = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i, j] = 1.0
            else:
                matrix[i, j] = cosine_map.get((variants[i], variants[j]), 0.0)

    fig, ax = plt.subplots(figsize=(8, 7))
    mask = np.zeros_like(matrix, dtype=bool)
    # mask upper triangle (mirror)
    # Actually show full matrix for clarity

    sns.heatmap(matrix, ax=ax, annot=True, fmt=".3f", cmap="RdYlGn",
                vmin=0, vmax=0.5, center=0.25,
                xticklabels=labels, yticklabels=labels,
                linewidths=1, linecolor="white",
                cbar_kws={"label": "Mean Cosine Similarity"})
    ax.set_title("GLM-4.7-Flash Cross-Technique Edit Vector Cosine Similarity",
                 fontsize=13, fontweight="bold")

    save_fig(fig, out_dir, "cosine_heatmap.svg")


# ---------- Graph 8: HarmBench ASR by category ----------

def gen_harmbench_bars(harmbench_dir, out_dir):
    """Grouped bar: ASR by category for base vs 4 abliterated."""
    # Hard-coded from README for accuracy and consistency
    categories = [
        ("Chem/Bio", 56),
        ("Copyright", 100),
        ("Cybercrime", 67),
        ("Harassment", 25),
        ("Harmful", 22),
        ("Illegal", 65),
        ("Misinfo", 65),
    ]

    base_asr = [5.4, 97.0, 35.8, 4.0, 4.5, 9.2, 56.9]
    # All abliterated are 100%
    ablit_asr = [100.0] * 7

    cat_labels = [f"{name}\n(n={n})" for name, n in categories]
    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 7))
    bars_base = ax.bar(x - width/2, base_asr, width, label="Base",
                       color=COLORS["base"], alpha=0.85, edgecolor="white")
    bars_ablit = ax.bar(x + width/2, ablit_asr, width, label="Abliterated (all 4)",
                        color=COLORS["hauhau"], alpha=0.85, edgecolor="white")

    # Value labels
    for bar, val in zip(bars_base, base_asr):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9, color=COLORS["base"])

    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels, fontsize=10)
    ax.set_ylabel("Attack Success Rate (%)", fontsize=12)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=11)
    ax.set_title("GLM-4.7-Flash HarmBench ASR by Category\n(All abliterated variants: 100%)",
                 fontsize=14, fontweight="bold")

    # Dashed line at 100%
    ax.axhline(y=100, color=COLORS["hauhau"], linestyle="--", alpha=0.4)

    save_fig(fig, out_dir, "harmbench_asr.svg")


# ---------- Graph 9: GSM8K empty rate correlation ----------

def gen_empty_rate_correlation(out_dir):
    """Scatter/bar: empty rate vs tensor scope."""
    models = ["Heretic", "Huihui", "HauhauCS", "Abliterix"]
    empty_rates = [4.9, 5.3, 11.8, 49.2]
    tensor_types = [3, 3, 8, 4]  # from README
    colors = [COLORS["heretic"], COLORS["huihui"], COLORS["hauhau"], COLORS["abliterix"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(tensor_types, empty_rates,
                         s=[e * 30 + 100 for e in empty_rates],
                         c=colors, alpha=0.8, edgecolors="white", linewidths=2, zorder=5)

    for name, tt, er in zip(models, tensor_types, empty_rates):
        ax.annotate(f"{name}\n{er}% empty", (tt, er),
                    textcoords="offset points", xytext=(15, 5),
                    fontsize=10, fontweight="bold",
                    color=colors[models.index(name)])

    ax.set_xlabel("Tensor Types Modified", fontsize=12)
    ax.set_ylabel("GSM8K Empty Response Rate (%)", fontsize=12)
    ax.set_title("GLM-4.7-Flash: Modification Breadth vs Reasoning Efficiency",
                 fontsize=14, fontweight="bold")
    ax.set_xlim(1, 10)
    ax.set_ylim(0, 55)

    save_fig(fig, out_dir, "empty_rate_correlation.svg")


# ---------- Graph 10: Edit magnitude distribution ----------

def gen_edit_distribution(forensics_dir, weight_dir, out_dir):
    """Violin/box plot: distribution of edit norms per technique."""
    all_norms = {}

    for v in ["heretic", "hauhau", "huihui"]:
        svd = load_json(forensics_dir / f"svd_{v}.json")
        if svd and "tensor_results" in svd:
            norms = []
            for r in svd["tensor_results"]:
                for k, val in r.items():
                    if isinstance(val, dict) and "frobenius_norm" in val:
                        n = val["frobenius_norm"]
                        if n > 0.01:
                            norms.append(n)
            if norms:
                all_norms[v] = norms

    svd = load_json(weight_dir / "svd_abliterix.json")
    if svd and "tensor_results" in svd:
        norms = []
        for r in svd["tensor_results"]:
            for k, val in r.items():
                if isinstance(val, dict) and "frobenius_norm" in val:
                    n = val["frobenius_norm"]
                    if n > 0.01:
                        norms.append(n)
        if norms:
            all_norms["abliterix"] = norms

    if not all_norms:
        print("  [SKIP] edit_distribution: no per-tensor data")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    plot_data = []
    plot_labels = []
    plot_colors = []
    for v in VARIANT_ORDER:
        if v in all_norms:
            plot_data.append(all_norms[v])
            plot_labels.append(VARIANT_LABELS[v])
            plot_colors.append(COLORS[v])

    if not plot_data:
        print("  [SKIP] edit_distribution: no data")
        return

    parts = ax.violinplot(plot_data, showmeans=True, showmedians=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(plot_colors[i])
        pc.set_alpha(0.7)

    ax.set_xticks(range(1, len(plot_labels) + 1))
    ax.set_xticklabels(plot_labels, fontsize=12)
    ax.set_ylabel("Edit Norm", fontsize=12)
    ax.set_title("GLM-4.7-Flash Distribution of Per-Tensor Edit Magnitudes",
                 fontsize=14, fontweight="bold")

    save_fig(fig, out_dir, "edit_distribution.svg")


# ---------- Graph 11: Venn diagram (4-way, shown as UpSet-style) ----------

def gen_venn_4way(forensics_dir, weight_dir, out_dir):
    """Venn diagram for 3-way overlap (heretic, hauhau, huihui) + annotation for abliterix."""
    if not HAS_VENN:
        print("  [SKIP] venn: matplotlib_venn not available")
        return

    panel = load_json(forensics_dir / "multi_model_panel.json")
    if not panel:
        print("  [SKIP] venn: no panel data")
        return

    base_h = set(panel.get("base->heretic_keys", []))
    base_hh = set(panel.get("base->hauhau_keys", []))
    base_hui = set(panel.get("base->huihui_keys", []))

    if not (base_h and base_hh and base_hui):
        print("  [SKIP] venn: missing key sets")
        return

    only_h = len(base_h - base_hh - base_hui)
    only_hh = len(base_hh - base_h - base_hui)
    only_hui = len(base_hui - base_h - base_hh)
    h_hh = len((base_h & base_hh) - base_hui)
    h_hui = len((base_h & base_hui) - base_hh)
    hh_hui = len((base_hh & base_hui) - base_h)
    all3 = len(base_h & base_hh & base_hui)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    v = venn3(subsets=(only_h, only_hh, h_hh, only_hui, h_hui, hh_hui, all3),
              set_labels=("Heretic", "HauhauCS", "Huihui"),
              set_colors=(COLORS["heretic"], COLORS["hauhau"], COLORS["huihui"]),
              alpha=0.6, ax=ax)

    ax.set_title("GLM-4.7-Flash Tensor Edit Overlap (Heretic, HauhauCS, Huihui)\n"
                 "Abliterix: 1,088 tensors (95.8% rank-1, routing control strategy)",
                 fontsize=13, fontweight="bold", pad=20)

    save_fig(fig, out_dir, "venn_overlap.svg")


# ---------- Main ----------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate GLM-4.7-Flash model card SVGs")
    parser.add_argument("--forensics-dir", type=Path, default=Path("/data/forensics"),
                        help="Directory with forensics results (default: /data/forensics)")
    parser.add_argument("--weight-dir", type=Path, default=Path("/data/weight"),
                        help="Directory with weight analysis results (default: /data/weight)")
    parser.add_argument("--lm-eval-dir", type=Path, default=Path("/data/lm_eval"),
                        help="Directory with lm-eval benchmark results (default: /data/lm_eval)")
    parser.add_argument("--harmbench-dir", type=Path, default=Path("/data/harmbench"),
                        help="Directory with HarmBench response files (default: /data/harmbench)")
    parser.add_argument("--output-dir", type=Path, default=Path("/output"),
                        help="Output directory for SVG files (default: /output)")
    args = parser.parse_args()

    forensics_dir = args.forensics_dir
    weight_dir = args.weight_dir
    lm_eval_dir = args.lm_eval_dir
    harmbench_dir = args.harmbench_dir
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GLM-4.7-Flash Model Card Graph Generator")
    print("=" * 60)

    print("\n[1/11] Benchmark comparison bars...")
    gen_benchmark_bars(lm_eval_dir, out_dir)

    print("\n[2/11] GSM8K reasoning efficiency...")
    gen_gsm8k_efficiency(out_dir)

    print("\n[3/11] Aggressiveness bars...")
    gen_aggressiveness(forensics_dir, weight_dir, out_dir)

    print("\n[4/11] Tensor type breakdown...")
    gen_tensor_type_breakdown(forensics_dir, weight_dir, out_dir)

    print("\n[5/11] Layer edit norm comparison...")
    gen_layer_comparison(forensics_dir, weight_dir, out_dir)

    print("\n[6/11] Expert heatmaps...")
    gen_expert_heatmaps(forensics_dir, weight_dir, out_dir)

    print("\n[7/11] Cross-technique cosine heatmap...")
    gen_cosine_heatmap(forensics_dir, weight_dir, out_dir)

    print("\n[8/11] HarmBench ASR bars...")
    gen_harmbench_bars(harmbench_dir, out_dir)

    print("\n[9/11] Empty rate correlation...")
    gen_empty_rate_correlation(out_dir)

    print("\n[10/11] Edit distribution violin...")
    gen_edit_distribution(forensics_dir, weight_dir, out_dir)

    print("\n[11/11] Venn overlap diagram...")
    gen_venn_4way(forensics_dir, weight_dir, out_dir)

    print("\n" + "=" * 60)
    print(f"Done. {len(list(out_dir.glob('*.svg')))} SVGs saved to {out_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
