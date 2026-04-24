#!/usr/bin/env python3
"""
Qwen Model Card Graph Generator

Generates SVG visualisations for all 5 Qwen model families from benchmark
results, weight analysis data, and HarmBench scores. All SVGs are prefixed
with the model label and written to a single output directory.

Model families:
  qwen35_2b, qwen35_4b, qwen35_9b, qwen35_27b, qwen3_4b

Data sources (mounted read-only inside Docker):
  FORENSICS_DIR/qwen35_2b/    fingerprint_*.json, layer_analysis_*.json,
  FORENSICS_DIR/qwen35_4b/    svd_*.json, multi_model_panel.json,
  ... etc                       technique_correlation.json, lm_eval_*.json,
                               kl_*.json
  HARMBENCH_DIR/               qwen*_responses.json (ASR by category)
  HARMBENCH_CLASSIFIED_DIR/    qwen*_classified.json (2B/4B/9B only)

Output: OUTPUT_DIR/{model}_{graph}.svg

Usage:
    docker run --rm --entrypoint python3 \
        -v "$(pwd)/results:/data/forensics:ro" \
        -v "$(pwd)/results/harmbench:/data/harmbench:ro" \
        -v "$(pwd)/results/harmbench_all/classified:/data/harmbench_classified:ro" \
        -v "$(pwd)/graphs/qwen_cards:/output" \
        abliterlitics-forensics:1.0.0 /generate_qwen_card_graphs.py
"""

import json
import sys
from collections import defaultdict
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
    "base":    "#95a5a6",
    "heretic": "#3498db",
    "hauhau":  "#e74c3c",
    "huihui":  "#2ecc71",
}

VARIANTS = ["heretic", "hauhau", "huihui"]
VARIANT_LABELS = {
    "heretic": "Heretic",
    "hauhau":  "HauhauCS",
    "huihui":  "Huihui",
}

# Pretty names for model cards
MODEL_TITLES = {
    "qwen35_2b":  "Qwen3.5-2B",
    "qwen35_4b":  "Qwen3.5-4B",
    "qwen35_9b":  "Qwen3.5-9B",
    "qwen35_27b": "Qwen3.5-27B",
    "qwen3_4b":   "Qwen3-4B",
}

# All models to process
MODELS = ["qwen35_2b", "qwen35_4b", "qwen35_9b", "qwen35_27b", "qwen3_4b"]

# HarmBench semantic categories in display order
HB_CATEGORIES = [
    "chemical_biological",
    "copyright",
    "cybercrime_intrusion",
    "harassment_bullying",
    "harmful",
    "illegal",
    "misinformation_disinformation",
]
HB_CAT_SHORT = {
    "chemical_biological":           "Chem/Bio",
    "copyright":                     "Copyright",
    "cybercrime_intrusion":          "Cybercrime",
    "harassment_bullying":           "Harassment",
    "harmful":                       "Harmful",
    "illegal":                       "Illegal",
    "misinformation_disinformation": "Misinfo",
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


def title(model):
    return MODEL_TITLES.get(model, model)


def prefix(model, graph_name):
    return f"{model}_{graph_name}.svg"


# ---------- Graph 1: Benchmark grouped bar ----------

# Tasks and where to find them in lm_eval JSON
LOGIT_TASKS = [
    ("MMLU",          "groups",  "mmlu",           "acc,none"),
    ("HellaSwag",     "results", "hellaswag",      "acc_norm,none"),
    ("ARC-Challenge",  "results", "arc_challenge",  "acc_norm,none"),
    ("WinoGrande",    "results", "winogrande",     "acc,none"),
    ("TruthfulQA",    "results", "truthfulqa_mc2",  "acc,none"),
    ("PiQA",          "results", "piqa",           "acc_norm,none"),
]


def gen_benchmark_bars(model, forensics_dir, out_dir):
    """Grouped bar chart: 4 variants across 7 tasks."""
    variants = ["base"] + VARIANTS
    data = {}
    model_dir = forensics_dir / model
    for v in variants:
        # Try canonical name first, then raw rerun, then raw
        candidates = [
            model_dir / f"lm_eval_{v}.json",
        ]
        if v != "base":
            # Also try raw rerun files
            import glob as _glob
            raws = sorted(_glob.glob(str(model_dir / f"lm_eval_{v}_rerun_raw_*.json")))
            candidates.extend(raws)
            raws2 = sorted(_glob.glob(str(model_dir / f"lm_eval_{v}_raw_*.json")))
            candidates.extend(raws2)

        for c in candidates:
            d = load_json(c)
            if d and "results" in d:
                data[v] = d
                break

    if len(data) < 3:
        print(f"  [SKIP] {model} benchmark_bars: not enough lm_eval data ({len(data)} files)")
        return

    task_names = []
    scores = {v: [] for v in variants}
    for tname, section, task_key, metric in LOGIT_TASKS:
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

    # Lambada (perplexity)
    task_names.append("Lambada")
    for v in variants:
        if v in data:
            try:
                ppl = data[v]["results"]["lambda_openai"]["perplexity,none"]
            except (KeyError, TypeError):
                try:
                    ppl = data[v]["results"]["lambada_openai"]["perplexity,none"]
                except (KeyError, TypeError):
                    ppl = 0
            scores[v].append(ppl)
        else:
            scores[v].append(0)

    n_tasks = len(task_names)
    n_vars = len(variants)
    x = np.arange(n_tasks)
    width = 0.18
    offsets = np.arange(n_vars) - (n_vars - 1) / 2

    fig, ax = plt.subplots(figsize=(16, 7))
    for i, v in enumerate(variants):
        label = "Base" if v == "base" else VARIANT_LABELS.get(v, v.title())
        ax.bar(x + offsets[i] * width, scores[v], width,
               label=label, color=COLORS.get(v, "#888"), alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(task_names, fontsize=11)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", fontsize=10)

    # Annotate Lambada as perplexity
    ax.annotate("(ppl, lower is better)",
                xy=(n_tasks - 1, 0), xytext=(n_tasks - 1, -8),
                ha="center", fontsize=8, color="gray")

    ax.set_title(f"{title(model)} Benchmark Comparison (7 tasks)",
                 fontsize=14, fontweight="bold")
    save_fig(fig, out_dir, prefix(model, "benchmark_comparison"))


# ---------- Graph 2: KL divergence bar ----------

def gen_kl_bars(model, forensics_dir, out_dir):
    """Bar chart: KL divergence per variant."""
    kl_data = {}
    for v in VARIANTS:
        d = load_json(forensics_dir / model / f"kl_{v}.json")
        if d and "kl_divergence_batchmean" in d:
            kl_data[v] = {
                "kl": d["kl_divergence_batchmean"],
                "interp": d.get("interpretation", ""),
            }

    if not kl_data:
        print(f"  [SKIP] {model} kl_bars: no KL data")
        return

    labels = [VARIANT_LABELS[v] for v in VARIANTS if v in kl_data]
    values = [kl_data[v]["kl"] for v in VARIANTS if v in kl_data]
    colors = [COLORS[v] for v in VARIANTS if v in kl_data]
    interps = [kl_data[v]["interp"] for v in VARIANTS if v in kl_data]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color=colors, alpha=0.9, edgecolor="white")

    for bar, val, interp in zip(bars, values, interps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.03,
                f"{val:.4f}\n({interp})", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("KL Divergence (batchmean)", fontsize=12)
    ax.set_title(f"{title(model)} KL Divergence from Base", fontsize=14, fontweight="bold")

    # Add quality bands
    ymax = max(values) * 1.4
    ax.set_ylim(0, ymax)
    if ymax > 0.1:
        ax.axhspan(0, 0.01, alpha=0.08, color="green", label="Excellent (<0.01)")
        ax.axhspan(0.01, 0.05, alpha=0.05, color="yellow", label="Very good (0.01-0.05)")
        ax.axhspan(0.05, 0.1, alpha=0.05, color="orange", label="Good (0.05-0.1)")
        ax.legend(fontsize=9, loc="upper right")

    save_fig(fig, out_dir, prefix(model, "kl_divergence"))


# ---------- Graph 3: Aggressiveness bar ----------

def gen_aggressiveness(model, forensics_dir, out_dir):
    """Bar chart: tensors changed per technique."""
    counts = {}

    panel = load_json(forensics_dir / model / "multi_model_panel.json")
    if panel:
        for v in VARIANTS:
            key = f"base->{v}"
            if key in panel.get("pairwise_changed_counts", {}):
                counts[v] = panel["pairwise_changed_counts"][key]

    # Fallback: read from fingerprints
    if not counts:
        for v in VARIANTS:
            fp = load_json(forensics_dir / model / f"fingerprint_{v}.json")
            if fp:
                counts[v] = fp.get("scope", {}).get("changed_tensors", 0)

    if not counts:
        print(f"  [SKIP] {model} aggressiveness: no data")
        return

    order = [v for v in VARIANTS if v in counts]
    labels = [VARIANT_LABELS[v] for v in order]
    values = [counts[v] for v in order]
    colors = [COLORS[v] for v in order]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color=colors, alpha=0.9, edgecolor="white")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.02,
                f"{val:,}", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylabel("Tensors Changed", fontsize=12)
    ax.set_title(f"{title(model)} Abliteration Aggressiveness", fontsize=14, fontweight="bold")
    save_fig(fig, out_dir, prefix(model, "aggressiveness"))


# ---------- Graph 4: Tensor type breakdown (dynamic) ----------

def gen_tensor_type_breakdown(model, forensics_dir, out_dir):
    """Grouped bar: which tensor types each technique modifies."""
    # Gather all tensor types across all variants
    all_types = set()
    type_counts = {}

    for v in VARIANTS:
        fp = load_json(forensics_dir / model / f"fingerprint_{v}.json")
        if fp and "targeting" in fp:
            tt = fp["targeting"].get("tensor_types", {})
            type_counts[v] = tt
            all_types.update(tt.keys())

    if not type_counts or not all_types:
        print(f"  [SKIP] {model} tensor_type_breakdown: no data")
        return

    # Sort types by total count across all variants (most edited first)
    type_order = sorted(all_types,
                        key=lambda t: sum(type_counts.get(v, {}).get(t, 0) for v in VARIANTS),
                        reverse=True)

    # Shorten type names for display
    def shorten(t):
        return t.replace(".weight", "").replace("_proj", "").replace("self_attn.", "attn_")

    short_names = [shorten(t) for t in type_order]
    n = len(type_order)
    x = np.arange(n)
    n_vars = len(VARIANTS)
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(12, n * 1.2), 7))

    for i, v in enumerate(VARIANTS):
        vals = [type_counts.get(v, {}).get(t, 0) for t in type_order]
        offset = (i - (n_vars - 1) / 2) * width
        ax.bar(x + offset, vals, width,
               label=VARIANT_LABELS[v], color=COLORS[v], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=9, rotation=25, ha="right")
    ax.set_ylabel("Tensors Modified", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_title(f"{title(model)} Tensor Type Targeting by Technique",
                 fontsize=14, fontweight="bold")

    save_fig(fig, out_dir, prefix(model, "tensor_type_breakdown"))


# ---------- Graph 5: Layer edit norm comparison ----------

def gen_layer_comparison(model, forensics_dir, out_dir):
    """Dual line plot: mean edit norm and relative edit by layer."""
    layer_data = {}

    for v in VARIANTS:
        d = load_json(forensics_dir / model / f"layer_analysis_{v}.json")
        if d and "layer_progression" in d:
            layer_data[v] = d["layer_progression"]

    if not layer_data:
        print(f"  [SKIP] {model} layer_comparison: no data")
        return

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    for v in VARIANTS:
        if v not in layer_data:
            continue
        lp = layer_data[v]
        layers = sorted(lp.keys(), key=lambda k: int(k))
        edit_norms = [lp[l].get("mean_edit_norm", 0) for l in layers]
        rel_edits = [lp[l].get("mean_relative_edit", 0) * 100 for l in layers]

        axes[0].plot(range(len(layers)), edit_norms,
                     label=VARIANT_LABELS[v], color=COLORS[v], alpha=0.8, linewidth=1.5)
        axes[1].plot(range(len(layers)), rel_edits,
                     label=VARIANT_LABELS[v], color=COLORS[v], alpha=0.8, linewidth=1.5)

    axes[0].set_ylabel("Mean Edit Norm", fontsize=12)
    axes[0].set_title(f"{title(model)} Layer-wise Edit Magnitude",
                      fontsize=14, fontweight="bold")
    axes[0].legend(fontsize=10)

    axes[1].set_ylabel("Mean Relative Edit (%)", fontsize=12)
    axes[1].set_xlabel("Layer", fontsize=12)
    axes[1].set_title("Layer-wise Relative Edit Magnitude",
                      fontsize=14, fontweight="bold")
    axes[1].legend(fontsize=10)

    # Get layer count for ticks
    first_lp = next(iter(layer_data.values()))
    all_layers = sorted(first_lp.keys(), key=lambda k: int(k))
    tick_step = max(1, len(all_layers) // 16)
    axes[1].set_xticks(range(0, len(all_layers), tick_step))
    axes[1].set_xticklabels([all_layers[i] for i in range(0, len(all_layers), tick_step)])

    fig.tight_layout()
    save_fig(fig, out_dir, prefix(model, "layer_comparison"))


# ---------- Graph 6: Edit magnitude distribution ----------

def gen_edit_distribution(model, forensics_dir, out_dir):
    """Violin plot: distribution of edit norms per technique."""
    all_norms = {}

    for v in VARIANTS:
        svd = load_json(forensics_dir / model / f"svd_{v}.json")
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

    if not all_norms:
        print(f"  [SKIP] {model} edit_distribution: no per-tensor data")
        return

    plot_data = []
    plot_labels = []
    plot_colors = []
    for v in VARIANTS:
        if v in all_norms:
            plot_data.append(all_norms[v])
            plot_labels.append(VARIANT_LABELS[v])
            plot_colors.append(COLORS[v])

    if not plot_data:
        print(f"  [SKIP] {model} edit_distribution: empty")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    parts = ax.violinplot(plot_data, showmeans=True, showmedians=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(plot_colors[i])
        pc.set_alpha(0.7)

    ax.set_xticks(range(1, len(plot_labels) + 1))
    ax.set_xticklabels(plot_labels, fontsize=12)
    ax.set_ylabel("Edit Norm", fontsize=12)
    ax.set_title(f"{title(model)} Distribution of Per-Tensor Edit Magnitudes",
                 fontsize=14, fontweight="bold")

    save_fig(fig, out_dir, prefix(model, "edit_distribution"))


# ---------- Graph 7: Cross-technique cosine heatmap ----------

def gen_cosine_heatmap(model, forensics_dir, out_dir):
    """3x3 heatmap of cross-technique cosine similarities."""
    variants = VARIANTS
    labels = [VARIANT_LABELS[v] for v in variants]

    tc = load_json(forensics_dir / model / "technique_correlation.json")
    if not tc or "pairwise_cosines" not in tc:
        print(f"  [SKIP] {model} cosine_heatmap: no correlation data")
        return

    cosine_map = {}
    for pair_key, pdata in tc["pairwise_cosines"].items():
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

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(matrix, ax=ax, annot=True, fmt=".3f", cmap="RdYlGn",
                vmin=0, vmax=1.0, center=0.5,
                xticklabels=labels, yticklabels=labels,
                linewidths=1, linecolor="white",
                cbar_kws={"label": "Mean Cosine Similarity"})
    ax.set_title(f"{title(model)} Cross-Technique Edit Vector Cosine Similarity",
                 fontsize=13, fontweight="bold")

    save_fig(fig, out_dir, prefix(model, "cosine_heatmap"))


# ---------- Graph 8: Venn overlap ----------

def gen_venn(model, forensics_dir, out_dir):
    """3-way Venn diagram of tensor edit overlap."""
    if not HAS_VENN:
        print(f"  [SKIP] {model} venn: matplotlib_venn not available")
        return

    panel = load_json(forensics_dir / model / "multi_model_panel.json")
    if not panel:
        print(f"  [SKIP] {model} venn: no panel data")
        return

    base_h = set(panel.get("base->heretic_keys", []))
    base_hh = set(panel.get("base->hauhau_keys", []))
    base_hui = set(panel.get("base->huihui_keys", []))

    if not (base_h and base_hh and base_hui):
        print(f"  [SKIP] {model} venn: missing key sets")
        return

    only_h = len(base_h - base_hh - base_hui)
    only_hh = len(base_hh - base_h - base_hui)
    only_hui = len(base_hui - base_h - base_hh)
    h_hh = len((base_h & base_hh) - base_hui)
    h_hui = len((base_h & base_hui) - base_hh)
    hh_hui = len((base_hh & base_hui) - base_h)
    all3 = len(base_h & base_hh & base_hui)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    venn3(subsets=(only_h, only_hh, h_hh, only_hui, h_hui, hh_hui, all3),
          set_labels=("Heretic", "HauhauCS", "Huihui"),
          set_colors=(COLORS["heretic"], COLORS["hauhau"], COLORS["huihui"]),
          alpha=0.6, ax=ax)

    ax.set_title(f"{title(model)} Tensor Edit Overlap",
                 fontsize=13, fontweight="bold", pad=20)

    save_fig(fig, out_dir, prefix(model, "venn_overlap"))


# ---------- Graph 9: HarmBench ASR by category ----------

def _extract_asr_from_classified(filepath):
    """Extract ASR by category from a classified JSON."""
    d = load_json(filepath)
    if not d:
        return None, None
    items = d.get("harmbench_classified", [])
    total_asr = d.get("complied_count", 0) / max(d.get("total_items", 1), 1) * 100
    cat_data = defaultdict(lambda: {"total": 0, "complied": 0})
    for i in items:
        cat = i.get("semantic_category", "unknown")
        cat_data[cat]["total"] += 1
        if not i.get("is_refusal", True):
            cat_data[cat]["complied"] += 1
    return total_asr, dict(cat_data)


def _extract_asr_from_responses(filepath):
    """Extract ASR by category from a raw responses JSON."""
    d = load_json(filepath)
    if not d:
        return None, None
    items = d.get("harmbench", [])
    refused = sum(1 for i in items if i.get("is_refusal"))
    total_asr = (len(items) - refused) / max(len(items), 1) * 100
    cat_data = defaultdict(lambda: {"total": 0, "complied": 0})
    for i in items:
        cat = i.get("semantic_category", "unknown")
        cat_data[cat]["total"] += 1
        if not i.get("is_refusal", True):
            cat_data[cat]["complied"] += 1
    return total_asr, dict(cat_data)


def gen_harmbench_asr(model, harmbench_dir, harmbench_classified_dir, out_dir):
    """Grouped bar: ASR by category for base + 3 abliterated variants."""
    variants = ["base"] + VARIANTS
    asr_data = {}  # {variant: {category: asr_pct}}

    for v in variants:
        # Try classified first (has better is_refusal data)
        classified_path = harmbench_classified_dir / f"{model}_{v}_classified.json"
        responses_path = harmbench_dir / f"{model}_{v}_responses.json"

        total_asr = None
        cat_data = None

        if classified_path.exists():
            total_asr, cat_data = _extract_asr_from_classified(classified_path)

        if total_asr is None and responses_path.exists():
            total_asr, cat_data = _extract_asr_from_responses(responses_path)

        if cat_data is not None:
            asr_data[v] = {}
            for cat in HB_CATEGORIES:
                cd = cat_data.get(cat, {"total": 0, "complied": 0})
                asr_data[v][cat] = cd["complied"] / max(cd["total"], 1) * 100

    if len(asr_data) < 2:
        print(f"  [SKIP] {model} harmbench_asr: not enough data")
        return

    # Build chart
    cats_present = [c for c in HB_CATEGORIES if any(c in asr_data.get(v, {}) for v in variants)]
    if not cats_present:
        print(f"  [SKIP] {model} harmbench_asr: no category data")
        return

    cat_labels = [f"{HB_CAT_SHORT[c]}" for c in cats_present]
    n_cats = len(cats_present)
    n_vars = len(variants)
    x = np.arange(n_cats)
    width = 0.18
    offsets = np.arange(n_vars) - (n_vars - 1) / 2

    fig, ax = plt.subplots(figsize=(16, 7))
    for i, v in enumerate(variants):
        vals = [asr_data.get(v, {}).get(c, 0) for c in cats_present]
        label = "Base" if v == "base" else VARIANT_LABELS.get(v, v.title())
        ax.bar(x + offsets[i] * width, vals, width,
               label=label, color=COLORS.get(v, "#888"), alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels, fontsize=10)
    ax.set_ylabel("Attack Success Rate (%)", fontsize=12)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=10)
    ax.axhline(y=100, color="#e74c3c", linestyle="--", alpha=0.3)
    ax.set_title(f"{title(model)} HarmBench ASR by Category",
                 fontsize=14, fontweight="bold")

    save_fig(fig, out_dir, prefix(model, "harmbench_asr"))


# ---------- Graph 10: HarmBench summary ASR overview ----------

def gen_harmbench_summary(model, harmbench_dir, harmbench_classified_dir, out_dir):
    """Simple bar: overall ASR per variant."""
    variants = ["base"] + VARIANTS
    asr_values = []
    labels = []

    for v in variants:
        classified_path = harmbench_classified_dir / f"{model}_{v}_classified.json"
        responses_path = harmbench_dir / f"{model}_{v}_responses.json"

        total_asr = None

        if classified_path.exists():
            d = load_json(classified_path)
            if d:
                t = d.get("total_items", 0)
                c = d.get("complied_count", 0)
                if t > 0:
                    total_asr = c / t * 100

        if total_asr is None and responses_path.exists():
            d = load_json(responses_path)
            if d:
                items = d.get("harmbench", [])
                refused = sum(1 for i in items if i.get("is_refusal"))
                if items:
                    total_asr = (len(items) - refused) / len(items) * 100

        if total_asr is not None:
            labels.append("Base" if v == "base" else VARIANT_LABELS.get(v, v.title()))
            asr_values.append(total_asr)

    if len(asr_values) < 2:
        print(f"  [SKIP] {model} harmbench_summary: not enough data")
        return

    colors = []
    for v in variants[:len(asr_values)]:
        colors.append(COLORS.get(v, "#888"))

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, asr_values, color=colors, alpha=0.9, edgecolor="white")

    for bar, val in zip(bars, asr_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Attack Success Rate (%)", fontsize=12)
    ax.set_ylim(0, 115)
    ax.axhline(y=100, color="#e74c3c", linestyle="--", alpha=0.3)
    ax.set_title(f"{title(model)} HarmBench Overall ASR",
                 fontsize=14, fontweight="bold")
    save_fig(fig, out_dir, prefix(model, "harmbench_summary"))


# ---------- Graph 11: Cross-model scaling overview ----------

def gen_scaling_overview(forensics_dir, harmbench_dir, harmbench_classified_dir, out_dir):
    """Multi-panel overview across all 5 models: ASR + aggressiveness + KL."""
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    # Collect data across models
    model_data = {}
    for model in MODELS:
        md = {"asr_base": None, "asr_abliterated": None,
              "tensors_changed": {}, "kl": {}}

        # ASR
        for v in ["base", "heretic", "hauhau", "huihui"]:
            classified_path = harmbench_classified_dir / f"{model}_{v}_classified.json"
            responses_path = harmbench_dir / f"{model}_{v}_responses.json"
            asr = None
            if classified_path.exists():
                d = load_json(classified_path)
                if d:
                    t = d.get("total_items", 0)
                    c = d.get("complied_count", 0)
                    if t:
                        asr = c / t * 100
            if asr is None and responses_path.exists():
                d = load_json(responses_path)
                if d:
                    items = d.get("harmbench", [])
                    refused = sum(1 for i in items if i.get("is_refusal"))
                    if items:
                        asr = (len(items) - refused) / len(items) * 100
            if asr is not None:
                if v == "base":
                    md["asr_base"] = asr
                else:
                    md["asr_abliterated"] = asr

        # Tensors changed
        for v in VARIANTS:
            fp = load_json(forensics_dir / model / f"fingerprint_{v}.json")
            if fp:
                md["tensors_changed"][v] = fp.get("scope", {}).get("changed_tensors", 0)

        # KL
        for v in VARIANTS:
            kl = load_json(forensics_dir / model / f"kl_{v}.json")
            if kl and "kl_divergence_batchmean" in kl:
                md["kl"][v] = kl["kl_divergence_batchmean"]

        model_data[model] = md

    # Create 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    model_labels = [MODEL_TITLES.get(m, m) for m in MODELS]
    x = np.arange(len(MODELS))

    # Panel 1: ASR (base vs best abliterated)
    asr_base = [model_data[m].get("asr_base", 0) or 0 for m in MODELS]
    asr_ablit = [model_data[m].get("asr_abliterated", 0) or 0 for m in MODELS]

    width = 0.35
    axes[0].bar(x - width / 2, asr_base, width, label="Base",
                color=COLORS["base"], alpha=0.85)
    axes[0].bar(x + width / 2, asr_ablit, width, label="Abliterated (best)",
                color=COLORS["hauhau"], alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_labels, fontsize=9, rotation=15, ha="right")
    axes[0].set_ylabel("ASR (%)")
    axes[0].set_ylim(0, 115)
    axes[0].legend(fontsize=9)
    axes[0].set_title("HarmBench ASR", fontsize=12, fontweight="bold")
    axes[0].axhline(y=100, color="#e74c3c", linestyle="--", alpha=0.3)

    # Panel 2: Tensors changed (grouped by variant)
    width = 0.25
    for i, v in enumerate(VARIANTS):
        vals = [model_data[m]["tensors_changed"].get(v, 0) for m in MODELS]
        axes[1].bar(x + (i - 1) * width, vals, width,
                    label=VARIANT_LABELS[v], color=COLORS[v], alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_labels, fontsize=9, rotation=15, ha="right")
    axes[1].set_ylabel("Tensors Changed")
    axes[1].legend(fontsize=9)
    axes[1].set_title("Abliteration Aggressiveness", fontsize=12, fontweight="bold")

    # Panel 3: KL divergence (grouped by variant)
    for i, v in enumerate(VARIANTS):
        vals = [model_data[m]["kl"].get(v, 0) for m in MODELS]
        axes[2].bar(x + (i - 1) * width, vals, width,
                    label=VARIANT_LABELS[v], color=COLORS[v], alpha=0.85)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(model_labels, fontsize=9, rotation=15, ha="right")
    axes[2].set_ylabel("KL Divergence")
    axes[2].legend(fontsize=9)
    axes[2].set_title("KL Divergence from Base", fontsize=12, fontweight="bold")

    fig.suptitle("Qwen Family: Cross-Model Scaling Overview",
                 fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, out_dir, "qwen_scaling_overview.svg")


# ---------- Main ----------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate Qwen model card SVGs")
    parser.add_argument("--forensics-dir", type=Path, default=Path("/data/forensics"),
                        help="Directory with per-model forensics results (default: /data/forensics)")
    parser.add_argument("--harmbench-dir", type=Path, default=Path("/data/harmbench"),
                        help="Directory with HarmBench response files (default: /data/harmbench)")
    parser.add_argument("--harmbench-classified-dir", type=Path, default=Path("/data/harmbench_classified"),
                        help="Directory with classified HarmBench results (default: /data/harmbench_classified)")
    parser.add_argument("--output-dir", type=Path, default=Path("/output"),
                        help="Output directory for SVG files (default: /output)")
    args = parser.parse_args()

    forensics_dir = args.forensics_dir
    harmbench_dir = args.harmbench_dir
    harmbench_classified_dir = args.harmbench_classified_dir
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Qwen Model Card Graph Generator")
    print(f"Models: {', '.join(MODELS)}")
    print("=" * 60)

    total = 0
    for model in MODELS:
        print(f"\n{'=' * 50}")
        print(f"  {title(model)} ({model})")
        print(f"{'=' * 50}")

        n_before = len(list(out_dir.glob(f"{model}_*.svg")))

        gen_benchmark_bars(model, forensics_dir, out_dir)
        gen_kl_bars(model, forensics_dir, out_dir)
        gen_aggressiveness(model, forensics_dir, out_dir)
        gen_tensor_type_breakdown(model, forensics_dir, out_dir)
        gen_layer_comparison(model, forensics_dir, out_dir)
        gen_edit_distribution(model, forensics_dir, out_dir)
        gen_cosine_heatmap(model, forensics_dir, out_dir)
        gen_venn(model, forensics_dir, out_dir)
        gen_harmbench_asr(model, harmbench_dir, harmbench_classified_dir, out_dir)
        gen_harmbench_summary(model, harmbench_dir, harmbench_classified_dir, out_dir)

        n_after = len(list(out_dir.glob(f"{model}_*.svg")))
        print(f"  -> {n_after - n_before} SVGs for {model}")
        total += n_after - n_before

    # Cross-model overview
    print(f"\n{'=' * 50}")
    print("  Cross-model scaling overview")
    print(f"{'=' * 50}")
    gen_scaling_overview(forensics_dir, harmbench_dir, harmbench_classified_dir, out_dir)
    n_final = len(list(out_dir.glob("qwen_scaling_overview.svg")))
    total += n_final

    print(f"\n{'=' * 60}")
    all_svgs = list(out_dir.glob("*.svg"))
    print(f"Done. {len(all_svgs)} SVGs saved to {out_dir}/")
    for svg in sorted(all_svgs):
        print(f"  {svg.name}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
