#!/usr/bin/env python3
"""
Gemma4-E2B Model Card Graph Generator

Generates SVG visualisations for the Gemma4-E2B abliteration comparison with
13 variants. Reads from the analysis results directory and produces all graphs
for the HuggingFace model card README.

Graph types:
  1. Benchmark Comparison (grouped bar, 7 tasks)
  2. Benchmark Delta (horizontal bar, deltas vs base)
  3. GSM8K Comparison (horizontal bar with empty counts)
  4. HarmBench ASR Summary (bar chart)
  5. HarmBench ASR by Category (grouped bar)
  6. KL Divergence (horizontal bar with ratings)
  7. Aggressiveness (tensors changed)
  8. Cosine Heatmap (13x13 cross-technique)
  9. Layer Edit Magnitude (line plot, all variants)
  10. Edit Distribution (violin plot)

Usage:
    python3 generate_gemma4_card_graphs.py \
        --results-dir comparisons/gemma4-e2b/results \
        --output-dir comparisons/gemma4-e2b/graphs
"""

from __future__ import annotations

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

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

# ---------- Variant definitions ----------

VARIANTS = [
    "coder3101",
    "duoneural",
    "ether4o4",
    "huihui-v1",
    "huihui-v2",
    "kasper",
    "llmfan46",
    "pew",
    "prithiv",
    "treadon",
    "trevorjs",
    "wangzhang",
    "wwtcyberlab",
]

VARIANT_LABELS = {
    "coder3101": "Coder3101",
    "duoneural": "Duoneural",
    "ether4o4": "EtherOpus",
    "huihui-v1": "Huihui-v1",
    "huihui-v2": "Huihui-v2",
    "kasper": "Kasper",
    "llmfan46": "LLMFan46",
    "pew": "PEW",
    "prithiv": "Prithiv",
    "treadon": "Treadon",
    "trevorjs": "TrevorJS",
    "wangzhang": "Wangzhang",
    "wwtcyberlab": "WWT CyberLab",
}

# Colour palette: ordered by aggressiveness for visual grouping
VARIANT_COLORS = {
    "llmfan46": "#2ecc71",
    "coder3101": "#27ae60",
    "kasper": "#3498db",
    "pew": "#2980b9",
    "duoneural": "#9b59b6",
    "huihui-v1": "#8e44ad",
    "prithiv": "#c39bd3",
    "treadon": "#e74c3c",
    "huihui-v2": "#e67e22",
    "trevorjs": "#f39c12",
    "wangzhang": "#d35400",
    "ether4o4": "#c0392b",
    "wwtcyberlab": "#7f8c8d",
}

BASE_COLOR = "#95a5a6"
LABEL = "Gemma4-E2B"

# HarmBench categories
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
    "chemical_biological": "Chem/Bio",
    "copyright": "Copyright",
    "cybercrime_intrusion": "Cybercrime",
    "harassment_bullying": "Harassment",
    "harmful": "Harmful",
    "illegal": "Illegal",
    "misinformation_disinformation": "Misinfo",
}

# Phase 1 tasks for benchmark comparison
LOGIT_TASKS = [
    ("MMLU", "results", "mmlu", "acc,none"),
    ("HellaSwag", "results", "hellaswag", "acc_norm,none"),
    ("ARC", "results", "arc_challenge", "acc,none"),
    ("WinoGrande", "results", "winogrande", "acc,none"),
    ("TQA-MC2", "results", "truthfulqa_mc2", "acc,none"),
    ("PiQA", "results", "piqa", "acc_norm,none"),
]


# ---------- Helpers ----------


def load_json(path: Path, quiet: bool = False) -> dict | None:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        if not quiet:
            print(f"  [WARN] File not found: {path}")
        return None
    except json.JSONDecodeError as e:
        print(f"  [WARN] JSON decode error in {path}: {e}")
        return None


def save_fig(fig: plt.Figure, out_dir: Path, name: str) -> None:
    path = out_dir / name
    fig.savefig(path, format="svg", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [OK] {name}")


# ---------- Graph 1: Benchmark Comparison (grouped bar) ----------


def gen_benchmark_comparison(results_dir: Path, out_dir: Path) -> None:
    """Grouped bar chart of Phase 1 benchmarks across all 14 models."""
    all_models = ["base"] + VARIANTS
    scores: dict[str, list[float]] = {v: [] for v in all_models}
    task_names = []

    for tname, section, task_key, metric in LOGIT_TASKS:
        task_names.append(tname)
        for v in all_models:
            if v == "base":
                candidates = [results_dir / "lm_eval" / "base_phase1_results.json"]
            else:
                # Try both hyphen and underscore variants for filename
                slug = v.replace("-", "_")
                candidates = [
                    results_dir / "lm_eval" / f"{slug}_phase1_results.json",
                    results_dir / "lm_eval" / f"{v}_phase1_results.json",
                ]

            val = 0.0
            for c in candidates:
                d = load_json(c)
                if d and "results" in d:
                    try:
                        val = d[section][task_key][metric] * 100
                        break
                    except (KeyError, TypeError):
                        pass
            scores[v].append(val)

    # Skip LAMBADA for the grouped bar (perplexity is a different scale)

    n_tasks = len(task_names)
    n_vars = len(all_models)
    x = np.arange(n_tasks)
    width = 0.9 / n_vars
    offsets = np.arange(n_vars) - (n_vars - 1) / 2

    fig, ax = plt.subplots(figsize=(20, 8))
    for i, v in enumerate(all_models):
        label = "Base" if v == "base" else VARIANT_LABELS.get(v, v)
        color = BASE_COLOR if v == "base" else VARIANT_COLORS.get(v, "#888")
        ax.bar(x + offsets[i] * width, scores[v], width,
               label=label, color=color, alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(task_names, fontsize=11)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 65)
    ax.legend(loc="upper right", fontsize=7, ncol=3)

    ax.set_title(f"{LABEL} Benchmark Comparison (Phase 1, Loglikelihood)",
                 fontsize=14, fontweight="bold")
    save_fig(fig, out_dir, f"{LABEL}_benchmark_comparison.svg")


# ---------- Graph 2: Benchmark Delta (horizontal bar) ----------


def gen_benchmark_delta(results_dir: Path, out_dir: Path) -> None:
    """Horizontal bar chart of deltas vs base for each variant."""
    # Load base scores
    base_data = load_json(results_dir / "lm_eval" / "base_phase1_results.json")
    if not base_data:
        print("  [SKIP] benchmark_delta: no base data")
        return

    tasks_delta = [
        ("MMLU", "results", "mmlu", "acc,none"),
        ("HellaSwag", "results", "hellaswag", "acc_norm,none"),
        ("ARC", "results", "arc_challenge", "acc,none"),
        ("WinoGrande", "results", "winogrande", "acc,none"),
        ("TQA-MC2", "results", "truthfulqa_mc2", "acc,none"),
        ("PiQA", "results", "piqa", "acc_norm,none"),
    ]

    base_scores = {}
    for tname, section, task_key, metric in tasks_delta:
        try:
            base_scores[tname] = base_data[section][task_key][metric] * 100
        except (KeyError, TypeError):
            base_scores[tname] = 0

    # Build delta data per variant
    delta_data: dict[str, dict[str, float]] = {}
    for v in VARIANTS:
        slug = v.replace("-", "_")
        candidates = [
            results_dir / "lm_eval" / f"{slug}_phase1_results.json",
            results_dir / "lm_eval" / f"{v}_phase1_results.json",
        ]
        d = None
        for c in candidates:
            d = load_json(c)
            if d and "results" in d:
                break
        if not d or "results" not in d:
            continue
        delta_data[v] = {}
        for tname, section, task_key, metric in tasks_delta:
            try:
                variant_score = d[section][task_key][metric] * 100
                delta_data[v][tname] = variant_score - base_scores[tname]
            except (KeyError, TypeError):
                delta_data[v][tname] = 0

    if not delta_data:
        print("  [SKIP] benchmark_delta: no variant data")
        return

    task_names = [t[0] for t in tasks_delta]
    n_tasks = len(task_names)
    n_vars = len(delta_data)
    variant_order = [v for v in VARIANTS if v in delta_data]

    fig, ax = plt.subplots(figsize=(12, 14))
    y = np.arange(n_tasks)
    height = 0.9 / n_vars
    offsets = np.arange(n_vars) - (n_vars - 1) / 2

    for i, v in enumerate(variant_order):
        vals = [delta_data[v].get(t, 0) for t in task_names]
        label = VARIANT_LABELS.get(v, v)
        color = VARIANT_COLORS.get(v, "#888")
        ax.barh(y + offsets[i] * height, vals, height,
                label=label, color=color, alpha=0.85, edgecolor="white")

    ax.set_yticks(y)
    ax.set_yticklabels(task_names, fontsize=11)
    ax.set_xlabel("Delta vs Base (percentage points)")
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    ax.set_title(f"{LABEL} Benchmark Delta vs Base",
                 fontsize=14, fontweight="bold")
    save_fig(fig, out_dir, f"{LABEL}_benchmark_delta.svg")


# ---------- Graph 3: GSM8K Comparison ----------


def gen_gsm8k_comparison(results_dir: Path, out_dir: Path) -> None:
    """Horizontal bar chart of GSM8K flexible-extract with empty count overlay."""
    all_models = ["base"] + VARIANTS
    data: list[tuple[str, float, int]] = []

    for v in all_models:
        if v == "base":
            path = results_dir / "lm_eval" / "google-base_gsm8k_lmeval_results.json"
        else:
            path = results_dir / "lm_eval" / f"{v}_gsm8k_lmeval_results.json"
        d = load_json(path)
        if not d or "results" not in d:
            continue
        try:
            flex = d["results"]["gsm8k"]["exact_match,flexible-extract"] * 100
            empty = 0
            # Count empty responses from samples
            samples_dir = results_dir / "lm_eval" / f"__tmp__model_{v}" / "gsm8k"
            if not samples_dir.exists():
                samples_dir = results_dir / "lm_eval" / "__model" / f"__tmp__model_{v}" / "gsm8k"
            data.append((v, flex, empty))
        except (KeyError, TypeError):
            continue

    if len(data) < 2:
        print("  [SKIP] gsm8k_comparison: not enough data")
        return

    # Sort by flex score descending
    data.sort(key=lambda x: x[1], reverse=True)

    labels = [VARIANT_LABELS.get(v, "Base") if v != "base" else "Base" for v, _, _ in data]
    values = [s for _, s, _ in data]
    colors = [BASE_COLOR if v == "base" else VARIANT_COLORS.get(v, "#888") for v, _, _ in data]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(range(len(labels)), values, color=colors, alpha=0.85, edgecolor="white")

    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("GSM8K Flexible-Extract Score (%)")
    ax.invert_yaxis()
    ax.set_title(f"{LABEL} GSM8K with Thinking Enabled",
                 fontsize=14, fontweight="bold")
    save_fig(fig, out_dir, f"{LABEL}_gsm8k_comparison.svg")


# ---------- Graph 4: HarmBench ASR Summary ----------


def gen_harmbench_summary(results_dir: Path, out_dir: Path) -> None:
    """Bar chart of overall HarmBench ASR for all 14 models."""
    all_models = ["base"] + VARIANTS
    asr_data: list[tuple[str, float, int]] = []

    for v in all_models:
        if v == "base":
            fname = "harmbench_base_responses.json"
        else:
            slug = v.replace("-", "_")
            fname = f"harmbench_{slug}_responses.json"
        path = results_dir / "harmbench" / fname
        d = load_json(path)
        if not d:
            # Try hyphenated version as fallback
            fname2 = f"harmbench_{v}_responses.json"
            path2 = results_dir / "harmbench" / fname2
            d = load_json(path2)
        if not d:
            continue
        items = d.get("harmbench", [])
        if not items:
            continue
        refused = sum(1 for i in items if i.get("is_refusal"))
        asr = (len(items) - refused) / len(items) * 100
        asr_data.append((v, asr, refused))

    if len(asr_data) < 2:
        print("  [SKIP] harmbench_summary: not enough data")
        return

    labels = ["Base" if v == "base" else VARIANT_LABELS.get(v, v) for v, _, _ in asr_data]
    values = [a for _, a, _ in asr_data]
    colors = [BASE_COLOR if v == "base" else VARIANT_COLORS.get(v, "#888") for v, _, _ in asr_data]

    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.bar(labels, values, color=colors, alpha=0.9, edgecolor="white")

    for bar, val, ref in zip(bars, values, [r for _, _, r in asr_data]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%\n({ref} ref)", ha="center", va="bottom",
                fontsize=8, fontweight="bold")

    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_ylim(0, 115)
    ax.axhline(y=100, color="#e74c3c", linestyle="--", alpha=0.3)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    ax.set_title(f"{LABEL} HarmBench Overall ASR",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, out_dir, f"{LABEL}_harmbench_summary.svg")


# ---------- Graph 5: HarmBench ASR by Category ----------


def gen_harmbench_asr_category(results_dir: Path, out_dir: Path) -> None:
    """Grouped bar: ASR by category for all 14 models."""
    all_models = ["base"] + VARIANTS
    asr_data: dict[str, dict[str, float]] = {}

    for v in all_models:
        if v == "base":
            fname = "harmbench_base_responses.json"
        else:
            slug = v.replace("-", "_")
            fname = f"harmbench_{slug}_responses.json"
        path = results_dir / "harmbench" / fname
        d = load_json(path)
        if not d:
            fname2 = f"harmbench_{v}_responses.json"
            path2 = results_dir / "harmbench" / fname2
            d = load_json(path2)
        if not d:
            continue
        items = d.get("harmbench", [])
        cat_data: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "complied": 0})
        for item in items:
            cat = item.get("semantic_category", "unknown")
            cat_data[cat]["total"] += 1
            if not item.get("is_refusal", True):
                cat_data[cat]["complied"] += 1

        asr_data[v] = {}
        for cat in HB_CATEGORIES:
            cd = cat_data.get(cat, {"total": 0, "complied": 0})
            asr_data[v][cat] = cd["complied"] / max(cd["total"], 1) * 100

    if len(asr_data) < 2:
        print("  [SKIP] harmbench_asr_category: not enough data")
        return

    cats_present = [c for c in HB_CATEGORIES if any(c in asr_data.get(v, {}) for v in all_models)]
    if not cats_present:
        print("  [SKIP] harmbench_asr_category: no category data")
        return

    cat_labels = [HB_CAT_SHORT[c] for c in cats_present]
    n_cats = len(cats_present)
    n_vars = len(all_models)
    x = np.arange(n_cats)
    width = 0.9 / n_vars
    offsets = np.arange(n_vars) - (n_vars - 1) / 2

    fig, ax = plt.subplots(figsize=(20, 8))
    for i, v in enumerate(all_models):
        vals = [asr_data.get(v, {}).get(c, 0) for c in cats_present]
        label = "Base" if v == "base" else VARIANT_LABELS.get(v, v)
        color = BASE_COLOR if v == "base" else VARIANT_COLORS.get(v, "#888")
        ax.bar(x + offsets[i] * width, vals, width,
               label=label, color=color, alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels, fontsize=10)
    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_ylim(0, 115)
    ax.axhline(y=100, color="#e74c3c", linestyle="--", alpha=0.3)
    ax.legend(fontsize=6, ncol=4, loc="upper left")
    ax.set_title(f"{LABEL} HarmBench ASR by Category",
                 fontsize=14, fontweight="bold")
    save_fig(fig, out_dir, f"{LABEL}_harmbench_asr.svg")


# ---------- Graph 6: KL Divergence ----------


def gen_kl_divergence(results_dir: Path, out_dir: Path) -> None:
    """Horizontal bar chart of KL divergence with colour-coded ratings."""
    kl_data: list[tuple[str, float, str]] = []

    for v in VARIANTS:
        path = results_dir / "kl" / f"kl_{v}.json"
        d = load_json(path)
        if d and "kl_divergence_batchmean" in d:
            kl = d["kl_divergence_batchmean"]
            if kl < 0.1:
                rating = "very good"
            elif kl < 0.4:
                rating = "moderate"
            elif kl < 1.0:
                rating = "significant"
            else:
                rating = "heavy"
            kl_data.append((v, kl, rating))

    if not kl_data:
        print("  [SKIP] kl_divergence: no data")
        return

    # Sort by KL ascending
    kl_data.sort(key=lambda x: x[1])

    labels = [f"{VARIANT_LABELS.get(v, v)}\n({r})" for v, _, r in kl_data]
    values = [kl for _, kl, _ in kl_data]

    # Colour by rating
    rating_colors = {
        "very good": "#2ecc71",
        "moderate": "#3498db",
        "significant": "#f39c12",
        "heavy": "#e74c3c",
    }
    colors = [rating_colors.get(r, "#888") for _, _, r in kl_data]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(range(len(labels)), values, color=colors, alpha=0.9, edgecolor="white")

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9, fontweight="bold")

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("KL Divergence (batchmean)")
    ax.invert_yaxis()

    # Rating bands
    if max(values) > 0.1:
        ax.axvspan(0, 0.1, alpha=0.05, color="green")
        ax.axvspan(0.1, 0.4, alpha=0.03, color="blue")
        ax.axvspan(0.4, 1.0, alpha=0.03, color="orange")
        if max(values) > 1.0:
            ax.axvspan(1.0, max(values) * 1.1, alpha=0.03, color="red")

    ax.set_title(f"{LABEL} KL Divergence from Base",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, out_dir, f"{LABEL}_kl_divergence.svg")


# ---------- Graph 7: Aggressiveness ----------


def gen_aggressiveness(results_dir: Path, out_dir: Path) -> None:
    """Bar chart: tensors changed per variant."""
    counts: list[tuple[str, int]] = []

    panel = load_json(results_dir / "multi_model_panel.json")
    if panel:
        for v in VARIANTS:
            key = f"base->{v}"
            if key in panel.get("pairwise_changed_counts", {}):
                counts.append((v, panel["pairwise_changed_counts"][key]))

    # Fallback: fingerprints
    if not counts:
        for v in VARIANTS:
            fp = load_json(results_dir / v / f"fingerprint_{v}.json")
            if fp:
                n = fp.get("scope", {}).get("changed_tensors", 0)
                if n:
                    counts.append((v, n))

    if not counts:
        print("  [SKIP] aggressiveness: no data")
        return

    # Sort by count ascending
    counts.sort(key=lambda x: x[1])

    labels = [VARIANT_LABELS.get(v, v) for v, _ in counts]
    values = [c for _, c in counts]
    colors = [VARIANT_COLORS.get(v, "#888") for v, _ in counts]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(range(len(labels)), values, color=colors, alpha=0.9, edgecolor="white")

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.02, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=10, fontweight="bold")

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Tensors Changed")
    ax.invert_yaxis()
    ax.set_title(f"{LABEL} Abliteration Aggressiveness",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, out_dir, f"{LABEL}_aggressiveness.svg")


# ---------- Graph 8: Cosine Heatmap ----------


def gen_cosine_heatmap(results_dir: Path, out_dir: Path) -> None:
    """13x13 heatmap of cross-technique cosine similarities."""
    # Collect all pairwise cosine means
    cosine_map: dict[tuple[str, str], float] = {}
    for corr_file in sorted(results_dir.glob("correlation_*_vs_*.json")):
        d = load_json(corr_file)
        if not d:
            continue
        # Extract variant names from filename
        stem = corr_file.stem.replace("correlation_", "")
        parts = stem.split("_vs_")
        if len(parts) != 2:
            continue
        a, b = parts[0], parts[1]
        pc = d.get("pairwise_cosines", {})
        # Get the mean cosine from the first entry
        for _key, pdata in pc.items():
            cosine_map[(a, b)] = pdata.get("mean", 0)
            cosine_map[(b, a)] = pdata.get("mean", 0)
            break

    if not cosine_map:
        print("  [SKIP] cosine_heatmap: no correlation data")
        return

    n = len(VARIANTS)
    matrix = np.zeros((n, n))
    for i in range(n):
        matrix[i, i] = 1.0
        for j in range(n):
            if i != j:
                matrix[i, j] = cosine_map.get((VARIANTS[i], VARIANTS[j]), 0.0)

    labels = [VARIANT_LABELS.get(v, v) for v in VARIANTS]

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(matrix, ax=ax, annot=True, fmt=".2f", cmap="RdYlGn",
                vmin=0, vmax=1.0, center=0.5,
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, linecolor="white",
                annot_kws={"size": 7},
                cbar_kws={"label": "Mean Cosine Similarity"})
    ax.set_title(f"{LABEL} Cross-Technique Edit Vector Cosine Similarity",
                 fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    fig.tight_layout()
    save_fig(fig, out_dir, f"{LABEL}_cosine_heatmap.svg")


# ---------- Graph 9: Layer Edit Magnitude ----------


def gen_layer_comparison(results_dir: Path, out_dir: Path) -> None:
    """Line plot: mean edit norm by layer for all variants."""
    fig, ax = plt.subplots(figsize=(16, 8))

    has_data = False
    for v in VARIANTS:
        d = load_json(results_dir / v / f"layer_analysis_{v}.json")
        if not d or "layer_progression" not in d:
            continue
        lp = d["layer_progression"]
        layers = sorted(lp.keys(), key=lambda k: int(k))
        edit_norms = [lp[l].get("mean_edit_norm", 0) for l in layers]

        label = VARIANT_LABELS.get(v, v)
        color = VARIANT_COLORS.get(v, "#888")
        ax.plot(range(len(layers)), edit_norms,
                label=label, color=color, alpha=0.7, linewidth=1.5)
        has_data = True

    if not has_data:
        print("  [SKIP] layer_comparison: no data")
        plt.close(fig)
        return

    ax.set_ylabel("Mean Edit Norm", fontsize=12)
    ax.set_xlabel("Layer", fontsize=12)
    ax.legend(fontsize=7, ncol=3, loc="upper left")
    ax.set_title(f"{LABEL} Layer-wise Edit Magnitude",
                 fontsize=14, fontweight="bold")

    # Layer ticks
    first_lp = load_json(results_dir / VARIANTS[0] / f"layer_analysis_{VARIANTS[0]}.json")
    if first_lp and "layer_progression" in first_lp:
        all_layers = sorted(first_lp["layer_progression"].keys(), key=lambda k: int(k))
        tick_step = max(1, len(all_layers) // 16)
        ax.set_xticks(range(0, len(all_layers), tick_step))
        ax.set_xticklabels([f"L{all_layers[i]}" for i in range(0, len(all_layers), tick_step)])

    save_fig(fig, out_dir, f"{LABEL}_layer_comparison.svg")


# ---------- Graph 10: Edit Distribution ----------


def gen_edit_distribution(results_dir: Path, out_dir: Path) -> None:
    """Violin plot of per-tensor edit norms per variant."""
    all_norms: dict[str, list[float]] = {}

    for v in VARIANTS:
        svd = load_json(results_dir / v / f"svd_{v}.json")
        if not svd or "tensor_results" not in svd:
            continue
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
        print("  [SKIP] edit_distribution: no per-tensor data")
        return

    # Sort by median norm for display
    variant_order = sorted(all_norms.keys(), key=lambda v: np.median(all_norms[v]))
    plot_data = [all_norms[v] for v in variant_order]
    plot_labels = [VARIANT_LABELS.get(v, v) for v in variant_order]
    plot_colors = [VARIANT_COLORS.get(v, "#888") for v in variant_order]

    fig, ax = plt.subplots(figsize=(14, 8))
    parts = ax.violinplot(plot_data, showmeans=True, showmedians=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(plot_colors[i])
        pc.set_alpha(0.7)

    ax.set_xticks(range(1, len(plot_labels) + 1))
    ax.set_xticklabels(plot_labels, fontsize=9, rotation=45, ha="right")
    ax.set_ylabel("Edit Norm (Frobenius)", fontsize=12)
    ax.set_title(f"{LABEL} Distribution of Per-Tensor Edit Magnitudes",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, out_dir, f"{LABEL}_edit_distribution.svg")


# ---------- Graph 11: Tensor Type Breakdown ----------


def gen_tensor_type_breakdown(results_dir: Path, out_dir: Path) -> None:
    """Grouped bar: which tensor types each variant modifies."""
    all_types: set[str] = set()
    type_counts: dict[str, dict[str, int]] = {}

    for v in VARIANTS:
        fp = load_json(results_dir / v / f"fingerprint_{v}.json")
        if fp and "targeting" in fp:
            tt = fp["targeting"].get("tensor_types", {})
            type_counts[v] = tt
            all_types.update(tt.keys())

    if not type_counts or not all_types:
        print("  [SKIP] tensor_type_breakdown: no data")
        return

    def shorten(t: str) -> str:
        return t.replace(".weight", "").replace("_proj", "").replace("self_attn.", "attn_").replace("per_layer_", "pl_")

    type_order = sorted(all_types,
                        key=lambda t: sum(type_counts.get(v, {}).get(t, 0) for v in VARIANTS),
                        reverse=True)

    short_names = [shorten(t) for t in type_order]
    n = len(type_order)
    x = np.arange(n)
    variant_order = [v for v in VARIANTS if v in type_counts]
    n_vars = len(variant_order)
    width = 0.9 / max(n_vars, 1)

    fig, ax = plt.subplots(figsize=(max(14, n * 1.5), 8))

    for i, v in enumerate(variant_order):
        vals = [type_counts.get(v, {}).get(t, 0) for t in type_order]
        offset = (i - (n_vars - 1) / 2) * width
        ax.bar(x + offset, vals, width,
               label=VARIANT_LABELS[v], color=VARIANT_COLORS[v], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=9, rotation=25, ha="right")
    ax.set_ylabel("Tensors Modified", fontsize=12)
    ax.legend(fontsize=6, ncol=3)
    ax.set_title(f"{LABEL} Tensor Type Targeting by Variant",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, out_dir, f"{LABEL}_tensor_type_breakdown.svg")


# ---------- Main ----------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate Gemma4-E2B model card SVGs")
    parser.add_argument("--results-dir", type=Path,
                        default=Path("comparisons/gemma4-e2b/results"),
                        help="Results directory")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("comparisons/gemma4-e2b/graphs"),
                        help="Output directory for SVG files")
    args = parser.parse_args()

    results_dir = args.results_dir
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Gemma4-E2B Card Graph Generator")
    print(f"Results: {results_dir}")
    print(f"Output:  {out_dir}")
    print("=" * 60)

    gen_benchmark_comparison(results_dir, out_dir)
    gen_benchmark_delta(results_dir, out_dir)
    gen_gsm8k_comparison(results_dir, out_dir)
    gen_harmbench_summary(results_dir, out_dir)
    gen_harmbench_asr_category(results_dir, out_dir)
    gen_kl_divergence(results_dir, out_dir)
    gen_aggressiveness(results_dir, out_dir)
    gen_cosine_heatmap(results_dir, out_dir)
    gen_layer_comparison(results_dir, out_dir)
    gen_edit_distribution(results_dir, out_dir)
    gen_tensor_type_breakdown(results_dir, out_dir)

    all_svgs = sorted(out_dir.glob("*.svg"))
    print(f"\n{'=' * 60}")
    print(f"Done. {len(all_svgs)} SVGs saved to {out_dir}/")
    for svg in all_svgs:
        print(f"  {svg.name}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
