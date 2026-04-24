#!/usr/bin/env python3
"""
Comprehensive Weight Analysis Graph Generator for GLM-4.7-Flash Abliteration Forensics.

Produces 16+ SVG visualizations from the weight analysis JSON data:
  1. Pairwise cosine similarity (bar chart, 3 original + 3 Abliterix pairs)
  2. Tensor types modified (grouped bar)
  3. Scope comparison (donut charts)
  4. Edit magnitude distribution (violin + box)
  5. Layer-wise edit profile (2x2 per variant)
  6. Venn diagram (3-way technique overlap)
  7. Aggressiveness bar (tensors changed per variant)
  8. Expert heatmaps (one per variant — MoE-specific)
  9. SVD spectrum analysis (rank + energy)
 10. Edit density by layer (all variants overlaid)
 11. Mean norm shift by layer (all variants overlaid)
 12. Stacking/cosine scatter (edit direction alignment)
 13. Low-rank reconstruction curves
 14. Subspace alignment (principal angle histograms)
 15. Router edit heatmap
 16. Shared expert edit heatmap

Usage:
    python3 make_weight_graphs.py [--weight-dir DIR] [--out-dir DIR]
"""

import argparse
import json
import math
import os
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

# ── Configuration ──────────────────────────────────────────────────────

sns.set_theme(style="darkgrid", palette="muted", font_scale=1.0)

COLORS = {
    "heretic":   "#e74c3c",
    "hauhau":    "#3498db",
    "huihui":    "#2ecc71",
    "abliterix": "#f39c12",
    "base":      "#95a5a6",
    "overlap":   "#9b59b6",
}

LABELS = {
    "heretic":   "Heretic",
    "hauhau":    "HauhauCS",
    "huihui":    "Huihui",
    "abliterix": "Abliterix",
}

ALL_VARIANTS = ["heretic", "hauhau", "huihui", "abliterix"]
THREE_VARIANTS = ["heretic", "hauhau", "huihui"]


def load_json(path):
    """Load JSON from a file path."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"  [WARN] Could not load {path}: {e}", file=sys.stderr)
        return None


def save_svg(fig, out_dir, filename):
    """Save figure as SVG and close."""
    fig.savefig(os.path.join(out_dir, filename), format="svg", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  ✓ {filename}")


# ── 1. Pairwise Cosine Similarity ─────────────────────────────────────

def plot_pairwise_cosine(weight_dir, out_dir):
    """Bar chart of pairwise edit-direction cosine similarity for all 6 pairs."""
    data = load_json(os.path.join(weight_dir, "technique_correlation.json"))
    if not data:
        return

    pairs = data["pairwise_cosines"]

    # Also load Abliterix correlation files
    corr_files = {
        "heretic_vs_abliterix": load_json(os.path.join(weight_dir, "correlation_heretic_vs_abliterix.json")),
        "hauhau_vs_abliterix":  load_json(os.path.join(weight_dir, "correlation_hauhau_vs_abliterix.json")),
        "huihui_vs_abliterix":  load_json(os.path.join(weight_dir, "correlation_huihui_vs_abliterix.json")),
    }
    for k, v in corr_files.items():
        if v and "pairwise_cosines" in v:
            inner = v["pairwise_cosines"]
            for pair_name, stats in inner.items():
                pairs[pair_name] = stats

    # Order pairs logically
    pair_order = [
        "heretic_vs_hauhau", "heretic_vs_huihui", "hauhau_vs_huihui",
        "heretic_vs_abliterix", "hauhau_vs_abliterix", "huihui_vs_abliterix",
    ]

    labels = []
    means = []
    medians = []
    for p in pair_order:
        if p in pairs:
            short = p.replace("_vs_", "\nvs ").replace("heretic", "Heretic").replace("hauhau", "HauhauCS").replace("huihui", "Huihui").replace("abliterix", "Abliterix")
            labels.append(short)
            means.append(pairs[p]["mean"])
            medians.append(pairs[p]["median"])

    if not labels:
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width / 2, means, width, label="Mean cosine", color="#3498db", alpha=0.8, edgecolor="white")
    bars2 = ax.bar(x + width / 2, medians, width, label="Median cosine", color="#e74c3c", alpha=0.8, edgecolor="white")

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Cosine Similarity", fontsize=12)
    ax.set_title("Pairwise Edit Direction Cosine Similarity\n(No universal abliteration subspace — all values ≤ 0.5)",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, max(max(means), max(medians)) * 1.2)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="0.5 threshold")
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_svg(fig, out_dir, "weight_pairwise_cosine.svg")


# ── 2. Tensor Types Modified ──────────────────────────────────────────

def plot_tensor_types(weight_dir, out_dir):
    """Grouped bar chart of tensor types modified per variant."""
    data = load_json(os.path.join(weight_dir, "technique_correlation.json"))
    if not data:
        return

    per_key = data.get("per_key_details", [])

    # Count layers modified per category per variant
    cat_counts = defaultdict(lambda: defaultdict(int))
    for item in per_key:
        cat = item.get("category", "other")
        for v in THREE_VARIANTS:
            if item.get(f"{v}_changed", False):
                cat_counts[cat][v] += 1

    # Also include Abliterix from edit_abliterix.json
    abl = load_json(os.path.join(weight_dir, "edit_abliterix.json"))
    if abl and "per_key_details" in abl:
        for item in abl["per_key_details"]:
            if item.get("changed", False):
                cat = item.get("type", "other")
                # Normalize category
                if "experts" in cat:
                    parts = cat.split(".")
                    if len(parts) >= 3:
                        cat = "expert"
                    else:
                        cat = cat
                elif "shared_expert" in cat:
                    cat = "shared_expert"
                elif "gate" in cat and "mlp" in cat:
                    cat = "router"
                cat_counts[cat]["abliterix"] += 1

    # Sort by total modifications
    sorted_cats = sorted(cat_counts.keys(),
                         key=lambda c: sum(cat_counts[c].values()), reverse=True)
    top_cats = sorted_cats[:15]

    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(len(top_cats))
    width = 0.2

    variants_to_plot = THREE_VARIANTS + ["abliterix"]
    for i, v in enumerate(variants_to_plot):
        vals = [cat_counts[c][v] for c in top_cats]
        ax.bar(x + i * width, vals, width, label=LABELS.get(v, v),
               color=COLORS.get(v, "#888"), alpha=0.85, edgecolor="white")

    short = [c.replace("mlp.experts.*.", "exp.").replace(".weight", "")
              .replace("self_attn.", "attn.")
              .replace("mlp.shared_experts.", "shared.")
              .replace("mlp.gate.", "router.") for c in top_cats]
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Layers Modified", fontsize=11)
    ax.set_title("Tensor Types Modified by Each Abliteration Technique\n(per-layer count)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_svg(fig, out_dir, "weight_tensor_types.svg")


# ── 3. Scope Comparison (Donut) ───────────────────────────────────────

def plot_scope_comparison(weight_dir, out_dir):
    """Donut charts showing % of tensors modified per variant."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))

    for i, variant in enumerate(ALL_VARIANTS):
        d = load_json(os.path.join(weight_dir, f"fingerprint_{variant}.json"))
        if not d:
            axes[i].text(0.5, 0.5, "No data", ha="center", va="center", transform=axes[i].transAxes)
            continue

        scope = d["scope"]
        changed = scope["changed_pct"]
        unchanged = 100 - changed

        wedges, texts, autotexts = axes[i].pie(
            [changed, unchanged],
            labels=["Modified", "Unchanged"],
            colors=[COLORS[variant], "#ecf0f1"],
            autopct="%1.1f%%",
            startangle=90,
            pctdistance=0.8,
            wedgeprops=dict(width=0.4),
            textprops={"fontsize": 9},
        )
        autotexts[0].set_fontweight("bold")
        axes[i].set_title(f'{LABELS[variant]}\n({scope["changed_tensors"]:,} tensors)',
                          fontsize=11, fontweight="bold")

    fig.suptitle("Weight Modification Scope — Percentage of Tensors Changed",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_svg(fig, out_dir, "weight_scope.svg")


# ── 4. Edit Magnitude Distribution ────────────────────────────────────

def plot_edit_magnitude(weight_dir, out_dir):
    """Violin + box plot of edit magnitude distributions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    data_list = []
    labels_list = []
    rel_data = []
    for variant in ALL_VARIANTS:
        d = load_json(os.path.join(weight_dir, f"fingerprint_{variant}.json"))
        if not d:
            continue
        per_key = d["per_key"]
        norms = [item["edit_norm"] for item in per_key if item.get("edit_norm", 0) > 0]
        rels = [item["rel_norm"] for item in per_key if item.get("edit_norm", 0) > 0]
        data_list.append(norms)
        rel_data.append(rels)
        labels_list.append(LABELS[variant])

    if not data_list:
        plt.close(fig)
        return

    # Violin plot (absolute)
    parts = ax1.violinplot(data_list, positions=range(len(data_list)),
                           showmeans=True, showmedians=True)
    for i, pc in enumerate(parts["bodies"]):
        v = ALL_VARIANTS[i] if i < len(ALL_VARIANTS) else "other"
        pc.set_facecolor(COLORS.get(v, "#888"))
        pc.set_alpha(0.7)
    ax1.set_xticks(range(len(labels_list)))
    ax1.set_xticklabels(labels_list, fontsize=10)
    ax1.set_ylabel("Edit Norm (L2)", fontsize=11)
    ax1.set_title("Absolute Edit Magnitude", fontsize=12, fontweight="bold")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Box plot (relative)
    bp = ax2.boxplot(rel_data, tick_labels=labels_list, patch_artist=True, showmeans=True,
                     meanprops=dict(marker="D", markerfacecolor="white", markersize=6))
    for i, patch in enumerate(bp["boxes"]):
        v = ALL_VARIANTS[i] if i < len(ALL_VARIANTS) else "other"
        patch.set_facecolor(COLORS.get(v, "#888"))
        patch.set_alpha(0.6)
    ax2.set_ylabel("Relative Edit (edit_norm / base_norm)", fontsize=11)
    ax2.set_title("Relative Edit Magnitude", fontsize=12, fontweight="bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("Distribution of Edit Magnitudes (non-zero edits only)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_svg(fig, out_dir, "weight_magnitude.svg")


# ── 5. Layer-wise Edit Profile ────────────────────────────────────────

def plot_layer_profile(weight_dir, out_dir):
    """2x2 grid: edit norm + relative edit per layer per variant."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for i, variant in enumerate(ALL_VARIANTS):
        d = load_json(os.path.join(weight_dir, f"layer_{variant}.json"))
        if not d:
            axes[i].text(0.5, 0.5, "No data", ha="center", va="center", transform=axes[i].transAxes)
            continue

        prog = d["layer_progression"]
        layers = sorted(prog.keys(), key=int)
        edit_norms = [prog[l]["mean_edit_norm"] for l in layers]
        rel_edits = [prog[l]["mean_relative_edit"] for l in layers]

        ax = axes[i]
        ax2 = ax.twinx()

        ax.bar([int(l) for l in layers], edit_norms, color=COLORS[variant], alpha=0.6, label="Edit norm")
        ax2.plot([int(l) for l in layers], [r * 100 for r in rel_edits],
                 color="black", linewidth=1.5, label="Relative edit %")

        ax.set_xlabel("Layer", fontsize=10)
        ax.set_ylabel("Mean Edit Norm", fontsize=10, color=COLORS[variant])
        ax2.set_ylabel("Relative Edit %", fontsize=10, color="black")
        ax.set_title(f"{LABELS[variant]} — Layer Profile", fontsize=12, fontweight="bold")
        ax.set_xlim(-1, 47)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    fig.suptitle("Layer-wise Edit Profile Across Techniques", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_svg(fig, out_dir, "weight_layer_profile.svg")


# ── 6. Venn Diagram ───────────────────────────────────────────────────

def plot_venn(weight_dir, out_dir):
    """3-way Venn diagram of technique overlap."""
    if not HAS_VENN:
        print("  [SKIP] matplotlib-venn not available")
        return

    panel = load_json(os.path.join(weight_dir, "panel_comparison.json", "multi_model_panel.json"))
    if not panel:
        return

    base_h  = set(panel.get("base->heretic_keys", []))
    base_hh = set(panel.get("base->hauhau_keys", []))

    # Huihui keys from fingerprint
    fp_hui = load_json(os.path.join(weight_dir, "fingerprint_huihui.json"))
    base_hui = set()
    if fp_hui:
        base_hui = {item["key"] for item in fp_hui["per_key"] if item.get("changed", False)}

    if not (base_h and base_hh and base_hui):
        print("  [SKIP] Venn: not enough variant key sets")
        return

    only_h   = len(base_h - base_hh - base_hui)
    only_hh  = len(base_hh - base_h - base_hui)
    only_hui = len(base_hui - base_h - base_hh)
    h_hh     = len((base_h & base_hh) - base_hui)
    h_hui    = len((base_h & base_hui) - base_hh)
    hh_hui   = len((base_hh & base_hui) - base_h)
    all3     = len(base_h & base_hh & base_hui)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    v = venn3(
        subsets=(only_h, only_hh, h_hh, only_hui, h_hui, hh_hui, all3),
        set_labels=("Heretic", "HauhauCS", "Huihui"),
        set_colors=(COLORS["heretic"], COLORS["hauhau"], COLORS["huihui"]),
        alpha=0.7, ax=ax,
    )
    ax.set_title("GLM-4.7-Flash — Technique Edit Overlap\n(Which tensors each technique modifies)",
                 fontsize=14, fontweight="bold", pad=20)

    # Add totals annotation
    ax.text(0.5, -0.05,
            f"Heretic: {len(base_h):,} | HauhauCS: {len(base_hh):,} | Huihui: {len(base_hui):,} | All 3: {all3:,}",
            ha="center", transform=ax.transAxes, fontsize=10, style="italic")

    save_svg(fig, out_dir, "weight_venn.svg")


# ── 7. Aggressiveness Bar ────────────────────────────────────────────

def plot_aggressiveness(weight_dir, out_dir):
    """Bar chart of total tensors changed per variant."""
    counts = {}
    for variant in ALL_VARIANTS:
        d = load_json(os.path.join(weight_dir, f"fingerprint_{variant}.json"))
        if d and "scope" in d:
            counts[variant] = d["scope"]["changed_tensors"]

    if not counts:
        return

    total = 9491  # GLM-4.7-Flash total tensors
    variants = list(counts.keys())
    vals = [counts[v] for v in variants]
    pcts = [v / total * 100 for v in vals]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Raw count
    colors = [COLORS.get(v, "#888") for v in variants]
    bars = ax1.bar([LABELS.get(v, v) for v in variants], vals, color=colors, alpha=0.9, edgecolor="white")
    for bar, count in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.02,
                 f"{count:,}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Tensors Changed", fontsize=12)
    ax1.set_title("Abliteration Aggressiveness\n(Absolute Count)", fontsize=13, fontweight="bold")
    ax1.axhline(y=total, color="gray", linestyle="--", alpha=0.4, label=f"Total tensors ({total:,})")
    ax1.legend(fontsize=9)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Percentage
    bars2 = ax2.bar([LABELS.get(v, v) for v in variants], pcts, color=colors, alpha=0.9, edgecolor="white")
    for bar, pct in zip(bars2, pcts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{pct:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Tensors Changed (%)", fontsize=12)
    ax2.set_title("Abliteration Aggressiveness\n(% of All Tensors)", fontsize=13, fontweight="bold")
    ax2.set_ylim(0, max(pcts) * 1.25)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    save_svg(fig, out_dir, "weight_aggressiveness.svg")


# ── 8. Expert Heatmaps ───────────────────────────────────────────────

def plot_expert_heatmaps(weight_dir, out_dir):
    """Per-variant heatmap of expert edit magnitude (layer × expert)."""
    data = load_json(os.path.join(weight_dir, "expert_analysis.json"))
    if not data or "per_expert_details" not in data:
        return

    per_expert = data["per_expert_details"]

    for variant in THREE_VARIANTS:
        details = per_expert.get(variant)
        if not details:
            continue

        layers = sorted(set(info["layer"] for info in details.values()))
        experts = sorted(set(info["expert_id"] for info in details.values()))

        grid = np.zeros((len(experts), len(layers)))
        for info in details.values():
            ei = experts.index(info["expert_id"])
            li = layers.index(info["layer"])
            grid[ei, li] = info["total_edit_norm"]

        fig, ax = plt.subplots(figsize=(max(14, len(layers) * 0.35), max(8, len(experts) * 0.15)))
        sns.heatmap(grid, ax=ax, cmap="YlOrRd",
                     xticklabels=[f"L{l}" for l in layers],
                     yticklabels=[f"E{e}" for e in experts],
                     linewidths=0.1, cbar_kws={"label": "Total Edit Norm"})
        ax.set_title(f"Expert Edit Magnitude — {LABELS[variant]}\n(GLM-4.7-Flash, 64 routed experts × layers)",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Layer", fontsize=11)
        ax.set_ylabel("Expert ID", fontsize=11)
        ax.tick_params(axis="y", labelsize=6)
        ax.tick_params(axis="x", rotation=45, labelsize=7)

        save_svg(fig, out_dir, f"weight_expert_heatmap_{variant}.svg")


# ── 9. SVD Analysis ──────────────────────────────────────────────────

def plot_svd(weight_dir, out_dir):
    """SVD effective rank and energy by tensor type."""
    svd_data = load_json(os.path.join(weight_dir, "svd_heretic.json"))
    if not svd_data or "tensor_results" not in svd_data:
        return

    results = svd_data["tensor_results"]

    # Group by category
    by_cat = defaultdict(list)
    for r in results:
        cat = r.get("category", r.get("tensor_type", "other"))
        # Extract the variant comparison
        variant_key = r.get("heretic_vs_base")
        if not variant_key:
            # Try last key that looks like a variant comparison
            for k in r:
                if "_vs_base" in k:
                    variant_key = r[k]
                    break
        if variant_key and isinstance(variant_key, dict):
            rank = variant_key.get("effective_rank_90pct_energy", 0)
            energy = variant_key.get("energy_top1_pct", 0)
            frob = variant_key.get("frobenius_norm", 0)
            if frob > 0:  # skip unchanged
                by_cat[cat].append({"rank": rank, "energy": energy})

    if not by_cat:
        return

    cats = sorted(by_cat.keys(), key=lambda c: len(by_cat[c]), reverse=True)
    # Shorten names
    short_names = [c.replace("mlp.experts.", "exp.").replace("mlp.shared_experts.", "shared.")
                    .replace("self_attn.", "attn.").replace(".weight", "") for c in cats]

    ranks_90 = [np.mean([x["rank"] for x in by_cat[c]]) for c in cats]
    energies = [np.mean([x["energy"] for x in by_cat[c]]) for c in cats]
    counts = [len(by_cat[c]) for c in cats]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

    x = np.arange(len(cats))

    # Rank plot
    bars1 = ax1.bar(x, ranks_90, color="#9b59b6", alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Effective Rank @ 90% Energy", fontsize=11)
    ax1.set_title("SVD Effective Rank by Tensor Type (Heretic vs Base, changed tensors only)",
                  fontweight="bold", fontsize=12)
    # Add count annotations
    for i, (bar, cnt) in enumerate(zip(bars1, counts)):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"n={cnt}", ha="center", va="bottom", fontsize=7, color="gray")

    # Energy plot
    ax2.bar(x, energies, color="#f39c12", alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Top-1 SV Energy (%)", fontsize=11)
    ax2.set_title("SVD Top-1 Singular Value Energy by Tensor Type", fontweight="bold", fontsize=12)
    ax2.axhline(y=90, color="red", linestyle="--", alpha=0.5, label="90% line")
    ax2.axhline(y=50, color="orange", linestyle="--", alpha=0.5, label="50% line")
    ax2.legend(fontsize=9)

    fig.suptitle("SVD Technique Analysis — GLM-4.7-Flash", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_svg(fig, out_dir, "weight_svd.svg")


# ── 10. Edit Density by Layer (All Variants Overlaid) ─────────────────

def plot_layer_edit_density(weight_dir, out_dir):
    """Overlayed line chart of edit density by layer for all variants."""
    fig, ax = plt.subplots(figsize=(16, 6))

    for variant in ALL_VARIANTS:
        d = load_json(os.path.join(weight_dir, f"layer_{variant}.json"))
        if not d:
            continue
        prog = d["layer_progression"]
        layers = sorted(prog.keys(), key=int)
        density = [prog[l]["edit_density"] for l in layers]
        ax.plot([int(l) for l in layers], density, color=COLORS[variant],
                linewidth=2, label=LABELS[variant], alpha=0.85)

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Edit Density (%)", fontsize=12)
    ax.set_title("Edit Density by Layer — All Techniques Compared\n(% of tensors in each layer with non-zero edits)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xlim(-1, 47)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_svg(fig, out_dir, "weight_layer_edit_density.svg")


# ── 11. Mean Norm Shift by Layer (All Variants Overlaid) ──────────────

def plot_layer_norm_shift(weight_dir, out_dir):
    """Overlayed line chart of mean norm shift by layer for all variants."""
    fig, ax = plt.subplots(figsize=(16, 6))

    for variant in ALL_VARIANTS:
        d = load_json(os.path.join(weight_dir, f"layer_{variant}.json"))
        if not d:
            continue
        prog = d["layer_progression"]
        layers = sorted(prog.keys(), key=int)
        shift = [prog[l]["mean_norm_shift"] for l in layers]
        ax.plot([int(l) for l in layers], shift, color=COLORS[variant],
                linewidth=2, label=LABELS[variant], alpha=0.85)

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Mean Norm Shift", fontsize=12)
    ax.set_title("Mean Norm Shift by Layer — All Techniques Compared\n(Negative = weight shrinkage, Positive = growth)",
                 fontsize=13, fontweight="bold")
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_xlim(-1, 47)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_svg(fig, out_dir, "weight_layer_norm_shift.svg")


# ── 12. Stacking / Cosine Scatter ────────────────────────────────────

def plot_stacking(weight_dir, out_dir):
    """Stacking analysis: edit norm comparison + reconstruction ratio."""
    data = load_json(os.path.join(weight_dir, "stacking_analysis.json"))
    if not data or "per_tensor" not in data:
        return

    metrics = data.get("stacking_metrics", {})
    per_tensor = data["per_tensor"]

    # Fields are norm_D_a, norm_D_b, norm_D_r, a_changed, b_changed, ratio_Dr_Db
    both = [t for t in per_tensor if t.get("a_changed", False) and t.get("b_changed", False)]
    a_only = [t for t in per_tensor if t.get("a_changed", False) and not t.get("b_changed", False)]
    b_only = [t for t in per_tensor if not t.get("a_changed", False) and t.get("b_changed", False)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: scatter of Da vs Db norms (for overlapping tensors)
    if both:
        da = [t["norm_D_a"] for t in both]
        db = [t["norm_D_b"] for t in both]
        ax1.scatter(da, db, alpha=0.3, s=8, color="#9b59b6")
        max_val = max(max(da), max(db)) * 1.1
        ax1.plot([0, max_val], [0, max_val], "r--", alpha=0.5, label="y=x")
        ax1.set_xlabel("Heretic Edit Norm (Da)", fontsize=11)
        ax1.set_ylabel("HauhauCS Edit Norm (Db)", fontsize=11)
        ax1.set_title(f"Edit Norm Comparison (n={len(both):,})",
                      fontsize=12, fontweight="bold")
        ax1.legend(fontsize=10)
    else:
        ax1.text(0.5, 0.5, f"No overlapping edits\nHeretic-only: {len(a_only)}\nHauhauCS-only: {len(b_only)}",
                 ha="center", va="center", transform=ax1.transAxes, fontsize=12)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Right: reconstruction ratio histogram
    ratios = [t.get("ratio_Dr_Db", 0) for t in per_tensor
              if t.get("b_changed", False) and t.get("ratio_Dr_Db", 0) > 0]
    if ratios:
        # Clip extreme outliers for visibility
        clipped = np.clip(ratios, 0, 5)
        ax2.hist(clipped, bins=50, color="#f39c12", alpha=0.7, edgecolor="white")
        ax2.axvline(x=1.0, color="red", linestyle="--", alpha=0.5, label="Ratio=1.0")
        ax2.axvline(x=np.mean(ratios), color="blue", linestyle="--",
                     label=f"Mean={np.mean(ratios):.3f}")
    ax2.set_xlabel("Reconstruction Ratio (Dr / Db)", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title("Edit Reconstruction Ratio Distribution",
                  fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Summary text
    def _metric_mean(m):
        if isinstance(m, dict):
            return m.get("mean", 0)
        return m or 0

    cos_val = _metric_mean(metrics.get("cos_Da_Db", 0))
    r2_val = _metric_mean(metrics.get("r2", 0))
    slope_val = _metric_mean(metrics.get("slope", 0))
    summary = (f"cos(Da,Db)={cos_val:.4f}   R²={r2_val:.4f}   slope={slope_val:.4f}   "
               f"shared={len(both)}   Heretic-only={len(a_only)}   HauhauCS-only={len(b_only)}")
    fig.text(0.5, 0.01, summary, ha="center", fontsize=9, style="italic",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    save_svg(fig, out_dir, "weight_stacking.svg")


# ── 13. Low-Rank Reconstruction Curves ────────────────────────────────

def plot_lowrank(weight_dir, out_dir):
    """Line chart of reconstruction error vs rank for edit matrices."""
    data = load_json(os.path.join(weight_dir, "lowrank_reconstruction.json"))
    if not data or "per_tensor" not in data:
        return

    ranks = data.get("ranks_tested", [1, 2, 5, 10, 20])
    per_tensor = data["per_tensor"]

    # Filter to tensors with meaningful edits
    edited = [t for t in per_tensor if t.get("edit_norm_b", 0) > 0]
    if not edited:
        return

    # Compute mean reconstruction error at each rank
    rank_keys = [f"recon_b_rank{r}_error_pct" for r in ranks]
    mean_errors = []
    for rk in rank_keys:
        vals = [t.get(rk, 100) for t in edited]
        mean_errors.append(np.mean(vals))

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(ranks, mean_errors, "o-", color="#e74c3c", linewidth=2, markersize=8, label="HauhauCS edits")

    # Also try Heretic edits (from lowrank_heretic_vs_abliterix if available)
    for fname, label, color in [
        ("lowrank_heretic_vs_abliterix.json", "Heretic→Abliterix", "#9b59b6"),
        ("lowrank_hauhau_vs_abliterix.json", "HauhauCS→Abliterix", "#3498db"),
        ("lowrank_huihui_vs_abliterix.json", "Huihui→Abliterix", "#2ecc71"),
    ]:
        lr = load_json(os.path.join(weight_dir, fname))
        if lr and "per_tensor" in lr:
            lr_edited = [t for t in lr["per_tensor"] if t.get("edit_norm_b", 0) > 0]
            if lr_edited:
                lr_ranks = lr.get("ranks_tested", ranks)
                lr_errors = []
                for r in lr_ranks:
                    rk = f"recon_b_rank{r}_error_pct"
                    vals = [t.get(rk, 100) for t in lr_edited]
                    lr_errors.append(np.mean(vals))
                ax.plot(lr_ranks, lr_errors, "o--", color=color, linewidth=1.5,
                        markersize=6, alpha=0.7, label=label)

    ax.set_xlabel("Reconstruction Rank", fontsize=12)
    ax.set_ylabel("Mean Reconstruction Error (%)", fontsize=12)
    ax.set_title("Low-Rank Reconstruction of Edit Matrices\n(Lower = more compressible edits)",
                 fontsize=13, fontweight="bold")
    ax.set_xscale("log")
    ax.legend(fontsize=10)
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.3, label="50% error")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_svg(fig, out_dir, "weight_lowrank.svg")


# ── 14. Subspace Alignment ───────────────────────────────────────────

def plot_subspace(weight_dir, out_dir):
    """Histogram of principal angles between edit subspaces."""
    data = load_json(os.path.join(weight_dir, "subspace_alignment.json"))
    if not data:
        return

    angles_ab = data.get("all_principal_angles_ab", [])
    angles_ac = data.get("principal_angles_ac", [])
    angles_bc = data.get("principal_angles_bc", [])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    datasets = [
        (angles_ab, "Heretic vs HauhauCS", COLORS["overlap"]),
        (angles_ac, "Heretic vs Huihui", COLORS["heretic"]),
        (angles_bc, "HauhauCS vs Huihui", COLORS["huihui"]),
    ]

    for ax, (angles, title, color) in zip(axes, datasets):
        if angles:
            cos_angles = np.cos(angles)
            ax.hist(cos_angles, bins=50, color=color, alpha=0.7, edgecolor="white")
            mean_cos = np.mean(cos_angles)
            ax.axvline(x=mean_cos, color="red", linestyle="--",
                        label=f"Mean cos(θ)={mean_cos:.3f}")
        ax.set_xlabel("cos(Principal Angle)", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(f"Subspace Alignment — Principal Angles Between Edit Subspaces\n"
                 f"Global mean cos(θ)={data.get('global_mean_cosine_principal', 0):.4f}, "
                 f"Overlap fraction >0.9: {data.get('global_overlap_fraction_gt_0.9', 0):.3f}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    save_svg(fig, out_dir, "weight_subspace.svg")


# ── 15. Router Edit Heatmap ──────────────────────────────────────────

def plot_router_edits(weight_dir, out_dir):
    """Heatmap of router/gate weight edits by layer and variant."""
    data = load_json(os.path.join(weight_dir, "expert_analysis.json"))
    if not data or "router_edits" not in data:
        return

    router_edits = data["router_edits"]
    layers = sorted(router_edits.keys(), key=int)
    variants = THREE_VARIANTS

    grid = np.zeros((len(variants), len(layers)))
    for vi, v in enumerate(variants):
        for li, layer in enumerate(layers):
            info = router_edits.get(layer, {}).get(v, {})
            grid[vi, li] = info.get("edit_norm", 0)

    fig, ax = plt.subplots(figsize=(16, 4))
    sns.heatmap(grid, ax=ax, cmap="YlOrRd",
                 xticklabels=[f"L{l}" for l in layers],
                 yticklabels=[LABELS[v] for v in variants],
                 linewidths=0.5, cbar_kws={"label": "Edit Norm"})
    ax.set_title("Router (Gate) Weight Edits by Layer and Technique\n(MoE expert routing weights)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Layer", fontsize=11)
    ax.tick_params(axis="x", rotation=45, labelsize=7)

    save_svg(fig, out_dir, "weight_router_edits.svg")


# ── 16. Shared Expert Edits ──────────────────────────────────────────

def plot_shared_expert_edits(weight_dir, out_dir):
    """Heatmap of shared expert weight edits by layer and variant."""
    data = load_json(os.path.join(weight_dir, "expert_analysis.json"))
    if not data or "shared_expert_edits" not in data:
        return

    shared_edits = data["shared_expert_edits"]
    layers = sorted(shared_edits.keys(), key=int)
    variants = THREE_VARIANTS

    # Collect all tensor types within shared experts
    # Each layer entry has per-variant data, each variant has multiple tensor types
    # We'll aggregate per layer
    grid = np.zeros((len(variants), len(layers)))
    for vi, v in enumerate(variants):
        for li, layer in enumerate(layers):
            info = shared_edits.get(layer, {}).get(v, {})
            grid[vi, li] = info.get("edit_norm", 0)

    fig, ax = plt.subplots(figsize=(16, 4))
    sns.heatmap(grid, ax=ax, cmap="YlOrRd",
                 xticklabels=[f"L{l}" for l in layers],
                 yticklabels=[LABELS[v] for v in variants],
                 linewidths=0.5, cbar_kws={"label": "Edit Norm"})
    ax.set_title("Shared Expert Weight Edits by Layer and Technique\n(GLM-4.7 shared expert MLP weights)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Layer", fontsize=11)
    ax.tick_params(axis="x", rotation=45, labelsize=7)

    save_svg(fig, out_dir, "weight_shared_expert_edits.svg")


# ── Main ──────────────────────────────────────────────────────────────

PLOTS = [
    ("pairwise_cosine",     plot_pairwise_cosine),
    ("tensor_types",        plot_tensor_types),
    ("scope",               plot_scope_comparison),
    ("magnitude",           plot_edit_magnitude),
    ("layer_profile",       plot_layer_profile),
    ("venn",                plot_venn),
    ("aggressiveness",      plot_aggressiveness),
    ("expert_heatmaps",     plot_expert_heatmaps),
    ("svd",                 plot_svd),
    ("layer_edit_density",  plot_layer_edit_density),
    ("layer_norm_shift",    plot_layer_norm_shift),
    ("stacking",            plot_stacking),
    ("lowrank",             plot_lowrank),
    ("subspace",            plot_subspace),
    ("router_edits",        plot_router_edits),
    ("shared_expert_edits", plot_shared_expert_edits),
]


def main():
    parser = argparse.ArgumentParser(description="Generate weight analysis graphs for GLM-4.7-Flash")
    parser.add_argument("--weight-dir", required=True,
                        help="Directory containing weight analysis JSON files")
    parser.add_argument("--out-dir", default="./graphs",
                        help="Output directory for SVG files (default: ./graphs)")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Only generate specific plots (by name)")
    args = parser.parse_args()

    weight_dir = args.weight_dir
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    print(f"Generating weight analysis graphs...")
    print(f"  Data: {weight_dir}")
    print(f"  Output: {out_dir}")
    print()

    for name, func in PLOTS:
        if args.only and name not in args.only:
            continue
        print(f"[{name}]")
        try:
            func(weight_dir, out_dir)
        except Exception as e:
            print(f"  ✗ Error: {e}", file=sys.stderr)
        print()

    print(f"Done. Graphs saved to {out_dir}/")


if __name__ == "__main__":
    main()
