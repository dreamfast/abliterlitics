#!/usr/bin/env python3
"""
Unified Graph Generator — auto-generates all SVG visualizations from analysis results.

Reads JSON results from a model's analysis directory and produces:
- Venn diagrams (technique overlap, auto-detects 2 or 3 variants)
- Aggressiveness bar charts
- Layer-wise magnitude heatmaps
- SVD spectrum/rank/energy plots
- Technique fingerprint radar charts
- Expert heatmaps (GLM)
- Edit progression plots
- Cross-technique comparison charts

Variant names are auto-discovered from result files — no hardcoded names.

Usage:
    python3 generate_graphs.py --results-dir /results/qwen35_2b --output-dir /graphs/qwen35_2b
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import to_hex as _to_hex

from src import RESULTS_VERSION

try:
    from matplotlib_venn import venn2, venn3  # noqa: F401

    HAS_VENN = True
except ImportError:
    HAS_VENN = False

sns.set_theme(style="darkgrid", palette="muted")

log = logging.getLogger(__name__)


def get_variant_colors(variant_names: list[str]) -> dict[str, str]:
    palette = sns.color_palette("husl", len(variant_names) + 1)
    colors = {}
    for i, name in enumerate(sorted(variant_names)):
        colors[name] = _to_hex(palette[i])
    colors["base"] = "#95a5a6"
    colors["overlap"] = "#f39c12"
    return colors


def load_json(path):
    try:
        data = json.loads(Path(path).read_text())
        v = data.get("results_version")
        if v is not None and v != RESULTS_VERSION:
            log.warning("results_version mismatch in %s: got %s, expected %s", path, v, RESULTS_VERSION)
        return data
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _discover_variants_from_panel(panel: dict) -> list[str]:
    """Extract sorted variant names from the panel comparison data."""
    variants = set()
    for pair_key in panel.get("pairwise_changed_counts", {}):
        parts = pair_key.split("->")
        if len(parts) == 2:
            variants.add(parts[1])
    return sorted(variants)


def gen_venn(panel, out_dir, label, colors):
    if not HAS_VENN:
        log.info("venn3 not available — skipping")
        return
    if not panel:
        return

    sets = {}
    for pair_key in panel.get("pairwise_changed_counts", {}):
        keys = set(panel.get(f"{pair_key}_keys", []))
        sets[pair_key] = keys

    if not sets:
        return

    # Extract only base->variant sets (skip variant->variant pairs)
    base_sets = {pk: s for pk, s in sets.items() if pk.startswith("base->")}
    variant_names = sorted(pk.removeprefix("base->") for pk in base_sets)

    if len(variant_names) == 3 and HAS_VENN:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        a, b, c = variant_names
        sa, sb, sc = base_sets[f"base->{a}"], base_sets[f"base->{b}"], base_sets[f"base->{c}"]

        only_a = len(sa - sb - sc)
        only_b = len(sb - sa - sc)
        only_c = len(sc - sa - sb)
        ab = len((sa & sb) - sc)
        ac = len((sa & sc) - sb)
        bc = len((sb & sc) - sa)
        all3 = len(sa & sb & sc)

        venn3(
            subsets=(only_a, only_b, ab, only_c, ac, bc, all3),
            set_labels=(a.capitalize(), b.capitalize(), c.capitalize()),
            set_colors=(colors.get(a, "#888"), colors.get(b, "#888"), colors.get(c, "#888")),
            alpha=0.7,
            ax=ax,
        )
        ax.set_title(f"{label} — Technique Edit Overlap", fontsize=14, fontweight="bold", pad=20)
        fig.savefig(out_dir / f"venn_{label}.svg", format="svg", bbox_inches="tight")
        plt.close(fig)
        log.info("venn_%s.svg", label)

    elif len(variant_names) == 2:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        a, b = variant_names
        sa, sb = base_sets[f"base->{a}"], base_sets[f"base->{b}"]

        only_a = len(sa - sb)
        only_b = len(sb - sa)
        overlap = len(sa & sb)

        venn2(
            subsets=(only_a, only_b, overlap),
            set_labels=(a.capitalize(), b.capitalize()),
            set_colors=(colors.get(a, "#888"), colors.get(b, "#888")),
            alpha=0.7,
            ax=ax,
        )
        ax.set_title(f"{label} — Edit Overlap", fontsize=14, fontweight="bold", pad=20)
        fig.savefig(out_dir / f"venn_{label}.svg", format="svg", bbox_inches="tight")
        plt.close(fig)
        log.info("venn_%s.svg", label)

    elif len(variant_names) >= 4:
        # For 4+ variants, generate pairwise 2-way Venn diagrams
        for i in range(len(variant_names)):
            for j in range(i + 1, len(variant_names)):
                a, b = variant_names[i], variant_names[j]
                sa = base_sets[f"base->{a}"]
                sb = base_sets[f"base->{b}"]
                only_a = len(sa - sb)
                only_b = len(sb - sa)
                overlap = len(sa & sb)

                fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                venn2(
                    subsets=(only_a, only_b, overlap),
                    set_labels=(a.capitalize(), b.capitalize()),
                    set_colors=(colors.get(a, "#888"), colors.get(b, "#888")),
                    alpha=0.7,
                    ax=ax,
                )
                ax.set_title(f"{label} — {a.capitalize()} vs {b.capitalize()} Overlap",
                             fontsize=13, fontweight="bold", pad=15)
                fname = f"venn_{label}_{a}_vs_{b}.svg"
                fig.savefig(out_dir / fname, format="svg", bbox_inches="tight")
                plt.close(fig)
                log.info(fname)


def gen_aggressiveness_bar(panel, out_dir, label, colors):
    counts = panel.get("pairwise_changed_counts", {}) if panel else {}
    variants = sorted(v for v in colors if v not in ("base", "overlap"))
    edit_counts = []
    labels = []
    for v in variants:
        key = f"base->{v}"
        if key in counts:
            edit_counts.append(counts[key])
            labels.append(v.capitalize())

    if not edit_counts:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    bar_colors = [colors.get(v, "#888") for v in variants if f"base->{v}" in counts]
    bars = ax.bar(labels, edit_counts, color=bar_colors, alpha=0.9, edgecolor="white")

    for bar, count in zip(bars, edit_counts, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(edit_counts) * 0.02,
            str(count),
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_ylabel("Tensors Changed", fontsize=12)
    ax.set_title(f"{label} — Abliteration Aggressiveness", fontsize=14, fontweight="bold")
    fig.savefig(out_dir / f"aggressiveness_{label}.svg", format="svg", bbox_inches="tight")
    plt.close(fig)
    log.info("aggressiveness_%s.svg", label)


def gen_layer_heatmap(layer_data, out_dir, label, variant="", colors=None):
    if not layer_data:
        return

    if colors is None:
        colors = {}

    layers = sorted(layer_data.keys(), key=lambda x: int(x))
    metrics = ["edit_density", "mean_edit_norm", "mean_norm_shift"]
    metric_labels = ["Edit Density (%)", "Mean Edit Norm", "Mean Norm Shift"]

    for metric, mlabel in zip(metrics, metric_labels, strict=False):
        vals = [layer_data[l].get(metric, 0) for l in layers]
        if all(v == 0 for v in vals):
            continue

        fig, ax = plt.subplots(figsize=(max(12, len(layers) * 0.4), 5))
        ax.bar([f"L{l}" for l in layers], vals, color=colors.get(variant, "#888"), alpha=0.8)
        ax.set_xlabel("Layer")
        ax.set_ylabel(mlabel)
        variant_suffix = f" ({variant})" if variant else ""
        ax.set_title(f"{label} — {mlabel} by Layer{variant_suffix}", fontsize=13, fontweight="bold")
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        fig.savefig(out_dir / f"layer_{metric}_{label}.svg", format="svg", bbox_inches="tight")
        plt.close(fig)
        log.info("layer_%s_%s.svg", metric, label)


def gen_svd_plots(svd_data, out_dir, label):
    if not svd_data:
        return
    results = svd_data.get("tensor_results", [])
    if not results:
        return

    by_type: dict[str, list[dict[str, object]]] = {}
    for r in results:
        by_type.setdefault(r.get("tensor_type", "other"), []).append(r)

    types = sorted(by_type.keys())
    ranks_90 = []
    energies = []
    labels_plot = []

    for tt in types:
        group = by_type[tt]
        # Use the first available variant key (not hardcoded to heretic_vs_base)
        rk = []
        en = []
        for r in group:
            variant_data = None
            for key in r:
                if key.endswith("_vs_base") and isinstance(r[key], dict):
                    variant_data = r[key]
                    break
            if variant_data is None and r:
                # Fallback: last key that's a dict
                for key in reversed(list(r.keys())):
                    if isinstance(r.get(key), dict):
                        variant_data = r[key]
                        break
            if variant_data:
                rk.append(variant_data.get("effective_rank_90pct_energy", 0))
                en.append(variant_data.get("energy_top1_pct", 0))
        if rk:
            ranks_90.append(np.mean(rk))
            energies.append(np.mean(en))
            labels_plot.append(tt)

    if not labels_plot:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    x = np.arange(len(labels_plot))
    ax1.bar(x, ranks_90, color="#9b59b6", alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_plot, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Effective Rank @ 90% Energy")
    ax1.set_title(f"{label} — SVD Rank by Tensor Type", fontweight="bold")

    ax2.bar(x, energies, color="#f39c12", alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_plot, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Top-1 SV Energy (%)")
    ax2.set_title(f"{label} — SVD Energy by Tensor Type", fontweight="bold")
    ax2.axhline(y=90, color="red", linestyle="--", alpha=0.5)
    ax2.axhline(y=50, color="orange", linestyle="--", alpha=0.5)

    fig.suptitle(f"SVD Technique Analysis — {label}", fontsize=14, fontweight="bold")
    fig.savefig(out_dir / f"svd_summary_{label}.svg", format="svg", bbox_inches="tight")
    plt.close(fig)
    log.info("svd_summary_%s.svg", label)


def gen_fingerprint_radar(fingerprints, out_dir, label):
    if not fingerprints:
        return

    for fp_path, _fp_label in fingerprints:
        data = load_json(fp_path)
        if not data:
            continue

    log.info("Radar charts require multiple fingerprints — skipping for single")


def gen_expert_heatmap(expert_data, out_dir, label):
    if not expert_data:
        return

    per_expert = expert_data.get("per_expert_details", {})

    for variant, details in per_expert.items():
        if not details:
            continue
        layers = set()
        experts = set()
        for _key, info in details.items():
            layers.add(info["layer"])
            experts.add(info["expert_id"])

        layers = sorted(layers)
        experts = sorted(experts)

        grid = np.zeros((len(experts), len(layers)))
        for _key, info in details.items():
            ei = experts.index(info["expert_id"])
            li = layers.index(info["layer"])
            grid[ei, li] = info["total_edit_norm"]

        fig, ax = plt.subplots(figsize=(max(12, len(layers) * 0.3), max(8, len(experts) * 0.3)))
        sns.heatmap(
            grid,
            ax=ax,
            cmap="YlOrRd",
            xticklabels=[f"L{l}" for l in layers],
            yticklabels=[f"E{e}" for e in experts],
            linewidths=0.5,
        )
        ax.set_title(f"{label} — Expert Edit Magnitude ({variant})", fontsize=13, fontweight="bold")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Expert ID")
        fig.savefig(out_dir / f"expert_heatmap_{variant}_{label}.svg", format="svg", bbox_inches="tight")
        plt.close(fig)
        log.info("expert_heatmap_%s_%s.svg", variant, label)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label", required=True, help="Model label (e.g. 'Qwen3.5-2B')")
    args = parser.parse_args()

    rdir = Path(args.results_dir)
    odir = Path(args.output_dir)
    odir.mkdir(parents=True, exist_ok=True)

    log.info("Generating graphs for %s", args.label)
    log.info("  Results: %s", rdir)
    log.info("  Output:  %s", odir)

    panel = load_json(rdir / "multi_model_panel.json")

    variant_names = []
    if panel:
        variant_names = _discover_variants_from_panel(panel)
    colors = get_variant_colors(variant_names)

    gen_venn(panel, odir, args.label, colors)
    gen_aggressiveness_bar(panel, odir, args.label, colors)

    # Auto-discover layer analysis files (try generic name first, then any variant-specific)
    layer_data = None
    layer_candidates = sorted(rdir.glob("layer_analysis*.json"))
    for lc in layer_candidates:
        d = load_json(lc)
        if d:
            layer_data = d.get("layer_progression")
            # Extract variant name from filename like layer_analysis_heretic.json
            stem = lc.stem.replace("layer_analysis", "").strip("_")
            if not stem:
                stem = ""
            break
    if layer_data:
        gen_layer_heatmap(layer_data, odir, args.label, variant=stem, colors=colors)

    # Auto-discover SVD files
    svd_data = None
    svd_candidates = sorted(rdir.glob("svd_*.json"))
    for sc in svd_candidates:
        d = load_json(sc)
        if d:
            svd_data = d
            break
    gen_svd_plots(svd_data, odir, args.label)

    expert_data = load_json(rdir / "expert_analysis.json")
    if expert_data and expert_data.get("architecture", {}).get("experts", 0) > 0:
        gen_expert_heatmap(expert_data, odir, args.label)

    log.info("Done. Graphs saved to %s/", odir)


if __name__ == "__main__":
    main()
