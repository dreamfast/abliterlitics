#!/usr/bin/env python3
"""
Unified Graph Generator — auto-generates all SVG visualizations from analysis results.

Reads JSON results from a model's analysis directory and produces:
- Venn diagrams (technique edit overlap; 2 variants → one 2-way Venn, 3 → one
  3-way Venn, 4+ → pairwise 2-way Venns)
- Aggressiveness bar charts
- Layer-wise magnitude bar charts
- SVD rank/energy plots
- Expert heatmaps (MoE models only; dense models are skipped)

Variant names are auto-discovered from result files — no hardcoded names.

All output SVGs are stamped with the ``RESULTS_VERSION`` from ``src/__init__.py``
in their SVG metadata (``Description: results_version=N``), so consumers can tell
which result schema version produced them.

Usage:
    python3 generate_graphs.py --results-dir /results/qwen35_2b --output-dir /graphs/qwen35_2b
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src import RESULTS_VERSION

try:
    from matplotlib_venn import venn2, venn3

    HAS_VENN = True
except ImportError:
    HAS_VENN = False

# Okabe-Ito color-blind-safe palette (distinguishable under deuteranopia,
# protanopia, and tritanopia). Replaces seaborn "husl", which is hue-uniform
# and CVD-hostile — the code-review agents require color-blind-friendly
# palettes for the published SVG graphs.
OKABE_ITO: tuple[str, ...] = (
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#000000",  # black
)

# Neutral gray fallback for unknown variant names — CVD-safe by construction.
DEFAULT_COLOR = "#888888"

sns.set_theme(style="darkgrid", palette="colorblind")

log = logging.getLogger(__name__)


def get_variant_colors(variant_names: list[str]) -> dict[str, str]:
    """Map variant names to color-blind-safe colors from the Okabe-Ito palette."""
    colors: dict[str, str] = {}
    for i, name in enumerate(sorted(variant_names)):
        colors[name] = OKABE_ITO[i % len(OKABE_ITO)]
    colors["base"] = "#95a5a6"
    colors["overlap"] = "#E69F00"
    return colors


def get_color(colors: dict[str, str], name: str) -> str:
    """Look up a variant color with a neutral fallback for unknown names."""
    return colors.get(name, DEFAULT_COLOR)


def load_json(path: Path | str) -> dict | None:
    """Load a results JSON file, logging (never silently swallowing) errors.

    Returns None (after a log.warning with path + reason) when the file is
    missing or contains corrupt JSON — callers skip that input. Other
    OS-level read failures are re-raised with the path in the message.
    """
    p = Path(path)
    try:
        data: dict = json.loads(p.read_text())
    except FileNotFoundError:
        log.warning("results file not found — skipping: %s", p)
        return None
    except json.JSONDecodeError as e:
        log.warning("corrupt JSON in %s (%s) — skipping", p, e)
        return None
    except OSError as e:
        raise RuntimeError(f"Failed to read results file {p}: {e}") from e
    v = data.get("results_version")
    if v is not None and v != RESULTS_VERSION:
        log.warning("results_version mismatch in %s: got %s, expected %s", p, v, RESULTS_VERSION)
    return data


def _save_fig(fig: matplotlib.figure.Figure, out_dir: Path, fname: str) -> None:
    """Save an SVG stamped with RESULTS_VERSION metadata, then close the figure.

    (matplotlib's SVG writer only accepts a fixed set of metadata keys —
    the version stamp rides on the supported ``Description`` key.)
    """
    fig.savefig(
        out_dir / fname,
        format="svg",
        bbox_inches="tight",
        metadata={
            "Creator": "abliterlitics generate_graphs",
            "Description": f"results_version={RESULTS_VERSION}",
        },
    )
    plt.close(fig)
    log.info("%s", fname)


def _discover_variants_from_panel(panel: dict) -> list[str]:
    """Extract sorted variant names from the panel comparison data."""
    variants = set()
    for pair_key in panel.get("pairwise_changed_counts", {}):
        parts = pair_key.split("->")
        if len(parts) == 2:
            variants.add(parts[1])
    return sorted(variants)


def _venn2(
    sa: set,
    sb: set,
    a: str,
    b: str,
    title: str,
    out_dir: Path,
    fname: str,
    colors: dict[str, str],
) -> None:
    """Render one 2-way Venn diagram (edit-set overlap of a vs b) and save it."""
    only_a = len(sa - sb)
    only_b = len(sb - sa)
    overlap = len(sa & sb)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    venn2(
        subsets=(only_a, only_b, overlap),
        set_labels=(a.capitalize(), b.capitalize()),
        set_colors=(get_color(colors, a), get_color(colors, b)),
        alpha=0.7,
        ax=ax,
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    _save_fig(fig, out_dir, fname)


def gen_venn(panel: dict | None, out_dir: Path, label: str, colors: dict[str, str]) -> None:
    """Venn diagrams of per-variant edited-tensor sets from the panel data.

    2 variants → one 2-way Venn; 3 variants → one 3-way Venn; 4+ variants →
    pairwise 2-way Venns (one file per pair).
    """
    if not panel:
        return

    sets: dict[str, set] = {}
    for pair_key in panel.get("pairwise_changed_counts", {}):
        keys = set(panel.get(f"{pair_key}_keys", []))
        sets[pair_key] = keys

    if not sets:
        return

    # Extract only base->variant sets (skip variant->variant pairs)
    base_sets = {pk: s for pk, s in sets.items() if pk.startswith("base->")}
    variant_names = sorted(pk.removeprefix("base->") for pk in base_sets)

    if not variant_names:
        return

    if not HAS_VENN:
        log.warning("matplotlib_venn unavailable — venn graphs skipped")
        return

    if len(variant_names) < 2:
        log.info("need at least 2 base->variant pairs for a venn — got %d", len(variant_names))
        return

    if len(variant_names) == 3:
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
            set_colors=(get_color(colors, a), get_color(colors, b), get_color(colors, c)),
            alpha=0.7,
            ax=ax,
        )
        ax.set_title(f"{label} — Technique Edit Overlap", fontsize=14, fontweight="bold", pad=20)
        _save_fig(fig, out_dir, f"venn_{label}.svg")

    elif len(variant_names) == 2:
        a, b = variant_names
        _venn2(
            base_sets[f"base->{a}"],
            base_sets[f"base->{b}"],
            a,
            b,
            f"{label} — Edit Overlap",
            out_dir,
            f"venn_{label}.svg",
            colors,
        )

    else:  # 4+ variants → pairwise 2-way Venns
        for i in range(len(variant_names)):
            for j in range(i + 1, len(variant_names)):
                a, b = variant_names[i], variant_names[j]
                _venn2(
                    base_sets[f"base->{a}"],
                    base_sets[f"base->{b}"],
                    a,
                    b,
                    f"{label} — {a.capitalize()} vs {b.capitalize()} Overlap",
                    out_dir,
                    f"venn_{label}_{a}_vs_{b}.svg",
                    colors,
                )


def gen_aggressiveness_bar(panel: dict | None, out_dir: Path, label: str, colors: dict[str, str]) -> None:
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
    bar_colors = [get_color(colors, v) for v in variants if f"base->{v}" in counts]
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
    _save_fig(fig, out_dir, f"aggressiveness_{label}.svg")


def gen_layer_heatmap(layer_data, out_dir, label, variant="", colors=None):
    """Bar charts of layer-wise edit metrics (one file per metric)."""
    if not layer_data:
        return

    if colors is None:
        colors = {}

    layers: list[tuple[int, str]] = []
    for key in layer_data:
        try:
            layers.append((int(key), key))
        except (TypeError, ValueError):
            log.warning("non-numeric layer key %r in layer_progression — skipping layer", key)
    layers.sort()

    metrics = ["edit_density", "mean_edit_norm", "mean_norm_shift"]
    metric_labels = ["Edit Density (%)", "Mean Edit Norm", "Mean Norm Shift"]

    for metric, mlabel in zip(metrics, metric_labels, strict=False):
        vals = [layer_data[key].get(metric, 0) for _num, key in layers]
        if all(v == 0 for v in vals):
            continue

        fig, ax = plt.subplots(figsize=(max(12, len(layers) * 0.4), 5))
        ax.bar([f"L{num}" for num, _key in layers], vals, color=get_color(colors, variant), alpha=0.8)
        ax.set_xlabel("Layer")
        ax.set_ylabel(mlabel)
        variant_suffix = f" ({variant})" if variant else ""
        ax.set_title(f"{label} — {mlabel} by Layer{variant_suffix}", fontsize=13, fontweight="bold")
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        # Include variant in filename so multiple variants don't overwrite each other
        fname_variant = f"_{variant}" if variant else ""
        _save_fig(fig, out_dir, f"layer_{metric}_{label}{fname_variant}.svg")


def gen_svd_plots(svd_data, out_dir, label, variant=""):
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
    ax1.bar(x, ranks_90, color="#CC79A7", alpha=0.8)  # Okabe-Ito reddish purple
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_plot, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Effective Rank @ 90% Energy")
    ax1.set_title(f"{label} — SVD Rank by Tensor Type", fontweight="bold")

    ax2.bar(x, energies, color="#E69F00", alpha=0.8)  # Okabe-Ito orange
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_plot, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Top-1 SV Energy (%)")
    ax2.set_title(f"{label} — SVD Energy by Tensor Type", fontweight="bold")
    ax2.axhline(y=90, color="red", linestyle="--", alpha=0.5)
    ax2.axhline(y=50, color="orange", linestyle="--", alpha=0.5)

    variant_suffix = f" ({variant})" if variant else ""
    # Include variant in filename so multiple variants don't overwrite each other
    fname_variant = f"_{variant}" if variant else ""
    fig.suptitle(f"SVD Technique Analysis — {label}{variant_suffix}", fontsize=14, fontweight="bold")
    _save_fig(fig, out_dir, f"svd_summary_{label}{fname_variant}.svg")


def gen_expert_heatmap(expert_data, out_dir, label):
    """Expert × layer heatmap of edit magnitudes (MoE models only)."""
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
        _save_fig(fig, out_dir, f"expert_heatmap_{variant}_{label}.svg")


def main():
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

    # Auto-discover layer analysis files (per-variant; rglob searches subdirs)
    layer_candidates = sorted(rdir.rglob("layer_analysis*.json"))
    for lc in layer_candidates:
        d = load_json(lc)
        if not d:
            continue
        layer_data = d.get("layer_progression")
        if not layer_data:
            continue
        # Extract variant name from filename like layer_analysis_t122.json
        stem = lc.stem.replace("layer_analysis", "").strip("_")
        gen_layer_heatmap(layer_data, odir, args.label, variant=stem, colors=colors)

    # Auto-discover SVD files (per-variant; rglob searches subdirs)
    svd_candidates = sorted(rdir.rglob("svd_*.json"))
    for sc in svd_candidates:
        d = load_json(sc)
        if not d:
            continue
        # Extract variant name from filename like svd_t122.json
        stem = sc.stem.replace("svd_", "").replace("svd", "").strip("_")
        gen_svd_plots(d, odir, args.label, variant=stem)

    # Auto-discover expert analysis files (MoE only; rglob searches subdirs)
    for ec in sorted(rdir.rglob("expert_analysis*.json")):
        expert_data = load_json(ec)
        if not expert_data:
            continue
        if "error" in expert_data:
            log.warning("expert analysis failed for %s — skipping: %s", ec, expert_data["error"])
            continue
        if expert_data.get("architecture", {}).get("experts", 0) > 0:
            gen_expert_heatmap(expert_data, odir, args.label)
        else:
            log.info("dense model (0 experts) — skipping expert heatmap: %s", ec)

    log.info("Done. Graphs saved to %s/", odir)


if __name__ == "__main__":
    main()
