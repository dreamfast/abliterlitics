#!/usr/bin/env python3
"""
Text-encoder report generator — SVG graphs + self-contained HTML report.

Reads te_<variant>.json results (written by run_analysis.py) and produces:
- te_layer_profile.svg   — per-layer MSE (log) with the consumption-point marker.
                           THE money chart: shows where abliteration damage lands
                           relative to what the DiT reads.
- te_damage_ranking.svg  — damage in quant-floor units by stage, with the
                           "one quantization step" reference line.
- te_category_gradient.svg — per-category MSE on the DiT-input stream
                           (taboo-concentration of transmitted damage).
- te_layer_heatmap.svg   — variants x layers heatmap of log10 MSE.
- te_report.html         — self-contained HTML (SVGs embedded base64) with
                           tables + auto-generated key findings.

Usage:
    python3 generate_report.py --results-dir /results/te-zimage/text_encoder --output-dir /graphs
"""

from __future__ import annotations

import base64
import html
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.text_encoder import TE_RESULTS_VERSION

log = logging.getLogger(__name__)

# The layer each backend's DiT consumes from the TE (empirically verified
# against ComfyUI model classes). Used to draw the consumption-point marker.
BACKEND_CONSUMED_LAYER: dict[str, int | None] = {
    "zimage": 34,  # Qwen3_4BModel(layer="hidden", layer_idx=-2) of 36 layers
    "ltx2": None,  # aggregates ALL layers through the projection — no single point
}

# Streams shown in the damage-ranking chart (the transmission pipeline).
PIPELINE_STREAMS = ["consumed_hidden", "cap_embedded", "refined_context", "video_proj", "audio_proj"]

COLORS = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860", "#da8bc3", "#8c8c8c"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_results(results_dir: Path) -> list[dict[str, Any]]:
    """Load all te_*.json result files.

    Each JSON's ``variants`` list carries one entry PER STAGE for the same
    model (run_analysis emits a VariantReport per stage) — entries are merged
    by name so each model appears once with all its stage streams.

    Returns: [{name, dtype, backend, reference, streams: {stream_name: stream_dict}}]
    """
    merged: dict[str, dict[str, Any]] = {}
    for f in sorted(results_dir.glob("te_*.json")):
        if f.name == "te_report.json":
            continue
        try:
            data = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Failed to parse %s: %s", f, exc)
            continue
        name = f.stem.replace("te_", "", 1)
        # Per-variant comparison reference: a variant NAME (e.g. "base_int8")
        # when reference-name was passed, else the global base filename.
        env = data.get("env", {})
        ref = env.get("reference") or data.get("base_id", "")
        ref = Path(ref).name if ref else "base"
        floor = data.get("reference_floors", {}).get("quant", "fakequant_int8")
        for var in data.get("variants", []):
            entry = merged.setdefault(
                name,
                {
                    "name": name,
                    "variant_id": var.get("variant_id", name),
                    "dtype": var.get("dtype", ""),
                    "gguf": var.get("gguf"),
                    "backend": data.get("backend", ""),
                    "reference": ref,
                    "floor": floor,
                    "streams": {},
                },
            )
            for sr in var.get("streams", []):
                if sr["stream"] in entry["streams"]:
                    log.warning("Duplicate stream name %s in %s — overwriting (format drift?)", sr["stream"], f.name)
                entry["streams"][sr["stream"]] = sr
    if not merged:
        raise FileNotFoundError(f"No te_*.json variant results found in {results_dir}")
    variants = list(merged.values())
    log.info("Loaded %d variant results from %s", len(variants), results_dir)
    return variants


def _is_reference_variant(var: dict[str, Any]) -> bool:
    """Non-abliterated quantizations of the base (noise-floor references).

    Convention: variants named base_* — the pure-quant damage measurements.
    """
    return bool(var["name"].startswith("base_"))


# ---------------------------------------------------------------------------
# Variant grouping: family (base/heretic) x container (safetensors/GGUF)
# ---------------------------------------------------------------------------

# Quant display order within a group (bpw-descending for GGUF).
_SAFETENSORS_QUANT_ORDER: dict[str, float] = {
    "": 0.0,  # plain heretic / base (BF16)
    "int8": 1.0,
    "fp8": 2.0,
    "fp8_e4m3fn": 2.0,
    "mxfp8": 3.0,
    "nvfp4": 4.0,
    "int4": 5.0,
}


def _variant_group(var: dict[str, Any]) -> tuple[int, str]:
    """Group key for report tables: (order, header label).

    Groups by family (pure-quantization reference vs abliterated) and
    container format; dual-run combined entries get their own trailing
    group (they duplicate the primary variant's streams — the decomposition
    table is their canonical view).
    """
    name = var["name"]
    # Container from persisted GGUFMeta when present, else from the name
    # (divergence summaries carry names only).
    is_gguf = bool(var.get("gguf")) or "_gguf_" in name
    if name.endswith("__vs_base"):
        return (4, "Abliterated quant — combined vs global base (dual runs)")
    if _is_reference_variant(var):
        if is_gguf:
            return (1, "Base model — GGUF containers (pure quantization)")
        return (0, "Base model — safetensors quants (pure quantization)")
    if is_gguf:
        return (3, "Abliterated (heretic) — GGUF containers")
    return (2, "Abliterated (heretic) — safetensors")


def _group_sort_key(var: dict[str, Any]) -> tuple[float, str]:
    """Sort within a group: quant tier (safetensors) or bpw-descending (GGUF)."""
    if var.get("gguf"):
        return (-float(var["gguf"].get("bpw", 0) or 0), var["name"])
    base = var["name"].replace("__vs_base", "")
    _prefix, _sep, suffix = base.partition("_")
    if "_gguf_" in base:
        # Name-only GGUF entries (divergence): order by known quant ladder.
        gguf_order = {
            "f16": 0.0,
            "q8_0": 1.0,
            "q6_k": 2.0,
            "q5_k_s": 3.0,
            "q5_k_m": 4.0,
            "q4_k_s": 5.0,
            "q4_k_m": 6.0,
            "q3_k_m": 7.0,
        }
        gguf_suffix = base.rsplit("_gguf_", 1)[1] if "_gguf_" in base else ""
        return (gguf_order.get(gguf_suffix, 50.0), var["name"])
    return (_SAFETENSORS_QUANT_ORDER.get(suffix, 50.0), var["name"])


def _group_ordered(variants: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Variants ordered by group header, then quant tier within group."""
    return sorted(variants, key=lambda v: (_variant_group(v)[0], _group_sort_key(v), v["name"]))


def _group_row(label: str, colspan: int = 9) -> str:
    return f'<tr class="group-row"><td colspan="{colspan}">{html.escape(label)}</td></tr>'


def _display(name: str) -> str:
    """Human-friendly variant name: dual-run suffix -> annotation."""
    return name.replace("__vs_base", " (vs base)")


def _layer_index(stream_name: str) -> int | None:
    if stream_name.startswith("layer_"):
        try:
            return int(stream_name.split("_")[1])
        except (IndexError, ValueError):
            return None
    return None


# ---------------------------------------------------------------------------
# Graph 1: per-layer MSE profile with consumption point
# ---------------------------------------------------------------------------


def graph_layer_profile(variants: list[dict[str, Any]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    backend = variants[0]["backend"]
    consumed = BACKEND_CONSUMED_LAYER.get(backend)

    # Readability curation: the safetensors story only (base quants as the
    # grey noise-floor band, heretic variants colored). GGUF containers have
    # their own views (ladder graph + grouped heatmap) — 14 more colored
    # lines here would be unreadable spaghetti.
    variants = [
        v for v in variants if not v["name"].endswith("__vs_base") and not v.get("gguf") and "_gguf_" not in v["name"]
    ]
    for i, var in enumerate(variants):
        pts = sorted(
            (idx, sr["metrics"]["mse"])
            for sr in var["streams"].values()
            if (idx := _layer_index(sr["stream"])) is not None
        )
        if not pts:
            continue
        xs, ys = zip(*pts, strict=True)
        # Clamp zeros for the log axis (consistent with the heatmap)
        ys = tuple(max(v, 1e-12) for v in ys)
        if _is_reference_variant(var):
            # Non-abliterated quants: the real quantization noise floor
            ax.plot(xs, ys, marker="o", ms=2, lw=1.2, ls="--", color="#95a5a6", alpha=0.8, label=_display(var["name"]))
        else:
            ax.plot(xs, ys, marker="o", ms=3, lw=1.5, color=COLORS[i % len(COLORS)], label=_display(var["name"]))

    if consumed is not None:
        ax.axvline(consumed, color="#2ecc71", lw=2, ls="--", alpha=0.9)
        ax.annotate(
            f"DiT reads layer {consumed}",
            xy=(consumed, ax.get_ylim()[1]),
            xytext=(8, -14),
            textcoords="offset points",
            color="#2ecc71",
            fontsize=10,
            fontweight="bold",
            va="top",
        )

    ax.set_yscale("log")
    ax.set_xlabel("encoder layer")
    ax.set_ylabel("MSE vs base (log)")
    ax.set_title(f"Per-layer conditioning damage — {backend}")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)
    log.info("Wrote %s", out)


# ---------------------------------------------------------------------------
# Graph 2: damage ranking by pipeline stage
# ---------------------------------------------------------------------------


def graph_damage_ranking(variants: list[dict[str, Any]], out: Path) -> None:
    """Transmitted damage at the DiT input, grouped and log-scaled.

    Readability redesign: one bar per variant (the headline refined_context
    stream — earlier pipeline stages live in the damage table), grouped by
    family×container with separator lines, log x-axis (damage spans 0.001×
    to 800×), and a stable family palette matching the ladder graph.
    """
    floor_label = variants[0].get("floor", "fakequant_int8")
    variants = _group_ordered([v for v in variants if not v["name"].endswith("__vs_base")])
    rows: list[tuple[dict[str, Any], float]] = []
    for var in variants:
        sr = var["streams"].get("refined_context") or var["streams"].get("cap_embedded")
        if sr and sr.get("damage_quant_units") is not None:
            rows.append((var, sr["damage_quant_units"]))

    if not rows:
        log.warning("No damage data — skipping %s", out.name)
        return

    def _bar_color(var: dict[str, Any]) -> str:
        is_gguf = bool(var.get("gguf")) or "_gguf_" in var["name"]
        if _is_reference_variant(var):
            return "#8c8c8c" if is_gguf else "#b8b8b8"
        return "#c44e52" if is_gguf else "#4c72b0"

    fig, ax = plt.subplots(figsize=(10, max(2, 0.42 * len(rows) + 2.0)))
    y = np.arange(len(rows))
    values = [d for _, d in rows]
    colors = [_bar_color(v) for v, _ in rows]
    ax.barh(y, values, color=colors, alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlim(max(min(values) * 0.5, 1e-3), max(values) * 1.5)
    ax.axvline(1.0, color="#e74c3c", lw=1.5, ls="--")
    ax.text(1.0, len(rows) - 0.2, " one quant step", color="#e74c3c", fontsize=9, va="top")

    ax.set_yticks(y)
    ax.set_yticklabels([_display(v["name"]) for v, _ in rows], fontsize=9)
    ax.invert_yaxis()
    # Group separators + inline group labels
    group_ids = [_variant_group(v)[0] for v, _ in rows]
    for i in range(1, len(rows)):
        if group_ids[i] != group_ids[i - 1]:
            ax.axhline(i - 0.5, color="#555555", lw=1.2, alpha=0.7)
    # Compact group captions at the right edge of each block's first row
    seen: set[int] = set()
    for i, gid in enumerate(group_ids):
        if gid not in seen:
            seen.add(gid)
            short = {
                0: "base quants",
                1: "base GGUF",
                2: "heretic",
                3: "heretic GGUF",
            }.get(gid, "")
            if short:
                ax.annotate(
                    short,
                    xy=(0.99, i / max(len(rows) - 1, 1)),
                    xycoords="axes fraction",
                    xytext=(-2, 4),
                    textcoords="offset points",
                    ha="right",
                    fontsize=8,
                    color="#555555",
                    style="italic",
                )
    ax.set_xlabel(f"DiT-input damage ({floor_label} floor units, log)")
    ax.set_title("Transmitted damage at the DiT input (lower = more faithful)")
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)
    log.info("Wrote %s", out)


# ---------------------------------------------------------------------------
# Graph 3: per-category gradient on the DiT-input stream
# ---------------------------------------------------------------------------


def _di_input_stream(var: dict[str, Any]) -> dict[str, Any] | None:
    for name in ("refined_context", "cap_embedded", "consumed_hidden"):
        sr: dict[str, Any] | None = var["streams"].get(name)
        if sr and sr.get("per_category"):
            return sr
    return None


def graph_category_gradient(variants: list[dict[str, Any]], out: Path) -> None:
    """Damage by prompt category, split into base/heretic panels.

    One bar group per category; base quants (the noise floor) on the left
    panel, abliterated variants on the right; family palette matches the
    other graphs. Log y-axis — extreme-category MSE spans orders of
    magnitude above innocent.
    """
    usable = _group_ordered([v for v in variants if not v["name"].endswith("__vs_base") and _di_input_stream(v)])
    if not usable:
        log.warning("No per-category data — skipping %s", out.name)
        return

    cats = ["innocent", "mild", "violent", "sexual", "extreme", "multilingual"]
    present = [c for c in cats if any(c in (_di_input_stream(v) or {}).get("per_category", {}) for v in usable)]
    if not present:
        return

    ref_panel = [v for v in usable if _is_reference_variant(v)]
    heretic_panel = [v for v in usable if not _is_reference_variant(v)]

    def _bar_color(var: dict[str, Any]) -> str:
        is_gguf = bool(var.get("gguf")) or "_gguf_" in var["name"]
        if _is_reference_variant(var):
            return "#8c8c8c" if is_gguf else "#b8b8b8"
        return "#c44e52" if is_gguf else "#4c72b0"

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    x = np.arange(len(present))

    for ax, panel, title in (
        (axes[0], ref_panel, "Base quants (noise floor)"),
        (axes[1], heretic_panel, "Abliterated (heretic)"),
    ):
        width = 0.8 / max(len(panel), 1)
        for i, var in enumerate(panel):
            sr = _di_input_stream(var)
            assert sr is not None
            vals = [max(sr["per_category"].get(c, {}).get("mse", 0.0), 1e-12) for c in present]
            ax.bar(
                x + i * width,
                vals,
                width * 0.92,
                color=_bar_color(var),
                label=_display(var["name"]),
            )
        ax.set_xticks(x + width * (len(panel) - 1) / 2)
        ax.set_xticklabels(present, rotation=20)
        ax.set_title(title, fontsize=11)
        ax.grid(True, axis="y", alpha=0.25)

    sr0 = _di_input_stream(usable[0])
    stream_name = sr0["stream"] if sr0 else ""
    axes[0].set_ylabel("MSE vs base (log)")
    axes[0].set_yscale("log")
    # No per-series legends: with 12 variants per panel the legend is pure
    # clutter and the figure's story is the category GRADIENT (taboo
    # concentration), not per-variant identification — the damage table,
    # ranking graph, and heatmap carry identity.
    axes[0].annotate(
        f"{len(ref_panel)} base quants (grey)",
        xy=(0.03, 0.97),
        xycoords="axes fraction",
        fontsize=9,
        color="#666666",
        va="top",
    )
    axes[1].annotate(
        f"{len(heretic_panel)} heretic variants\n(blue = safetensors, red = GGUF)",
        xy=(0.03, 0.97),
        xycoords="axes fraction",
        fontsize=9,
        color="#666666",
        va="top",
    )
    fig.suptitle(f"Damage by prompt category — DiT-input stream ({stream_name})", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)
    log.info("Wrote %s", out)


# ---------------------------------------------------------------------------
# Graph 4: layer heatmap (variants x layers, log10 MSE)
# ---------------------------------------------------------------------------


def graph_layer_heatmap(variants: list[dict[str, Any]], out: Path) -> None:
    # Grouped ordering: base safetensors -> base GGUF -> heretic safetensors
    # -> heretic GGUF -> dual runs, quant tier within each group.
    variants = _group_ordered([v for v in variants if not v["name"].endswith("__vs_base")])
    all_layers: set[int] = set()
    for var in variants:
        all_layers.update(idx for sr in var["streams"].values() if (idx := _layer_index(sr["stream"])) is not None)
    if not all_layers:
        log.warning("No layer streams — skipping %s", out.name)
        return
    layers = sorted(all_layers)

    mat = np.full((len(variants), len(layers)), np.nan)
    for i, var in enumerate(variants):
        for sr in var["streams"].values():
            if (idx := _layer_index(sr["stream"])) is not None:
                mat[i, layers.index(idx)] = np.log10(max(sr["metrics"]["mse"], 1e-12))

    fig, ax = plt.subplots(figsize=(max(10, 0.3 * len(layers)), 1.2 + 0.6 * len(variants)))
    im = ax.imshow(mat, aspect="auto", cmap="magma")
    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels([_display(v["name"]) for v in variants], fontsize=9)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([str(ly) for ly in layers], fontsize=7)
    ax.set_xlabel("encoder layer")
    ax.set_title("log10 MSE vs base (grouped: base quants, then abliterated)")
    # Horizontal separators between groups
    group_ids = [_variant_group(v)[0] for v in variants]
    for i in range(1, len(variants)):
        if group_ids[i] != group_ids[i - 1]:
            ax.axhline(i - 0.5, color="white", lw=2.5)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)
    log.info("Wrote %s", out)


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Graph 5: GGUF quantization ladder (damage vs bits-per-weight)
# ---------------------------------------------------------------------------

# Safetensors quants plotted as open reference markers at NOMINAL bpw
# (GGUF bpw is measured from file bytes; safetensors formats carry format-
# specific scale overhead that makes file-size bpw non-comparable — the
# nominal values are labeled as such in the legend).
_SAFETENSORS_NOMINAL_BPW = {
    "base_int8": 8.5,
    "base_fp8": 8.0,
    "base_mxfp8": 8.0,
    "base_nvfp4": 4.25,
    "base_int4": 4.5,
}


def graph_gguf_ladder(variants: list[dict[str, Any]], out: Path) -> None:
    """Damage (log y) vs bits-per-weight for GGUF containers + quant anchors."""
    by_name = {v["name"]: v for v in variants}
    gguf_base: list[tuple[float, float, str]] = []  # (bpw, damage, label)
    gguf_heretic: list[tuple[float, float, str]] = []
    safetensors: list[tuple[float, float, str]] = []

    for var in variants:
        if var["name"].endswith("__vs_base"):
            continue
        sr = var["streams"].get("refined_context") or var["streams"].get("cap_embedded")
        if not sr or sr.get("damage_quant_units") is None:
            continue
        dmg = float(sr["damage_quant_units"])
        if var.get("gguf"):
            bpw = float(var["gguf"].get("bpw", 0))
            label = var["gguf"].get("quant_label", var["name"])
            if _is_reference_variant(var):
                gguf_base.append((bpw, dmg, label))
            else:
                gguf_heretic.append((bpw, dmg, label))
        elif var["name"] in _SAFETENSORS_NOMINAL_BPW and _is_reference_variant(var):
            safetensors.append((_SAFETENSORS_NOMINAL_BPW[var["name"]], dmg, var["name"].replace("base_", "")))

    if not gguf_base and not gguf_heretic:
        log.info("No GGUF ladder data — skipping %s", out.name)
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    if gguf_base:
        ax.scatter(
            [b for b, _, _ in gguf_base],
            [d for _, d, _ in gguf_base],
            color="#8c8c8c",
            marker="o",
            s=70,
            zorder=3,
            label="GGUF base container (vs BF16 base)",
        )
        for b, d, lab in gguf_base:
            ax.annotate(lab, (b, d), textcoords="offset points", xytext=(6, 4), fontsize=8, color="#666")
    # Heretic combined entries (vs global base) are the apples-to-apples
    # companion to the base rungs; fall back to the primary (pure-ablation)
    # run only when the dual is missing, and note it via marker style.
    heretic_dual: list[tuple[float, float, str]] = []
    for b, d, lab in gguf_heretic:
        v_name = next(
            (
                v["name"]
                for v in variants
                if (v.get("gguf") or {}).get("quant_label") == lab and v["name"].startswith("heretic")
            ),
            None,
        )
        dual = by_name.get(f"{v_name}__vs_base") if v_name else None
        sr = (dual or {}).get("streams", {}).get("refined_context") if dual else None
        if sr and sr.get("damage_quant_units") is not None:
            heretic_dual.append((b, float(sr["damage_quant_units"]), lab))
        else:
            heretic_dual.append((b, d, lab))
    if heretic_dual:
        ax.scatter(
            [b for b, _, _ in heretic_dual],
            [d for _, d, _ in heretic_dual],
            color="#c44e52",
            marker="s",
            s=60,
            zorder=3,
            label="GGUF heretic (combined vs BF16 base)",
        )
        for b, d, lab in heretic_dual:
            ax.annotate(lab, (b, d), textcoords="offset points", xytext=(6, -10), fontsize=8, color="#a33")
    if safetensors:
        ax.scatter(
            [b for b, _, _ in safetensors],
            [d for _, d, _ in safetensors],
            facecolors="none",
            edgecolors="#4c72b0",
            marker="D",
            s=70,
            zorder=3,
            label="safetensors quant (nominal bpw)",
        )
        for b, d, lab in safetensors:
            ax.annotate(lab, (b, d), textcoords="offset points", xytext=(6, 5), fontsize=8, color="#4c72b0")

    ax.axhline(1.0, color="#e74c3c", lw=1.5, ls="--")
    ax.text(15.9, 1.15, "one quant step", color="#e74c3c", fontsize=9, ha="right")
    ax.set_yscale("log")
    ax.set_xlabel("bits per weight (GGUF: measured from file; safetensors: nominal)")
    ax.set_ylabel("DiT-input damage (quant-floor units, log)")
    ax.set_title("GGUF quantization ladder — conditioning damage vs compression")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)
    log.info("Wrote %s", out)


def _load_divergence_steps(results_dir: Path) -> dict[str, dict[str, Any]]:
    """Per-step divergence trajectories keyed by variant name.

    Complements _load_divergence (which reads the summaries): this keeps the
    full latent_div/velocity_div lists per prompt for the per-step graph.
    """
    out: dict[str, dict[str, Any]] = {}
    for f in sorted(results_dir.glob("te_dit_divergence_*.json")):
        try:
            d = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Failed to parse %s: %s", f, exc)
            continue
        name = d.get("variant_name", f.stem.replace("te_dit_divergence_", ""))
        out[name] = {
            "n_steps": d.get("n_steps", 20),
            "noise_control_final": d.get("noise_control_final"),
            "prompts": d.get("per_prompt", []),
        }
    return out


def graph_divergence_steps(steps_data: dict[str, dict[str, Any]], out: Path) -> None:
    """Mean per-step latent divergence curves, grouped family colors.

    Answers 'does the damage compound, cancel, or plateau across sampling?'
    Log y: latent divergence grows over ~3 orders of magnitude from step 2
    to the final step while staying far below the noise-control anchor.
    """
    if not steps_data:
        log.info("No per-step divergence data — skipping %s", out.name)
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    # Deterministic group ordering: base safetensors, base GGUF, heretic
    # safetensors, heretic GGUF (matches the other grouped views).
    names = sorted(
        steps_data,
        key=lambda n: (_variant_group({"name": n})[0], _group_sort_key({"name": n, "gguf": None})),
    )
    for name in names:
        entry = steps_data[name]
        prompts = entry.get("prompts", [])
        curves = [p["latent_div"] for p in prompts if p.get("latent_div")]
        if not curves:
            continue
        n = min(len(c) for c in curves)
        mean_curve = np.mean([c[:n] for c in curves], axis=0)
        xs = np.arange(1, n + 1)
        is_gguf = "_gguf_" in name
        if name.startswith("base_"):
            color, lw, alpha = ("#8c8c8c" if is_gguf else "#b8b8b8"), 1.0, 0.7
        else:
            color, lw, alpha = ("#c44e52" if is_gguf else "#4c72b0"), 1.4, 0.9
        ax.plot(xs, np.clip(mean_curve, 1e-6, None), color=color, lw=lw, alpha=alpha, label=_display(name))

    ctrl = next((e.get("noise_control_final") for e in steps_data.values() if e.get("noise_control_final")), None)
    if ctrl:
        ax.axhline(ctrl, color="#2ecc71", lw=2, ls="--", alpha=0.9)
        ax.annotate(
            "different-seed anchor ('another image')",
            xy=(0.02, ctrl),
            xycoords=("axes fraction", "data"),
            xytext=(4, -12),
            textcoords="offset points",
            color="#2ecc71",
            fontsize=9,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Euler step")
    ax.set_ylabel("latent divergence vs base (log)")
    ax.set_title("Per-step divergence — does conditioning damage compound or plateau?")
    ax.grid(True, which="both", alpha=0.25)
    # Legend outside right; ~27 series — compact font
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5), ncol=2)
    fig.tight_layout()
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out)


def _svg_data_uri(path: Path) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(path.read_bytes()).decode()


def _fmt(v: float | None, spec: str = ".4f") -> str:
    return "—" if v is None else format(v, spec)


def _table_rows(variants: list[dict[str, Any]]) -> str:
    rows = []
    current_group: tuple[int, str] | None = None
    for var in _group_ordered(variants):
        group = _variant_group(var)
        if group != current_group:
            current_group = group
            rows.append(_group_row(group[1]))
        for sr in var["streams"].values():
            if _layer_index(sr["stream"]) is not None:
                continue
            m = sr["metrics"]
            dmg = sr.get("damage_quant_units")
            dmg_s = "N/A" if dmg is None else (f"{dmg:.4f}" if dmg < 100 else f"{dmg:.1f}")
            dmg_cls = ""
            if dmg is not None:
                if dmg >= 10:
                    dmg_cls = ' class="dmg-high"'
                elif dmg <= 1.0:
                    dmg_cls = ' class="dmg-low"'
            name_cell = (
                f'<span class="ref-var">{html.escape(_display(var["name"]))}</span>'
                if _is_reference_variant(var)
                else html.escape(_display(var["name"]))
            )
            # Damage with CI when available
            dmg_ci = sr.get("damage_quant_ci")
            if dmg_s not in ("N/A", "—") and dmg_ci:
                dmg_s = f"{dmg_s} <span class='ci'>[{dmg_ci[0]:.2f}, {dmg_ci[1]:.2f}]</span>"
            # Semantic units: pooled damage / meaning-change distance
            sem = sr.get("damage_semantic_units")
            sem_s = f"{sem:.2%}" if sem is not None else "—"
            rows.append(
                f"<tr><td>{name_cell}</td><td>{html.escape(var['reference'])}</td>"
                f"<td>{html.escape(sr['stage'])}</td>"
                f"<td>{html.escape(sr['stream'])}</td><td>{m['cka_linear']:.6f}</td>"
                f"<td>{m['cos_all']:.6f}</td><td>{m['mse']:.3e}</td>"
                f"<td{dmg_cls}>{dmg_s}</td><td>{sem_s}</td></tr>"
            )
    return "\n".join(rows)


def _decomposition_section(variants: list[dict[str, Any]]) -> str:
    """Quantization-vs-abliteration decomposition when matched data exists.

    For each abliterated variant V with quant suffix X (e.g. heretic_int8):
      - pure quant     = base_X vs base_bf16   (reference-variant result)
      - pure ablation  = V's primary run (vs base_X, via te_reference)
      - combined       = V's dual run (vs base_bf16, the __vs_base entry)
    The three are DISTINCT measurements — a single run only ever produces
    one of pure-ablation / combined, so missing counterparts render as "—".
    """
    by_name = {v["name"]: v for v in variants}
    rows: list[str] = []
    for var in variants:
        if _is_reference_variant(var) or var["name"].endswith("__vs_base"):
            continue
        # quant suffix from the variant name (e.g. heretic_int8 -> int8).
        # LONGEST FIRST — "mxfp8" must precede "fp8" or heretic_mxfp8
        # endswith("fp8") and resolves the wrong base (silent wrong table).
        # NOTE: keep in sync with variant naming conventions
        for suffix in (
            "fp8_e4m3fn",
            "mxfp8",
            "nvfp4",
            "int8",
            "int4",
            "fp8",
            "gguf_f16",
            "gguf_q8_0",
            "gguf_q6_k",
            "gguf_q5_k_s",
            "gguf_q5_k_m",
            "gguf_q4_k_s",
            "gguf_q4_k_m",
            "gguf_q3_k_m",
        ):
            if var["name"].endswith(suffix):
                break
        else:
            continue
        base_q = by_name.get(f"base_{suffix}")
        if base_q is None:
            continue

        # All three numbers must be measured on the SAME stream — pick the
        # first stream present in ALL participating artifacts.
        participants = [base_q, var] + (
            [by_name[f"{var['name']}__vs_base"]] if f"{var['name']}__vs_base" in by_name else []
        )
        shared = next(
            (
                s
                for s in ("refined_context", "cap_embedded", "consumed_hidden")
                if all(s in pt["streams"] for pt in participants)
            ),
            None,
        )

        def _dmg(v: dict[str, Any], stream: str | None = shared) -> float | None:
            if stream is None:
                return None
            sr = v["streams"].get(stream)
            if sr is not None:
                d = sr.get("damage_quant_units")
                return float(d) if d is not None else None
            return None

        pure_q = _dmg(base_q)
        # Pure ablation: this variant's primary run compared against base_X
        # (matched by variant NAME — stable namespace, not filenames).
        is_matched = var["reference"] == base_q["name"]
        pure_ab = _dmg(var) if is_matched else None
        # Combined: the dual run entry (variant vs global base)
        vs_base = by_name.get(f"{var['name']}__vs_base")
        combined = _dmg(vs_base) if vs_base is not None else None
        if pure_q is None and combined is None and pure_ab is None:
            continue

        def _cell(x: float | None) -> str:
            return "—" if x is None else (f"{x:.2f}×" if x < 100 else f"{x:.0f}×")

        rows.append(
            f"<tr><td>{html.escape(_display(var['name']))}</td>"
            f"<td>{_cell(pure_q)}</td><td>{_cell(pure_ab)}</td><td>{_cell(combined)}</td></tr>"
        )
    if not rows:
        return ""
    return f"""
<h2>Quantization vs abliteration decomposition</h2>
<p>Damage at the DiT input, in quant-floor units. Pure quant = base
quantization vs BF16 base (1.0× by construction when the floor IS that
file). Pure ablation = abliterated vs base at matched quant (isolates the
abliteration edit). Combined = abliterated quant vs BF16 base. A missing
column means that run wasn't performed.</p>
<table>
<tr><th>Variant</th><th>Pure quant</th><th>Pure ablation</th><th>Combined</th></tr>
{chr(10).join(rows)}
</table>"""


def _load_divergence(results_dir: Path) -> dict[str, dict[str, Any]]:
    """Load te_dit_divergence_*.json summaries keyed by variant name."""
    out: dict[str, dict[str, Any]] = {}
    for f in sorted(results_dir.glob("te_dit_divergence_*.json")):
        try:
            d = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Failed to parse %s: %s", f, exc)
            continue
        s = d.get("summary", {})
        name = d.get("variant_name", f.stem.replace("te_dit_divergence_", ""))
        if s.get("mean_final_latent_div") is not None:
            out[name] = s
    return out


def _gguf_qtype_mix(hist: dict[str, Any] | None) -> str:
    """Compact qtype-mix label, e.g. 'Q4_K + Q6_K' (F32 norms omitted)."""
    if not hist:
        return "—"
    quant = [(k, v) for k, v in hist.items() if k != "F32" and k != "F16"]
    quant.sort(key=lambda kv: -kv[1])
    if not quant:
        return "F16"
    return " + ".join(k for k, _ in quant[:3]) + ("…" if len(quant) > 3 else "")


def _gguf_section(variants: list[dict[str, Any]], divergence: dict[str, dict[str, Any]]) -> str:
    """GGUF quantization ladder — damage vs bits-per-weight for GGUF containers.

    Rows are primary runs: base_gguf_* vs the global BF16 base, heretic_gguf_*
    vs their matched base_gguf_* (pure ablation at quant). bpw and the qtype
    mix come from the persisted GGUFMeta (measured, not hardcoded).
    """
    gguf_vars = [v for v in variants if v.get("gguf") and not v["name"].endswith("__vs_base")]
    if not gguf_vars:
        return ""
    rows_base: list[tuple[float, str]] = []
    rows_heretic: list[tuple[float, str]] = []
    for var in gguf_vars:
        g = var["gguf"]
        sr = var["streams"].get("refined_context") or var["streams"].get("cap_embedded")
        dmg = sr.get("damage_quant_units") if sr else None
        ci = sr.get("damage_quant_ci") if sr else None
        sem = sr.get("damage_semantic_units") if sr else None
        div = divergence.get(var["name"], {}).get("div_control_ratio")
        dmg_txt = f"{dmg:.2f}×" if dmg is not None else "—"
        if dmg is not None and ci:
            dmg_txt += f' <span class="ci">[{ci[0]:.1f}, {ci[1]:.1f}]</span>'
        sem_txt = f"{sem:.2%}" if sem is not None else "—"
        div_txt = f"{div * 100:.1f}%" if div is not None else "—"
        ref_cls = ' class="ref-var"' if _is_reference_variant(var) else ""
        # Heretic rows annotated with their matched-quant reference (span
        # appended AFTER escaping the name so the markup stays live)
        label = html.escape(_display(var["name"]))
        if not _is_reference_variant(var):
            ref = var.get("reference", "")
            if ref:
                label += f" <span class='ci'>vs {html.escape(_display(ref))}</span>"
        row_html = (
            # label is pre-escaped above (name escaped, span appended raw)
            f"<tr><td{ref_cls}>{label}</td>"
            f"<td>{html.escape(_gguf_qtype_mix(g.get('qtype_histogram')))}</td>"
            f"<td>{g.get('bpw', 0):.2f}</td>"
            f"<td>{dmg_txt}</td><td>{sem_txt}</td><td>{div_txt}</td></tr>"
        )
        (rows_base if _is_reference_variant(var) else rows_heretic).append((float(g.get("bpw", 0)), row_html))
    rows_base.sort(key=lambda r: -r[0])
    rows_heretic.sort(key=lambda r: -r[0])
    blocks = []
    if rows_base:
        blocks.append(_group_row("Base containers — pure GGUF quantization vs BF16 base", colspan=6))
        blocks.extend(r[1] for r in rows_base)
    if rows_heretic:
        blocks.append(_group_row("Heretic containers — pure ablation at matched quant", colspan=6))
        blocks.extend(r[1] for r in rows_heretic)
    return f"""
<h2>GGUF quantization ladder</h2>
<p>Damage for llama.cpp GGUF containers (ComfyUI-GGUF qwen3 path: vendored
city96 block dequant, eager bf16 storage — same rounding ComfyUI's lazy
dequant applies at bf16 compute). base_gguf rows measure the pure container
vs the BF16 safetensors base; heretic_gguf rows measure pure ablation at
matched quant. bpw is measured from the file (storage bytes × 8 / params).
The rightmost column is the DiT's response vs the noise-control anchor.</p>
<table>
<tr><th>Variant</th><th>Qtype mix</th><th>bpw</th><th>DiT-input damage</th><th>Semantic</th><th>DiT response</th></tr>
{chr(10).join(blocks)}
</table>"""


def _directional_section(results_dir: Path) -> str:
    """Directional analysis section — direction or noise?

    Reads te_directional_*.json (run_te_directional.sh). Reports the layer
    the DiT actually consumes (backend's consumed layer) plus the peak-z
    layer: cos(delta, taboo_direction) with permutation z, orthogonal
    energy fraction, and toward-innocent z.
    """
    rows: list[str] = []
    files = sorted(results_dir.glob("te_directional_*.json"))
    for f in files:
        try:
            d = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Failed to parse %s: %s", f, exc)
            continue
        name = d.get("variant_name", f.stem.replace("te_directional_", ""))
        consumed = 34 if d.get("backend", "zimage") != "ltx2" else None  # zimage consumed layer
        layers = {lr["layer"]: lr for lr in d.get("layers", [])}
        # peak-z layer + consumed layer stats
        peak = max(layers.values(), key=lambda lr: lr["cos_z"]) if layers else None
        con = layers.get(consumed) if consumed is not None else None
        if peak is None or con is None:
            continue

        def _fmt_cell(lr: dict[str, Any]) -> str:
            return f"{lr['cos_taboo_mean']:+.3f} <span class='ci'>(z {lr['cos_z']:.0f})</span>"

        ref_cls = ' class="ref-var"' if name.startswith("base_") else ""
        rows.append(
            f"<tr><td{ref_cls}>{html.escape(_display(name))}</td>"
            f"<td>{con['layer']}</td><td>{_fmt_cell(con)}</td>"
            f"<td>{100 * con['orth_fraction']:.1f}%</td>"
            f"<td>{con['toward_innocent_mean']:+.1e} <span class='ci'>(z {con['toward_innocent_z']:+.1f})</span></td>"
            f"<td>{peak['layer']}</td><td>{_fmt_cell(peak)}</td></tr>"
        )
    if not rows:
        return ""
    return f"""
<h2>Directional analysis — is the abliteration delta a direction or noise?</h2>
<p>The mechanistic claim behind abliteration is that the edit maps taboo
representations toward benign ones. For each variant (vs its matched-quant
reference), the per-layer ablation delta Δ is compared to the BASE model's
taboo direction d (mean(taboo) − mean(innocent), unit): cos(Δ, d) with a
prompt-permutation z-score, the fraction of Δ energy orthogonal to d, and
the signed movement of taboo prompts toward the innocent centroid.
<b>Read:</b> small |cos| with ~99% orthogonal energy and |z<sub>TI</sub>| ≲ 3
means the delta is NOT a taboo→harmless mapping — it is mostly
geometry-preserving noise concentrated on taboo prompts.</p>
<table>
<tr><th>Variant</th><th>DiT layer</th><th>cos(Δ, d) @ DiT layer</th><th>Orthogonal energy</th><th>Toward-innocent</th><th>Peak-z layer</th><th>cos @ peak</th></tr>
{chr(10).join(rows)}
</table>"""


def _ab_section(variants: list[dict[str, Any]], divergence: dict[str, dict[str, Any]], results_dir: Path) -> str:
    """A/B image gallery + generation-timing section.

    Reads ab_images/manifest.json (written by run_ab_test.py; also imported
    into the te_ab_images DB table). Each prompt renders a comparison row:
    the SAME seed sampled with only the text encoder swapped. Captions carry
    the per-variant metrics (quant label, DiT-input damage, DiT response,
    cold/warm generation time) so the visual delta reads against the
    measured delta.
    """
    manifest_path = results_dir / "ab_images" / "manifest.json"
    if not manifest_path.exists():
        return ""
    try:
        manifest = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Failed to parse A/B manifest: %s", exc)
        return ""

    by_name = {v["name"]: v for v in variants}
    # Warm-pass images preferred for the gallery (identical seed; cold pass
    # exists for timing).
    images: dict[str, dict[str, dict[str, Any]]] = {}
    for img in manifest.get("images", []):
        label = img.get("label") or "unlabeled"
        slot = images.setdefault(label, {})
        slot[img["variant"]] = img  # warm overwrites cold (later in list)

    if not images:
        return ""

    def _variant_caption(var_name: str, img: dict[str, Any]) -> str:
        var = by_name.get(var_name, {})
        sr = var.get("streams", {}).get("refined_context") or var.get("streams", {}).get("cap_embedded") or {}
        dmg = sr.get("damage_quant_units")
        ref = var.get("reference", "")
        gguf = var.get("gguf") or {}
        bits = f" · {gguf['bpw']:.1f} bpw" if gguf else ""
        div = divergence.get(var_name, {}).get("div_control_ratio")
        lines = [f"<b>{html.escape(_display(var_name))}</b>{bits}"]
        if ref:
            lines.append(f"<span class='ci'>vs {html.escape(ref)}</span>")
        m: list[str] = []
        if dmg is not None:
            m.append(f"damage {dmg:.1f}×")
        if div is not None:
            m.append(f"DiT {div * 100:.1f}%")
        t = manifest.get("timing", {}).get(var_name, {}).get("passes", {})
        cold = t.get("cold", {}).get("first_image_seconds")
        warm = t.get("warm", {}).get("first_image_seconds")
        if cold and warm:
            m.append(f"{cold:.1f}s → {warm:.1f}s")
        if m:
            lines.append(html.escape(" · ".join(m)))
        return "<br/>".join(lines)

    blocks: list[str] = []
    for label in sorted(images):
        row_imgs: list[str] = []
        checks: list[str] = []
        for var_name in sorted(images[label]):
            img = images[label][var_name]
            pth = Path(img.get("path", ""))
            local = results_dir / "ab_images" / var_name / pth.name if pth.name else None
            if local is None or not local.exists():
                # manifest paths may be absolute host paths — try the leaf
                alt = results_dir / "ab_images" / var_name / pth.name
                if not alt.exists():
                    continue
                local = alt
            b64 = base64.b64encode(local.read_bytes()).decode()
            row_imgs.append(
                f'<figure class="ab-cell"><img src="data:image/webp;base64,{b64}" alt="{html.escape(var_name)}"/>'
                f"<figcaption>{_variant_caption(var_name, img)}</figcaption></figure>"
            )
            checks = img.get("checks", [])
        if not row_imgs:
            continue
        check_html = ""
        if checks:
            check_html = "<p class='meta'>Adherence checks: " + "; ".join(html.escape(chk) for chk in checks) + "</p>"
        blocks.append(f"<h3>{html.escape(label)}</h3>{check_html}<div class='ab-row'>{''.join(row_imgs)}</div>")

    timing_rows: list[str] = []
    for var_name in sorted(manifest.get("timing", {})):
        t = manifest["timing"][var_name]
        passes = t.get("passes", {})
        cold = passes.get("cold", {})
        warm = passes.get("warm", {})
        ref_cls = ' class="ref-var"' if var_name.startswith("base_") else ""
        timing_rows.append(
            f"<tr><td{ref_cls}>{html.escape(_display(var_name))}</td>"
            f"<td>{html.escape(t.get('te_name', '').split('/')[-1])}</td>"
            f"<td>{html.escape(t.get('loader', ''))}</td>"
            f"<td>{_fmt(cold.get('first_image_seconds'), '.1f')}</td>"
            f"<td>{_fmt(warm.get('first_image_seconds'), '.1f')}</td>"
            f"<td>{_fmt(cold.get('mean_image_seconds'), '.1f')}</td>"
            f"<td>{_fmt(warm.get('mean_image_seconds'), '.1f')}</td></tr>"
        )

    timing_html = ""
    if timing_rows:
        timing_html = f"""
<h3>Generation timing (seconds, RTX 5090, 1024px, 9 steps)</h3>
<p>Cold = first image after the text encoder is freshly loaded into VRAM
(model-load time included); warm = first image on the cached model. The
delta is the practical cost of a TE swap in a running session.</p>
<table>
<tr><th>Variant</th><th>TE file</th><th>Loader</th><th>Cold 1st img</th><th>Warm 1st img</th><th>Cold mean</th><th>Warm mean</th></tr>
{chr(10).join(timing_rows)}
</table>"""

    return f"""
<h2>A/B image evidence — same seed, only the text encoder swapped</h2>
<p>Paired generations from the fixed seed {manifest.get("seed", "?")} through the
real ComfyUI stack (Z-Image Turbo, 9 steps, cfg 1, euler/simple). Caption
metrics: DiT-input damage (quant-floor units), DiT response (% of a
seed-change), and cold→warm first-image time. Visual differences at
matched seed are the image-level manifestation of the conditioning
perturbation; identical-looking images at different damage levels
demonstrate DiT robustness.</p>
{chr(10).join(blocks)}
{timing_html}"""


def _key_findings(variants: list[dict[str, Any]], backend: str) -> list[str]:
    findings: list[str] = []
    consumed = BACKEND_CONSUMED_LAYER.get(backend)

    # CKA/MSE disconnect (methodological highlight): geometry-preserving
    # damage. Scan non-dual variants for high MSE with CKA still >= 0.98.
    disconnect = []
    for var in variants:
        if var["name"].endswith("__vs_base"):
            continue
        sr = var["streams"].get("refined_context")
        if not sr:
            continue
        dmg = sr.get("damage_quant_units")
        cka = sr.get("metrics", {}).get("cka_linear")
        if dmg is not None and dmg > 50 and cka is not None and cka >= 0.98:
            disconnect.append((var["name"], dmg, cka))
    if disconnect:
        worst = max(disconnect, key=lambda t: t[1])
        disconnect_names = {n for n, _, _ in disconnect}
        gguf_ct = sum(1 for n in disconnect_names if "_gguf_" in n)
        findings.append(
            f"<b>CKA/MSE disconnect</b>: {len(disconnect)} variants (incl. {gguf_ct} GGUF) exceed "
            f"50× quant-floor MSE damage at the DiT input while CKA stays ≥ 0.98 — worst: "
            f"<b>{html.escape(_display(worst[0]))}</b> at {worst[1]:.0f}× damage with CKA {worst[2]:.4f}. "
            "The perturbation destroys magnitude alignment but preserves representational "
            "geometry — consistent with the DiT's weak response to it."
        )

    # GGUF variants are aggregated into summary lines (16 individual bullets
    # would drown the findings list) — full detail in the GGUF ladder section.
    gguf_vars = [v for v in variants if v.get("gguf") and not v["name"].endswith("__vs_base")]
    gguf_names = {v["name"] for v in gguf_vars}
    if gguf_vars:
        base_rungs = [
            (v["gguf"].get("bpw", 0), v["streams"].get("refined_context", {}).get("damage_quant_units"), v["name"])
            for v in gguf_vars
            if _is_reference_variant(v)
            and v["streams"].get("refined_context", {}).get("damage_quant_units") is not None
        ]
        measured: list[tuple[float, Any]] = [(float(bpw), dmg) for bpw, dmg, _n in base_rungs]
        if measured:
            best_rung: tuple[float, Any] = min(measured, key=lambda t: t[1])
            worst_rung: tuple[float, Any] = max(measured, key=lambda t: t[1])
            lo_bpws = [b for b, d in measured if d == best_rung[1]]
            findings.append(
                f"<b>GGUF ladder (base containers, {len(measured)} rungs)</b>: pure-container DiT-input damage "
                f"spans <b>{best_rung[1]:.2f}×</b> (bpw {max(lo_bpws):.1f}) to <b>{worst_rung[1]:.0f}×</b> "
                f"(bpw {worst_rung[0]:.1f}) quant floor — the full ladder is in the GGUF section below."
            )
        heretic_ab = [
            v["streams"].get("refined_context", {}).get("damage_quant_units")
            for v in gguf_vars
            if not _is_reference_variant(v)
        ]
        heretic_ab = [d for d in heretic_ab if d is not None]
        if heretic_ab:
            findings.append(
                f"<b>GGUF × heretic</b>: pure-ablation damage at matched quant spans "
                f"<b>{min(heretic_ab):.1f}×–{max(heretic_ab):.1f}×</b> quant floor across the {len(heretic_ab)} "
                f"rungs — decomposition table + ladder graph show the quant/abliteration interaction."
            )

    for var in variants:
        if var["name"] in gguf_names:
            continue  # aggregated above
        # Dual-run combined entries are surfaced in the decomposition table —
        # skip here to avoid duplicating the primary variant's findings.
        if var["name"].endswith("__vs_base"):
            continue
        sr = var["streams"].get("refined_context") or var["streams"].get("cap_embedded")
        dmg_val = sr.get("damage_quant_units") if sr else None
        if _is_reference_variant(var):
            # Pure quantization floor: how much the quant alone perturbs conditioning
            if dmg_val is not None:
                findings.append(
                    f"<b>{html.escape(_display(var['name']))}</b> (pure quantization): DiT-input damage "
                    f"{dmg_val:.2f}× quant floor — the reference damage level for its quant format."
                )
            continue
        # Peak layer damage
        layer_data = [
            (idx, sr["metrics"]["mse"], sr.get("damage_quant_units"))
            for sr in var["streams"].values()
            if (idx := _layer_index(sr["stream"])) is not None
        ]
        if layer_data and consumed is not None:
            peak = max(layer_data, key=lambda t: t[1])
            at_consumed = next((t for t in layer_data if t[0] == consumed), None)
            if at_consumed and peak[2] is not None and at_consumed[2] is not None:
                filt = peak[2] / at_consumed[2] if at_consumed[2] > 0 else float("inf")
                findings.append(
                    f"<b>{html.escape(_display(var['name']))}</b>: peak damage {peak[2]:.1f}× quant floor at "
                    f"layer {peak[0]}; the DiT reads layer {consumed}, where damage is "
                    f"{at_consumed[2]:.2f}× — a <b>{filt:.0f}× lower-damage</b> layer than the encoder's peak."
                )
        # DiT-input damage + taboo ratio
        sr = var["streams"].get("refined_context") or var["streams"].get("cap_embedded")
        if sr and sr.get("per_category"):
            pc = sr["per_category"]
            inn, ext = pc.get("innocent", {}).get("mse"), pc.get("extreme", {}).get("mse")
            if inn is not None and ext is not None and inn > 0:
                sem = sr.get("damage_semantic_units")
                sem_txt = f"; {sem:.2%} of a meaning-change" if sem is not None else ""
                findings.append(
                    f"<b>{html.escape(_display(var['name']))}</b>: DiT-input damage "
                    f"{_fmt(sr.get('damage_quant_units'))}× quant step{sem_txt}; taboo-concentration "
                    f"(extreme vs innocent MSE) = <b>{ext / inn:.1f}×</b>."
                )
    return findings


def _divergence_section(divergence_data: dict[str, dict[str, Any]]) -> str:
    """DiT-step divergence section — does the DiT respond to the perturbation?

    Rows come from the te_dit_divergence_*.json summaries (written by
    run_dit_divergence, preloaded by _load_divergence) — final-latent
    divergence per variant against the noise-control anchor
    ("completely different image" level).
    """
    rows: list[str] = []
    current_group: tuple[int, str] | None = None
    # Group-ordered iteration (family × container × quant tier) — plain
    # alphabetical would flip-flop the group headers (base_gguf_* sorts
    # between base_fp8 and base_int4).
    entries = sorted(
        divergence_data.items(),
        key=lambda kv: (_variant_group({"name": kv[0]})[0], _group_sort_key({"name": kv[0], "gguf": None})),
    )
    for name, s in entries:
        div = s.get("mean_final_latent_div")
        ctrl = s.get("noise_control_final")
        ratio = s.get("div_control_ratio")
        if div is None or ctrl is None or ratio is None:
            continue
        # Group headers keyed by variant-name convention (base_* = reference)
        pseudo = {"name": name}
        group = _variant_group(pseudo)
        if group != current_group:
            current_group = group
            rows.append(_group_row(group[1], colspan=4))
        ref_cls = ' class="ref-var"' if name.startswith("base_") else ""
        rows.append(
            f"<tr><td{ref_cls}>{html.escape(_display(name))}</td>"
            f"<td>{div:.5f}</td><td>{ctrl:.3f}</td>"
            f"<td><b>{ratio * 100:.1f}%</b></td></tr>"
        )
    if not rows:
        return ""
    return f"""
<h2>DiT-step divergence — does the diffusion model respond?</h2>
<p>Paired flow-matching trajectories from <b>identical noise</b> (seed 42,
20 Euler steps, 1024px latents), one with base conditioning and one with the
variant's. Divergence is the final-latent relative difference; the
<b>noise-control anchor</b> is the same conditioning with a different seed —
the "completely different image" level. A variant at ~1% of the anchor means
the DiT's output is ~100x closer to the base image than to a different
image.</p>
<table>
<tr><th>Variant</th><th>Final latent divergence</th><th>Noise-control anchor</th><th>Relative response</th></tr>
{chr(10).join(rows)}
</table>"""


def generate_html(variants: list[dict[str, Any]], results_dir: Path, out: Path) -> None:
    backend = variants[0]["backend"]
    floor_label = variants[0].get("floor", "fakequant_int8")
    graphs_dir = out.parent
    svgs = {
        n: graphs_dir / f"te_{n}.svg"
        for n in (
            "layer_profile",
            "damage_ranking",
            "category_gradient",
            "layer_heatmap",
            "gguf_ladder",
            "divergence_steps",
        )
    }
    embedded = "".join(
        f'<figure><img src="{_svg_data_uri(p)}" alt="{name}"/><figcaption>{cap}</figcaption></figure>'
        for name, p, cap in (
            ("layer_profile", svgs["layer_profile"], "Where the damage lands vs where the DiT reads"),
            (
                "damage_ranking",
                svgs["damage_ranking"],
                "Transmitted damage at the DiT input, grouped by family (log scale)",
            ),
            ("category_gradient", svgs["category_gradient"], "Taboo-concentration — base quants vs abliterated panels"),
            ("layer_heatmap", svgs["layer_heatmap"], "Full per-layer damage map"),
            ("gguf_ladder", svgs["gguf_ladder"], "GGUF quantization ladder (damage vs bits-per-weight)"),
            (
                "divergence_steps",
                svgs["divergence_steps"],
                "Per-step latent divergence — compounding vs plateau (log scale)",
            ),
        )
        if p.exists()
    )
    findings = "\n".join(f"<li>{f}</li>" for f in _key_findings(variants, backend))
    decomposition = _decomposition_section(variants)
    divergence_data = _load_divergence(results_dir)
    divergence = _divergence_section(divergence_data)
    gguf_section = _gguf_section(variants, divergence_data)
    ab_section = _ab_section(variants, divergence_data, results_dir)
    directional = _directional_section(results_dir)
    has_gguf = any(v.get("gguf") for v in variants)
    gguf_method = (
        "<br/><b>GGUF methodology.</b> llama.cpp containers are loaded through ComfyUI-GGUF's "
        "qwen3 path (city96 block dequantization, Apache-2.0, vendored verbatim; ordered "
        "LLAMA_SD_MAP name remap; no llama_permute for qwen3; token-embedding fp16 detour for "
        "large vocabularies). Weights are eagerly dequantized to bf16 storage — the same "
        "fp16-scale-to-bf16 rounding ComfyUI's lazy dequant applies at bf16 compute — then "
        "analyzed with fp32 model compute like every other variant."
        if has_gguf
        else ""
    )

    html_doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>Text-Encoder Conditioning Fidelity Report</title>
<style>
 body {{ font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; margin: 0 auto;
        max-width: 1080px; padding: 24px; color: #1a1a2e; background: #fafafa; }}
 h1 {{ border-bottom: 3px solid #4c72b0; padding-bottom: 8px; }}
 h2 {{ margin-top: 40px; color: #4c72b0; }}
 table {{ border-collapse: collapse; width: 100%; font-size: 13px; margin: 16px 0; }}
 th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: right; }}
 th {{ background: #4c72b0; color: white; text-align: left; }}
 tr:nth-child(even) {{ background: #f4f6fa; }}
 .dmg-high {{ color: #c0392b; font-weight: bold; }}
 .dmg-low {{ color: #27ae60; }}
 .ref-var {{ color: #7f8c8d; font-style: italic; }}
 .group-row td {{ background: #dde6f2; color: #2c3e66; font-weight: bold; text-align: left; }}
 .ab-row {{ display: flex; flex-wrap: wrap; gap: 12px; align-items: flex-start; margin: 12px 0 28px 0; }}
 .ab-cell {{ flex: 1 1 300px; max-width: 420px; margin: 0; text-align: center;
            background: white; border: 1px solid #e0e0e0; border-radius: 8px; padding: 8px; }}
 .ab-cell img {{ max-width: 100%; height: auto; border-radius: 4px; }}
 .ab-cell figcaption {{ font-size: 12px; color: #333; margin-top: 6px; line-height: 1.5; }}
 .ci {{ color: #888; font-size: 11px; white-space: nowrap; }}
 figure {{ margin: 32px 0; text-align: center; }}
 figure img {{ max-width: 100%; background: white; border: 1px solid #e0e0e0; border-radius: 8px; }}
 figcaption {{ font-size: 13px; color: #666; margin-top: 6px; }}
 .meta {{ color: #666; font-size: 13px; }}
 .findings li {{ margin: 10px 0; line-height: 1.5; }}
 .foot {{ margin-top: 48px; padding: 16px; background: #eef2f7; border-radius: 8px;
         font-size: 12.5px; color: #444; line-height: 1.6; }}
</style></head><body>
<h1>Text-Encoder Conditioning Fidelity Report</h1>
<p class="meta">Backend: <b>{html.escape(backend)}</b> · Variants: {len(variants)} ·
Results: {html.escape(str(results_dir))} · te_results_version: {TE_RESULTS_VERSION}</p>

<h2>Key findings</h2>
<ol class="findings">{findings}</ol>
{decomposition}
{gguf_section}
{directional}
{divergence}
{ab_section}
<h2>Pipeline damage table</h2>
<table>
<tr><th>Variant</th><th>vs reference</th><th>Stage</th><th>Stream</th><th>CKA (linear)</th><th>Cos (all)</th><th>MSE</th><th>Damage (quant units)</th><th>Dmg (semantic)</th></tr>
{_table_rows(variants)}
</table>

<h2>Visualizations</h2>
{embedded}

<div class="foot">
<b>Methodology.</b> Conditioning tensors are captured with ComfyUI's own model classes at three
pipeline stages (raw hidden states → projection → connector/refiner). Fidelity is measured by
CKA (representational similarity) and MSE; the headline number expresses MSE damage in units of
<b>one quantization step of the noise floor</b> ({floor_label}) (damage &lt; 1 = below the perturbation
a known-good compression would introduce). Prompt suite is stratified
(innocent / mild / violent / sexual / extreme / multilingual). Pad positions are excluded.
<br/><b>Semantic unit definition.</b> "One meaning-change" is the mean squared
distance between the mean-pooled conditioning of <i>distinct prompts from the
base model</i> — all C(n,2) pairs of the 60-prompt suite (1,770 pairs;
uniform subsample at ≤4,000, seed-fixed). It is therefore an <i>intra-suite
content-change</i> scale: how far conditioning moves when the described
scene changes entirely, under the same tokenizer/encoder. It is NOT
synonym/antonym-calibrated and not cross-encoder; a variant at 1% of a
meaning-change means its perturbation is 100× smaller than swapping the
prompt for an unrelated one.{gguf_method}
<br/><b>Scope.</b> This measures conditioning <i>perturbation</i> — how much the modified text
encoder shifts what the DiT sees. It does not measure uncensoring efficacy in either direction.
</div>
</body></html>"""
    out.write_text(html_doc)
    log.info("Wrote %s", out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Generate TE graphs + HTML report")
    p.add_argument("--results-dir", required=True, help="Directory containing te_*.json")
    p.add_argument("--output-dir", required=True, help="Directory for graphs + te_report.html")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = load_results(results_dir)

    graph_layer_profile(variants, out_dir / "te_layer_profile.svg")
    graph_damage_ranking(variants, out_dir / "te_damage_ranking.svg")
    graph_category_gradient(variants, out_dir / "te_category_gradient.svg")
    graph_layer_heatmap(variants, out_dir / "te_layer_heatmap.svg")
    graph_gguf_ladder(variants, out_dir / "te_gguf_ladder.svg")
    graph_divergence_steps(_load_divergence_steps(results_dir), out_dir / "te_divergence_steps.svg")
    generate_html(variants, results_dir, out_dir / "te_report.html")
    log.info("Report generation complete")


if __name__ == "__main__":
    main()
