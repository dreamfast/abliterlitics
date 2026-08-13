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
    floor_label = variants[0].get("floor", "fakequant_int8")
    rows: list[tuple[str, str, float]] = []
    for var in variants:
        if var["name"].endswith("__vs_base"):
            continue  # combined measurements live in the decomposition table
        for stream in PIPELINE_STREAMS:
            sr = var["streams"].get(stream)
            if sr and sr.get("damage_quant_units") is not None:
                rows.append((var["name"], stream, sr["damage_quant_units"]))

    if not rows:
        log.warning("No damage data — skipping %s", out.name)
        return

    fig, ax = plt.subplots(figsize=(10, max(2, 0.5 * len(rows) + 1.5)))
    labels = [f"{_display(v)}  ·  {s}" for v, s, _ in rows]
    values = [d for _, _, d in rows]
    colors = [COLORS[sorted({v for v, _, _ in rows}).index(v) % len(COLORS)] for v, _, _ in rows]

    y = np.arange(len(rows))
    ax.barh(y, values, color=colors, alpha=0.85)
    ax.axvline(1.0, color="#e74c3c", lw=1.5, ls="--")
    ax.text(1.0, len(rows) - 0.2, " one quant step", color="#e74c3c", fontsize=9, va="top")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(f"damage ({floor_label} floor units)")
    ax.set_title("Transmitted damage by pipeline stage (lower = more faithful)")
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
    usable = [v for v in variants if _di_input_stream(v)]
    if not usable:
        log.warning("No per-category data — skipping %s", out.name)
        return

    cats = ["innocent", "mild", "violent", "sexual", "extreme", "multilingual"]
    present = [c for c in cats if any(c in (_di_input_stream(v) or {}).get("per_category", {}) for v in usable)]
    if not present:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(present))
    width = 0.8 / len(usable)

    for i, var in enumerate(usable):
        sr = _di_input_stream(var)
        assert sr is not None
        vals = [max(sr["per_category"].get(c, {}).get("mse", 0.0), 1e-12) for c in present]
        ax.bar(x + i * width, vals, width * 0.92, color=COLORS[i % len(COLORS)], label=_display(var["name"]))

    sr0 = _di_input_stream(usable[0])
    stream_name = sr0["stream"] if sr0 else ""
    ax.set_xticks(x + width * (len(usable) - 1) / 2)
    ax.set_xticklabels(present, rotation=20)
    ax.set_ylabel("MSE vs base")
    ax.set_title(f"Damage by prompt category — DiT-input stream ({stream_name})")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)
    log.info("Wrote %s", out)


# ---------------------------------------------------------------------------
# Graph 4: layer heatmap (variants x layers, log10 MSE)
# ---------------------------------------------------------------------------


def graph_layer_heatmap(variants: list[dict[str, Any]], out: Path) -> None:
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
    ax.set_xticklabels(layers, fontsize=7)
    ax.set_xlabel("encoder layer")
    ax.set_title("log10 MSE vs base")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)
    log.info("Wrote %s", out)


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------


def _svg_data_uri(path: Path) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(path.read_bytes()).decode()


def _fmt(v: float | None, spec: str = ".4f") -> str:
    return "—" if v is None else format(v, spec)


def _table_rows(variants: list[dict[str, Any]]) -> str:
    rows = []
    for var in variants:
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
            rows.append(
                f"<tr><td>{name_cell}</td><td>{html.escape(var['reference'])}</td>"
                f"<td>{html.escape(sr['stage'])}</td>"
                f"<td>{html.escape(sr['stream'])}</td><td>{m['cka_linear']:.6f}</td>"
                f"<td>{m['cos_all']:.6f}</td><td>{m['mse']:.3e}</td>"
                f"<td{dmg_cls}>{dmg_s}</td></tr>"
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
        # quant suffix from the variant name (e.g. heretic_int8 -> int8)
        # NOTE: keep in sync with variant naming conventions
        for suffix in ("int8", "fp8", "nvfp4", "fp8_e4m3fn", "gguf_q8_0", "gguf_q6_k", "gguf_q4_k"):
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


def _key_findings(variants: list[dict[str, Any]], backend: str) -> list[str]:
    findings: list[str] = []
    consumed = BACKEND_CONSUMED_LAYER.get(backend)

    for var in variants:
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
                findings.append(
                    f"<b>{html.escape(_display(var['name']))}</b>: DiT-input damage "
                    f"{_fmt(sr.get('damage_quant_units'))}× quant step; taboo-concentration "
                    f"(extreme vs innocent MSE) = <b>{ext / inn:.1f}×</b>."
                )
    return findings


def generate_html(variants: list[dict[str, Any]], results_dir: Path, out: Path) -> None:
    backend = variants[0]["backend"]
    floor_label = variants[0].get("floor", "fakequant_int8")
    graphs_dir = out.parent
    svgs = {
        n: graphs_dir / f"te_{n}.svg" for n in ("layer_profile", "damage_ranking", "category_gradient", "layer_heatmap")
    }
    embedded = "".join(
        f'<figure><img src="{_svg_data_uri(p)}" alt="{name}"/><figcaption>{cap}</figcaption></figure>'
        for name, p, cap in (
            ("layer_profile", svgs["layer_profile"], "Where the damage lands vs where the DiT reads"),
            ("damage_ranking", svgs["damage_ranking"], "Transmitted damage, quant-floor units"),
            ("category_gradient", svgs["category_gradient"], "Taboo-concentration of transmitted damage"),
            ("layer_heatmap", svgs["layer_heatmap"], "Full per-layer damage map"),
        )
        if p.exists()
    )
    findings = "\n".join(f"<li>{f}</li>" for f in _key_findings(variants, backend))
    decomposition = _decomposition_section(variants)

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
<h2>Pipeline damage table</h2>
<table>
<tr><th>Variant</th><th>vs reference</th><th>Stage</th><th>Stream</th><th>CKA (linear)</th><th>Cos (all)</th><th>MSE</th><th>Damage (quant units)</th></tr>
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
    generate_html(variants, results_dir, out_dir / "te_report.html")
    log.info("Report generation complete")


if __name__ == "__main__":
    main()
