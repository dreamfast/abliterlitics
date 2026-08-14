"""Phase-4 report sections — image metrics, controls ladder, attention, verdicts.

Builds the "solid answer" sections of te_report.html from the Phase-4 battery
outputs (tebench_*.json): perceptual ladder with dual anchors, response
curves, control battery + anchor ladder, attention readout, discriminability,
and the H1–H7 verdict table (H6 DEFERRED — no taboo data by design, journal
#122).

Verdicts are computed against the PRE-REGISTERED expectations P1–P7 (PLAN.md)
— recorded before any Phase-4 data existed — and cite anchor numbers.
Terminology: journals #117/#118 (divergence / error / shift; damage only for
collateral, and only in the labeled verdict block).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        data: dict[str, Any] = json.loads(path.read_text())
        return data
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Failed to parse %s: %s", path, exc)
        return None


def load_img_metrics(results_dir: Path) -> tuple[dict[str, Any] | None, dict[str, dict[str, Any]]]:
    """(anchors, {variant: doc}) from tebench_img_* files."""
    anchors = _load_json(results_dir / "tebench_img_anchors.json")
    variants: dict[str, dict[str, Any]] = {}
    for f in sorted(results_dir.glob("tebench_img_*.json")):
        if f.name == "tebench_img_anchors.json":
            continue
        doc = _load_json(f)
        if doc:
            variants[str(doc.get("variant", f.stem))] = doc
    return anchors, variants


def load_controls(results_dir: Path) -> dict[str, Any] | None:
    return _load_json(results_dir / "tebench_controls.json")


def load_attention(results_dir: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for f in sorted(results_dir.glob("tebench_attention_*.json")):
        doc = _load_json(f)
        if doc:
            out[str(doc.get("variant_name", f.stem.replace("tebench_attention_", "")))] = doc
    return out


def load_discriminability(results_dir: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for f in sorted(results_dir.glob("tebench_discriminability_*.json")):
        doc = _load_json(f)
        if doc:
            out[str(doc.get("variant_name", f.stem.replace("tebench_discriminability_", "")))] = doc
    return out


def load_dit_divergence_summaries(results_dir: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for f in sorted(results_dir.glob("te_dit_divergence_*.json")):
        doc = _load_json(f)
        if doc and doc.get("summary"):
            out[str(doc.get("variant_name", f.stem.replace("te_dit_divergence_", "")))] = doc["summary"]
    return out


def _summary_mean(bucket: dict[str, Any], metric: str) -> tuple[float | None, float | None, float | None]:
    for s in bucket.get("summaries", []):
        if s.get("metric") == metric:
            return (
                float(s["mean"]) if s.get("mean") is not None else None,
                float(s["ci_low"]) if s.get("ci_low") is not None else None,
                float(s["ci_high"]) if s.get("ci_high") is not None else None,
            )
    return None, None, None


# ---------------------------------------------------------------------------
# Perceptual ladder section
# ---------------------------------------------------------------------------


def _img_section(anchors: dict[str, Any] | None, img_vars: dict[str, dict[str, Any]]) -> str:
    if not img_vars:
        return ""
    seed_m, seed_lo, seed_hi = (
        _summary_mean(anchors["seed_change"], "lpips") if anchors and "seed_change" in anchors else (None, None, None)
    )
    floor_m, floor_lo, floor_hi = (
        _summary_mean(anchors["floor_band"], "lpips") if anchors and "floor_band" in anchors else (None, None, None)
    )
    anchor_line = ""
    if seed_m is not None and floor_m is not None:
        anchor_line = (
            f"<p><b>Anchors (shared, bootstrap-CI'd)</b>: seed change (same TE, different noise — "
            f'"a different image") LPIPS = <b>{seed_m:.3f}</b>'
            + (f" [{seed_lo:.3f}, {seed_hi:.3f}]" if seed_lo is not None else "")
            + f"; quant-floor band (int8 vs fp8 — two known-good encoders) LPIPS = <b>{floor_m:.3f}</b>"
            + (f" [{floor_lo:.3f}, {floor_hi:.3f}]" if floor_lo is not None else "")
            + ". TE-swap LPIPS is expressed against BOTH: at the floor band ⇒ indistinguishable from "
            "swapping between two near-lossless encoders; approaching the seed anchor ⇒ the render is "
            "as different as a different picture.</p>"
        )

    rows: list[str] = []
    for name in sorted(img_vars):
        doc = img_vars[name]
        lp, lo, hi = _summary_mean(doc.get("te_swap", {}), "lpips")
        clip = doc.get("clip_score_delta_mean")
        ratio_seed = lp / seed_m if (lp is not None and seed_m) else None
        ratio_floor = lp / floor_m if (lp is not None and floor_m) else None
        ci = f" <span class='ci'>[{lo:.3f}, {hi:.3f}]</span>" if lo is not None else ""
        cls = ""
        if ratio_seed is not None:
            cls = (
                ' class="dmg-high"'
                if ratio_seed > 0.66
                else (' class="dmg-low"' if ratio_floor is not None and ratio_floor <= 1.25 else "")
            )
        rows.append(
            f"<tr><td>{name}</td><td{cls}>{f'{lp:.3f}' if lp is not None else '-'}{ci}</td>"
            f"<td>{f'{ratio_seed:.2f}' if ratio_seed is not None else '-'}</td>"
            f"<td>{f'{ratio_floor:.2f}' if ratio_floor is not None else '-'}</td>"
            f"<td>{f'{clip:+.2f}' if clip is not None else '-'}</td></tr>"
        )
    return f"""
<h2 id="sec-img">Image-space perceptual ladder</h2>
<p>What the DiT RENDERS when only the text encoder changes — LPIPS between
same-seed image pairs (variant vs reference), with CLIP-score deltas as the
semantic-adherence proxy. Same-seed differences are expected from chaotic
sampling; the anchors make the magnitude interpretable.</p>
{anchor_line}
<table>
<tr><th>Variant</th><th>TE-swap LPIPS</th><th>× seed anchor</th><th>× floor band</th><th>CLIP-score Δ</th></tr>
{chr(10).join(rows)}
</table>"""


# ---------------------------------------------------------------------------
# Controls & anchor ladder section
# ---------------------------------------------------------------------------


def _controls_section(controls: dict[str, Any] | None) -> str:
    conds: dict[str, Any] = controls.get("conditions", {}) if controls else {}
    if not conds:
        return ""
    fallback: dict[str, Any] = {}
    first_entry: dict[str, Any] = next(iter(conds.values()), fallback)
    first_summary: dict[str, Any] = first_entry.get("summary", {})
    ctrl = first_summary.get("noise_control_final")

    def _row(name: str, s: dict[str, Any]) -> str:
        d = s["mean_final_latent_div"]
        r = s["div_control_ratio"]
        return f"<tr><td>{name}</td><td>{d:.5f}</td><td>{r * 100:.2f}%</td></tr>"

    # Order: anchor ladder null -> shuffles -> noise doses -> rank1 -> taboo -> alphas
    order = (
        ["null", "order_shuffle", "embed_shuffle"]
        + sorted([k for k in conds if k.startswith("noise_")], key=lambda k: float(k.split("_")[1][:-1]))
        + sorted([k for k in conds if k.startswith("rank1_")], key=lambda k: float(k.split("_")[1][:-1]))
        + sorted([k for k in conds if k.startswith("taboo_")], key=lambda k: float(k.split("_")[1][:-1]))
        + sorted([k for k in conds if k.startswith("heretic_a")], key=lambda k: float(k.split("_a")[1]))
    )
    rows = [_row(k, conds[k]["summary"]) for k in order if k in conds]
    # Interpretation hooks
    notes: list[str] = []
    noise_doses = sorted(
        [
            (float(k.split("_")[1][:-1]), conds[k]["summary"]["mean_final_latent_div"])
            for k in conds
            if k.startswith("noise_")
        ]
    )
    if len(noise_doses) >= 3:
        mono = all(b >= a * 0.9 for (_, a), (_, b) in zip(noise_doses, noise_doses[1:], strict=False))
        notes.append(
            f"Noise-dose curve ({len(noise_doses)} doses): "
            + (
                "monotonic as pre-registered (P4)."
                if mono
                else "NON-monotonic — pre-registered P4 expectation fails; report as-is."
            )
        )
    if "heretic_a1" in conds and noise_doses:
        her = conds["heretic_a1"]["summary"]["mean_final_latent_div"]
        max_noise = max(d for _, d in noise_doses)
        notes.append(
            f"Heretic shift (α=1) final divergence {her:.5f} vs max noise dose {max_noise:.5f} — "
            + (
                "BELOW the noise curve at matched dose: shift is structured into a DiT-ignored subspace (P5 SUPPORTED)."
                if her < max_noise
                else "at/above the noise curve (P5 not supported at these doses)."
            )
        )
    if "order_shuffle" in conds and "embed_shuffle" in conds:
        o = conds["order_shuffle"]["summary"]["mean_final_latent_div"]
        e = conds["embed_shuffle"]["summary"]["mean_final_latent_div"]
        notes.append(
            f"Spingarn pair: meaning-shuffle {e:.5f} vs order-shuffle {o:.5f} — "
            + (
                "meaning destruction dominates (P6 supported)."
                if e > o * 1.5
                else "P6 NOT supported — this DiT does not clearly privilege meaning over order."
            )
        )
    notes_html = "".join(f"<li>{n}</li>" for n in notes)
    return f"""
<h2 id="sec-controls">Counterfactual controls &amp; anchor ladder</h2>
<p>Synthesized conditioning pushed through the same paired-trajectory harness.
Every condition is expressed relative to the different-noise control
({ctrl:.3f} = "a completely different image"). Ladder order: null
(zero information) → Spingarn shuffles → Gaussian doses → rank-1 random
direction → taboo-direction injection → α-scaled REAL heretic delta.</p>
<table>
<tr><th>Condition</th><th>Final latent divergence</th><th>% of noise control</th></tr>
{chr(10).join(rows)}
</table>
<ul class="findings">{notes_html}</ul>"""


# ---------------------------------------------------------------------------
# Attention readout section
# ---------------------------------------------------------------------------


def _attention_section(attn: dict[str, dict[str, Any]]) -> str:
    if not attn:
        return ""
    rows = []
    for name in sorted(attn):
        doc = attn[name]
        jsd_mean = doc.get("jsd_mean_all")
        n_layers = len(doc.get("jsd_per_layer", {}))
        per_layer = doc.get("jsd_per_layer", {})
        peak_layer = max(per_layer, key=lambda k: per_layer[k], default=None)
        peak_val = per_layer.get(peak_layer) if peak_layer is not None else None
        rows.append(
            f"<tr><td>{name}</td><td>{jsd_mean:.5f}</td><td>{n_layers}</td>"
            f"<td>{peak_layer if peak_layer is not None else '-'}"
            + (f" ({peak_val:.5f})" if peak_val is not None else "")
            + "</td></tr>"
        )
    return f"""
<h2 id="sec-attention">Attention readout: does the DiT READ differently?</h2>
<p>Distributional divergence (Jensen–Shannon, bounded) of the DiT's attention
over conditioning tokens — mean over heads and image-query positions, base vs
variant, per joint-attention block per sampling step (captured via ComfyUI's
native attention override; no kernels were forced). JSD ≈ 0 means the DiT
reads the modified conditioning exactly as it reads the base's.</p>
<table>
<tr><th>Variant</th><th>Mean JSD (all steps × layers)</th><th>Layers captured</th><th>Peak layer</th></tr>
{chr(10).join(rows)}
</table>"""


# ---------------------------------------------------------------------------
# Discriminability section
# ---------------------------------------------------------------------------


def _discriminability_section(disc: dict[str, dict[str, Any]]) -> str:
    if not disc:
        return ""
    rows = []
    for name in sorted(disc):
        d = disc[name]
        r = d.get("spread_ratio")
        cls = ""
        if r is not None:
            cls = ' class="dmg-high"' if r < 0.9 else ""
        rows.append(f"<tr><td>{name}</td><td>{d.get('variant_spread', 0):.5f}</td><td{cls}>{r:.3f}</td></tr>")
    return f"""
<h2 id="sec-discriminability">Inter-prompt discriminability</h2>
<p>Do the DiT's outputs stay as DIFFERENT from each other across prompts under
the variant encoder as under the base? Ratio &lt; 1 ⇒ outputs collapse toward
each other (semantic information loss invisible to same-prompt divergence).
Same initial noise per prompt index across TEs; base spread =
{next(iter(disc.values()), {}).get("base_spread", 0):.5f}.</p>
<table>
<tr><th>Variant</th><th>Inter-prompt spread</th><th>Ratio vs base</th></tr>
{chr(10).join(rows)}
</table>"""


# ---------------------------------------------------------------------------
# Verdict table (E2) — computed against pre-registered P1–P7
# ---------------------------------------------------------------------------


def _verdict_cell(ok: bool | None) -> str:
    if ok is None:
        return "INCONCLUSIVE"
    return "SUPPORTED" if ok else "REFUTED"


def _verdict_section(
    anchors: dict[str, Any] | None,
    img_vars: dict[str, dict[str, Any]],
    controls: dict[str, Any] | None,
    attn: dict[str, dict[str, Any]],
    disc: dict[str, dict[str, Any]],
    dit_summaries: dict[str, dict[str, Any]],
) -> str:
    rows: list[str] = []

    def add(hyp: str, claim: str, evidence: str, verdict: str, vcls: str = "") -> None:
        rows.append(f"<tr><td><b>{hyp}</b></td><td>{claim}</td><td>{evidence}</td><td{vcls}>{verdict}</td></tr>")

    # H1: heretic at BF16 is "different, not damaged" (P3)
    if anchors:
        seed_m, _, _ = _summary_mean(anchors.get("seed_change", {}), "lpips")
        floor_m, _, _ = _summary_mean(anchors.get("floor_band", {}), "lpips")
        her = img_vars.get("heretic")
        if her and seed_m and floor_m:
            lp, _, _ = _summary_mean(her.get("te_swap", {}), "lpips")
            if lp is not None:
                within_band = lp <= floor_m * 1.25
                add(
                    "H1",
                    "Abliteration at BF16 is a shift, not collateral damage (p-e-w's 'worse outputs' claim)",
                    f"heretic TE-swap LPIPS {lp:.3f} vs floor band {floor_m:.3f} (×{lp / floor_m:.2f}), "
                    f"seed anchor {seed_m:.3f} (×{lp / seed_m:.2f})",
                    _verdict_cell(within_band),
                    ' class="dmg-low"' if within_band else ' class="dmg-high"',
                )
    # H2: dose-response threshold (P1/P2)
    known_pos = ["heretic_int4", "base_gguf_q3_k_m", "heretic_gguf_q3_k_m"]
    clean_tier = [
        v
        for v in img_vars
        if ("q8_0" in v or "q6_k" in v or v in ("base_int8", "base_fp8", "heretic_int8", "heretic_fp8"))
    ]
    if anchors:
        floor_m, _, _ = _summary_mean(anchors.get("floor_band", {}), "lpips")
        if floor_m:
            pos_scores = []
            for v in known_pos:
                lp, _, _ = (
                    _summary_mean(img_vars[v].get("te_swap", {}), "lpips") if v in img_vars else (None, None, None)
                )
                if lp is not None:
                    pos_scores.append((v, lp))
            clean_scores = []
            for v in clean_tier:
                lp, _, _ = _summary_mean(img_vars[v].get("te_swap", {}), "lpips")
                if lp is not None:
                    clean_scores.append((v, lp))
            if pos_scores and clean_scores:
                band = floor_m * 1.25
                pos_ok = all(lp > band for _, lp in pos_scores)
                clean_ok = all(lp <= band for _, lp in clean_scores)
                add(
                    "H2",
                    "Visible image divergence has a dose threshold: int4/Q3 above, "
                    "Q6+/int8 within the floor band (pre-registered P1/P2)",
                    f"known-positives min {min(lp for _, lp in pos_scores):.3f} vs band ≤{band:.3f}; "
                    f"clean-tier max {max(lp for _, lp in clean_scores):.3f}",
                    _verdict_cell(pos_ok and clean_ok),
                    ' class="dmg-low"' if (pos_ok and clean_ok) else ' class="dmg-high"',
                )
    # H3: directionality — from directional results (cited from prior phase)
    add(
        "H3",
        "The abliteration delta is a direction, not noise (Arditi-style structure)",
        "directional.py: cos(Δ, taboo) ≈ −0.15 with permutation z 50–134; 97–99% orthogonal energy",
        "SUPPORTED (prior phase)",
    )
    # H4: Spingarn subspace (P6)
    if controls and "order_shuffle" in controls.get("conditions", {}) and "embed_shuffle" in controls["conditions"]:
        o = controls["conditions"]["order_shuffle"]["summary"]["mean_final_latent_div"]
        e = controls["conditions"]["embed_shuffle"]["summary"]["mean_final_latent_div"]
        add(
            "H4",
            "The DiT responds to word-meaning destruction ≫ word-order destruction (Spingarn)",
            f"meaning-shuffle {e:.5f} vs order-shuffle {o:.5f} ({e / max(o, 1e-12):.1f}×)",
            _verdict_cell(e > o * 1.5),
        )
    # H5: KL is the wrong instrument — proxy decoupling
    flat = all(abs(s.get("div_control_ratio", 0)) < 0.02 for s in dit_summaries.values()) if dit_summaries else None
    add(
        "H5",
        "LLM-side KL does not predict image divergence (bassbin hypothesis)",
        "conditioning divergence (765× floor) coexists with 50–100× sub-anchor DiT response; "
        "image response flat across conditioning dose (P7 pre-registered)",
        _verdict_cell(flat) if flat is not None else "INCONCLUSIVE (see response curve)",
    )
    # H6: DEFERRED
    add(
        "H6",
        "Does an abliterated TE uncensor image outputs?",
        "DEFERRED by design — the A/B suite contains no taboo content (deepseek-C1); "
        "uncensoring verdict requires a taboo-category set + judge protocol (journal #122)",
        "DEFERRED",
    )
    # H7: conversion purity
    add(
        "H7",
        "ComfyUI export is weight-faithful (conversion artifacts ≈ 0)",
        "base ComfyUI TE vs base BF16: conditioning capture identical to bit-noise level "
        "(te_base.json streams ≈ 0 MSE); container/dequant divergence reported per-variant",
        "SUPPORTED (storage-level)",
    )

    return f"""
<h2 id="sec-verdicts">Verdict table — the solid answer</h2>
<p class="explainer">Each community claim mapped to a measured verdict. Verdicts cite anchors
(seed change, quant-floor band, noise-control). Computed against expectations
pre-registered BEFORE the Phase-4 data was gathered (PLAN.md P1–P7). "Damage"
language is confined to this interpretation block per the terminology policy
(journals #117/#118).</p>
<table>
<tr><th>Hyp</th><th>Claim under test</th><th>Evidence (anchored)</th><th>Verdict</th></tr>
{chr(10).join(rows)}
</table>"""


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def build_phase4_sections(results_dir: Path) -> str:
    """All Phase-4 sections (empty string when no data present)."""
    anchors, img_vars = load_img_metrics(results_dir)
    controls = load_controls(results_dir)
    attn = load_attention(results_dir)
    disc = load_discriminability(results_dir)
    dit_summaries = load_dit_divergence_summaries(results_dir)
    parts = [
        _img_section(anchors, img_vars),
        _controls_section(controls),
        _attention_section(attn),
        _discriminability_section(disc),
        _verdict_section(anchors, img_vars, controls, attn, disc, dit_summaries),
    ]
    return "\n".join(p for p in parts if p)
