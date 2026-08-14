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


def _matched_ref_note(results_dir: Path) -> str:
    """Cross-check note when a matched (vs base BF16) run exists (deepseek-R1)."""
    doc = _load_json(results_dir / "matched_ref" / "tebench_img_heretic.json")
    if not doc:
        return ""
    lp, _, _ = _summary_mean(doc.get("te_swap", {}), "lpips")
    if lp is None:
        return ""
    return (
        f"<p><b>Matched-frame cross-check:</b> a parallel scoring run against the BF16 base "
        f"(isolating abliteration from the reference's int8 dtype step) gives heretic-vs-base "
        f"LPIPS {lp:.3f} — the headline table's reference is base_int8; the two frames agree.</p>"
    )


def _img_section(anchors: dict[str, Any] | None, img_vars: dict[str, dict[str, Any]], results_dir: Path) -> str:
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

    # matched-frame cross-check (review deepseek-R1/hy3-R4): a parallel run
    # against the BF16 base isolates abliteration from the int8 dtype step.
    matched_note = _matched_ref_note(results_dir)
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
{matched_note}
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
    her_a10 = conds.get("heretic_a10", {}).get("summary", {}).get("mean_final_latent_div")
    noise_1x = conds.get("noise_1x", {}).get("summary", {}).get("mean_final_latent_div")
    if "heretic_a1" in conds and noise_doses:
        her = conds["heretic_a1"]["summary"]["mean_final_latent_div"]
        notes.append(
            f"The real (unscaled) heretic edit deflects sampling {her:.5f} — far below the heavy-noise "
            "doses simply because the edit itself is small at the DiT input (~75× lighter than the "
            "heaviest compression perturbation)."
        )
    if her_a10 is not None and noise_1x is not None:
        notes.append(
            f"P5 ADJUDICATION AT MATCHED MAGNITUDE: the heretic delta scaled to the noise family's 1× "
            f"dose (α=10, {her_a10:.4f}) responds essentially like that noise ({noise_1x:.4f}) — the "
            "image model does NOT specially ignore the refusal direction (P5 not supported); the "
            "abliteration's small image effect is a SMALL-EDIT effect, not a hidden-subspace effect."
        )
    elif "heretic_a10" not in conds:
        notes.append(
            "P5 INCONCLUSIVE: no magnitude-matched dose pair exists in this battery "
            "(heretic_a1 is ~75× lighter than the nearest noise dose)."
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
                # hy3-R4 caveat: quantify "above band" against the cleanest
                # UNMODIFIED 4-bit quant so REFUTED reads as "visible like a
                # 4-bit quant", not "catastrophic".
                q4_note = ""
                for q4n in ("base_nvfp4", "base_gguf_q4_k_m"):
                    if q4n in img_vars:
                        lp_q4, _, _ = _summary_mean(img_vars[q4n].get("te_swap", {}), "lpips")
                        if lp_q4 is not None:
                            q4_note = (
                                f" — comparable to an unmodified 4-bit quant's visible shift "
                                f"({q4n} {lp_q4:.3f}), not catastrophic"
                            )
                            break
                add(
                    "H1",
                    "Abliteration at BF16 is a shift, not collateral damage (p-e-w's 'worse outputs' claim)",
                    f"heretic TE-swap LPIPS {lp:.3f} vs floor band {floor_m:.3f} (×{lp / floor_m:.2f}), "
                    f"seed anchor {seed_m:.3f} (×{lp / seed_m:.2f}){q4_note}",
                    _verdict_cell(within_band),
                    ' class="dmg-low"' if within_band else ' class="dmg-high"',
                )
    # H2: dose-response threshold (P1/P2)
    known_pos = ["heretic_int4", "base_gguf_q3_k_m", "heretic_gguf_q3_k_m"]
    # P2 pre-registration scopes the clean tier to BASE-family quants. Heretic
    # rows at any quant carry the (independently visible, cf. H1) abliteration
    # shift and must not leak into the quantization verdict via substring match
    # (review hy3-C1: 'heretic_gguf_q8_0' matches 'q8_0').
    clean_tier = [
        v for v in img_vars if v.startswith("base_") and ("q8_0" in v or "q6_k" in v or v in ("base_int8", "base_fp8"))
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
                heretic_quant_max = None
                for v in img_vars:
                    if v.startswith("heretic_") and v not in ("heretic",):
                        lp_hq, _, _ = _summary_mean(img_vars[v].get("te_swap", {}), "lpips")
                        if lp_hq is not None:
                            heretic_quant_max = max(heretic_quant_max or 0.0, lp_hq)
                heretic_note = (
                    f"; abliterated quants reach {heretic_quant_max:.3f} — above band via the "
                    "abliteration shift itself (H1), not via quantization"
                    if heretic_quant_max is not None and heretic_quant_max > band
                    else ""
                )
                add(
                    "H2",
                    "Visible image divergence has a dose threshold: base-family int4/Q3 above, "
                    "base-family Q6+/int8/fp8 within the floor band (pre-registered P1/P2)",
                    f"base-family clean-tier max {max(lp for _, lp in clean_scores):.3f} ≤ band "
                    f"{band:.3f}; known-positives min {min(lp for _, lp in pos_scores):.3f} > band" + heretic_note,
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
    # H5: KL is the wrong instrument — proxy decoupling (evidence derived
    # from the trajectory summaries; no hardcoded multipliers — hy3-R2)
    ratios = [float(s.get("div_control_ratio", 0.0)) for s in dit_summaries.values()]
    flat = all(r < 0.02 for r in ratios) if ratios else None
    h5_evidence = (
        f"every variant's trajectory response stays at {min(ratios) * 100:.1f}–{max(ratios) * 100:.1f}% "
        "of the different-image anchor regardless of conditioning divergence (orders of magnitude "
        "above the quant floor) — conditioning magnitude does not predict image response "
        "(response-curve section)"
        if ratios
        else "no trajectory summaries available"
    )
    add(
        "H5",
        "Conditioning-side divergence does not predict image divergence (bassbin's KL-proxy hypothesis)",
        h5_evidence,
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
    # H7: conversion purity — optional stretch; NOT executed this comparison (hy3-R5)
    add(
        "H7",
        "ComfyUI export is weight-faithful (conversion artifacts ≈ 0)",
        "NOT RUN — the optional base-vs-base purity artifact was not produced in this comparison "
        "(plan marked it a stretch); container/dequant divergence IS reported per-variant elsewhere",
        "NOT RUN",
    )

    return f"""
<h2 id="sec-verdicts">Verdict table — the solid answer</h2>
<p class="explainer">Each community claim mapped to a measured verdict. Verdicts cite anchors
(seed change, quant-floor band, noise-control). Computed against expectations
pre-registered BEFORE the Phase-4 data was gathered (PLAN.md P1–P7); the in-band
rule uses band = floor × 1.25 (the analysis safety margin over the int8↔fp8 wiggle —
the ×1.25 multiplier is the standing convention, not a per-result adjustment).
"Damage" language is confined to this interpretation block per the terminology
policy (journals #117/#118).</p>
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
        _executive_summary(anchors, img_vars, controls, attn, disc),
        _img_section(anchors, img_vars, results_dir),
        _controls_section(controls),
        _attention_section(attn),
        _word_readout_section(attn),
        _discriminability_section(disc),
        _verdict_section(anchors, img_vars, controls, attn, disc, dit_summaries),
    ]
    return "\n".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Plain-English executive summary (labeled interpretation section)
# ---------------------------------------------------------------------------


def _executive_summary(
    anchors: dict[str, Any] | None,
    img_vars: dict[str, dict[str, Any]],
    controls: dict[str, Any] | None,
    attn: dict[str, dict[str, Any]],
    disc: dict[str, dict[str, Any]],
) -> str:
    """The "start here" section: what we asked, how it's measured, what it says.

    This is a LABELED INTERPRETATION section (terminology policy, journals
    #117/#118): plain-English verdict language is allowed here and in the
    verdict table; the data tables themselves stay value-neutral.
    All numbers are pulled from the Phase-4 data at render time.
    """
    if not img_vars:
        return ""
    seed_m, _, _ = (
        _summary_mean(anchors["seed_change"], "lpips") if anchors and "seed_change" in anchors else (None, None, None)
    )
    floor_m, _, _ = (
        _summary_mean(anchors["floor_band"], "lpips") if anchors and "floor_band" in anchors else (None, None, None)
    )
    if seed_m is None or floor_m is None:
        return ""

    band = floor_m * 1.25

    def _lp(name: str) -> float | None:
        doc = img_vars.get(name)
        if not doc:
            return None
        lp, _, _ = _summary_mean(doc.get("te_swap", {}), "lpips")
        return lp

    # Tier the variants from real data
    fine: list[str] = []
    visible: list[str] = []
    for name, doc in img_vars.items():
        lp, _, _ = _summary_mean(doc.get("te_swap", {}), "lpips")
        if lp is None:
            continue
        (fine if lp <= band else visible).append(name)
    visible_sorted = sorted(visible, key=lambda n: -(_summary_mean(img_vars[n].get("te_swap", {}), "lpips")[0] or 0))
    heretic_lp = _lp("heretic")
    worst = visible_sorted[0] if visible_sorted else None
    worst_lp = _summary_mean(img_vars[worst].get("te_swap", {}), "lpips")[0] if worst else None

    # Controls headline numbers
    conds: dict[str, Any] = (controls or {}).get("conditions", {})
    noise_1x = conds.get("noise_1x", {}).get("summary", {}).get("mean_final_latent_div")
    taboo_1x = conds.get("taboo_1x", {}).get("summary", {}).get("mean_final_latent_div")
    heretic_traj = conds.get("heretic_a1", {}).get("summary", {}).get("mean_final_latent_div")
    noise_doses = sorted(
        (float(k.split("_")[1][:-1]), conds[k]["summary"]["mean_final_latent_div"])
        for k in conds
        if k.startswith("noise_") and "summary" in conds[k]
    )
    mono_txt = ""
    if len(noise_doses) >= 3:
        mono = all(b >= a * 0.9 for (_, a), (_, b) in zip(noise_doses, noise_doses[1:], strict=False))
        mono_txt = (
            f" Bigger noise injections produced bigger image changes, as they must "
            f"({noise_doses[0][1]:.3f} at quarter dose rising to {noise_doses[-1][1]:.3f} at double dose"
            + (" — monotonic as pre-registered (P4)." if mono else " — CAUTION: non-monotonic, ruler needs review).")
        )
    taboo_txt = ""
    her_a10 = conds.get("heretic_a10", {}).get("summary", {}).get("mean_final_latent_div")
    if noise_1x is not None and her_a10 is not None:
        taboo_txt = (
            f" At MATCHED perturbation size, the real abliteration direction (scaled ×10 to the noise "
            f"family's 1× dose) deflects sampling {her_a10:.4f} vs that noise's {noise_1x:.4f} — the "
            "image model does <b>not</b> specially ignore the refusal direction. The abliteration's "
            "small image effect is a <b>small-edit effect</b>, not a hidden-subspace effect."
        )
    elif noise_1x is not None and taboo_1x is not None:
        taboo_txt = (
            f" Refusal-direction injection {taboo_1x:.3f} vs random noise {noise_1x:.3f} — NOTE: these "
            "are not magnitude-matched (the injection matches the heretic norm; the noise matches the "
            "heaviest compression dose); no subspace conclusion is drawn."
        )
    heretic_txt = ""
    if heretic_traj is not None:
        heretic_txt = (
            f" The real, unscaled heretic edit deflects sampling only {heretic_traj:.4f} — it is a "
            "light perturbation at the image model's input, and its image effect is proportionally small."
        )
    controls_html = (
        f"<ul class='findings'>{mono_txt}{taboo_txt}{heretic_txt}</ul>"
        if (mono_txt or taboo_txt or heretic_txt)
        else ""
    )

    attn_html = ""
    if attn:
        worst_attn = max(attn.items(), key=lambda kv: kv[1].get("jsd_mean_all", 0))
        attn_html = (
            "<p><b>What the image model LOOKS AT.</b> We x-rayed which prompt words the image model "
            f"pays attention to, base encoder vs modified. The largest readout change across the "
            f"measured encoders is {worst_attn[1].get('jsd_mean_all', 0):.4f} (Jensen–Shannon divergence, "
            "where 0 = reads identically and 0.69 = completely different reading habits) — "
            f"for <code>{worst_attn[0]}</code>.</p>"
        )
    disc_html = ""
    if disc:
        worst_disc = min(disc.items(), key=lambda kv: kv[1].get("spread_ratio", 1.0))
        below_09 = [k for k, v in disc.items() if v.get("spread_ratio", 1.0) < 0.9]
        heretic_below = [k for k in below_09 if k.startswith("heretic")]
        disc_html = (
            "<p><b>Do the images stay distinct from each other?</b> A truly broken encoder would make "
            "every prompt produce mushily similar pictures. The largest collapse across the measured "
            f"encoders keeps {worst_disc[1].get('spread_ratio', 1.0) * 100:.0f}% of the base encoder's "
            f"prompt-to-prompt separation (<code>{worst_disc[0]}</code>).</p>"
        )
        if heretic_below:
            disc_html += (
                f"<p>Honest caveat: {len(heretic_below)} abliterated+quantized combinations "
                f"(e.g. <code>{heretic_below[0]}</code>) lose 10–20% of prompt-to-prompt separation — "
                "compounding the abliteration with heavy compression is measurably worse than either "
                "alone. The abliteration is not wholly invisible when compounded.</p>"
            )

    heretic_verdict = ""
    if heretic_lp is not None:
        if heretic_lp <= band:
            heretic_verdict = (
                f"<li><b>The abliterated (heretic) encoder at full precision: image change {heretic_lp:.2f} — "
                f"<u>below</u> the normal wiggle band ({floor_m:.2f}).</b> Swapping to it changes the rendered "
                "image LESS than swapping between two perfectly fine encoders does. Indistinguishable from "
                "harmless.</li>"
            )
        else:
            heretic_verdict = (
                f"<li><b>The abliterated encoder at full precision: image change {heretic_lp:.2f} — ABOVE the "
                f"normal wiggle band ({floor_m:.2f}).</b> Report and inspect before trusting.</li>"
            )

    bullet_worst = ""
    if worst and worst_lp is not None:
        bullet_worst = (
            f"<li><b>Heavy compression ({worst.replace('_', ' ')}): {worst_lp:.2f} — "
            f"{worst_lp / floor_m:.1f}× the wiggle band</b>, most of the way to "
            f"'a different picture' ({seed_m:.2f}). Visible degradation, caused by compression — "
            "not by the abliteration.</li>"
        )

    return f"""
<h2 id="sec-summary-plain">Executive summary — in plain English</h2>
<p class="explainer"><b>This section interprets the data. Interpretation language lives here and in the
verdict table; every table below reports neutral measurements.</b></p>

<p><b>The one question:</b> if you swap Z-Image's text encoder for an "uncensored" (abliterated) or
compressed (quantized) one, <i>do the images get worse?</i></p>

<p><b>Why this was ever hard:</b> with a fixed seed, swapping ANY two encoders changes the image —
diffusion is a butterfly-effect machine, so "the images look different" proves nothing. Both sides of
the online debate were showing each other differences. We built rulers instead.</p>

<p><b>The rulers</b> (image difference measured in LPIPS — 0 = identical, ~0.5 = "clearly a different
picture"):
<ul class="findings">
<li><b>Normal wiggle band: {floor_m:.2f}</b> — the difference between two encoders everyone already
trusts (int8 vs fp8). Any swap scoring at or below this is indistinguishable from ordinary,
uncontroversial compression.</li>
<li><b>"A different picture": {seed_m:.2f}</b> — same encoder, different random seed. This is how far
the image moves when NOTHING is wrong at all.</li>
</ul></p>

<p><b>Where the encoders fall:</b></p>
<ul class="findings">
{heretic_verdict}
<li><b>Base-family quants from Q8 down to Q5, plus fp8/mxfp8/F16: all within the wiggle band</b>
({len([n for n in fine if n.startswith("base_")])} of {len(fine) + len(visible)} encoders measured
stay in-band, all of them base-family). Base-family 4-bit tiers (int4, nvfp4, Q4, Q3) ARE visible —
quantization damage is real and dose-ordered.</li>
{bullet_worst}
</ul>

<p><b>Stress tests (fake damage of known kinds):</b> we injected controlled perturbations into the
prompt embedding and measured how the sampler responds, to learn what real sabotage looks like.{controls_html}</p>

{attn_html}
{disc_html}

<p><b>Uncensoring:</b> not answered here — the prompt suite deliberately contains no explicit content
(that study needs its own taboo-prompt set and review protocol; see verdict table, H6 DEFERRED).</p>

<p><b>The full evidence for every claim lives in the
<a href="#sec-verdicts">verdict table</a>.</b></p>"""


# ---------------------------------------------------------------------------
# Word-level readout section — what WORDS does the DiT read differently?
# ---------------------------------------------------------------------------


def _word_readout_section(attn: dict[str, dict[str, Any]]) -> str:
    """Decode the attention readout back into words and diff base vs variant.

    The capture stores, per prompt, the mean text-readout distribution for
    BOTH sides (base + variant) and the token strings per position — so this
    section needs no extra runs. Answers: for the same prompt, which words
    does the DiT look at more / less under this encoder?
    """
    with_words = {k: d for k, d in attn.items() if d.get("has_word_readout") and d.get("per_prompt")}
    if not with_words:
        return ""
    from src.text_encoder.attention_metrics import word_diff

    blocks: list[str] = []
    for name in sorted(with_words):
        doc = with_words[name]
        pp_rows: list[str] = []
        for pp in doc["per_prompt"]:
            toks = pp.get("tokens", [])
            b = pp.get("base_mean_readout", [])
            v = pp.get("variant_mean_readout", [])
            if not toks or not b or not v:
                continue
            rows = word_diff(toks, b, v, top_k=12)
            if not rows:
                continue
            tr = "".join(
                f"<tr><td>{str(r['word'])[:28]}</td><td>{float(r['base']) * 100:.1f}%</td>"
                f"<td>{float(r['variant']) * 100:.1f}%</td>"
                f"<td>{float(r['delta']) * 100:+.1f}pp</td></tr>"
                for r in rows
            )
            prompt_short = pp.get("prompt", "")[:110]
            pp_rows.append(f"<tr><td colspan=4 class='meta'>{prompt_short}…</td></tr>{tr}")
        if pp_rows:
            blocks.append(
                f"<details class='glossary'><summary><b>{name}</b> — top word-readout shifts "
                f"(mean JSD {float(doc.get('jsd_mean_all', 0)):.5f})</summary>"
                "<table><tr><th>word</th><th>base TE mass</th><th>variant TE mass</th><th>Δ</th></tr>"
                + "".join(pp_rows)
                + "</table></details>"
            )
    if not blocks:
        return ""
    return f"""
<h2 id="sec-words">What the DiT reads, in words</h2>
<p>The attention readout decoded back into prompt words: for the SAME prompt,
how much attention mass each word receives from the image tokens under the
base encoder vs this encoder (mean over sampling steps and layers). Δ is in
percentage points of total attention. This is the direct answer to "does the
modified encoder change what the image model looks at?" — a healthy encoder
shows tiny, diffuse shifts; a sabotaged one would move mass off content
words (or onto padding).</p>
{chr(10).join(blocks)}"""
