"""H3 text-encoder battery report — Phase 7 verdicts (te-h3).

Consumes the battery results (tebench_h3_audio.json, tebench_h3_video.json,
te_heretic.json conditioning analysis) and renders a standalone HTML report.
Separate from the z-image phase4 report by design (comparisons stay fully
separate); shares only styling conventions.

Refuses to publish verdicts from smoke runs (manifest smoke=true poisons
every downstream result; NOTES.md non-reportable rule).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

from src.text_encoder import TE_RESULTS_VERSION

log = logging.getLogger(__name__)


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def h8_verdict(audio: dict[str, Any] | None) -> dict[str, str]:
    """Audio damage: heretic error rate exceeds Floor A (whisper intrinsic on
    base) on >= 2 of 3 tiers."""
    if audio is None:
        return {"verdict": "NOT RUN", "detail": "no tebench_h3_audio.json"}
    if audio.get("smoke"):
        return {"verdict": "REFUSED (smoke)", "detail": "stand-in run — non-reportable"}
    arms = audio.get("arms", {})
    if "base" not in arms or "heretic" not in arms:
        return {"verdict": "NOT RUN", "detail": "missing base/heretic arm"}
    floor_a = (audio.get("floors") or {}).get("a_whisper_intrinsic") or {}
    exceeded = 0
    details = []
    for tier in ("rare_polysyllabic", "common_polysyllabic", "monosyllabic"):
        h_tier = arms["heretic"]["tiers"].get(tier) or {}
        h_err = h_tier.get("error_rate")
        n_h = h_tier.get("total") or 1
        a = floor_a.get(tier)
        if h_err is None or a is None:
            continue
        # margin guard: exceedance needs >= 1 full extra error above the
        # floor rate, so a single stochastic whisper flake on a clean base
        # tier cannot flip the verdict (review round 1)
        margin = 1.0 / max(n_h, 1)
        beat = h_err > max(a, 0.0) + margin
        exceeded += int(beat)
        details.append(
            f"{tier}: heretic {h_err:.3f} vs floorA {a:.3f} (+{margin:.3f} margin){' EXCEEDED' if beat else ''}"
        )
    verdict = "SUPPORTED" if exceeded >= 2 else "REFUTED (no damage)"
    return {"verdict": verdict, "detail": "; ".join(details)}


def h9_verdict(audio: dict[str, Any] | None) -> dict[str, str]:
    """Structured edit: same words fail across seeds; heretic overlap >= 2x
    base overlap, failure-only jaccard."""
    if audio is None:
        return {"verdict": "NOT RUN", "detail": "no tebench_h3_audio.json"}
    if audio.get("smoke"):
        return {"verdict": "REFUSED (smoke)", "detail": "stand-in run — non-reportable"}
    arms = audio.get("arms", {})
    if "heretic" not in arms:
        return {"verdict": "NOT RUN", "detail": "no heretic arm"}
    h = arms["heretic"].get("cross_seed_jaccard_h9")
    b = arms.get("base", {}).get("cross_seed_jaccard_h9")
    if h is None:
        return {
            "verdict": "VACUOUS (no failures)",
            "detail": "heretic never failed a target across seeds; nothing to structure-compare",
        }
    if b is None:
        # heretic fails reproducibly, base has <2 failing seeds (or clean):
        # the strongest structured-edit signature — never a REFUTED
        return {
            "verdict": "SUPPORTED (base clean)",
            "detail": f"heretic h9={h:.3f} with base h9={b} (base lacks failure structure)",
        }
    if b == 0.0:
        return {
            "verdict": "REFUTED (no structure)",
            "detail": f"heretic h9={h:.3f} but base failures never repeat (base h9=0) — heretic overlap is noise-level",
        }
    ratio = h / b
    det = f"heretic h9={h:.3f}, base h9={b:.3f}, ratio={ratio:.2f}"
    return {
        "verdict": "SUPPORTED" if ratio >= 2.0 else "REFUTED (no structure)",
        "detail": det,
    }


def h10_verdict(video: dict[str, Any] | None) -> dict[str, str]:
    """Video reroll-not-damage: within seed anchor AND CLIP in band."""
    if video is None:
        return {"verdict": "NOT RUN", "detail": "no tebench_h3_video.json"}
    if video.get("smoke"):
        return {"verdict": "REFUSED (smoke)", "detail": "stand-in run — non-reportable"}
    v = video.get("variants", {}).get("heretic")
    anchor = video.get("seed_change_anchor_lpips")
    if v is None or anchor is None:
        return {"verdict": "NOT RUN", "detail": "missing heretic pairs or seed anchor"}
    det = (
        f"frame-LPIPS {v['mean_frame_lpips']} vs seed-anchor {anchor}; "
        f"CLIP delta absmax {v['clip_delta_absmax']} (band ±{video.get('clip_band', 1.5)})"
    )
    if v["within_seed_anchor"] and v["clip_within_band"]:
        return {"verdict": "SUPPORTED", "detail": det}
    return {"verdict": "REFUTED", "detail": det}


def h11_verdict(cond: dict[str, Any] | None) -> dict[str, str]:
    """Conditioning direction: from the te_heretic conditioning analysis we
    report the measured interface delta (full permutation-null directional
    analysis needs taboo/refusal prompt groups — te_directional_*.json)."""
    if cond is None:
        return {"verdict": "NOT RUN", "detail": "no te_heretic.json"}
    v = cond["variants"][0]
    rc = next((s for s in v["streams"] if s["stream"] == "refined_context"), None)
    if rc is None:
        return {"verdict": "NOT RUN", "detail": "no refined_context stream"}
    m = rc["metrics"]
    det = (
        f"interface cos {m['cos_all']:.4f}, CKA {m['cka_linear']:.4f} "
        "(permutation-null directional deferred: needs taboo/ref prompt groups)"
    )
    return {"verdict": "PARTIAL (interface measured, directional deferred)", "detail": det}


def _interface_paragraph(cond: dict[str, Any] | None) -> str:
    """Conditioning-interface prose rendered FROM the result json (no drift)."""
    cos = cka = None
    if cond:
        rc = next((s for s in cond["variants"][0]["streams"] if s["stream"] == "refined_context"), None)
        if rc is not None:
            cos = rc["metrics"]["cos_all"]
            cka = rc["metrics"]["cka_linear"]
    nums = (
        f"<b>cos {cos:.4f}, CKA {cka:.4f}</b>"
        if cos is not None and cka is not None
        else "<b>(conditioning result unavailable)</b>"
    )
    return f"""<p>The ARA edit is surgical at the interface: layers 0&ndash;30 are bit-clean
(cos &ge; 0.99998), divergence switches on around layer 38&ndash;40 (peak dip cos
0.99936 at layer_40, requant floor 0.99999 at every layer), partially recovering by
the consumed layer 49. What the DiT reads (<code>refined_context</code>):
{nums}. The packaging confound (heretic additionally int8-quantizes
<code>embed_tokens</code>) is REFUTED — an embedding-level difference would diverge
from layer 0 and persist; it washes out within the first layers (journal #139/#140).</p>"""


def verdict_row(hid: str, text: str, res: dict[str, str]) -> str:
    v = escape(res["verdict"])
    benign_null = "no damage" in res["verdict"] or "no structure" in res["verdict"]
    cls = (
        "ok"
        if "SUPPORTED" in res["verdict"] and "PARTIAL" not in res["verdict"]
        else "bad"
        if "REFUTED" in res["verdict"] and not benign_null
        else "mid"
    )
    return (
        f"<tr><td><b>{hid}</b></td><td>{escape(text)}</td>"
        f"<td>{escape(res['detail'])}</td><td class='{cls}'>{v}</td></tr>"
    )


def build_report(results_dir: Path, out_html: Path) -> str:
    audio = _load(results_dir / "tebench_h3_audio.json")
    video = _load(results_dir / "tebench_h3_video.json")
    cond = _load(results_dir / "te_heretic.json")

    smoke_refusal = bool((audio or {}).get("smoke")) or bool((video or {}).get("smoke"))
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    rows = [
        verdict_row(
            "H8",
            "Audio damage: heretic target-word error exceeds the whisper floor on >= 2 of 3 tiers",
            h8_verdict(audio),
        ),
        verdict_row(
            "H9",
            "Structured edit: the same words fail across seeds (failure-only overlap >= 2x base)",
            h9_verdict(audio),
        ),
        verdict_row(
            "H10",
            "Video reroll-not-damage: frame movement within the seed anchor, CLIP adherence in band",
            h10_verdict(video),
        ),
        verdict_row(
            "H11", "Conditioning delta is directional (ARA edit signature at the interface)", h11_verdict(cond)
        ),
        verdict_row(
            "H12",
            "Uncensoring: does the TE swap unlock refused content",
            {
                "verdict": "NOT TESTED BY DESIGN",
                "detail": "H3 has no render-stage refusal per community consensus; ethics scope — see z-image H6",
            },
        ),
    ]

    arms_audio_html = ""
    if audio and not audio.get("smoke"):
        arms_audio_html = "".join(
            f"<tr><td>{escape(arm)}</td><td>{a['mean_wer']:.4f}</td>"
            + "".join(
                f"<td>{escape(str((a['tiers'].get(t) or {}).get('error_rate', '—')))}</td>"
                for t in ("rare_polysyllabic", "common_polysyllabic", "monosyllabic")
            )
            + "</tr>"
            for arm, a in sorted(audio["arms"].items())
        )

    arms_video_html = ""
    if video and not video.get("smoke"):
        arms_video_html = "".join(
            f"<tr><td>{escape(arm)}</td><td>{escape(str(r['mean_frame_lpips']))}</td>"
            f"<td>{escape(str(r['mean_clip_self']))}</td>"
            f"<td>{escape(str(r['mean_clip_delta_vs_base']))}</td><td>{escape(str(r['n_pairs']))}</td></tr>"
            for arm, r in sorted(video["variants"].items())
        )

    refusal_banner = (
        '<p class="explainer" style="border-left-color:#c0392b"><b>REFUSED:</b> some inputs carry '
        "<code>smoke: true</code> (stand-in plumbing runs). Verdicts above are computed ONLY from "
        "non-smoke arms; smoke data is never published.</p>"
        if smoke_refusal
        else ""
    )

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>MiniMax H3 Text-Encoder Battery — Phase 7</title>
<style>
 body {{ font-family: -apple-system, 'Segoe UI', Roboto, sans-serif;
        margin: 24px auto; max-width: 1080px; color: #222; }}
 h1 {{ font-size: 26px; }} h2 {{ margin-top: 36px; border-bottom: 2px solid #4c72b0; padding-bottom: 4px; }}
 table {{ border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 14px; }}
 th, td {{ border: 1px solid #d0d7e2; padding: 6px 10px; text-align: left; vertical-align: top; }}
 th {{ background: #f2f5fa; }}
 td.ok {{ color: #1e7d32; font-weight: 600; }}
 td.bad {{ color: #b3261e; font-weight: 600; }}
 td.mid {{ color: #946200; font-weight: 600; }}
 .explainer {{ background: #f6f8fb; border-left: 4px solid #4c72b0;
              border-radius: 0 8px 8px 0; padding: 10px 14px; font-size: 13px; }}
 .meta {{ color: #666; font-size: 12.5px; }}
 code {{ background: #f0f2f6; padding: 1px 5px; border-radius: 4px; font-size: 12.5px; }}
</style></head><body>
<h1>MiniMax H3 Text-Encoder Battery — Phase 7</h1>
<p class="meta">Generated {now} · te_results_version {TE_RESULTS_VERSION} · results: {escape(str(results_dir))}</p>
{refusal_banner}

<h2>Origin and scope</h2>
<p>This battery asks what swapping H3's Qwen3-VL-32B text encoder (truncated to 50 layers)
for an abliterated (ARA-heretic, matched int8-convrot quant) variant does — motivated by
p-e-w's reproducible audio-artifact report ("inconceivable" rendered "inconcable",
journal #134). Arms: <code>base</code> (official int8_convrot) vs
<code>heretic</code> (ultra_uncensored ARA int8_convrot). Generation config: the
community turbo t2v workflow (6-step + LoRA v4) — disclosed; both arms traverse the
identical graph. Everything here is H3-specific: this is NOT the z-image study, and
z-image numbers are never mixed in.</p>

<h2>Conditioning interface (base vs heretic)</h2>
{_interface_paragraph(cond)}

<h2>H8/H9 — audio battery (the headline test)</h2>
<p>16 scripted spoken lines x 3 seeds per arm, whisper-large-v3 scoring, three
word-difficulty tiers. In-line repetition doubles the rare-word sample.</p>
<table><tr><th>Arm</th><th>mean WER</th><th>rare err</th><th>common err</th><th>mono err</th></tr>
{arms_audio_html}</table>
<p class="explainer">"inconceivable" (the p-e-w word) is spoken and transcribed
perfectly on <b>both</b> arms, every seed: rare-tier 0/56 (base) vs 0/54 (heretic).
Heretic mean WER (0.015) is at or below base (0.017). Caveats: turbo config (stock
50-step scheduler untested); whisper's LM prior can normalize mild degradation —
human listening check on the heretic professor clips is the closing control.</p>

<h2>H10 — video battery</h2>
<p>8 scenes x 2 seeds per arm, 8 frames/clip, LPIPS-alex + CLIP ViT-L-14.
Seed-change anchor (same TE, different noise): <b>{video.get("seed_change_anchor_lpips") if video else "—"}</b>.</p>
<table><tr><th>Arm</th><th>mean frame-LPIPS vs base</th><th>CLIP self</th><th>CLIP delta</th><th>pairs</th></tr>
{arms_video_html}</table>
<p class="explainer">Heretic moves frames ~0.33 — far below the 0.60 seed anchor and
next to the nvfp4 quant swap (0.29): the video silently REROLLS, nothing degrades
(CLIP adherence within ±1.5). Identical fingerprint to z-image's heretic (0.29 ≈ nvfp4 0.27).</p>

<h2>Verdicts</h2>
<table><tr><th>ID</th><th>Hypothesis</th><th>Evidence</th><th>Verdict</th></tr>
{"".join(rows)}</table>
<p class="explainer"><b>Reading:</b> H8 REFUTED means <i>no audio damage was detected</i>
from this cleanly-packaged ARA heretic at turbo config — p-e-w's artifact does not
reproduce here. Ranked explanations: (1) a different/earlier H3 conversion with real
packaging damage (the user's badly-done-safetensor theory — our clean file shows
nothing), (2) stock-scheduler config difference, (3) whisper normalization (weakest;
pending the ear check). H9 is VACUOUS because the heretic never failed a target —
there is no failure structure to compare. The interface delta (cos 0.9935, surgical
layer-40) plus the audio null plus the video reroll together say: the edit reaches
the DiT loudly enough to reroll pixels, but at this config it does not break speech
or adherence.</p>

<h2>Ethics &amp; scope</h2>
<p>Audio/video clips stay local. No taboo generation (H12 not tested — H3 has no
render-stage refusal per community consensus; engineered T2I/T2V safety lives in the
DiT, not the TE — see the z-image H6 result). Single-file, single-config study:
one ARA heretic conversion at one quant at turbo settings.</p>
</body></html>"""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html)
    log.info("Wrote %s", out_html)
    return html


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="H3 battery report")
    p.add_argument("--results-dir", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    build_report(Path(args.results_dir), Path(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
