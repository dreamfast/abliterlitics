"""Perturbation-ladder summary with the named reroll-fraction metric.

Reads ``tebench_img_*.json`` (+ ``tebench_img_anchors.json``) written by
``run_img_metrics.py`` and emits the ranked ladder every comparison report
leads with:

    swap (same seed, same prompt)        LPIPS    n   reroll fraction
    ...
    SEED CHANGE (anchor)                 0.5158  168   100.0%

**Reroll fraction** = arm's te_swap LPIPS mean / seed-change anchor LPIPS
mean — "the TE swap moves the image by X% of what a whole new seed does."
One scalar per arm; the abliteration's rung on the compression ladder.

Outputs (to --out-dir):
    ladder_summary.json    machine-readable (per-arm lpips/reroll + anchors)
    ladder_summary.md      ranked markdown table

Usage:
    python3 -m src.text_encoder.ladder_summary \
        --results-dir comparisons/te-krea2/results/te-krea2/text_encoder \
        --out-dir comparisons/te-krea2/results/te-krea2/text_encoder
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def load_ladder(results_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Return ({arm: {lpips_mean, n}}, anchors dict) from tebench_img_* files."""
    arms: dict[str, dict[str, Any]] = {}
    anchors: dict[str, Any] = {}
    for f in sorted(results_dir.glob("tebench_img_*.json")):
        name = f.stem[len("tebench_img_") :]
        data = json.loads(f.read_text())
        if name == "anchors":
            anchors = data
            continue
        vals = data.get("te_swap", {}).get("values", {}).get("lpips", [])
        if vals:
            arms[name] = {"lpips_mean": sum(vals) / len(vals), "n": len(vals)}
    return arms, anchors


def anchor_lpips(anchors: dict[str, Any]) -> tuple[float | None, int | None]:
    """Seed-change anchor (LPIPS mean, n_pairs) — (None, None) when absent/incomplete."""
    for s in anchors.get("seed_change", {}).get("summaries", []):
        if s.get("metric") == "lpips" and s.get("mean") is not None:
            return float(s["mean"]), int(s.get("n", 0)) or None
    return None, None


def build_summary(
    arms: dict[str, dict[str, Any]],
    seed_anchor: float | None,
    seed_anchor_n: int | None = None,
    reference_arm: str = "heretic",
) -> dict[str, Any]:
    """Ranked ladder with reroll fractions (None anchor -> fraction null)."""
    ranked = sorted(arms.items(), key=lambda kv: kv[1]["lpips_mean"])
    rows = [
        {
            "arm": name,
            "lpips_mean": round(v["lpips_mean"], 4),
            "n": v["n"],
            "reroll_fraction_pct": (round(100 * v["lpips_mean"] / seed_anchor, 1) if seed_anchor is not None else None),
        }
        for name, v in ranked
    ]
    return {
        "seed_change_anchor_lpips": round(seed_anchor, 4) if seed_anchor is not None else None,
        "seed_change_anchor_n": seed_anchor_n,
        "ladder": rows,
        "reference_arm": reference_arm,
        "reference_reroll_pct": next((r["reroll_fraction_pct"] for r in rows if r["arm"] == reference_arm), None),
    }


def render_markdown(summary: dict[str, Any], title: str) -> str:
    """Ranked ladder table in markdown (the comparison's one-sentence metric)."""
    lines = [f"# {title}", "", "## Perturbation ladder (LPIPS vs base bf16, same seed+prompt)", ""]
    anchor = summary["seed_change_anchor_lpips"]
    lines.append("| swap | LPIPS | n | reroll fraction |")
    lines.append("|---|---:|---:|---:|")
    for r in summary["ladder"]:
        pct = f"{r['reroll_fraction_pct']:.1f}%" if r["reroll_fraction_pct"] is not None else "—"
        lines.append(f"| {r['arm']} | {r['lpips_mean']:.4f} | {r['n']} | {pct} |")
    if anchor is not None:
        n_seed = summary.get("seed_change_anchor_n")
        n_cell = str(n_seed) if n_seed is not None else "—"
        lines.append(f"| **SEED CHANGE (anchor)** | **{anchor:.4f}** | {n_cell} | **100.0%** |")
    reference_arm = summary.get("reference_arm", "heretic")
    ref = summary.get("reference_reroll_pct")
    if ref is not None:
        lines += [
            "",
            f"**The whole {reference_arm} bf16 edit moves the image by {ref}% of a full seed change** "
            "(the reroll fraction — one scalar per model for community-facing summaries).",
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Perturbation ladder + reroll-fraction summary")
    p.add_argument("--results-dir", required=True, help="Directory with tebench_img_*.json")
    p.add_argument("--out-dir", required=True, help="Where to write ladder_summary.{json,md}")
    p.add_argument("--title", default="Text-encoder perturbation ladder")
    p.add_argument("--reference-arm", default="heretic", help="Arm named in the headline sentence")
    args = p.parse_args()

    results_dir, out_dir = Path(args.results_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arms, anchors = load_ladder(results_dir)
    if not arms:
        raise SystemExit(f"No tebench_img_*.json arms found in {results_dir}")
    seed_anchor, seed_anchor_n = anchor_lpips(anchors)
    if seed_anchor is None:
        log.warning("No seed-change anchor found — reroll fractions will be null (rerun after anchors)")
    elif seed_anchor_n is not None and seed_anchor_n < 5:
        log.warning("Seed-change anchor has only n=%d pairs — reroll fractions not yet trustworthy", seed_anchor_n)

    summary = build_summary(arms, seed_anchor, seed_anchor_n, reference_arm=args.reference_arm)
    (out_dir / "ladder_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "ladder_summary.md").write_text(render_markdown(summary, args.title))

    log.info("Ladder: %d arms | %s reroll = %s%%", len(arms), args.reference_arm, summary["reference_reroll_pct"])
    for r in summary["ladder"]:
        pct = f"{r['reroll_fraction_pct']:.1f}%" if r["reroll_fraction_pct"] is not None else "—"
        log.info("  %-28s %.4f  %s", r["arm"], r["lpips_mean"], pct)


if __name__ == "__main__":
    main()
