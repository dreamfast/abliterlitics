"""Consolidate the te-h3 audio-damage investigation into one reportable JSON.

Reads the probe outputs under results/te-h3/text_encoder/ :
  probe_audio_ladder_deltas.txt   (turbo ladder + delta-direction + word summary)
  probe_stock_ladder.txt          (stock-config ladder + delta-direction)
  probe_dnsmos.json               (DNSMOS paired quality stats)
  probe_mos_accent.json           (accent distribution)
  probe_word_utterly.json         (word-level forensics pairs)

Writes tebench_h3_audio_damage.json : metadata + per-instrument results +
overall verdict + a flat ``rows`` list consumed by import_te_battery_results.

Usage:
    python3 scripts/consolidate_h3_audio_damage.py \
        --results-dir comparisons/te-h3/results/te-h3/text_encoder
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

RESULTS_VERSION = 1

_LADDER_RE = re.compile(r"^(?P<bucket>[\w:]+)\s+(?P<n>\d+)\s+(?P<mean>[-\d.]+)±(?P<std>[\d.]+)$")
_DELTA_RE = re.compile(
    r"^(?P<scope>\w+)\s+(?P<set>[\w ()\-]+?)\s+(?P<n>\d+)\s+(?P<cos>[-\d.]+)±(?P<cos_sd>[\d.]+)\s+(?P<pc1>[\d.]+)$"
)


def parse_ladder_block(text: str) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Parse ladder + delta-direction sections of a probe output.

    Returns (ladder_rows, delta_rows) with parsed n/mean/std.
    """
    ladder: dict[str, dict[str, Any]] = {}
    deltas: dict[str, dict[str, Any]] = {}
    in_delta = False
    for line in text.splitlines():
        line = line.rstrip()
        if line.startswith("--- delta-direction"):
            in_delta = True
            continue
        if not in_delta:
            m = _LADDER_RE.match(line.strip())
            if m:
                ladder[m["bucket"]] = {
                    "n": int(m["n"]),
                    "mean": float(m["mean"]),
                    "std": float(m["std"]),
                }
        else:
            m = _DELTA_RE.match(line.strip())
            if m:
                key = f"{m['scope']}:{m['set'].strip()}"
                deltas[key] = {
                    "n": int(m["n"]),
                    "mean_pairwise_cos": float(m["cos"]),
                    "cos_sd": float(m["cos_sd"]),
                    "pc1_var": float(m["pc1"]),
                }
    return ladder, deltas


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results-dir", required=True, type=Path)
    args = p.parse_args()
    rd = args.results_dir

    turbo_txt = (rd / "probe_audio_ladder_deltas.txt").read_text()
    stock_txt = (rd / "probe_stock_ladder.txt").read_text()
    dnsmos = json.loads((rd / "probe_dnsmos.json").read_text())
    accent = json.loads((rd / "probe_mos_accent.json").read_text())
    word = json.loads((rd / "probe_word_utterly.json").read_text())

    ladder = {"turbo": {}, "stock": {}}
    deltas = {"turbo": {}, "stock": {}}
    ladder["turbo"], deltas["turbo"] = parse_ladder_block(turbo_txt)
    ladder["stock"], deltas["stock"] = parse_ladder_block(stock_txt)

    # word forensics summary
    te_pairs = [x["mel_cos"] for x in word["pairs"] if x["pair"].startswith("te_swap")]
    sd_pairs = [x["mel_cos"] for x in word["pairs"] if x["pair"].startswith("seed_")]
    word_summary = {
        "word": word["word"],
        "label": word["label"],
        "te_swap_mel_cos_mean": sum(te_pairs) / len(te_pairs),
        "seed_noise_mel_cos_mean": sum(sd_pairs) / len(sd_pairs),
        "n_te_swap": len(te_pairs),
        "n_seed": len(sd_pairs),
    }

    # dnsmos paired stats
    dnsmos_pairs = dnsmos.get("stats", {}).get("paired_heretic_minus_base", {})
    nvfp4_pairs = dnsmos.get("stats", {}).get("paired_base_nvfp4_minus_base", {})

    # accent native pct per arm
    accent_stats = accent.get("accent_stats", {})
    native_pct = {}
    for arm, counts in accent_stats.items():
        total = sum(counts.values()) or 1
        native_pct[arm] = 100.0 * counts.get("Native", 0) / total

    # ---- DB rows ------------------------------------------------------------
    rows: list[dict[str, Any]] = []

    def add(
        config: str,
        instrument: str,
        metric: str,
        value: float,
        n: int,
        ci: tuple[float, float] | None = None,
    ) -> None:
        row: dict[str, Any] = {
            "battery": "h3_audio_damage",
            "config": config,
            "instrument": instrument,
            "metric_name": metric,
            "metric_value": round(float(value), 6),
            "n": int(n),
        }
        if ci is not None:
            row["ci_low"], row["ci_high"] = round(ci[0], 6), round(ci[1], 6)
        rows.append(row)

    for cfg in ("turbo", "stock"):
        lad = ladder[cfg]
        buckets = (
            "te_swap:speech:frame",
            "floor:speech:frame",
            "seed_base:speech:frame",
            "te_swap:ambient:frame",
            "floor:ambient:frame",
        )
        for bucket in buckets:
            if bucket in lad:
                metric = bucket.replace(":", "_")
                add(cfg, "audio_vae_ladder", metric, lad[bucket]["mean"], lad[bucket]["n"])
        dl = deltas[cfg]
        for key, metric in (
            ("speech:te_swap (heretic-base)", "te_swap_mean_pairwise_cos"),
            ("speech:seed_base (s-42)", "seed_base_mean_pairwise_cos"),
            ("speech:seed_heretic (s-42)", "seed_heretic_mean_pairwise_cos"),
            ("speech:floor (nvfp4-base)", "floor_mean_pairwise_cos"),
        ):
            if key in dl:
                add(cfg, "delta_direction", metric, dl[key]["mean_pairwise_cos"], dl[key]["n"])
        te_delta = dl.get("speech:te_swap (heretic-base)")
        if te_delta:
            add(cfg, "delta_direction", "te_swap_pc1_var", te_delta["pc1_var"], te_delta["n"])

    add(
        "turbo",
        "word_forensics",
        "te_swap_mel_cos_mean",
        word_summary["te_swap_mel_cos_mean"],
        word_summary["n_te_swap"],
    )
    add(
        "turbo",
        "word_forensics",
        "seed_noise_mel_cos_mean",
        word_summary["seed_noise_mel_cos_mean"],
        word_summary["n_seed"],
    )
    for field in ("SIG", "OVRL", "P808"):
        if field in dnsmos_pairs:
            d = dnsmos_pairs[field]
            add("turbo", "dnsmos", f"paired_heretic_minus_base_{field.lower()}", d["mean"], d["n"], tuple(d["ci95"]))
        if field in nvfp4_pairs:
            d = nvfp4_pairs[field]
            add("turbo", "dnsmos", f"paired_nvfp4_minus_base_{field.lower()}", d["mean"], d["n"], tuple(d["ci95"]))
    for arm in sorted(native_pct):
        n_arm = sum(accent_stats[arm].values())
        add("turbo", "accent", f"native_pct_{arm}", native_pct[arm], n_arm)

    # ---- verdict ------------------------------------------------------------
    null_conditions: dict[str, bool] = {}
    for cfg in ("turbo", "stock"):
        dl = deltas[cfg]
        te_cos = dl.get("speech:te_swap (heretic-base)", {}).get("mean_pairwise_cos")
        seed_cos = max(
            dl.get("speech:seed_base (s-42)", {}).get("mean_pairwise_cos", 0.0),
            dl.get("speech:seed_heretic (s-42)", {}).get("mean_pairwise_cos", 0.0),
        )
        if te_cos is not None:
            null_conditions[f"no_tilt_{cfg}"] = te_cos < seed_cos
    null_conditions["dnsmos_ci_zero"] = all(d["ci95"][0] <= 0.0 <= d["ci95"][1] for d in dnsmos_pairs.values())
    null_conditions["word_inside_seed_noise"] = (
        word_summary["te_swap_mel_cos_mean"] >= word_summary["seed_noise_mel_cos_mean"] - 0.02
    )
    lad_t = ladder["turbo"]
    if "te_swap:speech:frame" in lad_t and "seed_base:speech:frame" in lad_t and "floor:speech:frame" in lad_t:
        null_conditions["swap_near_quant_control"] = abs(
            lad_t["te_swap:speech:frame"]["mean"] - lad_t["floor:speech:frame"]["mean"]
        ) < abs(lad_t["te_swap:speech:frame"]["mean"] - lad_t["seed_base:speech:frame"]["mean"])

    expected_conditions = 5
    complete = len(null_conditions) >= expected_conditions
    verdict = (
        "REFUTED (noise injection, not damage)"
        if complete and all(null_conditions.values())
        else "PARTIAL (some instruments flag; inspect conditions)"
        if complete
        else "INCOMPLETE (absent conditions — probe format drift?)"
    )

    out = {
        "metadata": {
            "tool": "h3-audio-damage-investigation",
            "results_version": RESULTS_VERSION,
            "comparison": "te-h3",
            "variant": "heretic",
            "architecture": "qwen3vl_32b_h3_te",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trigger": "p-e-w voice-quality report ('utterly' out of place) after WER-clean battery",
            "provenance": {
                "probes": [
                    "scripts/probe_h3_audio_vae_distance.py",
                    "scripts/probe_h3_delta_direction.py",
                    "scripts/probe_h3_word_forensics.py",
                    "scripts/probe_h3_dnsmos.py",
                    "scripts/probe_h3_mos_accent.py",
                ],
                "journals": [147, 148, 149],
                "agents_lesson": 43,
            },
        },
        "smoke": False,
        "configs": {
            "turbo": "community turbo t2v (6-step + LoRA v4) — ab_videos/ (197 clips)",
            "stock": "official res_multistep/20-step simple, no LoRA — ab_videos_stock/ (144 clips, smoke=false)",
        },
        "instruments": {
            "audio_vae_ladder": ladder,
            "delta_direction": deltas,
            "word_forensics_utterly": word_summary,
            "dnsmos_paired": dnsmos_pairs,
            "dnsmos_paired_nvfp4": nvfp4_pairs,
            "accent_native_pct": native_pct,
            "human_blind_listening": {
                "verdict": "no difference ('both sound fine, just some variation')",
                "rater": "publisher (DreamFast)",
            },
        },
        "null_conditions": null_conditions,
        "verdict": verdict,
        "rows": rows,
    }

    out_path = rd / "tebench_h3_audio_damage.json"
    out_path.write_text(json.dumps(out, indent=2))
    log.info("wrote %s (%d rows, verdict: %s)", out_path, len(rows), verdict)
    for k, v in null_conditions.items():
        log.info("  condition %-28s %s", k, "PASS(null)" if v else "FLAG")


if __name__ == "__main__":
    main()
