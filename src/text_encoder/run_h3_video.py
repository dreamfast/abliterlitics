"""H3 video battery runner — frame-LPIPS + CLIP adherence over ab_videos.

H17/H10 (video reroll-not-damage): when only the TE changes at matched
seed, do frames move within the seed-change anchor, and does CLIP adherence
stay within +-1.5 of base? Uses the shared LPIPS/CLIP meters; frames via
PyAV. Output: tebench_h3_video.json (+ smoke passthrough).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.text_encoder import TE_RESULTS_VERSION
from src.text_encoder.h3_video import (
    adherence_probe_text,
    extract_frames,
    pair_clips,
    score_pair_lpips,
)
from src.text_encoder.image_metrics import CLIPMeter, LPIPSMeter

log = logging.getLogger(__name__)

FRAME_K = 8
CLIP_BAND = 1.5


def clip_score_frames(frames: list[np.ndarray], prompt: str, meter: CLIPMeter) -> float:
    """Mean CLIP score of frames vs the adherence probe text."""
    from PIL import Image

    scores = [meter.clip_score(meter.preprocess_pil(Image.fromarray(f)), prompt) for f in frames]
    return float(np.mean(scores))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="H3 video battery: frame-LPIPS + CLIP adherence")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--suite", required=True, help="video prompt suite (label filter)")
    p.add_argument("--frame-k", type=int, default=FRAME_K)
    p.add_argument("--device", default="cuda")
    p.add_argument("--arms", default="", help="Comma-separated arm filter (default all)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    import yaml

    manifest = json.loads(Path(args.manifest).read_text())
    suite = yaml.safe_load(Path(args.suite).read_text())
    video_labels = {e.get("label") or "" for e in suite.get("prompts", [])}

    clips = [c for c in manifest.get("clips", []) if c.get("label") in video_labels]
    if args.arms:
        keep = {a.strip() for a in args.arms.split(",")}
        clips = [c for c in clips if c["variant"] in keep]
    if not clips:
        log.error("No video-suite clips in manifest")
        return 1
    grouped = pair_clips(clips)
    arms = [a for a in grouped["arms"] if a != "base"] if "base" in grouped["arms"] else grouped["arms"]
    log.info("Video battery: %d buckets, arms %s", len(grouped["buckets"]), grouped["arms"])

    if args.dry_run:
        for (label, seed), bucket in sorted(grouped["buckets"].items()):
            log.info("[dry-run] %s seed%d: arms %s", label, seed, sorted(bucket))
        return 0

    lpips_meter = LPIPSMeter(device=args.device)
    clip_meter = CLIPMeter(device=args.device)

    # --- seed-change anchor: base seed pairs (needs >= 2 base seeds) ---
    base_buckets = {(label, seed): bucket for (label, seed), bucket in grouped["buckets"].items() if "base" in bucket}
    by_label_base: dict[str, dict[int, Any]] = {}
    for (label, seed), bucket in base_buckets.items():
        by_label_base.setdefault(label, {})[seed] = bucket["base"]

    seed_anchor_vals: list[float] = []
    base_seeds = sorted({s for label_map in by_label_base.values() for s in label_map})
    for label, seed_map in by_label_base.items():
        ss = sorted(seed_map)
        for i in range(len(ss) - 1):
            fa = extract_frames(Path(seed_map[ss[i]]["path"]), args.frame_k)
            fb = extract_frames(Path(seed_map[ss[i + 1]]["path"]), args.frame_k)
            v = score_pair_lpips(fa, fb, lpips_meter)
            seed_anchor_vals.append(v)
            log.info("anchor seed-change %s s%d~s%d: lpips=%.4f", label, ss[i], ss[i + 1], v)
    seed_anchor = round(float(np.mean(seed_anchor_vals)), 4) if seed_anchor_vals else None

    # --- per-variant te_swap: (label, seed) matched vs base ---
    variant_out: dict[str, Any] = {}
    for variant in arms:
        vals: list[float] = []
        clip_deltas: list[float] = []
        clip_self: list[float] = []
        n_pairs = 0
        for key, bucket in sorted(grouped["buckets"].items()):
            if variant not in bucket or "base" not in bucket:
                continue
            fa = extract_frames(Path(bucket["base"]["path"]), args.frame_k)
            fv = extract_frames(Path(bucket[variant]["path"]), args.frame_k)
            v = score_pair_lpips(fa, fv, lpips_meter)
            vals.append(v)
            prompt = bucket[variant].get("prompt") or bucket["base"].get("prompt", "")
            probe = adherence_probe_text(prompt)
            cb = clip_score_frames(fa, probe, clip_meter)
            cv = clip_score_frames(fv, probe, clip_meter)
            clip_deltas.append(cv - cb)
            clip_self.append(cv)
            n_pairs += 1
            log.info("%-12s %s seed%d: lpips=%.4f clip=%.1f (base %.1f)", variant, key[0], key[1], v, cv, cb)
        if vals:
            variant_out[variant] = {
                "n_pairs": n_pairs,
                "mean_frame_lpips": round(float(np.mean(vals)), 4),
                "lpips_values": [round(v, 4) for v in vals],
                "mean_clip_self": round(float(np.mean(clip_self)), 2),
                "mean_clip_delta_vs_base": round(float(np.mean(clip_deltas)), 2),
                "clip_delta_absmax": round(float(np.max(np.abs(clip_deltas))), 2) if clip_deltas else None,
                "within_seed_anchor": (bool(np.mean(vals) <= seed_anchor) if seed_anchor is not None else None),
                "clip_within_band": (bool(np.max(np.abs(clip_deltas)) <= CLIP_BAND) if clip_deltas else None),
            }

    result = {
        "te_results_version": TE_RESULTS_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "frame_k": args.frame_k,
        "smoke": bool(manifest.get("smoke", False)),
        "seed_change_anchor_lpips": seed_anchor,
        "base_seeds": base_seeds,
        "variants": variant_out,
        "clip_band": CLIP_BAND,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    log.info("Wrote %s (anchor=%s)", out_path, seed_anchor)
    return 0


if __name__ == "__main__":
    sys.exit(main())
