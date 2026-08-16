"""Probe: absolute speech-QUALITY (MOS) + accent distributions per arm — H3 speech clips.

The distance/direction instruments measure *change*; this measures *quality*
and *accent* absolutely, per clip, with no reference:

1. MOS (Mean Opinion Score 1-5) via the UTMOS22 w2v2 regressor
   (sarulab-speech/MSR-UTMOS_w2v2_fold0). If the heretic arm's clips score
   systematically lower than base, that is a quality shift invisible to the
   distance ladder. Reports per-arm mean±sd + bootstrap CI on the paired
   (same line+seed) heretic-base difference.
2. Accent distribution via dima806/speech-accent-classification (Common-Voice
   accent classes). Tests the accent-drift explanation for "sounds worse /
   more north american": does the heretic arm's accent distribution shift?

Usage (container, network for first download):
    python3 scripts/probe_h3_mos_accent.py \
        --manifest comparisons/te-h3/results/te-h3/ab_videos/manifest.json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("probe")

SR = 32000
BASE_SEED = 42


def decode_mono16k(path: Path) -> np.ndarray:
    import av
    from scipy.signal import resample_poly

    c = av.open(str(path))
    stream = next(s for s in c.streams if s.type == "audio")
    frames = []
    for frame in c.decode(stream):
        arr = frame.to_ndarray()
        if arr.ndim == 1:
            arr = arr[None, :]
        frames.append(arr.astype(np.float32))
    c.close()
    wav = np.concatenate(frames, axis=1).mean(axis=0)  # mono 32k
    return resample_poly(wav, 1, SR // 16000).astype(np.float32)


def load_mos():
    """UTMOS22-strong needs torchaudio (absent in the te image; NGC-torch
    binary incompatible with stock wheels) — superseded by
    scripts/probe_h3_dnsmos.py (DNSMOS). Kept as a graceful no-op so this
    probe degrades to accent-only instead of crashing."""
    log.info("MOS via UTMOS disabled — use scripts/probe_h3_dnsmos.py (DNSMOS)")
    return None, None, ""


def load_accent():
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    repo = "dima806/speech-accent-classification"
    model = AutoModelForAudioClassification.from_pretrained(repo)
    proc = AutoFeatureExtractor.from_pretrained(repo)
    model.eval()
    labels = {int(k): v for k, v in model.config.id2label.items()}
    log.info("accent model loaded: %s (%d classes)", repo, len(labels))
    return model, proc, labels


@torch.no_grad()
def accent_logits(model, proc, wav16k: np.ndarray) -> np.ndarray:
    inputs = proc(wav16k, sampling_rate=16000, return_tensors="pt")
    out = model(**{k: v for k, v in inputs.items() if k in ("input_values", "attention_mask")})
    return torch.softmax(out.logits, dim=-1).flatten()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", default="")
    p.add_argument("--skip-accent", action="store_true")
    args = p.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    clips = [c for c in manifest.get("clips", manifest.get("images", [])) if c.get("targets")]
    comp_root = str(manifest_path.parents[3])
    log.info("%d speech clips", len(clips))

    mos_model, mos_proc, mos_repo = load_mos()
    acc_model = acc_proc = acc_labels = None
    if not args.skip_accent:
        try:
            acc_model, acc_proc, acc_labels = load_accent()
        except Exception as exc:  # noqa: BLE001
            log.warning("accent model unavailable: %s", exc)

    rows = []
    for i, c in enumerate(clips):
        raw = c["path"]
        path = next((q for q in (raw, raw.replace("/comp", comp_root)) if Path(q).exists()), None)
        if path is None:
            log.warning("clip missing: %s", raw)
            continue
        wav = decode_mono16k(Path(path))
        row = {
            "variant": str(c["variant"]),
            "label": str(c["label"]),
            "seed": int(c["seed"]),
            "path": str(path),
        }
        if acc_model is not None:
            probs = accent_logits(acc_model, acc_proc, wav)
            top = int(probs.argmax())
            row["accent"] = acc_labels.get(top, str(top))
            row["accent_p"] = float(probs[top])
        rows.append(row)
        if (i + 1) % 25 == 0 or i + 1 == len(clips):
            log.info("scored %d/%d", i + 1, len(clips))

    result: dict = {"mos_repo": mos_repo, "n": len(rows), "rows": rows}

    if mos_model is not None:
        print(f"\n--- MOS by arm ({mos_repo}) ---")
        by_arm = defaultdict(list)
        for r in rows:
            by_arm[r["variant"]].append(r["mos"])
        stats = {}
        for arm, vals in sorted(by_arm.items()):
            v = np.array(vals)
            stats[arm] = {"mean": float(v.mean()), "sd": float(v.std()), "n": len(v)}
            print(f"{arm:12} {v.mean():.3f} ± {v.std():.3f}  (n={len(v)})")
        # paired diff: heretic - base, same line+seed
        for a, b in (("heretic", "base"), ("base_nvfp4", "base")):
            pa = {(r["label"], r["seed"]): r["mos"] for r in rows if r["variant"] == a}
            pb = {(r["label"], r["seed"]): r["mos"] for r in rows if r["variant"] == b}
            keys = sorted(set(pa) & set(pb))
            if len(keys) >= 5:
                diffs = np.array([pa[k] - pb[k] for k in keys])
                boot = [float(np.mean(diffs[np.random.randint(0, len(diffs), len(diffs))])) for _ in range(2000)]
                lo, hi = np.percentile(boot, [2.5, 97.5])
                print(
                    f"paired {a} - {b}: mean {diffs.mean():+.3f} MOS "
                    f"(95% CI {lo:+.3f}..{hi:+.3f}, n={len(keys)}) "
                    f"{'*SHIFT*' if lo > 0 or hi < 0 else '(CI straddles 0)'}"
                )
                stats[f"paired_{a}_minus_{b}"] = {
                    "mean": float(diffs.mean()),
                    "ci95": [float(lo), float(hi)],
                    "n": len(keys),
                }
        result["mos_stats"] = stats

    if acc_model is not None:
        print("\n--- accent distribution by arm ---")
        acc_stats: dict = {}
        for arm in sorted({r["variant"] for r in rows}):
            counts = Counter(r["accent"] for r in rows if r["variant"] == arm)
            top = counts.most_common(4)
            acc_stats[arm] = dict(counts)
            print(f"{arm:12} {top}")
        result["accent_stats"] = acc_stats

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
        log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
