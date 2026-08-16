"""Probe: DNSMOS (Microsoft P.808/P.835 no-reference MOS) per H3 speech clip.

The quality axis the distance/direction instruments cannot see: each clip is
scored INDEPENDENTLY (no reference) on the human-calibrated 1-5 MOS scale:
SIG (speech distortion), BAK (background noise), OVRL (overall), P808.
If the heretic arm scored systematically lower than base -> real quality
damage invisible to the distance ladder.

Uses the vendored official implementation (scripts/vendor_dnsmos.py from
microsoft/DNS-Challenge DNSMOS/dnsmos_local.py) with the ONNX models
mirrored on HF (chaanks/DNSMOS model_v8.onnx + mohammed-bahumaish/dnsmos
sig_bak_ovr.onnx). Audio decoded in-memory (av) — no wav temp files.

Usage (container):
    python3 scripts/probe_h3_dnsmos.py \
        --manifest comparisons/te-h3/results/te-h3/ab_videos/manifest.json \
        --out comparisons/te-h3/results/te-h3/text_encoder/probe_dnsmos.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

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
        arr = frame.to_ndarray()  # type: ignore[union-attr]
        if arr.ndim == 1:
            arr = arr[None, :]
        frames.append(arr.astype(np.float32))
    c.close()
    wav = np.concatenate(frames, axis=1).mean(axis=0)  # mono 32k
    out: np.ndarray = resample_poly(wav, 1, SR // 16000).astype(np.float32)
    return out


def fetch_onnx(repo: str, fname: str, cache: Path) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, fname, cache_dir=str(cache)))


def score_clip(scorer: Any, audio: np.ndarray) -> dict[str, float]:
    """In-memory version of vendor ComputeScore.__call__ (9.01s hops, 1s stride)."""
    fs = 16000
    len_samples = int(9.01 * fs)
    audio = audio.copy()
    while len(audio) < len_samples:
        audio = np.append(audio, audio)
    num_hops = int(np.floor(len(audio) / fs) - 9.01) + 1
    sig_r, bak_r, ovr_r, sig, bak, ovr, p808 = [], [], [], [], [], [], []
    for idx in range(num_hops):
        seg = audio[int(idx * fs) : int((idx + 9.01) * fs)]
        if len(seg) < len_samples:
            continue
        x = seg.astype("float32")[np.newaxis, :]
        p808_x = np.array(scorer.audio_melspec(audio=seg[:-160])).astype("float32")[np.newaxis, :, :]
        p808.append(float(scorer.p808_onnx_sess.run(None, {"input_1": p808_x})[0][0][0]))
        s_r, b_r, o_r = scorer.onnx_sess.run(None, {"input_1": x})[0][0]
        s_p, b_p, o_p = scorer.get_polyfit_val(s_r, b_r, o_r, False)
        sig_r.append(float(s_r))
        bak_r.append(float(b_r))
        ovr_r.append(float(o_r))
        sig.append(float(s_p))
        bak.append(float(b_p))
        ovr.append(float(o_p))
    return {
        "SIG": float(np.mean(sig)),
        "BAK": float(np.mean(bak)),
        "OVRL": float(np.mean(ovr)),
        "P808": float(np.mean(p808)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", default="")
    args = p.parse_args()

    scripts_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(scripts_dir))
    from vendor_dnsmos import ComputeScore

    cache = Path("/root/.cache/huggingface")
    sbo = fetch_onnx("mohammed-bahumaish/dnsmos", "sig_bak_ovr.onnx", cache)
    p808 = fetch_onnx("chaanks/DNSMOS", "model_v8.onnx", cache)
    scorer = ComputeScore(str(sbo), str(p808))
    log.info("DNSMOS sessions ready")

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    clips = [c for c in manifest.get("clips", manifest.get("images", [])) if c.get("targets")]
    comp_root = str(manifest_path.parents[3])
    log.info("%d speech clips", len(clips))

    rows: list[dict[str, Any]] = []
    for i, c in enumerate(clips):
        raw = c["path"]
        path = next((q for q in (raw, raw.replace("/comp", comp_root)) if Path(q).exists()), None)
        if path is None:
            log.warning("clip missing: %s", raw)
            continue
        scores = score_clip(scorer, decode_mono16k(Path(path)))
        rows.append({"variant": str(c["variant"]), "label": str(c["label"]), "seed": int(c["seed"]), **scores})
        if (i + 1) % 25 == 0 or i + 1 == len(clips):
            log.info("scored %d/%d", i + 1, len(clips))

    print("\n--- DNSMOS by arm ---")
    print(f"{'arm':12} {'SIG':>12} {'BAK':>12} {'OVRL':>12} {'P808':>12}")
    stats: dict[str, Any] = {}
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_arm[str(r["variant"])].append(r)
    for arm, rs in sorted(by_arm.items()):
        line = f"{arm:12}"
        stats[arm] = {}
        for field in ("SIG", "BAK", "OVRL", "P808"):
            v = np.array([r[field] for r in rs])
            stats[arm][field] = {"mean": float(v.mean()), "sd": float(v.std()), "n": len(v)}
            line += f" {v.mean():.3f}±{v.std():.3f}"
        print(line)

    print("\n--- paired diffs (variant - base, same line+seed, bootstrap 95% CI) ---")
    for a in ("heretic", "base_nvfp4"):
        pa = {(str(r["label"]), int(r["seed"])): r for r in rows if str(r["variant"]) == a}
        pb = {(str(r["label"]), int(r["seed"])): r for r in rows if str(r["variant"]) == "base"}
        keys = sorted(set(pa) & set(pb))
        if len(keys) < 5:
            continue
        for field in ("SIG", "OVRL", "P808"):
            d = np.array([pa[k][field] - pb[k][field] for k in keys])
            boot = [float(np.mean(d[np.random.randint(0, len(d), len(d))])) for _ in range(2000)]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            flag = "*SHIFT*" if lo > 0 or hi < 0 else "(ci~0)"
            print(f"{a:12} {field:5} {d.mean():+.4f}  [{lo:+.4f},{hi:+.4f}]  n={len(keys)}  {flag}")
            stats.setdefault(f"paired_{a}_minus_base", {})[field] = {
                "mean": float(d.mean()),
                "ci95": [float(lo), float(hi)],
                "n": len(keys),
            }

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        payload = {"model": "DNSMOS P.808+P.835 (standard)", "rows": rows, "stats": stats}
        Path(args.out).write_text(json.dumps(payload, indent=2))
        log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
