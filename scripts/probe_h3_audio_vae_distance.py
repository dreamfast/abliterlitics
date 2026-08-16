"""Probe: H3 audio neural-distance ladder via the MiniMax H3 audio VAE encoder.

Re-encodes every generated clip's audio through MiniMaxH3AudioVAE.encode
(DAC-lineage codec, 32 kHz stereo -> [32, 2, T] @ 40 fps — the same latent
space the DiT generates audio in) and scores pair distances:

    te_swap      base <-> heretic, same seed+line      (the claim under test)
    seed_change  same arm, seed 42 <-> 43/44           (normal-variation anchor)
    floor        base <-> base_nvfp4, same seed        (quant noise floor)

Whisper-free by construction: this measures the waveform, not a transcript.
If te_swap sits inside the seed_change envelope -> "different voice, not
damaged voice" (reroll fingerprint). If it exceeds it -> real acoustic
degradation signal worth escalating to per-word formant forensics.

Usage (container):
    python3 scripts/probe_h3_audio_vae_distance.py \
        --manifest comparisons/te-h3/results/te-h3/ab_videos/manifest.json \
        --comfyui /home/maxx/comfyui-docker/comfyui \
        --vae /home/maxx/extra/comfyui-models/vae/minimax_h3_audio_vae_fp32.safetensors
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import av
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("probe")


def decode_audio(path: Path, sample_rate: int = 32000) -> np.ndarray:
    """Decode first audio stream -> float32 [2, L] at sample_rate."""
    c = av.open(str(path))
    stream = next((s for s in c.streams if s.type == "audio"), None)
    if stream is None:
        raise ValueError(f"no audio stream: {path}")
    frames: list[np.ndarray] = []
    for frame in c.decode(stream):
        arr = frame.to_ndarray()  # [ch, n] float planar or [n, ch] packed
        if arr.ndim == 1:
            arr = arr[None, :]
        elif arr.shape[0] > arr.shape[1] if arr.ndim == 2 else False:  # packed
            arr = arr.T
        frames.append(arr.astype(np.float32))
    c.close()
    wav = np.concatenate([f if f.ndim == 2 else f[None] for f in frames], axis=1)
    if wav.shape[0] == 1:
        wav = np.repeat(wav, 2, axis=0)
    if wav.shape[0] > 2:
        wav = wav[:2]
    return wav


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--manifest", required=True)
    p.add_argument("--comfyui", default="/home/maxx/comfyui-docker/comfyui")
    p.add_argument("--vae", default="/home/maxx/extra/comfyui-models/vae/minimax_h3_audio_vae_fp32.safetensors")
    p.add_argument("--limit", type=int, default=0, help="cap clips (debug)")
    args = p.parse_args()

    import sys

    sys.path.insert(0, str(Path(args.comfyui).resolve()))
    import comfy.options

    comfy.options.enable_args_parsing(True)
    old_argv = sys.argv
    sys.argv = ["comfy", "--cpu"]
    try:
        from comfy.ldm.minimax.audio_vae import MiniMaxH3AudioVAE
    finally:
        sys.argv = old_argv

    vae = MiniMaxH3AudioVAE()
    sd = {}
    from safetensors import safe_open

    with safe_open(args.vae, framework="pt", device="cpu") as f:
        for k in list(f.keys()):
            sd[k] = f.get_tensor(k)
    vae.load_state_dict(sd, strict=True)
    vae.eval()
    log.info("audio VAE loaded (%d tensors)", len(sd))

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    clips = manifest.get("clips", manifest.get("images", []))
    if args.limit:
        clips = clips[: args.limit]
    # /comp was the comparison-dir mount at generation time
    # (manifest at <comp>/results/<name>/ab_videos/manifest.json -> parents[3])
    comp_root = str(manifest_path.parents[3])

    # encode every clip; remember speech lines (audio battery: targets set)
    latents: dict[tuple[str, str, int], torch.Tensor] = {}
    speech_labels: set[str] = set()
    for i, c in enumerate(clips):
        key = (str(c["variant"]), str(c["label"]), int(c["seed"]))
        if c.get("targets"):
            speech_labels.add(str(c["label"]))
        raw = c["path"]
        candidates = [raw, raw.replace("/comp", comp_root), raw.replace("/comp", "/app"), raw.replace("/comp", ".")]
        path = next((p for p in candidates if Path(p).exists()), None)
        if path is None:
            raise FileNotFoundError(f"clip not found under any root: {raw}")
        wav = decode_audio(path)  # [2, L]
        with torch.no_grad():
            z = vae.encode(torch.from_numpy(wav).unsqueeze(0))  # [1, 32, 2, T]
        latents[key] = z[0].float()
        if (i + 1) % 20 == 0 or i + 1 == len(clips):
            log.info("encoded %d/%d", i + 1, len(clips))

    # mine pairs
    by_vl: dict[tuple[str, str], dict[int, torch.Tensor]] = defaultdict(dict)
    for (v, l, s), z in latents.items():
        by_vl[(v, l)][s] = z

    def pair_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
        pa, pb = a.flatten(1).mean(-1), b.flatten(1).mean(-1)  # pooled [32*2]
        cos_p = torch.cosine_similarity(pa, pb, dim=0).item()
        t = min(a.shape[-1], b.shape[-1])
        fa, fb = a[..., :t].flatten(1).T, b[..., :t].flatten(1).T  # [T, 32*2]
        cos_f = torch.cosine_similarity(fa, fb, dim=1).mean().item()
        l2_p = (pa - pb).norm().item()
        return {"pooled_cos": cos_p, "frame_cos": cos_f, "pooled_l2": l2_p}

    buckets: dict[str, list[dict[str, float]]] = defaultdict(list)

    def add(kind: str, label: str, a: torch.Tensor, b: torch.Tensor) -> None:
        scope = "speech" if label in speech_labels else "ambient"
        buckets[f"{kind}:{scope}"].append(pair_metrics(a, b))

    for (v, l), seeds in sorted(by_vl.items()):
        for s, z in sorted(seeds.items()):
            # te_swap: base <-> heretic same seed
            if v == "base" and s in by_vl.get(("heretic", l), {}):
                add("te_swap", l, z, by_vl[("heretic", l)][s])
            # floor: base <-> base_nvfp4 same seed
            if v == "base" and s in by_vl.get(("base_nvfp4", l), {}):
                add("floor", l, z, by_vl[("base_nvfp4", l)][s])
            # seed_change: same arm, this seed vs base seed 42
            if s != 42 and 42 in seeds:
                add(f"seed_change_{v}", l, by_vl[(v, l)][42], z)

    print(f"\n{'bucket':22} {'n':>3}  {'pooled_cos':>10}  {'frame_cos':>10}  {'pooled_l2':>10}")
    for name in sorted(buckets):
        ms = buckets[name]
        for field in ("pooled_cos", "frame_cos", "pooled_l2"):
            pass
        pc = np.array([m["pooled_cos"] for m in ms])
        fc = np.array([m["frame_cos"] for m in ms])
        l2 = np.array([m["pooled_l2"] for m in ms])

        def fmt(x: np.ndarray) -> str:
            return f"{x.mean():.4f}±{x.std():.4f}"

        print(f"{name:22} {len(ms):>3}  {fmt(pc):>14}  {fmt(fc):>14}  {fmt(l2):>14}")


if __name__ == "__main__":
    main()
