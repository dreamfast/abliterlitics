"""Probe: delta-direction consistency — the tilt-vs-reroll discriminator for H3 audio.

Question: does the heretic TE edit leave a SYSTEMATIC direction in the audio-VAE
latent (distribution tilt = damage), or do base->heretic changes point in random
directions like ordinary reroll noise?

Method (all in the MiniMaxH3AudioVAE latent space, pooled per clip to [64]):
1. encode every clip, pool latent over time
2. build delta SETS:
     te_swap   pooled(heretic) - pooled(base), per line+seed     (TE signal)
     seed_*    pooled(arm,s)   - pooled(arm,42), per line        (reroll null)
     floor     pooled(nvfp4)   - pooled(base),   per line+seed   (quant control)
3. for each set: mean pairwise cosine among deltas + PC1 explained variance.
   Random directions -> mean pairwise cos ~ 0, PC1 var ~ 1/dim.
   A shared direction -> mean pairwise cos >> 0, PC1 var large.

Also prints the distance ladder (pooled/frame cos per bucket) as a refresh
with the nvfp4 speech top-up included.

Usage (container):
    python3 scripts/probe_h3_delta_direction.py \
        --manifest comparisons/te-h3/results/te-h3/ab_videos/manifest.json \
        --comfyui /comfyui --vae /home/maxx/extra/comfyui-models/vae/minimax_h3_audio_vae_fp32.safetensors
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any  # noqa: F401 - kept for probe extensions

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("probe")

SR = 32000
BASE_SEED = 42


def decode_audio(path: Path) -> np.ndarray:
    import av

    c = av.open(str(path))
    stream = next(s for s in c.streams if s.type == "audio")
    frames: list[np.ndarray] = []
    for frame in c.decode(stream):
        arr = frame.to_ndarray()
        if arr.ndim == 1:
            arr = arr[None, :]
        frames.append(arr.astype(np.float32))
    c.close()
    wav = np.concatenate(frames, axis=1)
    if wav.shape[0] == 1:
        wav = np.repeat(wav, 2, axis=0)
    return wav[:2]


def mean_pairwise_cos(x: torch.Tensor) -> tuple[float, float]:
    """Mean/std of pairwise cosine over rows of x (n>=2)."""
    n = x.shape[0]
    if n < 2:
        return float("nan"), float("nan")
    xn = torch.nn.functional.normalize(x, dim=1)
    sim = xn @ xn.T
    iu = torch.triu_indices(n, n, offset=1)
    vals = sim[iu[0], iu[1]]
    return float(vals.mean()), float(vals.std())


def pc1_variance(x: torch.Tensor) -> float:
    """Explained variance ratio of the first principal component."""
    if x.shape[0] < 2:
        return float("nan")
    xc = x - x.mean(dim=0, keepdim=True)
    _, s, _ = torch.linalg.svd(xc, full_matrices=False)
    return float((s[0] ** 2) / (s**2).sum())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--manifest", required=True)
    p.add_argument("--comfyui", default="/comfyui")
    p.add_argument("--vae", default="/home/maxx/extra/comfyui-models/vae/minimax_h3_audio_vae_fp32.safetensors")
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
    from safetensors import safe_open

    with safe_open(args.vae, framework="pt", device="cpu") as f:
        sd = {k: f.get_tensor(k) for k in list(f.keys())}
    vae.load_state_dict(sd, strict=True)
    vae.eval()
    log.info("audio VAE loaded (%d tensors)", len(sd))

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    clips = manifest.get("clips", manifest.get("images", []))
    comp_root = str(manifest_path.parents[3])

    speech_labels = {str(c["label"]) for c in clips if c.get("targets")}
    latents: dict[tuple[str, str, int], torch.Tensor] = {}
    pooled: dict[tuple[str, str, int], torch.Tensor] = {}
    for i, c in enumerate(clips):
        key = (str(c["variant"]), str(c["label"]), int(c["seed"]))
        raw = c["path"]
        path = next((q for q in (raw, raw.replace("/comp", comp_root)) if Path(q).exists()), None)
        if path is None:
            raise FileNotFoundError(f"clip not found: {raw}")
        with torch.no_grad():
            z = vae.encode(torch.from_numpy(decode_audio(Path(path))).unsqueeze(0))[0].float()  # [32,2,T]
        latents[key] = z
        pooled[key] = z.mean(dim=-1).flatten()  # [64]
        if (i + 1) % 40 == 0 or i + 1 == len(clips):
            log.info("encoded %d/%d", i + 1, len(clips))

    by_vl: dict[tuple[str, str], dict[int, torch.Tensor]] = defaultdict(dict)
    for (v, l, s), z in latents.items():
        by_vl[(v, l)][s] = z
    by_vl_p: dict[tuple[str, str], dict[int, torch.Tensor]] = defaultdict(dict)
    for (v, l, s), z in pooled.items():
        by_vl_p[(v, l)][s] = z

    # ---- ladder refresh -----------------------------------------------------
    def frame_cos(a: torch.Tensor, b: torch.Tensor) -> float:
        t = min(a.shape[-1], b.shape[-1])
        fa, fb = a[..., :t].flatten(1).T, b[..., :t].flatten(1).T
        return float(torch.cosine_similarity(fa, fb, dim=1).mean())

    def pooled_cos(a: torch.Tensor, b: torch.Tensor) -> float:
        pa, pb = a.mean(dim=-1).flatten(), b.mean(dim=-1).flatten()  # [32,2,T] -> [64]
        return float(torch.cosine_similarity(pa, pb, dim=0))

    buckets: dict[str, list[float]] = defaultdict(list)

    def add(kind: str, label: str, a: torch.Tensor, b: torch.Tensor) -> None:
        scope = "speech" if label in speech_labels else "ambient"
        buckets[f"{kind}:{scope}:frame"].append(frame_cos(a, b))
        buckets[f"{kind}:{scope}:pooled"].append(pooled_cos(a, b))

    for (v, l), seeds in sorted(by_vl.items()):
        for s, z in sorted(seeds.items()):
            if v == "base" and s in by_vl.get(("heretic", l), {}):
                add("te_swap", l, z, by_vl[("heretic", l)][s])
            if v == "base" and s in by_vl.get(("base_nvfp4", l), {}):
                add("floor", l, z, by_vl[("base_nvfp4", l)][s])
            if s != BASE_SEED and BASE_SEED in seeds:
                add(f"seed_{v}", l, by_vl[(v, l)][BASE_SEED], z)

    print(f"\n--- distance ladder (n=197 clips incl. nvfp4 speech top-up) ---")
    print(f"{'bucket':32} {'n':>3}  {'mean±std'}")
    for name in sorted(buckets):
        vals = np.array(buckets[name])
        print(f"{name:32} {len(vals):>3}  {vals.mean():.4f}±{vals.std():.4f}")

    # ---- delta direction ----------------------------------------------------
    def te_deltas(scope: str) -> torch.Tensor:
        labels = speech_labels if scope == "speech" else {l for (_, l) in by_vl_p} - speech_labels
        rows = []
        for (v, l), seeds in sorted(by_vl_p.items()):
            if v != "base" or l not in labels:
                continue
            for s, zb in seeds.items():
                zh = by_vl_p.get(("heretic", l), {}).get(s)
                if zh is not None:
                    rows.append(zh - zb)
        return torch.stack(rows) if rows else torch.empty(0, 64)

    def seed_deltas(arm: str, scope: str) -> torch.Tensor:
        labels = speech_labels if scope == "speech" else {l for (_, l) in by_vl_p} - speech_labels
        rows = []
        for (v, l), seeds in sorted(by_vl_p.items()):
            if v != arm or l not in labels or BASE_SEED not in seeds:
                continue
            for s, z in sorted(seeds.items()):
                if s != BASE_SEED:
                    rows.append(z - seeds[BASE_SEED])
        return torch.stack(rows) if rows else torch.empty(0, 64)

    def floor_deltas(scope: str) -> torch.Tensor:
        labels = speech_labels if scope == "speech" else {l for (_, l) in by_vl_p} - speech_labels
        rows = []
        for (v, l), seeds in sorted(by_vl_p.items()):
            if v != "base" or l not in labels:
                continue
            for s, zb in seeds.items():
                zn = by_vl_p.get(("base_nvfp4", l), {}).get(s)
                if zn is not None:
                    rows.append(zn - zb)
        return torch.stack(rows) if rows else torch.empty(0, 64)

    print("\n--- delta-direction consistency (pooled [64] latents) ---")
    print(f"{'delta set':30} {'n':>3}  {'pairwise cos':>14}  {'PC1 var':>8}   random-null reference: cos~0, PC1~{1/64:.3f}")
    for scope in ("speech", "ambient"):
        for name, mat in (
            ("te_swap (heretic-base)", te_deltas(scope)),
            ("seed_base (s-42)", seed_deltas("base", scope)),
            ("seed_heretic (s-42)", seed_deltas("heretic", scope)),
            ("floor (nvfp4-base)", floor_deltas(scope)),
        ):
            if mat.numel() == 0:
                print(f"{scope + ' ' + name:30}    0  (no data)")
                continue
            m, sd = mean_pairwise_cos(mat)
            pc1 = pc1_variance(mat)
            print(f"{scope + ' ' + name:30} {mat.shape[0]:>3}  {m:>8.4f}±{sd:<5.4f}  {pc1:>8.3f}")


if __name__ == "__main__":
    main()
