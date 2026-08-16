"""Probe: per-word acoustic forensics for a flagged word across seeds/arms.

For one prompt label (default prof_inconceivable) x {base, heretic} x seeds:
1. whisper-large-v3 word timestamps -> locate the flagged word (fuzzy match)
2. slice the waveform segment (+/- 50 ms guard)
3. pure-torch acoustic features per segment:
     - log-mel spectrogram (80 mels, 16 kHz) -> cross-segment cos/mse
     - F0 median + spread (autocorrelation, 60-400 Hz, voiced frames)
     - spectral flatness median (buzz/noise artifact indicator)
4. paired comparisons:
     te_swap   base <-> heretic, same seed     (TE signal)
     seed_*    within-arm, base seed vs others (noise null)

If te_swap distances sit inside the within-arm seed distances, the flagged
word's variance is noise-dominated (draw luck, not a TE lesion). F0/flatness
deltas outside the seed band indicate systematic voice-quality shift.

Usage (container):
    python3 scripts/probe_h3_word_forensics.py \
        --clips-dir comparisons/te-h3/results/te-h3/ab_videos \
        --label prof_inconceivable --word utterly --seeds 42,43,44
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("probe")

SR = 32000


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


def _fuzzy(token: str, word: str) -> bool:
    t = token.strip().strip(".,!?\"'()-").lower()
    w = word.lower()
    return w in t or t in w or (len(t) > 3 and t.startswith(w[: max(3, len(w) - 2)]))


def find_word(pipe: Any, wav: np.ndarray, word: str) -> tuple[float, float] | None:
    mono = wav.mean(axis=0)
    out = pipe(mono, return_timestamps="word")
    for chunk in out.get("chunks", []):
        if _fuzzy(str(chunk.get("text", "")), word):
            ts = chunk["timestamp"]
            return float(ts[0]), float(ts[1])
    return None


# ---- pure-torch features ----------------------------------------------------


def _mel_filterbank(n_mels: int, n_fft: int, sr: int, fmax: float) -> torch.Tensor:
    def hz2mel(f: np.ndarray) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m: np.ndarray) -> np.ndarray:
        return 700.0 * (10 ** (m / 2595.0) - 1.0)

    mels = np.linspace(float(hz2mel(np.array([0.0]))), float(hz2mel(np.array([fmax]))), n_mels + 2)
    freqs = mel2hz(mels)
    bins = np.floor((n_fft + 1) * freqs / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        left, center, right = int(bins[i]), int(bins[i + 1]), int(bins[i + 2])
        for k in range(left, center):
            if center != left:
                fb[i, k] = (k - left) / (center - left)
        for k in range(center, right):
            if right != center:
                fb[i, k] = (right - k) / (right - center)
    return torch.from_numpy(fb).float()


_FB16K = _mel_filterbank(80, 2048, SR, 16000.0)


def logmel(mono: np.ndarray) -> torch.Tensor:
    """Log-mel at native 32 kHz (n_fft 2048 -> 16 kHz Nyquist band, no resample)."""
    x = torch.from_numpy(mono.astype(np.float32))
    spec = torch.stft(x, n_fft=2048, hop_length=512, window=torch.hann_window(2048), return_complex=True, center=True)
    power = spec.abs() ** 2
    mel = _FB16K @ power
    return 10.0 * torch.log10(mel + 1e-8)  # [n_mels, T]


def mel_distance(a_mono: np.ndarray, b_mono: np.ndarray) -> dict[str, float]:
    ma, mb = logmel(a_mono), logmel(b_mono)
    t = min(ma.shape[-1], mb.shape[-1])
    va, vb = ma[:, :t].flatten(), mb[:, :t].flatten()
    return {
        "mel_cos": F.cosine_similarity(va, vb, dim=0).item(),
        "mel_mse": F.mse_loss(va, vb).item(),
    }


def f0_track(mono: np.ndarray, sr: int = SR) -> tuple[float, float]:
    """(median F0 Hz, IQR) via frame-wise autocorrelation over voiced frames."""
    x = torch.from_numpy(mono.astype(np.float32))
    frame, hop = int(0.025 * sr), int(0.010 * sr)
    if x.numel() < frame:
        return float("nan"), float("nan")
    f0s: list[float] = []
    lag_min, lag_max = int(sr / 400), int(sr / 60)
    for start in range(0, x.numel() - frame + 1, hop):
        seg = x[start : start + frame]
        if seg.abs().max() < 1e-3:
            continue
        seg = seg - seg.mean()
        n = seg.numel()
        padded = F.pad(seg.view(1, 1, -1), (0, n - 1))
        ac = F.conv1d(padded, seg.view(1, 1, -1)).flatten()  # full autocorr, len 2n-1
        if ac.numel() <= lag_max:
            continue
        ac = ac / (ac[0].abs().clamp_min(1e-8) + 1e-8)
        window = ac[lag_min : min(lag_max, ac.numel() - 1)]
        if window.numel() == 0:
            continue
        peak = window.argmax()
        if window[peak] > 0.5:  # voiced + clear periodicity
            f0s.append(sr / (lag_min + peak.item()))
    if not f0s:
        return float("nan"), float("nan")
    q = np.percentile(f0s, [25, 50, 75])
    return float(q[1]), float(q[2] - q[0])


def spectral_flatness(mono: np.ndarray, sr: int = SR) -> float:
    x = torch.from_numpy(mono.astype(np.float32))
    spec = torch.stft(x, n_fft=1024, hop_length=256, window=torch.hann_window(1024), return_complex=True, center=True)
    power = spec.abs() + 1e-10
    flat = power.log().mean(dim=0).exp() / power.mean(dim=0)  # [T]
    return float(flat.median())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--clips-dir", required=True)
    p.add_argument("--label", default="prof_inconceivable")
    p.add_argument("--word", default="utterly")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--model", default="openai/whisper-large-v3")
    p.add_argument("--out", default="")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    base = Path(args.clips_dir)

    from transformers import pipeline

    pipe = pipeline("automatic-speech-recognition", model=args.model, chunk_length_s=30, device="cpu")

    segs: dict[tuple[str, int], np.ndarray] = {}
    feats: dict[tuple[str, int], dict[str, float]] = {}
    for arm in ("base", "heretic"):
        for seed in seeds:
            path = base / arm / f"{args.label}__seed{seed}.mp4"
            if not path.exists():
                log.warning("missing clip: %s", path)
                continue
            wav = decode_audio(path)
            span = find_word(pipe, wav, args.word)
            if span is None:
                log.warning("%s: word %r not found", path.name, args.word)
                continue
            s0 = max(0.0, span[0] - 0.05)
            s1 = min(wav.shape[1] / SR, span[1] + 0.05)
            mono = wav[:, int(s0 * SR) : int(s1 * SR)].mean(axis=0)
            segs[(arm, seed)] = mono
            f0, iqr = f0_track(mono)
            feats[(arm, seed)] = {"f0_med": f0, "f0_iqr": iqr, "flatness": spectral_flatness(mono)}
            log.info(
                "%s/%s: %r at %.2f-%.2fs | F0 %.0f Hz (IQR %.0f) | flatness %.4f",
                arm, seed, args.word, span[0], span[1], f0, iqr, feats[(arm, seed)]["flatness"],
            )

    print(f"\nword = {args.word!r}, label = {args.label}")
    print(f"{'pair':36} {'mel_cos':>8}  {'mel_mse':>8}")
    rows: list[dict[str, Any]] = []

    def dist(kind: str, a: tuple[str, int], b: tuple[str, int]) -> None:
        if a in segs and b in segs:
            m = mel_distance(segs[a], segs[b])
            rows.append({"pair": kind, "a": list(a), "b": list(b), **m})
            print(f"{kind:36} {m['mel_cos']:>8.4f}  {m['mel_mse']:>8.2f}")

    for seed in seeds:
        dist(f"te_swap seed{seed} (base vs heretic)", ("base", seed), ("heretic", seed))
    for arm in ("base", "heretic"):
        for seed in seeds[1:]:
            dist(f"seed_{arm} {seeds[0]} vs {seed}", (arm, seeds[0]), (arm, seed))

    print(f"\n{'clip':14} {'F0 med':>7} {'F0 IQR':>7} {'flatness':>9}")
    for (arm, seed), f in sorted(feats.items()):
        print(f"{arm}/{seed:<10} {f['f0_med']:>7.1f} {f['f0_iqr']:>7.1f} {f['flatness']:>9.4f}")

    te = [r for r in rows if r["pair"].startswith("te_swap")]
    sd = [r for r in rows if r["pair"].startswith("seed_")]
    if te and sd:
        te_cos = np.array([r["mel_cos"] for r in te])
        sd_cos = np.array([r["mel_cos"] for r in sd])
        verdict = "INSIDE seed noise -> draw luck" if te_cos.mean() >= sd_cos.mean() - 0.02 else "BELOW seed noise -> TE signal"
        print(
            f"\nte_swap mel_cos mean {te_cos.mean():.4f} | within-arm seed mel_cos mean {sd_cos.mean():.4f} | {verdict}"
        )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"word": args.word, "label": args.label, "pairs": rows, "features": {f"{k[0]}/{k[1]}": v for k, v in feats.items()}}, indent=2))
        log.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
