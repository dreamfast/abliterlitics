"""H3 video-space metrics — frame-sampled LPIPS + CLIP adherence.

Pure helpers here (frame sampling, pairing); model meters are reused from
image_metrics (LPIPSMeter / CLIPMeter). The runner (run_h3_video.py) wires
them over the ab_videos manifest:

- te_swap pairs: same (label, seed), base vs variant -> mean frame-LPIPS
  ("how much does the video move when only the TE changes")
- seed_change anchor: base seed A vs base seed B, same label -> mean
  frame-LPIPS (the reroll scale — a different noise draw)
- CLIP adherence: per-frame clip_score vs the prompt text (the H3 3-field
  prompt is long; we score against the integrated description's first
  sentence as the adherence probe, disclosed)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import torch


def sample_frame_indices(n_frames: int, k: int) -> list[int]:
    """Deterministic uniform frame indices (k frames across n, always sorted).

    Includes the first and last frame; k is clamped to n.
    """
    if n_frames <= 0:
        raise ValueError(f"n_frames must be > 0, got {n_frames}")
    k = max(1, min(k, n_frames))
    if k == 1:
        return [n_frames // 2]
    step = (n_frames - 1) / (k - 1)
    return sorted({min(n_frames - 1, round(i * step)) for i in range(k)})


def extract_frames(path: Path, k: int) -> list[np.ndarray]:
    """Decode the video stream and return k uniformly-sampled RGB frames
    as uint8 [H, W, 3] arrays (via PyAV — no cv2/ffmpeg in the image)."""
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        n = stream.frames or 0
        if n <= 0:
            # some containers lack a frame count — count via full decode
            frames_all = [f.to_ndarray(format="rgb24") for f in container.decode(stream)]
            n = len(frames_all)
            idx = set(sample_frame_indices(n, k))
            return [frames_all[i] for i in sorted(idx)]
        idx = set(sample_frame_indices(n, k))
        out: list[np.ndarray] = []
        for i, frame in enumerate(container.decode(stream)):
            if i in idx:
                out.append(frame.to_ndarray(format="rgb24"))
            if i > max(idx):
                break
    if len(out) != len(idx):
        # frame count lied; fall back to what we got, uniformly
        if not out:
            raise ValueError(f"No frames decoded from {path}")
        want = sample_frame_indices(len(out), min(k, len(out)))
        out = [out[i] for i in want]
    return out


def frame_to_tensor(frame: np.ndarray) -> torch.Tensor:
    """uint8 [H,W,3] -> float [1,3,H,W] in [-1,1] (LPIPS convention)."""
    t = torch.from_numpy(frame.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return t * 2.0 - 1.0


def adherence_probe_text(prompt: str) -> str:
    """The first sentence of the integrated description as the CLIP probe.

    The full H3 3-field prompt (camera grammar, soundscape, music) is far
    beyond CLIP's 77-token window; the first sentence carries the subject.
    """
    m = re.search(r"integrated_multimodal_description:\s*(.+?)(?:\n|$)", prompt, re.S)
    body = m.group(1).strip() if m else prompt
    first = re.split(r"(?<=[.!?])\s+", body)[0]
    return first.strip()


def pair_clips(manifest_clips: list[dict[str, Any]]) -> dict[str, Any]:
    """Group manifest clips into (label, seed) buckets per arm.

    Returns {"buckets": {(label, seed): {arm: clip}}, "arms": [...]}.
    """
    buckets: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for c in manifest_clips:
        key = (c["label"], c["seed"])
        buckets.setdefault(key, {})[c["variant"]] = c
    arms = sorted({c["variant"] for c in manifest_clips})
    return {"buckets": buckets, "arms": arms}


def score_pair_lpips(
    frames_a: list[np.ndarray],
    frames_b: list[np.ndarray],
    lpips_meter: Any,
) -> float:
    """Mean frame-LPIPS between two frame lists (index-matched)."""
    if len(frames_a) != len(frames_b):
        raise ValueError(f"frame count mismatch: {len(frames_a)} vs {len(frames_b)}")
    vals = [lpips_meter.score(frame_to_tensor(a), frame_to_tensor(b)) for a, b in zip(frames_a, frames_b, strict=True)]
    return float(np.mean(vals))
