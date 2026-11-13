"""Attention capture harness — record what the DiT READS from conditioning.

Phase 4 T2.2 (B0 spike GO — journal #123); generalized in phase 9 for
multi-sequence DiTs (krea2 single-stream, H3 packed audio+video).

Uses ComfyUI's native ``transformer_options["optimized_attention_override"]``
hook (checked by ``wrap_attn`` BEFORE any kernel dispatch): no monkey-patching,
no kernel forcing; bypasses flash/sage/xformers for the capture run only.
``wrap_attn`` unwraps ``AttentionTensorContainer`` args itself
(``container.take()``) before calling the override, so the override ALWAYS
receives plain tensors — z-image (Lumina2), krea2 (optimized_attention_masked,
``[B, H, L, D]`` via skip_reshape) and H3 (containers around ``[1, H, L, D]``)
all arrive in the same convention.

Layout contract: TEXT KEYS OCCUPY THE LEADING ``text_len`` positions of the
key axis (true for Lumina2 JointAttention, krea2 ``torch.cat((context, img))``
and H3's PackedLayout ``[text | ... | audio | video]``). Query segments name
the non-text spans whose attention TO those text keys we record, e.g.::

    krea2:  configure(text_len, {"img": (text_len, text_len + img_len)})
    h3:     configure(text_len, {"audio": (a0, a1), "video": (v0, v1)})
    z-image: single "img" segment (previous behavior, preserved)

Per joint attention call the capture records:
- ``segment_readouts``: per configured segment, mean over heads + segment
  query rows of attention TO text-key columns -> [text_len] distribution
  ("what the image/audio/video stream reads from the prompt")
- ``text_readout``: row-count-weighted aggregate over segments (z-image
  compatibility: identical to the old img-only readout)
- ``text_entropy`` + top-10 token indices of the aggregate

Calls whose key length is shorter than the configured full joint sequence
carry no text keys in leading position (z-image noise_refiner blocks, H3's
internal token_refiner, krea2 txtfusion — txtfusion keys are the 12 taps,
not text positions) and are delegated untouched. Step/side identity comes from the caller resetting the capture
before each forward; call order within a forward identifies the layer.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch

log = logging.getLogger(__name__)


class AttentionCapture:
    """Stateful capture bound to one DiT forward (one step, one side).

    Lifecycle per prompt: ``configure(text_len, query_segments)`` once, then
    per forward: ``reset(step, side)`` -> model(...) with
    ``transformer_options=capture.transformer_options`` -> ``take_records()``.
    """

    def __init__(self, device: str = "cuda") -> None:
        self._device = device
        self._text_len = 0
        self._segments: dict[str, tuple[int, int]] = {}
        self._expected_full = 0  # max segment stop; calls below it are delegated
        self._step: int = -1
        self._side: str = "?"
        self.records: list[dict[str, Any]] = []
        self.transformer_options: dict[str, Any] = {
            "optimized_attention_override": self._override,
        }

    # -- lifecycle ---------------------------------------------------------

    def configure(self, text_len: int, query_segments: dict[str, tuple[int, int]]) -> None:
        """Set the sequence layout for the next forwards (per prompt).

        Args:
            text_len: number of LEADING key positions that are text tokens.
            query_segments: name -> (start, stop) spans of QUERY rows whose
                attention to the text keys is recorded. Spans are clipped to
                the observed query length at call time.
        """
        if text_len <= 0:
            raise ValueError(f"text_len must be positive, got {text_len}")
        for name, (a, b) in query_segments.items():
            if b <= a:
                raise ValueError(f"segment {name!r} empty: ({a}, {b})")
        self._text_len = text_len
        self._segments = dict(query_segments)
        self._expected_full = max(b for _a, b in self._segments.values())

    def reset(self, step: int, side: str) -> None:
        """Start a new forward; records gathered until the next reset/take."""
        self.records = []
        self._step = step
        self._side = side

    def take_records(self) -> list[dict[str, Any]]:
        recs = self.records
        self.records = []
        return recs

    # -- the override ------------------------------------------------------

    def _override(
        self,
        func: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        seq_k = int(k.shape[-2])
        if not self._segments or seq_k < self._expected_full:
            # no text keys present in this call (refiner/txtfusion/noise
            # blocks) or capture not configured: delegate untouched
            delegated: torch.Tensor = func(q, k, v, heads, *args, **kwargs)
            return delegated

        text_len = self._text_len
        head_dim = float(q.shape[-1])
        seq_q = int(q.shape[-2])
        with torch.no_grad():
            scores = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(head_dim)
            probs = torch.softmax(scores, dim=-1)  # [b, heads, seq_q, seq_k]
            seg_readouts: dict[str, torch.Tensor] = {}
            total_rows = 0
            agg = torch.zeros(text_len, dtype=torch.float32)
            for name, (a, b) in self._segments.items():
                a2, b2 = max(0, a), min(b, seq_q)
                if b2 <= a2:
                    continue  # segment outside this call's query range
                rows = probs[..., a2:b2, :text_len].mean(dim=(0, 1, 2)).to("cpu", torch.float32)
                seg_readouts[name] = rows
                n = b2 - a2
                agg += rows * n
                total_rows += n
            if not seg_readouts:
                delegated = func(q, k, v, heads, *args, **kwargs)
                return delegated
            readout = agg / max(total_rows, 1)  # row-weighted aggregate
            self.records.append(
                {
                    "step": self._step,
                    "side": self._side,
                    "call": len(self.records),
                    "text_len": text_len,
                    "seq_k": seq_k,
                    "segment_readouts": seg_readouts,
                    "text_readout": readout,
                    "text_entropy": float(-(readout.clamp(min=1e-12).log() * readout).sum()),
                    "top_tokens": torch.topk(readout, min(10, text_len)).indices.tolist(),
                }
            )
        out: torch.Tensor = func(q, k, v, heads, *args, **kwargs)
        return out
