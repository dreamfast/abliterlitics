"""Attention capture harness — record what the DiT READS from conditioning.

Phase 4 T2.2 (B0 spike GO — journal #123). Uses ComfyUI's native
``transformer_options["optimized_attention_override"]`` hook (checked by
``wrap_attn`` BEFORE any kernel dispatch): no monkey-patching, no kernel
forcing; bypasses flash/sage/xformers for the capture run only.

The override receives q/k/v as [b, heads, seq, head_dim] (skip_reshape=True)
from Lumina2 JointAttention. In the joint sequence, text (cap) tokens occupy
the LEADING positions, image tokens the tail (4096 at 1024px, patch 2).
Capture computes softmax(q@k^T) explicitly and records, per joint attention
call:

- text_readout: mean over heads + image-query rows of attention TO text-key
  columns -> [text_len] distribution ("what the image tokens read from the
  prompt")
- text_entropy + top-10 token indices
- the seq split for the analysis layer

Attention calls whose key length equals the image-only length are
noise_refiner blocks (no text keys) and are delegated untouched. Step/side
identity comes from the caller resetting the capture before each forward;
call order within a forward identifies the layer (noise_refiner first, then
main layers — the skipped refiner calls do not consume indices).
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch

log = logging.getLogger(__name__)


class AttentionCapture:
    """Stateful capture bound to one DiT forward (one step, one side).

    Lifecycle per forward: ``reset(step, side)`` -> model(...) with
    ``transformer_options=capture.transformer_options`` -> ``take_records()``.
    Requires ``configure(img_len=...)`` once before the first forward.
    """

    def __init__(self, img_len: int, device: str = "cuda") -> None:
        self._img_len = img_len
        self._device = device
        self._step: int = -1
        self._side: str = "?"
        self.records: list[dict[str, Any]] = []
        self.transformer_options: dict[str, Any] = {
            "optimized_attention_override": self._override,
        }

    # -- lifecycle ---------------------------------------------------------

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
        if seq_k <= self._img_len:
            # image-only block (noise_refiner): no text keys to read
            delegated: torch.Tensor = func(q, k, v, heads, *args, **kwargs)
            return delegated

        head_dim = float(q.shape[-1])
        text_len = seq_k - self._img_len
        with torch.no_grad():
            scores = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(head_dim)
            probs = torch.softmax(scores, dim=-1)  # [b, heads, seq_q, seq_k]
            img_to_text = probs[..., self._img_len :, :text_len]  # [b, heads, img, text]
            readout = img_to_text.mean(dim=(0, 1, 2)).to("cpu", torch.float32)  # [text_len]
            self.records.append(
                {
                    "step": self._step,
                    "side": self._side,
                    "call": len(self.records),
                    "text_len": text_len,
                    "img_len": self._img_len,
                    "text_readout": readout,
                    "text_entropy": float(-(readout.clamp(min=1e-12).log() * readout).sum()),
                    "top_tokens": torch.topk(readout, min(10, text_len)).indices.tolist(),
                }
            )
        out: torch.Tensor = func(q, k, v, heads, *args, **kwargs)
        return out
