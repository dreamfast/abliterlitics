"""Tests for the generalized AttentionCapture (phase 9).

Locks the wrap_attn override contract: text-leading keys, per-segment readouts
with row-weighted aggregate, delegation of sub-full-length calls (noise
refiner / token refiner / txtfusion), and output passthrough.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch

from src.text_encoder.attention_capture import AttentionCapture


def _fake_func(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, *a: Any, **kw: Any) -> torch.Tensor:
    return q.sum() + v.sum() + heads + len(a) + len(kw)


def _attn_probs(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    scores = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(float(q.shape[-1]))
    return torch.softmax(scores, dim=-1)


class TestConfigure:
    def test_empty_segment_rejected(self) -> None:
        c = AttentionCapture()
        with pytest.raises(ValueError, match="empty"):
            c.configure(5, {"img": (5, 5)})

    def test_zero_text_len_rejected(self) -> None:
        c = AttentionCapture()
        with pytest.raises(ValueError, match="positive"):
            c.configure(0, {"img": (0, 4)})


class TestDelegation:
    def test_unconfigured_delegates(self) -> None:
        c = AttentionCapture()
        q, k, v = (torch.randn(1, 2, 8, 4) for _ in range(3))
        out = c._override(_fake_func, q, k, v, 2)
        assert c.records == []
        assert float(out) == float(q.sum() + v.sum() + 2)

    def test_sub_full_seq_delegates(self) -> None:
        """The z-image noise_refiner case: image-only keys, seq_k < text+img."""
        c = AttentionCapture()
        c.configure(text_len=6, query_segments={"img": (6, 6 + 16)})
        c.reset(0, "base")
        q, k, v = (torch.randn(1, 2, 16, 4) for _ in range(3))  # seq_k=16 < 22
        c._override(_fake_func, q, k, v, 2)
        assert c.records == []

    def test_refiner_and_txtfusion_lengths_delegate(self) -> None:
        """seq_k == text_len (H3 token_refiner / krea2 txtfusion) delegates."""
        c = AttentionCapture()
        c.configure(text_len=8, query_segments={"img": (8, 8 + 12)})
        c.reset(0, "base")
        q, k, v = (torch.randn(1, 2, 8, 4) for _ in range(3))
        c._override(_fake_func, q, k, v, 2)
        assert c.records == []


class TestCapture:
    def test_single_segment_matches_manual(self) -> None:
        """z-image semantics preserved: mean over heads + img rows of text cols."""
        torch.manual_seed(0)
        text_len, img_len, heads, d = 5, 9, 3, 4
        c = AttentionCapture()
        c.configure(text_len, {"img": (text_len, text_len + img_len)})
        c.reset(2, "base")
        q, k, v = (torch.randn(1, heads, text_len + img_len, d) for _ in range(3))
        out = c._override(_fake_func, q, k, v, heads)
        # passthrough output
        assert float(out) == float(q.sum() + v.sum() + heads)
        assert len(c.records) == 1
        rec = c.records[0]
        probs = _attn_probs(q, k)
        manual = probs[..., text_len:, :text_len].mean(dim=(0, 1, 2))
        assert torch.allclose(rec["text_readout"], manual, atol=1e-6)
        assert rec["text_len"] == text_len
        assert rec["seq_k"] == text_len + img_len
        assert rec["step"] == 2 and rec["side"] == "base"

    def test_multi_segment_weighted_aggregate(self) -> None:
        """H3-style audio+video: aggregate = row-weighted mean; segments exact."""
        torch.manual_seed(1)
        text_len, a_len, v_len, heads, d = 4, 6, 10, 2, 4
        c = AttentionCapture()
        c.configure(
            text_len, {"audio": (text_len, text_len + a_len), "video": (text_len + a_len, text_len + a_len + v_len)}
        )
        c.reset(0, "variant")
        seq = text_len + a_len + v_len
        q, k, v = (torch.randn(1, heads, seq, d) for _ in range(3))
        c._override(_fake_func, q, k, v, heads)
        rec = c.records[0]
        probs = _attn_probs(q, k)
        man_a = probs[..., text_len : text_len + a_len, :text_len].mean(dim=(0, 1, 2))
        man_v = probs[..., text_len + a_len :, :text_len].mean(dim=(0, 1, 2))
        assert torch.allclose(rec["segment_readouts"]["audio"], man_a, atol=1e-6)
        assert torch.allclose(rec["segment_readouts"]["video"], man_v, atol=1e-6)
        manual_agg = (man_a * a_len + man_v * v_len) / (a_len + v_len)
        assert torch.allclose(rec["text_readout"], manual_agg, atol=1e-6)

    def test_reset_and_take(self) -> None:
        c = AttentionCapture()
        c.configure(3, {"img": (3, 8)})
        c.reset(0, "base")
        q, k, v = (torch.randn(1, 1, 8, 4) for _ in range(3))
        c._override(_fake_func, q, k, v, 1)
        c._override(_fake_func, q, k, v, 1)
        assert len(c.records) == 2
        recs = c.take_records()
        assert len(recs) == 2 and c.records == []
        c.reset(1, "variant")
        assert c.records == []

    def test_entropy_and_topk(self) -> None:
        c = AttentionCapture()
        c.configure(6, {"img": (6, 10)})
        c.reset(0, "base")
        q, k, v = (torch.randn(1, 1, 10, 4) for _ in range(3))
        c._override(_fake_func, q, k, v, 1)
        rec = c.records[0]
        r = rec["text_readout"]
        assert rec["text_entropy"] == pytest.approx(float(-(r.clamp(min=1e-12).log() * r).sum()), rel=1e-5)
        assert sorted(rec["top_tokens"]) == sorted(torch.topk(r, 6).indices.tolist())
