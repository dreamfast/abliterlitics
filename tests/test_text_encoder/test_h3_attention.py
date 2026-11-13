"""Regression tests for the run_h3_attention empty-segment guard.

The aggregation used to torch.stack the per-segment readout lists unguarded:
a capture layout that yields no record for a segment raised AFTER all GPU
work for the variant was done (audit: run_h3_attention.py:221). The guard
emits empty means + a loud error instead of crashing the battery.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from src.text_encoder.run_h3_attention import _segment_means


def _rec(audio: list[float] | None, video: list[float] | None) -> dict[str, Any]:
    segs: dict[str, torch.Tensor] = {}
    if audio is not None:
        segs["audio"] = torch.tensor(audio)
    if video is not None:
        segs["video"] = torch.tensor(video)
    return {"segment_readouts": segs}


def test_segment_means_normal_aggregation() -> None:
    recs_b = [_rec([1.0, 2.0], [3.0, 4.0]), _rec([3.0, 4.0], [5.0, 6.0])]
    recs_v = [_rec([5.0, 6.0], [7.0, 8.0]), _rec([7.0, 8.0], [9.0, 10.0])]
    out = _segment_means(recs_b, recs_v)
    assert out["audio"]["base_mean_readout"] == [2.0, 3.0]
    assert out["audio"]["variant_mean_readout"] == [6.0, 7.0]
    assert out["video"]["base_mean_readout"] == [4.0, 5.0]
    assert out["video"]["variant_mean_readout"] == [8.0, 9.0]


def test_segment_means_empty_record_lists_guarded(caplog: Any) -> None:
    """Empty capture for a whole variant must not torch.stack([])-crash."""
    with caplog.at_level(logging.ERROR):
        out = _segment_means([], [])
    assert out == {
        "audio": {"base_mean_readout": [], "variant_mean_readout": []},
        "video": {"base_mean_readout": [], "variant_mean_readout": []},
    }
    assert "missing from capture records" in caplog.text


def test_segment_means_missing_segment_key_guarded(caplog: Any) -> None:
    """Records present but NONE carrying a segment key (layout regression).

    Guard is all-or-nothing per segment: a paired A/B mean with one side
    missing is not emitted half-populated — both sides go empty + error.
    """
    recs_b = [_rec([1.0], [2.0]), _rec([3.0], [4.0])]
    recs_v = [_rec(None, [6.0]), _rec(None, [8.0])]  # no audio key at all
    with caplog.at_level(logging.ERROR):
        out = _segment_means(recs_b, recs_v)
    assert out["audio"] == {"base_mean_readout": [], "variant_mean_readout": []}
    assert out["video"]["base_mean_readout"] == [3.0]
    assert out["video"]["variant_mean_readout"] == [7.0]
    assert "missing from capture records" in caplog.text
    assert "audio" in caplog.text


def test_segment_means_error_names_the_segment(caplog: Any) -> None:
    with caplog.at_level(logging.ERROR):
        _segment_means([_rec(None, [1.0])], [_rec(None, [2.0])])  # audio missing BOTH sides
    assert "'audio'" in caplog.text
    assert "h3_dit.build_layout" in caplog.text
