"""Tests for h3_video.py — frame sampling, pairing, adherence probe."""

from __future__ import annotations

import numpy as np
import pytest

from src.text_encoder.h3_video import (
    adherence_probe_text,
    frame_to_tensor,
    pair_clips,
    sample_frame_indices,
)


class TestSampleFrameIndices:
    def test_includes_first_and_last(self) -> None:
        idx = sample_frame_indices(124, 8)
        assert idx[0] == 0
        assert idx[-1] == 123

    def test_deterministic(self) -> None:
        assert sample_frame_indices(100, 5) == sample_frame_indices(100, 5)

    def test_k_clamped(self) -> None:
        assert sample_frame_indices(3, 10) == [0, 1, 2]

    def test_k1_middle(self) -> None:
        assert sample_frame_indices(9, 1) == [4]

    def test_zero_raises(self) -> None:
        with pytest.raises(ValueError):
            sample_frame_indices(0, 4)


class TestFrameTensor:
    def test_shape_and_range(self) -> None:
        f = np.full((4, 4, 3), 255, dtype=np.uint8)
        t = frame_to_tensor(f)
        assert t.shape == (1, 3, 4, 4)
        assert t.min() >= -1.0 and t.max() <= 1.0
        assert abs(float(t.max()) - 1.0) < 1e-6  # white -> +1


class TestPairClips:
    def test_buckets_and_arms(self) -> None:
        clips = [
            {"label": "cat", "seed": 42, "variant": "base", "path": "a"},
            {"label": "cat", "seed": 42, "variant": "heretic", "path": "b"},
            {"label": "cat", "seed": 43, "variant": "base", "path": "c"},
        ]
        g = pair_clips(clips)
        assert g["arms"] == ["base", "heretic"]
        assert set(g["buckets"]) == {("cat", 42), ("cat", 43)}
        assert g["buckets"][("cat", 42)]["heretic"]["path"] == "b"


class TestAdherenceProbe:
    def test_first_sentence_of_integrated(self) -> None:
        prompt = (
            "integrated_multimodal_description: A black cat sits on a windowsill. "
            "The camera pushes in slowly. More text here.\n"
            "overall_soundscape: rain\nnon_diegetic_music: N/A"
        )
        assert adherence_probe_text(prompt) == "A black cat sits on a windowsill."

    def test_fallback_when_no_field(self) -> None:
        assert adherence_probe_text("A red circle on white background.") == "A red circle on white background."
