"""Tests for the krea2 + h3 DiT harness modules (no real multi-GB weights).

Covers: config derivation formulas against a synthetic checkpoint-shaped key
map, sigma schedules, time_shift_sigma (verbatim formula), and the H3 layout
span helper against the plain-t2va segment contract.
"""

from __future__ import annotations

import pytest
import torch

from src.text_encoder.backends.h3_dit import time_shift_sigma
from src.text_encoder.backends.krea2_dit import _count_blocks, _sigma_schedule, time_snr_shift


class TestSigmaSchedules:
    def test_descending_from_sigma_max_to_zero(self) -> None:
        sig = _sigma_schedule(6)
        assert float(sig[0]) == pytest.approx(1.0)  # shift(1) = sigma_max
        assert float(sig[-1]) == 0.0
        diffs = sig[1:-1] - sig[:-2]
        assert torch.all(diffs < 0)  # strictly descending into the terminal 0

    def test_ascending_reproduces_stored_grid(self) -> None:
        sig = _sigma_schedule(6, descending=False)
        # first sigma = shift(1/6): 1.15*(1/6)/(1+0.15/6)
        assert float(sig[0]) == pytest.approx(1.15 * (1 / 6) / (1 + 0.15 * (1 / 6)), rel=1e-6)
        diffs = sig[1:-1] - sig[:-2]
        assert torch.all(diffs > 0)  # ascending, terminal 0

    def test_time_snr_shift_identity(self) -> None:
        assert time_snr_shift(1.0, 0.3) == 0.3


class TestTimeShiftSigma:
    def test_identity_when_same_shift(self) -> None:
        assert time_shift_sigma(0.5, 12.0, 12.0) == pytest.approx(0.5, rel=1e-9)

    def test_verbatim_formula(self) -> None:
        sigma, fs, ts = 0.4, 12.0, 3.0
        base = sigma / (fs + sigma * (1.0 - fs))
        expect = ts * base / (1.0 + (ts - 1.0) * base)
        assert time_shift_sigma(sigma, fs, ts) == pytest.approx(expect, rel=1e-12)

    def test_roundtrip(self) -> None:
        # shifting to b then back to a recovers the input
        s = 0.6
        via = time_shift_sigma(s, 12.0, 3.0)
        back = time_shift_sigma(via, 3.0, 12.0)
        assert back == pytest.approx(s, rel=1e-9)


class TestCountBlocks:
    def test_counts_unique_indices(self) -> None:
        keys = [
            "model.diffusion_model.blocks.0.attn.wq.weight",
            "model.diffusion_model.blocks.0.attn.wk.weight",
            "model.diffusion_model.blocks.27.mlp.up.weight",
            "model.diffusion_model.txtfusion.projector.weight",
        ]
        assert _count_blocks(keys, r"blocks\.(\d+)\.") == 2


class TestKrea2ConfigFormulas:
    """The model_detection.py:949 formulas (weights -> cfg), as pure shape math."""

    def test_config_from_reference_shapes(self) -> None:
        head_dim = 128
        first_w = torch.zeros(6144, 16 * 4)  # features x channels*patch^2
        wq = torch.zeros(48 * head_dim, 6144)
        wk = torch.zeros(12 * head_dim, 6144)
        projector = torch.zeros(1, 12)
        prenorm = torch.zeros(2560)
        cfg = {
            "features": int(first_w.shape[0]),
            "channels": int(first_w.shape[1]) // 4,
            "patch": 2,
            "heads": int(wq.shape[0]) // head_dim,
            "kvheads": int(wk.shape[0]) // head_dim,
            "txtlayers": int(projector.shape[1]),
            "txtdim": int(prenorm.shape[0]),
        }
        assert cfg == {
            "features": 6144,
            "channels": 16,
            "patch": 2,
            "heads": 48,
            "kvheads": 12,
            "txtlayers": 12,
            "txtdim": 2560,
        }


class TestH3LayoutContract:
    """Plain-t2va PackedLayout: [text | audio (2*T rows) | video], spans from
    build_layout must match the segment table. Skipped when comfy is not
    importable (CI containers); exercised for real in the live smoke."""

    def test_layout_spans(self) -> None:
        pytest.importorskip("comfy", reason="comfy repo not mounted")
        import sys

        comfy_root = "/comfyui"
        from pathlib import Path

        if not Path(comfy_root, "comfy", "ldm", "minimax", "model.py").exists():
            pytest.skip("/comfyui not mounted")
        if comfy_root not in sys.path:
            sys.path.insert(0, comfy_root)
        from comfy.ldm.minimax.model import PackedLayout

        from src.text_encoder.backends import h3_dit

        layout = PackedLayout(text_len=37, latent_t=5, latent_h=32, latent_w=32, audio_t=80)
        kinds = [kind for _a, _b, kind in layout.segments]
        assert kinds == ["text", "audio", "video"]
        # audio rows = 2 per latent frame; video rows = latent_t * (32/2)^2
        a0, a1 = None, None
        v0, v1 = None, None
        row = 0
        for a, b, kind in layout.segments:
            if kind == "audio":
                a0, a1 = a, b
            if kind == "video":
                v0, v1 = a, b
            row = b
        assert (a0, a1) == (37, 37 + 160)
        assert (v0, v1) == (37 + 160, 37 + 160 + 5 * 256)
        assert layout.seq_len == row

        # build_layout wrapper picks exactly these spans
        _lay, spans = h3_dit.build_layout(None, 37, 5, 32, 32, 80)
        assert spans == {"audio": (a0, a1), "video": (v0, v1)}
