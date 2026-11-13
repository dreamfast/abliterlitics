"""Tests for backend infrastructure (BaseBackend, HookManager, FakeQuantizer).

These test the abstract infrastructure without loading any real model.
Model-specific backend tests live in test_backends_<model>.py
(h3, krea2, zimage).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.text_encoder.backends import BaseBackend, HookManager, get_backend, register_backend
from src.text_encoder.backends.base import (
    _pack_sequences,
    dtype_label,
    fake_quantize_weights,
)

# ---------------------------------------------------------------------------
# HookManager
# ---------------------------------------------------------------------------


class TestHookManager:
    def test_register_and_capture(self) -> None:
        """Hook should capture module output."""
        hm = HookManager()
        linear = nn.Linear(4, 2)
        hm.register("projected", "test_stream", linear)
        x = torch.randn(1, 4)
        _ = linear(x)
        captures = hm.get_captures()
        assert "projected" in captures
        assert "test_stream" in captures["projected"]
        assert captures["projected"]["test_stream"].shape == (1, 2)
        hm.clear()

    def test_clear_removes_hooks(self) -> None:
        """After clear, hooks should not fire."""
        hm = HookManager()
        linear = nn.Linear(4, 2)
        hm.register("test", "s", linear)
        hm.clear()
        _ = linear(torch.randn(1, 4))
        assert len(hm.get_captures()) == 0

    def test_captures_are_fp32(self) -> None:
        """Captured tensors should be float32 regardless of model dtype."""
        hm = HookManager()
        linear = nn.Linear(4, 2).to(torch.bfloat16)
        hm.register("test", "s", linear)
        _ = linear(torch.randn(1, 4, dtype=torch.bfloat16))
        captures = hm.get_captures()
        assert captures["test"]["s"].dtype == torch.float32
        hm.clear()


# ---------------------------------------------------------------------------
# FakeQuantizer
# ---------------------------------------------------------------------------


class TestFakeQuantize:
    def test_int8_changes_weights(self) -> None:
        """int8 quantization should alter weights."""
        model = nn.Linear(8, 8)
        orig = model.weight.data.clone()
        fake_quantize_weights(model, "int8")
        assert not torch.allclose(orig, model.weight.data, atol=1e-8)

    def test_fp8_changes_weights(self) -> None:
        """fp8 quantization should alter weights."""
        model = nn.Linear(8, 8)
        orig = model.weight.data.clone()
        fake_quantize_weights(model, "fp8")
        assert not torch.allclose(orig, model.weight.data, atol=1e-8)

    def test_preserves_dtype(self) -> None:
        """Original dtype should be preserved after quantization."""
        model = nn.Linear(8, 8).to(torch.bfloat16)
        fake_quantize_weights(model, "int8")
        assert model.weight.dtype == torch.bfloat16

    def test_zero_model_stays_zero(self) -> None:
        """A zero-initialized model should stay near-zero."""
        model = nn.Linear(4, 4)
        model.weight.data.zero_()
        model.bias.data.zero_()
        fake_quantize_weights(model, "int8")
        assert model.weight.data.abs().max() < 1e-10

    def test_unknown_type_raises(self) -> None:
        model = nn.Linear(4, 4)
        with pytest.raises(ValueError, match="Unknown quant_type"):
            fake_quantize_weights(model, "unknown")


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------


class _DummyBackend(BaseBackend):
    """Minimal concrete backend for testing the registry."""

    @property
    def name(self) -> str:
        return "dummy"

    @property
    def streams(self) -> list:
        return []

    def load(self, ckpt: str, dtype: str = "bf16", device: str = "cpu") -> None:
        pass

    def encode(self, prompts: list[str], categories: list[str], stages=None) -> list:
        return []

    def tokenize(self, prompts: list[str]) -> dict:
        return {"input_ids": []}


class TestBackendRegistry:
    def test_get_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("nonexistent")

    def test_register_and_get(self) -> None:
        register_backend("dummy_test", _DummyBackend)
        backend = get_backend("dummy_test")
        assert isinstance(backend, _DummyBackend)
        assert backend.name == "dummy"
        # Clean up
        from src.text_encoder.backends import BACKENDS

        del BACKENDS["dummy_test"]


# ---------------------------------------------------------------------------
# _pack_sequences (shared right-pad/mask helper)
# ---------------------------------------------------------------------------


class TestPackSequences:
    def test_pads_with_default_mask(self) -> None:
        seqs = [torch.ones(3, 4), torch.ones(5, 4) * 2.0]
        acts, mask = _pack_sequences(seqs, 4)
        assert acts.shape == (2, 5, 4)
        assert mask.shape == (2, 5)
        assert mask[0].tolist() == [1, 1, 1, 0, 0]
        assert mask[1].tolist() == [1] * 5
        assert torch.equal(acts[1], torch.full((5, 4), 2.0))
        assert torch.equal(acts[0, 3:], torch.zeros(2, 4))

    def test_explicit_masks_win_over_default(self) -> None:
        # 0 inside a sequence = backend-applied pad (e.g. z_image cap-pad)
        seqs = [torch.zeros(4, 2), torch.zeros(4, 2)]
        m0 = torch.tensor([1, 1, 0, 1], dtype=torch.int)
        m1 = torch.tensor([1, 1, 1, 1], dtype=torch.int)
        _, mask = _pack_sequences(seqs, 2, masks=[m0, m1])
        assert mask[0].tolist() == [1, 1, 0, 1]
        assert mask[1].tolist() == [1, 1, 1, 1]

    def test_masks_shorter_than_max_len_right_pad_with_zero(self) -> None:
        seqs = [torch.zeros(2, 2), torch.zeros(5, 2)]
        masks = [torch.ones(2, dtype=torch.int), torch.ones(5, dtype=torch.int)]
        _, mask = _pack_sequences(seqs, 2, masks=masks)
        assert mask[0].tolist() == [1, 1, 0, 0, 0]
        assert mask[1].tolist() == [1] * 5

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="no sequences"):
            _pack_sequences([], 4)


# ---------------------------------------------------------------------------
# dtype_label (shared artifact dtype ladder — one precedence for all backends)
# ---------------------------------------------------------------------------


class TestDtypeLabel:
    def test_fakequant_outranks_gguf(self) -> None:
        # the unified precedence (audit Bugs #2): krea2's order everywhere
        assert dtype_label("int8", "q4_k_m", False) == "fakequant_int8"

    def test_gguf(self) -> None:
        assert dtype_label(None, "q4_k_m", True) == "gguf_q4_k_m"

    def test_comfy_quant_label_override(self) -> None:
        assert dtype_label(None, None, True, comfy_quant_label="comfy_int8_tensorwise") == "comfy_int8_tensorwise"

    def test_comfy_quant_default_label(self) -> None:
        assert dtype_label(None, None, True) == "comfy_int8_convrot"

    def test_fallback(self) -> None:
        assert dtype_label(None, None, False) == "comfy_bf16"


# ---------------------------------------------------------------------------
# BaseBackend.apply_fake_quant declaration
# ---------------------------------------------------------------------------


class TestApplyFakeQuantStub:
    def test_stub_raises_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError, match="fake quantization"):
            _DummyBackend().apply_fake_quant("int8")

    def test_ltx2_inherits_stub(self) -> None:
        from src.text_encoder.backends.ltx2 import LTX2Backend

        assert "apply_fake_quant" not in LTX2Backend.__dict__  # inherits, not overrides
        with pytest.raises(NotImplementedError):
            LTX2Backend().apply_fake_quant("fp8")
