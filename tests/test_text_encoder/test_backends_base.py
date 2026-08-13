"""Tests for backend infrastructure (BaseBackend, HookManager, FakeQuantizer).

These test the abstract infrastructure without loading any real model.
Model-specific backend tests live in backends/test_ltx2.py etc.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.text_encoder.backends import BaseBackend, HookManager, get_backend, register_backend
from src.text_encoder.backends.base import fake_quantize_weights

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
