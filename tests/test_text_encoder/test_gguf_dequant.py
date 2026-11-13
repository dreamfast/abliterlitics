"""Tests for vendored GGUF dequantization (gguf_dequant.py).

Golden tests compare the vendored torch block functions against gguf's own
numpy reference (``gguf.quants.dequantize``) on bytes extracted from the
REAL qwen3-4b GGUF files — bit-exact agreement verified during development
(max diff 0.0 on a Q4_K tensor). File-dependent tests skip when the models
directory is absent.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from src.text_encoder.backends import gguf_dequant as gd

GGUF_DIR = Path(os.environ["TE_TEST_GGUF_DIR"]) if os.environ.get("TE_TEST_GGUF_DIR") else None

gguf = pytest.importorskip("gguf", reason="gguf package not installed")
quants = pytest.importorskip("gguf.quants", reason="gguf.quants not available")


def _np(arr: np.ndarray, dtype: Any = None) -> np.ndarray:
    return np.ascontiguousarray(arr, dtype=dtype)


class TestHandBuiltQ8:
    """Exact-value test on hand-assembled Q8_0 blocks (no gguf files needed)."""

    def test_exact_values(self) -> None:
        # Two Q8_0 blocks: [fp16 scale (2B) + 32 x int8 payload (32B)] each.
        d1 = np.float16(1.0).tobytes()  # 0x3C00
        d2 = np.float16(0.5).tobytes()  # 0x3800
        x1 = np.arange(-16, 16, dtype=np.int8).tobytes()
        x2 = np.ones(32, dtype=np.int8).tobytes()
        raw = np.frombuffer(d1 + x1 + d2 + x2, dtype=np.uint8)
        data = torch.from_numpy(_np(raw)).view(torch.uint8)

        out = gd.dequantize(data, gd.QTYPE_Q8_0, oshape=(2, 32), dtype=torch.float32)
        assert out.shape == (2, 32)
        assert out.dtype == torch.float32
        expected_row1 = torch.arange(-16, 16, dtype=torch.float32) * 1.0
        expected_row2 = torch.full((32,), 0.5, dtype=torch.float32)
        assert torch.equal(out[0], expected_row1)
        assert torch.equal(out[1], expected_row2)

    def test_bf16_target_dtype(self) -> None:
        d1 = np.float16(1.0).tobytes()
        x1 = np.full(32, 3, dtype=np.int8).tobytes()
        raw = np.frombuffer(d1 + x1, dtype=np.uint8)
        data = torch.from_numpy(_np(raw)).view(torch.uint8)
        out = gd.dequantize(data, gd.QTYPE_Q8_0, oshape=(1, 32), dtype=torch.bfloat16)
        assert out.dtype == torch.bfloat16
        assert torch.equal(out, torch.full((1, 32), 3.0, dtype=torch.bfloat16))


class TestTorchCompatPassthrough:
    def test_f16_reinterprets_and_casts(self) -> None:
        # Storage flat buffer of 16 fp16 values; logical shape (2, 8).
        vals = np.arange(16, dtype=np.float16)
        data = torch.from_numpy(np.ascontiguousarray(vals)).view(torch.float16)
        out = gd.dequantize(data, gd.QTYPE_F16, oshape=(2, 8), dtype=torch.bfloat16)
        assert out.shape == (2, 8)
        assert out.dtype == torch.bfloat16
        assert torch.equal(out, torch.arange(16, dtype=torch.float32).reshape(2, 8).to(torch.bfloat16))

    def test_f32_passthrough(self) -> None:
        vals = np.linspace(-1.0, 1.0, 12, dtype=np.float32)
        data = torch.from_numpy(np.ascontiguousarray(vals))
        out = gd.dequantize(data, gd.QTYPE_F32, oshape=(3, 4), dtype=torch.float32)
        assert torch.equal(out, data.reshape(3, 4))


class TestFailureModes:
    def test_unsupported_qtype_raises(self) -> None:
        data = torch.zeros(64, dtype=torch.uint8)
        # Q2_K (10) deliberately not vendored.
        with pytest.raises(ValueError, match="Unsupported GGUF qtype"):
            gd.dequantize(data, 10, oshape=(2, 32), dtype=torch.float32)

    def test_non_divisible_storage_raises(self) -> None:
        data = torch.zeros(33, dtype=torch.uint8)  # not a multiple of 34
        with pytest.raises(ValueError, match="not divisible"):
            gd.dequantize(data, gd.QTYPE_Q8_0, oshape=(1, 32), dtype=torch.float32)


def _real_tensor(file_name: str, tensor_name: str):  # type: ignore[no-untyped-def]
    reader = gguf.GGUFReader(str(GGUF_DIR / file_name))
    for t in reader.tensors:
        if t.name == tensor_name:
            return t
    raise AssertionError(f"{tensor_name} not found in {file_name}")


@pytest.mark.skipif(GGUF_DIR is None or not GGUF_DIR.exists(), reason="TE_TEST_GGUF_DIR not set or dir absent")
class TestGoldenAgainstNumpyReference:
    """Vendored torch dequant must reproduce gguf's numpy reference bit-exactly."""

    @pytest.mark.parametrize(
        ("file_name", "tensor_name"),
        [
            ("qwen3-4b-Q8_0.gguf", "blk.3.attn_v.weight"),
            ("qwen3-4b-Q6_K.gguf", "blk.0.attn_q.weight"),
            ("qwen3-4b-Q5_K_M.gguf", "blk.0.attn_q.weight"),  # Q5_K blocks
            ("qwen3-4b-Q4_K_M.gguf", "blk.0.attn_q.weight"),  # Q4_K blocks
            ("qwen3-4b-Q3_K_M.gguf", "blk.0.attn_q.weight"),  # Q3_K blocks
            ("qwen3-4b-Q3_K_M.gguf", "blk.0.ffn_gate.weight"),  # Q4_K blocks in Q3_K_M
        ],
    )
    def test_real_tensor_bit_exact(self, file_name: str, tensor_name: str) -> None:
        t = _real_tensor(file_name, tensor_name)
        assert t.tensor_type is not None
        qv = int(t.tensor_type)
        assert qv in gd.dequantize_functions, f"tensor qtype {qv} not in vendored set"

        oshape = tuple(int(v) for v in reversed(t.shape))
        data = torch.from_numpy(_np(t.data)).view(torch.uint8)

        mine = gd.dequantize(data, qv, oshape=oshape, dtype=torch.float32)
        ref = quants.dequantize(_np(t.data), t.tensor_type)
        ref_t = torch.from_numpy(_np(ref, dtype=np.float32)).reshape(oshape)

        assert mine.shape == ref_t.shape
        # fp32 target: verified bit-exact during development.
        assert torch.equal(mine, ref_t), f"max diff {(mine - ref_t).abs().max()}"

    def test_f16_file_tensor(self) -> None:
        t = _real_tensor("qwen3-4b-f16.gguf", "blk.0.attn_output.weight")
        oshape = tuple(int(v) for v in reversed(t.shape))
        data = torch.from_numpy(_np(t.data))
        out = gd.dequantize(data, int(t.tensor_type), oshape=oshape, dtype=torch.float32)
        ref = torch.from_numpy(_np(t.data, dtype=np.float32)).reshape(oshape)
        assert torch.equal(out, ref)
