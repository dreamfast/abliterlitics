"""Vendored GGUF block dequantization (torch) for the qtypes we measure.

Vendored VERBATIM from ComfyUI-GGUF ``dequant.py`` (c) City96, Apache-2.0
(https://github.com/city96/ComfyUI-GGUF), commit
6ea2651e7df66d7585f6ffee804b20e92fb38b8a (upstream HEAD at vendoring
time). Only the functions
needed by our qtype set (Q8_0, Q6_K, Q5_K, Q4_K, Q3_K + F16/F32/BF16
passthrough) are kept; unsupported qtypes fail loud. Do NOT "modernize"
the vendored block functions — byte-fidelity with ComfyUI-GGUF's runtime
dequantization is the point of this module.

Deviation from upstream (documented, deliberate):
- Upstream ``dequantize`` only handles quantized qtypes (F16/F32 would
  KeyError); torch-compatible types are routed around it by
  ``dequantize_tensor``. Our standalone ``dequantize`` performs that
  branch itself (reshape to original shape + dtype cast), so callers
  cannot hit the KeyError path.
- qtypes are passed as int values (``GGMLQuantizationType`` is an
  IntEnum: F32=0, F16=1, Q3_K=11, Q4_K=12, Q5_K=13, Q6_K=14, Q8_0=8,
  BF16=30) so this module has no runtime dependency on the ``gguf``
  package. IntEnum members hash equal to their int value, so callers
  can pass either.

Storage-order contract (CRITICAL — see gguf_loader.py): ``data`` must be
the RAW GGUF storage tensor exactly as ``gguf.GGUFReader`` serves it
(uint8/flat/storage-shaped). Block structure is decoded from the flat
buffer; ``oshape`` (the reversed/original HF shape) is applied LAST.
Never reshape storage to ``oshape`` before calling — that scrambles the
blocks silently.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch

log = logging.getLogger(__name__)

# GGMLQuantizationType int values (IntEnum) for the qtypes we support.
QTYPE_F32 = 0
QTYPE_F16 = 1
QTYPE_Q3_K = 11
QTYPE_Q4_K = 12
QTYPE_Q5_K = 13
QTYPE_Q6_K = 14
QTYPE_Q8_0 = 8
QTYPE_BF16 = 30

# GGML_QUANT_SIZES subset: qtype -> (block_size, type_size_in_bytes).
GGUF_QUANT_SIZES: dict[int, tuple[int, int]] = {
    QTYPE_Q8_0: (32, 34),
    QTYPE_Q3_K: (256, 110),
    QTYPE_Q4_K: (256, 144),
    QTYPE_Q5_K: (256, 176),
    QTYPE_Q6_K: (256, 210),
}

# Types stored in a torch-native dtype (pass-through, no block decode).
TORCH_COMPATIBLE_QTYPES = frozenset({QTYPE_F32, QTYPE_F16, QTYPE_BF16})

QTYPE_NAMES: dict[int, str] = {
    QTYPE_F32: "F32",
    QTYPE_F16: "F16",
    QTYPE_Q3_K: "Q3_K",
    QTYPE_Q4_K: "Q4_K",
    QTYPE_Q5_K: "Q5_K",
    QTYPE_Q6_K: "Q6_K",
    QTYPE_Q8_0: "Q8_0",
    QTYPE_BF16: "BF16",
}

QK_K = 256
K_SCALE_SIZE = 12


# --- vendored from ComfyUI-GGUF/dequant.py (city96, Apache-2.0) ---


def to_uint32(x: torch.Tensor) -> torch.Tensor:
    # no uint32 :(
    x = x.view(torch.uint8).to(torch.int32)
    return (x[:, 0] | x[:, 1] << 8 | x[:, 2] << 16 | x[:, 3] << 24).unsqueeze(1)


def split_block_dims(blocks: torch.Tensor, *args: int) -> tuple[torch.Tensor, ...]:
    n_max = blocks.shape[1]
    dims = list(args) + [n_max - sum(args)]
    return torch.split(blocks, dims, dim=1)


def get_scale_min(scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n_blocks = scales.shape[0]
    scales = scales.view(torch.uint8)
    scales = scales.reshape((n_blocks, 3, 4))

    d, m, m_d = torch.split(scales, scales.shape[-2] // 3, dim=-2)

    sc = torch.cat([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], dim=-1)
    min = torch.cat([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], dim=-1)

    return (sc.reshape((n_blocks, 8)), min.reshape((n_blocks, 8)))


def dequantize_blocks_BF16(blocks: torch.Tensor, block_size: int, type_size: int, dtype: torch.dtype | None = None) -> torch.Tensor:
    return (blocks.view(torch.int16).to(torch.int32) << 16).view(torch.float32)


def dequantize_blocks_Q8_0(blocks: torch.Tensor, block_size: int, type_size: int, dtype: torch.dtype | None = None) -> torch.Tensor:
    d, x = split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)
    x = x.view(torch.int8)
    return (d * x)


def dequantize_blocks_Q6_K(blocks: torch.Tensor, block_size: int, type_size: int, dtype: torch.dtype | None = None) -> torch.Tensor:
    n_blocks = blocks.shape[0]

    ql, qh, scales, d, = split_block_dims(blocks, QK_K // 2, QK_K // 4, QK_K // 16)

    scales = scales.view(torch.int8).to(dtype)
    d = d.view(torch.float16).to(dtype)
    d = (d * scales).reshape((n_blocks, QK_K // 16, 1))

    ql = ql.reshape((n_blocks, -1, 1, 64)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))
    qh = (qh & 0x03).reshape((n_blocks, -1, 32))
    q = (ql | (qh << 4)).to(torch.int8) - 32
    q = q.reshape((n_blocks, QK_K // 16, -1))

    return (d * q).reshape((n_blocks, QK_K))


def dequantize_blocks_Q5_K(blocks: torch.Tensor, block_size: int, type_size: int, dtype: torch.dtype | None = None) -> torch.Tensor:
    n_blocks = blocks.shape[0]

    d, dmin, scales, qh, qs = split_block_dims(blocks, 2, 2, K_SCALE_SIZE, QK_K // 8)

    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)

    sc, m = get_scale_min(scales)

    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))

    ql = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([i for i in range(8)], device=d.device, dtype=torch.uint8).reshape((1, 1, 8, 1))
    ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh = (qh & 0x01).reshape((n_blocks, -1, 32))
    q = (ql | (qh << 4))

    return (d * q - dm).reshape((n_blocks, QK_K))


def dequantize_blocks_Q4_K(blocks: torch.Tensor, block_size: int, type_size: int, dtype: torch.dtype | None = None) -> torch.Tensor:
    n_blocks = blocks.shape[0]

    d, dmin, scales, qs = split_block_dims(blocks, 2, 2, K_SCALE_SIZE)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)

    sc, m = get_scale_min(scales)

    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))

    qs = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1, 32))

    return (d * qs - dm).reshape((n_blocks, QK_K))


def dequantize_blocks_Q3_K(blocks: torch.Tensor, block_size: int, type_size: int, dtype: torch.dtype | None = None) -> torch.Tensor:
    n_blocks = blocks.shape[0]

    hmask, qs, scales, d = split_block_dims(blocks, QK_K // 8, QK_K // 4, 12)
    d = d.view(torch.float16).to(dtype)

    lscales, hscales = scales[:, :8], scales[:, 8:]
    lscales = lscales.reshape((n_blocks, 1, 8)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 2, 1))
    lscales = lscales.reshape((n_blocks, 16))
    hscales = hscales.reshape((n_blocks, 1, 4)) >> torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 4, 1))
    hscales = hscales.reshape((n_blocks, 16))
    scales = (lscales & 0x0F) | ((hscales & 0x03) << 4)
    scales = (scales.to(torch.int8) - 32)

    dl = (d * scales).reshape((n_blocks, 16, 1))

    ql = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))
    qh = hmask.reshape(n_blocks, -1, 1, 32) >> torch.tensor([i for i in range(8)], device=d.device, dtype=torch.uint8).reshape((1, 1, 8, 1))
    ql = ql.reshape((n_blocks, 16, QK_K // 16)) & 3
    qh = (qh.reshape((n_blocks, 16, QK_K // 16)) & 1) ^ 1
    q = (ql.to(torch.int8) - (qh << 2).to(torch.int8))

    return (dl * q).reshape((n_blocks, QK_K))


BlockFn = Callable[[torch.Tensor, int, int, "torch.dtype | None"], torch.Tensor]

# to_uint32 + dequantize_blocks_BF16 are vendored for upstream byte-fidelity
# even though no vendored qtype currently calls them (Q5_0/Q5_1/IQ* dropped;
# BF16 refused in dequantize() pending corpus validation).
dequantize_functions: dict[int, BlockFn] = {
    QTYPE_BF16: dequantize_blocks_BF16,
    QTYPE_Q8_0: dequantize_blocks_Q8_0,
    QTYPE_Q6_K: dequantize_blocks_Q6_K,
    QTYPE_Q5_K: dequantize_blocks_Q5_K,
    QTYPE_Q4_K: dequantize_blocks_Q4_K,
    QTYPE_Q3_K: dequantize_blocks_Q3_K,
}


# --- end vendored ---


def is_torch_compatible(qtype: int | None) -> bool:
    """True for qtypes stored natively (F32/F16/BF16) — no block decode needed."""
    return qtype is None or int(qtype) in TORCH_COMPATIBLE_QTYPES


def dequantize(data: torch.Tensor, qtype: int, oshape: torch.Size | tuple[int, ...], dtype: torch.dtype | None = None) -> torch.Tensor:
    """Dequantize a raw GGUF storage tensor back to ``oshape``.

    Args:
        data: RAW storage tensor as served by ``gguf.GGUFReader`` (uint8 /
            storage dtype, storage shape). Must NOT be reshaped beforehand.
        qtype: ``GGMLQuantizationType`` member or its int value.
        oshape: ORIGINAL (HF) shape — the reversed storage shape.
        dtype: Target dtype (e.g. ``torch.bfloat16``). NOTE: when omitted
            (None) on F16/F32 input, the returned tensor can be a VIEW
            aliasing the caller's buffer — pass an explicit dtype (or copy)
            if the buffer is mmap-backed and will be released.

    Returns:
        Tensor of shape ``oshape`` and dtype ``dtype`` (default: keep the
        block function's output dtype, typically fp16-scaled fp32).
    """
    qv = int(qtype)
    if is_torch_compatible(qv):
        if qv == QTYPE_BF16:
            # No BF16 GGUF corpus to validate reader byte semantics against
            # (gguf-py may serve raw uint8 here — reshape would silently
            # reinterpret). Fail loud; revisit when a BF16 container shows up.
            raise NotImplementedError("BF16 GGUF containers are not validated — refusing to guess byte layout")
        # F16/F32: flat reinterpret to the original shape, then cast.
        # (upstream routes this around dequantize(); we branch here so the
        # standalone call cannot KeyError on torch-native types)
        return data.reshape(oshape).to(dtype if dtype is not None else data.dtype)
    if qv not in dequantize_functions:
        raise ValueError(
            f"Unsupported GGUF qtype {qv} ({QTYPE_NAMES.get(qv, '?')}) — this build "
            "vendored dequant only for F16/F32/BF16/Q8_0/Q6_K/Q5_K/Q4_K/Q3_K"
        )

    block_size, type_size = GGUF_QUANT_SIZES[qv]

    rows = data.reshape((-1, data.shape[-1])).view(torch.uint8)

    n_blocks = rows.numel() // type_size
    if n_blocks * type_size != rows.numel():
        raise ValueError(
            f"Storage size {rows.numel()} not divisible by type_size {type_size} for qtype {qv} — "
            "data is not raw GGUF storage (was it reshaped/transposed before dequantize?)"
        )
    blocks = rows.reshape((n_blocks, type_size))
    blocks = dequantize_functions[qv](blocks, block_size, type_size, dtype)
    out = blocks.reshape(oshape)
    if dtype is not None and out.dtype != dtype:
        out = out.to(dtype)
    return out
