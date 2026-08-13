"""GGUF text-encoder loader: llama.cpp container -> HF-named state dict.

Replicates ComfyUI-GGUF's qwen3 load path (city96, Apache-2.0) with eager
dequantization instead of lazy matmul-time ops:

1. ``gguf.GGUFReader`` -> raw storage tensors (NEVER reshaped before
   dequantization — block structure is decoded from the raw buffer, the
   original HF shape is applied last; see gguf_dequant.py).
2. Original shape: ``comfy.gguf.orig_shape.<tensor>`` metadata when present
   (mirrors ComfyUI-GGUF ``get_orig_shape``), else the REVERSED storage
   shape (llama.cpp transposed-storage convention).
3. ``token_embd.weight`` with vocab >= 64K takes ComfyUI's fp16 detour
   (an OOM workaround there; kept for fidelity — the fp16 intermediate is
   immediately cast to the target dtype).
4. Name remap via the vendored order-preserving ``LLAMA_SD_MAP`` — the
   order is load-bearing: ``attn_q_norm.`` must be replaced before
   ``attn_q`` or the substring collision garbles Qwen3 q/k_norm keys.
5. Fail-loud key-set validation against the qwen3 layout derived from the
   file's own ``qwen3.block_count`` metadata (no lm_head — encoder-only).

Dequantization target: eager, storage-truth dtype (bf16 by default) —
same convention as the safetensors backends (weights stored at their
native precision class, fp32 compute in the model). Deviation from
ComfyUI production (lazy dequant at compute dtype) is bounded and
documented in the report methodology.
"""

from __future__ import annotations

import logging
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.text_encoder.backends import gguf_dequant as gd

log = logging.getLogger(__name__)

# Vendored VERBATIM from ComfyUI-GGUF/loader.py (city96, Apache-2.0).
# INSERTION ORDER IS LOAD-BEARING — sd_map_replace applies replacements
# sequentially, so attn_q_norm./attn_k_norm. MUST precede attn_q/attn_k.
LLAMA_SD_MAP = {
    "blk.": "model.layers.",
    "attn_norm": "input_layernorm",
    "attn_q_norm.": "self_attn.q_norm.",
    "attn_k_norm.": "self_attn.k_norm.",
    "attn_v_norm.": "self_attn.v_norm.",
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
    "ffn_gate": "mlp.gate_proj",
    "ffn_norm": "post_attention_layernorm",
    "token_embd": "model.embed_tokens",
    "output_norm": "model.norm",
    "output.weight": "lm_head.weight",
}


def sd_map_replace(raw_sd: dict[str, torch.Tensor], key_map: dict[str, str]) -> dict[str, torch.Tensor]:
    """Apply ordered substring replacements to keys (ComfyUI-GGUF semantics)."""
    sd: dict[str, torch.Tensor] = {}
    for k, v in raw_sd.items():
        for s, d in key_map.items():
            k = k.replace(s, d)
        sd[k] = v
    return sd


# GGUF tensor-name suffixes per layer (llama.cpp layout) and their HF
# counterparts after LLAMA_SD_MAP — used to build the expected key set.
_LAYER_TENSOR_SUFFIXES = (
    "attn_norm.weight",
    "attn_q.weight",
    "attn_q_norm.weight",
    "attn_k.weight",
    "attn_k_norm.weight",
    "attn_v.weight",
    "attn_output.weight",
    "ffn_gate.weight",
    "ffn_down.weight",
    "ffn_up.weight",
    "ffn_norm.weight",
)


def expected_qwen3_keys(n_layers: int) -> set[str]:
    """HF key set for a qwen3 encoder-only model with ``n_layers`` layers."""
    keys = {"model.embed_tokens.weight", "model.norm.weight"}
    for i in range(n_layers):
        for suffix in _LAYER_TENSOR_SUFFIXES:
            keys.add(f"model.layers.{i}.{_map_single(suffix)}")
    return keys


def _map_single(gguf_suffix: str) -> str:
    """Map one GGUF tensor-name suffix through LLAMA_SD_MAP (ordered)."""
    k = gguf_suffix
    for s, d in LLAMA_SD_MAP.items():
        k = k.replace(s, d)
    return k


# Quant-label tokens for filename parsing, longest-first so e.g. q4_k_m
# wins over q4_k.
_QUANT_LABEL_RE = re.compile(
    r"(bf16|f16|f32|q8_0|q6_k|q5_k_s|q5_k_m|q5_k|q4_k_s|q4_k_m|q4_k|q3_k_l|q3_k_s|q3_k_m|q3_k|q2_k)"
)


@dataclass(frozen=True)
class GGUFMeta:
    """Provenance of a loaded GGUF text encoder.

    ``quant_label`` is derived from the FILENAME (e.g. "q4_k_m") — unique
    per quant recipe — with a qtype-histogram fallback. It feeds the
    artifact dtype label ("gguf_q4_k_m") and report rows.
    """

    quant_label: str
    bpw: float  # exact bits-per-weight from storage bytes
    qtype_histogram: dict[str, int]
    arch: str
    n_tensors: int
    n_params: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "quant_label": self.quant_label,
            "bpw": self.bpw,
            "qtype_histogram": dict(self.qtype_histogram),
            "arch": self.arch,
            "n_tensors": self.n_tensors,
            "n_params": self.n_params,
        }


def _get_meta_int(reader: Any, key: str) -> int | None:
    field = reader.get_field(key)
    if field is None:
        return None
    try:
        return int(field.parts[field.data[-1]].item())
    except (IndexError, AttributeError, TypeError, ValueError) as exc:
        log.warning("Could not read GGUF metadata %s: %s", key, exc)
        return None


def _get_orig_shape(reader: Any, tensor_name: str) -> tuple[int, ...] | None:
    """Original (HF) shape metadata — mirrors ComfyUI-GGUF get_orig_shape."""
    import gguf

    field_key = f"comfy.gguf.orig_shape.{tensor_name}"
    field = reader.get_field(field_key)
    if field is None:
        return None
    if (
        len(field.types) != 2
        or field.types[0] != gguf.GGUFValueType.ARRAY
        or field.types[1] != gguf.GGUFValueType.INT32
    ):
        raise TypeError(f"Bad original shape metadata for {field_key}: expected ARRAY of INT32, got {field.types!r}")
    return tuple(int(field.parts[part_idx][0]) for part_idx in field.data)


def _quant_label_from_name(path: Path, histogram_by_numel: dict[int, int]) -> str:
    """Derive the quant label from the filename, histogram fallback."""
    m = _QUANT_LABEL_RE.search(path.stem.lower())
    if m:
        return m.group(1)
    # Fallback: dominant quantized qtype by element count.
    quantized = {qv: n for qv, n in histogram_by_numel.items() if qv not in gd.TORCH_COMPATIBLE_QTYPES}
    if quantized:
        best = max(quantized, key=lambda qv: quantized[qv])
        return gd.QTYPE_NAMES.get(best, f"qtype{best}").lower()
    return "f16" if 1 in histogram_by_numel else "unknown"


def load_gguf_te_state_dict(
    te_file: str | Path,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[dict[str, torch.Tensor], GGUFMeta]:
    """Load a qwen3 GGUF text encoder into an HF-named state dict.

    Args:
        te_file: Path to the .gguf file (llama.cpp container, arch qwen3).
        dtype: Storage-truth target dtype for all tensors (bf16 default —
            same precision class as the safetensors backends).

    Returns:
        (state_dict, GGUFMeta). Keys match the safetensors ComfyUI layout
        (model.layers.*, model.embed_tokens.weight, model.norm.weight).

    Raises:
        ImportError: gguf package missing.
        FileNotFoundError: te_file missing.
        ValueError: wrong architecture, key-set mismatch, or unsupported qtype.
    """
    try:
        import gguf
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError(
            "The 'gguf' package is required for GGUF text-encoder loading (pip install gguf==0.19.0)"
        ) from exc

    te_file = Path(te_file)
    if not te_file.exists():
        raise FileNotFoundError(f"GGUF text encoder not found: {te_file}")

    if dtype not in (torch.bfloat16, torch.float32):
        # Storage-truth convention (same as the safetensors backends): GGUF
        # weights land at bf16 in the pipeline; fp32 is allowed for debug /
        # golden tests only. Fail loud on anything else.
        raise ValueError(f"GGUF loader supports dtype bf16 (pipeline) or fp32 (debug), got {dtype}")

    reader = gguf.GGUFReader(str(te_file))

    arch_field = reader.get_field("general.architecture")
    if arch_field is None:
        raise ValueError(f"{te_file.name}: missing general.architecture metadata")
    arch = bytes(arch_field.parts[arch_field.data[-1]]).decode("utf-8")
    if arch != "qwen3":
        raise ValueError(
            f"{te_file.name}: architecture {arch!r} not supported — this loader "
            "replicates ComfyUI-GGUF's qwen3 path only (fidelity contract)"
        )

    sd: dict[str, torch.Tensor] = {}
    qtype_count: dict[str, int] = {}
    numel_by_qtype: dict[int, int] = {}
    total_storage_bytes = 0
    total_numel = 0

    # Metadata read BEFORE the reader is released.
    n_layers_meta = _get_meta_int(reader, "qwen3.block_count")
    vocab_meta = _get_meta_int(reader, "qwen3.vocab_size")  # 151936 for qwen3-4b
    hidden_meta = _get_meta_int(reader, "qwen3.embedding_length")  # 2560 (hidden dim, NOT vocab)

    try:
        for tensor in reader.tensors:
            name = tensor.name
            if tensor.tensor_type is None:  # assert would be stripped under -O
                raise ValueError(f"{te_file.name}: tensor {name} has no qtype")
            qv = int(tensor.tensor_type)
            qname = gd.QTYPE_NAMES.get(qv, f"qtype{qv}")
            qtype_count[qname] = qtype_count.get(qname, 0) + 1

            # Original HF shape: metadata preferred, reversed storage fallback.
            oshape = _get_orig_shape(reader, name)
            shape_source = "metadata"
            if oshape is None:
                oshape = tuple(int(v) for v in reversed(tensor.shape))
                shape_source = "reversed"

            numel = int(np.prod(oshape)) if oshape else int(tensor.data.shape[0])
            numel_by_qtype[qv] = numel_by_qtype.get(qv, 0) + numel
            total_storage_bytes += int(tensor.data.nbytes)
            total_numel += numel

            # RAW storage tensor — never reshaped before dequantization.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
                data = torch.from_numpy(tensor.data)

            if name == "token_embd.weight" and oshape[0] >= 64 * 1024:
                # ComfyUI fp16 detour for large embedding tables (OOM
                # workaround there) — fp16 intermediate, then target dtype.
                out = gd.dequantize(data, qv, oshape=oshape, dtype=torch.float16).to(dtype)
                log.debug("token_embd: fp16 detour (vocab %d) -> %s", oshape[0], dtype)
            else:
                out = gd.dequantize(data, qv, oshape=oshape, dtype=dtype)
            if out.dtype == data.dtype:
                out = out.clone()  # detach from the mmap'd reader buffer
            sd[name] = out
            if shape_source == "metadata":
                log.debug("tensor %s used orig_shape metadata %s", name, oshape)
    finally:
        # Release the mmap'd reader (output tensors own their memory).
        del reader

    # Ordered name remap (order-preserving vendored map).
    sd = sd_map_replace(sd, LLAMA_SD_MAP)

    # Fail-loud key-set validation against the file's own block count.
    n_layers = n_layers_meta if n_layers_meta is not None else _infer_layer_count(sd)
    expected = expected_qwen3_keys(n_layers)
    actual = set(sd.keys())
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"{te_file.name}: GGUF key set does not match the qwen3 layout "
            f"({n_layers} layers, {len(expected)} expected keys). "
            f"Missing ({len(missing)}): {missing[:6]}{'...' if len(missing) > 6 else ''} "
            f"Unexpected ({len(extra)}): {extra[:6]}{'...' if len(extra) > 6 else ''}"
        )
    if "lm_head.weight" in actual:
        raise ValueError(f"{te_file.name}: unexpected lm_head.weight — encoder-only layout required")

    # Embedding dims vs file metadata (transpose-gotcha guard):
    # [vocab_size, embedding_length] — embedding_length is the HIDDEN dim.
    emb = sd["model.embed_tokens.weight"]
    if emb.ndim != 2:
        raise ValueError(f"{te_file.name}: model.embed_tokens.weight is {emb.ndim}-D — expected 2-D")
    if vocab_meta is not None and emb.shape[0] != vocab_meta:
        raise ValueError(
            f"{te_file.name}: model.embed_tokens.weight shape {tuple(emb.shape)} inconsistent with "
            f"qwen3.vocab_size={vocab_meta} — transposed storage handling bug?"
        )
    if hidden_meta is not None and emb.shape[1] != hidden_meta:
        raise ValueError(
            f"{te_file.name}: model.embed_tokens.weight shape {tuple(emb.shape)} inconsistent with "
            f"qwen3.embedding_length={hidden_meta} — transposed storage handling bug?"
        )

    bpw = (total_storage_bytes * 8.0 / total_numel) if total_numel else 0.0
    quant_label = _quant_label_from_name(te_file, numel_by_qtype)

    # Filename/content consistency guard (reviewer #3): a mislabeled file
    # would make the artifact dtype label lie while bpw tells the truth.
    quantized = {q: n for q, n in numel_by_qtype.items() if q not in gd.TORCH_COMPATIBLE_QTYPES}
    if quantized:
        dominant = gd.QTYPE_NAMES.get(max(quantized, key=lambda q: quantized[q]), "").lower()
        family = quant_label.rsplit("_", 1)[0] if quant_label.startswith("q") else quant_label
        if family in {"q3", "q4", "q5", "q6", "q8"}:
            family = family + "_k" if family != "q8" else "q8_0"
        if dominant and family != dominant and not dominant.startswith(family.split("_")[0]):
            log.warning(
                "GGUF filename label %r inconsistent with dominant qtype %r — check the file name!",
                quant_label,
                dominant,
            )

    meta = GGUFMeta(
        quant_label=quant_label,
        bpw=round(bpw, 3),
        qtype_histogram=dict(sorted(qtype_count.items(), key=lambda kv: -kv[1])),
        arch=arch,
        n_tensors=len(sd),
        n_params=total_numel,
    )
    log.info(
        "Loaded GGUF TE %s: label=%s bpw=%.2f tensors=%d params=%.2fB qtypes=%s",
        te_file.name,
        meta.quant_label,
        meta.bpw,
        meta.n_tensors,
        meta.n_params / 1e9,
        meta.qtype_histogram,
    )
    return sd, meta


def _infer_layer_count(sd: dict[str, torch.Tensor]) -> int:
    """Infer layer count from mapped keys when metadata is absent."""
    counts = [int(k.split(".")[2]) for k in sd if k.startswith("model.layers.") and k.count(".") >= 3]
    return (max(counts) + 1) if counts else 0
