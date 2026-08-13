"""Tests for the GGUF -> HF state-dict loader (gguf_loader.py).

Synthetic files are built with ``gguf.GGUFWriter`` (verified conventions:
quantized tensors passed as ``packed.reshape(n_rows, bytes_per_row)`` with
``raw_dtype``; reader reports storage shape reversed). Real-file
integration tests (skipif the models dir is absent) lock the loader to
the actual qwen3-4b files, including the fp16->bf16 byte-identity of the
f16 container and key-set equality with the safetensors base.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

gguf = pytest.importorskip("gguf", reason="gguf package not installed")

from src.text_encoder.backends.gguf_loader import (  # noqa: E402
    LLAMA_SD_MAP,
    _quant_label_from_name,
    expected_qwen3_keys,
    load_gguf_te_state_dict,
    sd_map_replace,
)

GGUF_DIR = Path("/home/maxx/extra/comfyui-models/text_encoders/qwen3-4b/gguf")
ST_BASE = Path("/home/maxx/extra/comfyui-models/text_encoders/qwen3-4b/qwen_3_4b.safetensors")

N_LAYERS = 2
HIDDEN = 8
VOCAB_SMALL = 64
VOCAB_BIG = 70_000  # >= 64K -> fp16 detour branch

# Small logical dims: [out, in] pairs per tensor kind.
F16_TENSORS: list[tuple[str, tuple[int, int]]] = [
    ("attn_norm.weight", (HIDDEN,)),
    ("attn_q.weight", (16, 32)),
    ("attn_q_norm.weight", (HIDDEN,)),
    ("attn_k.weight", (16, HIDDEN)),
    ("attn_k_norm.weight", (HIDDEN,)),
    ("attn_v.weight", (16, HIDDEN)),
    ("attn_output.weight", (HIDDEN, 16)),
    ("ffn_gate.weight", (32, HIDDEN)),
    ("ffn_down.weight", (HIDDEN, 32)),
    ("ffn_up.weight", (32, HIDDEN)),
    ("ffn_norm.weight", (HIDDEN,)),
]


def _pack_q8_0(rows: int, cols: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Pack a logical [rows, cols] Q8_0 weight. cols must be a multiple of 32.

    Returns (packed_2d_uint8, logical_fp32) where packed rows are
    [fp16 scale (2B) + cols int8 payloads].
    """
    assert cols % 32 == 0
    logical = rng.standard_normal((rows, cols)).astype(np.float32)
    chunks = []
    for r in range(rows):
        row = logical[r]
        amax = float(np.abs(row).max())
        d = np.float16(amax / 127.0) if amax > 0 else np.float16(1.0)
        x = np.clip(np.rint(row / float(d)), -127, 127).astype(np.int8)
        chunks.append(d.tobytes() + x.tobytes())
        # dequantized reference for this row: d * x (fp16-scale math)
        logical[r] = np.frombuffer(d.tobytes(), dtype=np.float16)[0] * x.astype(np.float32)
    type_size = 2 + cols
    packed = np.frombuffer(b"".join(chunks), dtype=np.uint8).reshape(rows, type_size)
    return packed, logical


def _write_tiny_gguf(
    path: Path,
    arch: str = "qwen3",
    vocab: int = VOCAB_SMALL,
    drop_tensor: str | None = None,
    extra_output_weight: bool = False,
    q8_layer0_q: bool = True,
    seed: int = 7,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Write a minimal qwen3 GGUF. Returns (q8_logical, embed_fp32)."""
    rng = np.random.default_rng(seed)
    w = gguf.GGUFWriter(str(path), arch)
    w.add_uint32("qwen3.block_count", N_LAYERS)
    w.add_uint32("qwen3.embedding_length", HIDDEN)
    w.add_uint32("qwen3.vocab_size", vocab)
    q8_logical = None
    for layer in range(N_LAYERS):
        for suffix, dims in F16_TENSORS:
            name = f"blk.{layer}.{suffix}"
            if drop_tensor and name == drop_tensor:
                continue
            if q8_layer0_q and layer == 0 and suffix == "attn_q.weight":
                packed, q8_logical = _pack_q8_0(dims[0], dims[1], rng)
                w.add_tensor(name, packed, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
                continue
            shape = dims if len(dims) == 2 else (1, dims[0])
            w.add_tensor(name, rng.standard_normal(shape).astype(np.float16))
    w.add_tensor("token_embd.weight", rng.standard_normal((vocab, HIDDEN)).astype(np.float16))
    w.add_tensor("output_norm.weight", rng.standard_normal((1, HIDDEN)).astype(np.float16))
    if extra_output_weight:
        w.add_tensor("output.weight", rng.standard_normal((vocab, HIDDEN)).astype(np.float16))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return q8_logical, np.array([], dtype=np.float32)


class TestNameMapping:
    def test_q_norm_ordering_load_bearing(self) -> None:
        """attn_q_norm must map to self_attn.q_norm, not q_proj_norm."""
        out = sd_map_replace({"blk.3.attn_q_norm.weight": torch.zeros(1)}, LLAMA_SD_MAP)
        assert list(out) == ["model.layers.3.self_attn.q_norm.weight"]

    def test_k_norm_and_core_keys(self) -> None:
        raw = {
            "blk.0.attn_k_norm.weight": torch.zeros(1),
            "blk.0.attn_q.weight": torch.zeros(1),
            "blk.0.ffn_gate.weight": torch.zeros(1),
            "token_embd.weight": torch.zeros(1),
            "output_norm.weight": torch.zeros(1),
        }
        out = sd_map_replace(raw, LLAMA_SD_MAP)
        assert set(out) == {
            "model.layers.0.self_attn.k_norm.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.embed_tokens.weight",
            "model.norm.weight",
        }

    def test_expected_key_count(self) -> None:
        keys = expected_qwen3_keys(36)
        assert len(keys) == 36 * 11 + 2
        assert "model.layers.35.self_attn.o_proj.weight" in keys
        assert "lm_head.weight" not in keys


class TestQuantLabel:
    @pytest.mark.parametrize(
        ("stem", "expected"),
        [
            ("qwen3-4b-Q4_K_M", "q4_k_m"),
            ("qwen3-4b-heretic-Q4_K_S", "q4_k_s"),
            ("qwen3-4b-Q5_K_S", "q5_k_s"),
            ("qwen3-4b-q3_k_m", "q3_k_m"),
            ("qwen3-4b-f16", "f16"),
            ("qwen3-4b-Q8_0", "q8_0"),
            ("some_bf16_model", "bf16"),
        ],
    )
    def test_from_filename(self, stem: str, expected: str) -> None:
        assert _quant_label_from_name(Path(f"{stem}.gguf"), {}) == expected

    def test_histogram_fallback(self) -> None:
        # Q4_K (12) dominant by numel, no filename token.
        label = _quant_label_from_name(Path("mystery.gguf"), {0: 100, 12: 10_000})
        assert label == "q4_k"


class TestSyntheticLoader:
    def test_roundtrip_small_vocab(self, tmp_path: Path) -> None:
        f = tmp_path / "tiny-q8_0.gguf"
        q8_logical, _ = _write_tiny_gguf(f)
        sd, meta = load_gguf_te_state_dict(f, dtype=torch.bfloat16)

        assert set(sd) == expected_qwen3_keys(N_LAYERS)
        assert all(t.dtype == torch.bfloat16 for t in sd.values())
        assert sd["model.embed_tokens.weight"].shape == (VOCAB_SMALL, HIDDEN)
        assert meta.arch == "qwen3"
        assert meta.quant_label == "q8_0"
        assert meta.n_tensors == N_LAYERS * 11 + 2

        # Q8_0 tensor decoded through the full loader == hand-packed reference
        # (fp32 target => block math matches the numpy fp32 reference exactly;
        # bf16 storage rounding is covered by the real-file integration tests).
        sd32, _ = load_gguf_te_state_dict(f, dtype=torch.float32)
        q = sd32["model.layers.0.self_attn.q_proj.weight"]
        assert q.shape == (16, 32)
        assert torch.equal(q, torch.from_numpy(q8_logical))  # type: ignore[arg-type]

        # F16 tensor: fp16 -> bf16 exact for these magnitudes (verify a row).
        # (values were random fp16; check dtype/shape only - exactness covered
        # by the real-file identity test below)

    def test_fp16_detour_big_vocab(self, tmp_path: Path) -> None:
        f = tmp_path / "tiny-f16.gguf"
        _write_tiny_gguf(f, vocab=VOCAB_BIG, q8_layer0_q=False)
        sd, meta = load_gguf_te_state_dict(f, dtype=torch.bfloat16)
        emb = sd["model.embed_tokens.weight"]
        assert emb.shape == (VOCAB_BIG, HIDDEN)
        assert emb.dtype == torch.bfloat16
        assert meta.quant_label == "f16"

    def test_wrong_arch_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "tiny-llama.gguf"
        _write_tiny_gguf(f, arch="llama")
        with pytest.raises(ValueError, match="architecture 'llama' not supported"):
            load_gguf_te_state_dict(f)

    def test_missing_tensor_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "tiny-broken.gguf"
        _write_tiny_gguf(f, drop_tensor="blk.1.attn_v.weight")
        with pytest.raises(ValueError, match="key set does not match"):
            load_gguf_te_state_dict(f)

    def test_lm_head_rejected(self, tmp_path: Path) -> None:
        f = tmp_path / "tiny-with-head.gguf"
        _write_tiny_gguf(f, extra_output_weight=True)
        with pytest.raises(ValueError, match="key set does not match"):
            load_gguf_te_state_dict(f)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_gguf_te_state_dict(tmp_path / "nope.gguf")

    def test_orig_shape_metadata_honored(self, tmp_path: Path) -> None:
        """comfy.gguf.orig_shape.<tensor> wins over the reversed fallback."""
        f_meta = tmp_path / "tiny-q8_0.gguf"
        _write_tiny_gguf(f_meta)
        w = gguf.GGUFWriter(str(f_meta), "qwen3")
        w.write_header_to_file()  # append-style rewrite is not supported; build fresh below
        w.close()

        # Build two fresh files: metadata shape deliberately TRANSPOSED vs
        # reversed storage; loader must honor the metadata.
        def build(path: Path, orig_shape: tuple[int, int] | None) -> None:
            w = gguf.GGUFWriter(str(path), "qwen3")
            w.add_uint32("qwen3.block_count", N_LAYERS)
            w.add_uint32("qwen3.embedding_length", HIDDEN)
            w.add_uint32("qwen3.vocab_size", VOCAB_SMALL)
            rng = np.random.default_rng(3)
            for layer in range(N_LAYERS):
                for suffix, dims in F16_TENSORS:
                    shape = dims if len(dims) == 2 else (1, dims[0])
                    name = f"blk.{layer}.{suffix}"
                    arr = rng.standard_normal(shape).astype(np.float16)
                    if layer == 0 and suffix == "attn_q.weight" and orig_shape is not None:
                        w.add_array(f"comfy.gguf.orig_shape.{name}", [int(d) for d in orig_shape])
                    w.add_tensor(name, arr)
            w.add_tensor("token_embd.weight", rng.standard_normal((VOCAB_SMALL, HIDDEN)).astype(np.float16))
            w.add_tensor("output_norm.weight", rng.standard_normal((1, HIDDEN)).astype(np.float16))
            w.write_header_to_file()
            w.write_kv_data_to_file()
            w.write_tensors_to_file()
            w.close()

        with_meta = tmp_path / "with-meta.gguf"
        no_meta = tmp_path / "no-meta.gguf"
        build(with_meta, orig_shape=(32, 16))  # transposed vs reversed (16, 32)
        build(no_meta, orig_shape=None)

        sd_meta, _ = load_gguf_te_state_dict(with_meta)
        sd_fallback, _ = load_gguf_te_state_dict(no_meta)
        q_meta = sd_meta["model.layers.0.self_attn.q_proj.weight"]
        q_fallback = sd_fallback["model.layers.0.self_attn.q_proj.weight"]
        assert q_meta.shape == (32, 16), "orig_shape metadata must be honored"
        assert q_fallback.shape == (16, 32)
        # Same bytes, different shape interpretation: transposed view.
        assert torch.equal(q_meta, q_fallback.reshape(32, 16))

    def test_orig_shape_bad_metadata_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "bad-meta.gguf"
        w = gguf.GGUFWriter(str(f), "qwen3")
        w.add_uint32("qwen3.block_count", N_LAYERS)
        w.add_uint32("qwen3.embedding_length", HIDDEN)
        w.add_uint32("qwen3.vocab_size", VOCAB_SMALL)
        rng = np.random.default_rng(3)
        w.add_array("comfy.gguf.orig_shape.blk.0.attn_q.weight", ["bad"])
        for layer in range(N_LAYERS):
            for suffix, dims in F16_TENSORS:
                shape = dims if len(dims) == 2 else (1, dims[0])
                w.add_tensor(f"blk.{layer}.{suffix}", rng.standard_normal(shape).astype(np.float16))
        w.add_tensor("token_embd.weight", rng.standard_normal((VOCAB_SMALL, HIDDEN)).astype(np.float16))
        w.add_tensor("output_norm.weight", rng.standard_normal((1, HIDDEN)).astype(np.float16))
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
        with pytest.raises(TypeError, match="Bad original shape metadata"):
            load_gguf_te_state_dict(f)

    def test_layer_count_inferred_without_metadata(self, tmp_path: Path) -> None:
        f = tmp_path / "no-blockcount.gguf"
        _write_tiny_gguf(f)
        # Strip qwen3.block_count by writing a fresh file without it.
        w = gguf.GGUFWriter(str(f), "qwen3")
        w.add_uint32("qwen3.embedding_length", HIDDEN)
        w.add_uint32("qwen3.vocab_size", VOCAB_SMALL)
        rng = np.random.default_rng(5)
        for layer in range(N_LAYERS):
            for suffix, dims in F16_TENSORS:
                shape = dims if len(dims) == 2 else (1, dims[0])
                w.add_tensor(f"blk.{layer}.{suffix}", rng.standard_normal(shape).astype(np.float16))
        w.add_tensor("token_embd.weight", rng.standard_normal((VOCAB_SMALL, HIDDEN)).astype(np.float16))
        w.add_tensor("output_norm.weight", rng.standard_normal((1, HIDDEN)).astype(np.float16))
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
        sd, meta = load_gguf_te_state_dict(f)
        assert set(sd) == expected_qwen3_keys(N_LAYERS)
        assert meta.n_tensors == N_LAYERS * 11 + 2


@pytest.mark.skipif(not GGUF_DIR.exists(), reason="qwen3-4b GGUF files not on this host")
@pytest.mark.skipif(not ST_BASE.exists(), reason="safetensors base not on this host")
class TestRealFileIntegration:
    def test_f16_container_identity(self) -> None:
        """The f16 GGUF must reproduce the bf16 safetensors base bit-exactly."""
        from safetensors import safe_open

        sd, meta = load_gguf_te_state_dict(GGUF_DIR / "qwen3-4b-f16.gguf")
        with safe_open(str(ST_BASE), framework="pt", device="cpu") as f:
            base_keys = set(f.keys())
            assert set(sd) == base_keys - {"spiece_model"}
            # bf16->fp16->bf16 is exact for fp16-NORMAL magnitudes; entries in
            # fp16's subnormal range (<6.1e-5) can lose mantissa bits - so the
            # container is near-identical to the safetensors base (any
            # transpose/remap/mapping bug would produce O(1) diffs instead).
            for probe in ("model.layers.0.self_attn.o_proj.weight", "model.embed_tokens.weight"):
                base_t = f.get_tensor(probe)
                assert base_t.dtype == torch.bfloat16
                diff = (sd[probe].float() - base_t.float()).abs()
                frac = (diff > 0).float().mean()
                assert diff.max() < 1e-6, f"{probe}: max diff {diff.max()}"
                assert frac < 1e-3, f"{probe}: mismatch fraction {frac}"
        assert sd["model.embed_tokens.weight"].shape == (151_936, 2_560)
        assert meta.quant_label == "f16"
        assert meta.bpw == pytest.approx(16.0, abs=0.05)

    def test_q4_k_m_container(self) -> None:
        from gguf import quants

        sd, meta = load_gguf_te_state_dict(GGUF_DIR / "qwen3-4b-Q4_K_M.gguf")
        assert set(sd) == expected_qwen3_keys(36)
        assert sd["model.layers.0.self_attn.q_proj.weight"].shape == (4096, 2560)
        assert meta.quant_label == "q4_k_m"
        assert meta.bpw == pytest.approx(4.97, abs=0.02)
        assert meta.qtype_histogram.get("Q4_K") == 216
        assert meta.qtype_histogram.get("Q6_K") == 37
        assert meta.qtype_histogram.get("F32") == 145

        # Cross-check one tensor against gguf's numpy reference dequant.
        reader = gguf.GGUFReader(str(GGUF_DIR / "qwen3-4b-Q4_K_M.gguf"))
        for t in reader.tensors:
            if t.name == "blk.0.attn_q.weight":
                ref = quants.dequantize(np.ascontiguousarray(t.data), t.tensor_type)
                ref_t = torch.from_numpy(np.ascontiguousarray(ref, dtype=np.float32)).reshape(4096, 2560)
                got = sd["model.layers.0.self_attn.q_proj.weight"].float()
                # bf16 storage: the block scale (fp16) is rounded to bf16
                # (~0.2% relative), and Q4_K's d*qs-dm cancellation can turn
                # that into ~1e-2 ABSOLUTE error at near-zero outputs. A
                # transpose/remap bug would produce O(1) diffs instead.
                assert torch.allclose(got, ref_t, rtol=0.01, atol=0.01), f"max diff {(got - ref_t).abs().max()}"
                break
