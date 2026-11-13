"""Tests for the Z-Image backend (no real weights required).

Covers: registry, stream specs (declared == _build_spec — the spec-drift
contract guard), DiT conditioning loading (missing/valid keys), fake-quant,
tokenize, the dtype-label ladder via _make_artifact (incl. the unified
fakequant > gguf precedence), and the full encode() path (all stages,
cap-pad masking, stage filtering, spec equality) against fake
model/tokenizer/conditioning modules mirroring the comfy structure.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from src.text_encoder.artifact import Stage, Stream
from src.text_encoder.backends import get_backend
from src.text_encoder.backends.gguf_loader import GGUFMeta
from src.text_encoder.backends.z_image import (
    CONSUMED_LAYER_IDX,
    CONTEXT_DIM,
    HIDDEN_DIM,
    N_LAYERS,
    PAD_TOKENS_MULTIPLE,
    ZImageBackend,
    _build_spec,
    _load_dit_conditioning_sd,
)


class _FakeLayer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class _FakeTransformerModel(nn.Module):
    def __init__(self, dim: int, n_layers: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(16, dim)
        self.layers = nn.ModuleList([_FakeLayer(dim) for _ in range(n_layers)])


class _FakeTransformer(nn.Module):
    def __init__(self, dim: int, n_layers: int) -> None:
        super().__init__()
        self.model = _FakeTransformerModel(dim, n_layers)


class _FakeQwen3(nn.Module):
    """Mirrors the comfy z_image model nesting: .qwen3_4b.transformer.model."""

    def __init__(self, dim: int, n_layers: int) -> None:
        super().__init__()
        self.transformer = _FakeTransformer(dim, n_layers)


class _FakeModel:
    def __init__(self, seq_len: int) -> None:
        self.qwen3_4b = _FakeQwen3(HIDDEN_DIM, N_LAYERS)
        self._seq_len = seq_len

    def encode_token_weights(self, tokens: Any) -> tuple[torch.Tensor, None, dict]:  # noqa: ANN401
        # run a real forward through the fake layers so HookManager fires
        x = torch.randn(1, self._seq_len, HIDDEN_DIM)
        for layer in self.qwen3_4b.transformer.model.layers:
            x = layer(x)
        # ZImageTokenizer pads to min_length=1 only: seq length == real tokens
        return x, None, {}


class _FakeTokenizer:
    def __init__(self, n_tokens: int = 9) -> None:
        self.n_tokens = n_tokens

    def tokenize_with_weights(self, text: str) -> dict:  # noqa: ANN401
        entries = [(5 + i, 1.0) for i in range(self.n_tokens)]
        return {"qwen3_4b": [entries]}


class _FakeCapEmbedder(nn.Module):
    """RMSNorm+Linear stand-in: [1, L, 2560] -> [1, L, 3840] (pad features)."""

    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))  # parameters() must be non-empty

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.pad(x, (0, CONTEXT_DIM - x.shape[-1]))


class _FakeRope(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, pos_ids: torch.Tensor) -> torch.Tensor:
        # shape only needs dim 1 (movedim(1, 2) in the caller)
        return torch.ones(1, pos_ids.shape[1], 2, 4)


class _FakeRefinerLayer(nn.Module):
    """Stand-in for comfy's JointTransformerBlock call signature."""

    def forward(
        self,
        x: torch.Tensor,
        mod: Any = None,  # noqa: ARG002
        freqs_cis: Any = None,  # noqa: ARG002
        transformer_options: Any = None,  # noqa: ARG002
    ) -> torch.Tensor:
        return x


def _make_backend(seq_len: int = 9) -> ZImageBackend:
    b = ZImageBackend()
    b.model = _FakeModel(seq_len)
    b.tokenizer = _FakeTokenizer(seq_len)
    b.cap_embedder = _FakeCapEmbedder()
    b.rope_embedder = _FakeRope()
    b.context_refiner = nn.ModuleList([_FakeRefinerLayer(), _FakeRefinerLayer()])
    b.cap_pad_token = torch.randn(CONTEXT_DIM)
    b.ckpt_id = "fake"
    return b


def _fake_gguf_meta(label: str = "q4_k_m") -> GGUFMeta:
    return GGUFMeta(
        quant_label=label,
        bpw=4.8,
        qtype_histogram={"Q4_K": 100},
        arch="qwen3",
        n_tensors=400,
        n_params=4_000_000_000,
    )


class TestRegistry:
    def test_zimage_registered(self) -> None:
        backend = get_backend("zimage")
        assert isinstance(backend, ZImageBackend)
        assert backend.name == "zimage"


class TestStreams:
    def test_names_match_constants(self) -> None:
        b = get_backend("zimage")
        names = [s.name for s in b.streams]
        assert names[:N_LAYERS] == [f"layer_{i:02d}" for i in range(N_LAYERS)]
        assert names[N_LAYERS:] == ["consumed_hidden", "cap_embedded", "refined_context"]

    def test_spec_fields(self) -> None:
        b = get_backend("zimage")
        by_name = {s.name: s for s in b.streams}
        assert by_name["layer_00"].dim == HIDDEN_DIM
        assert by_name["layer_00"].pooling.value == "mean"
        assert by_name["layer_00"].stage == Stage.RAW_HIDDEN
        assert by_name["layer_35"].layers == (35,)
        assert by_name["consumed_hidden"].layers == (CONSUMED_LAYER_IDX,)
        assert by_name["consumed_hidden"].pooling.value == "none"
        assert by_name["cap_embedded"].stage == Stage.PROJECTED
        assert by_name["cap_embedded"].dim == CONTEXT_DIM
        assert by_name["refined_context"].stage == Stage.CONNECTOR_OUT
        assert by_name["refined_context"].dim == CONTEXT_DIM
        assert {s.stage for s in b.streams} == {Stage.RAW_HIDDEN, Stage.PROJECTED, Stage.CONNECTOR_OUT}

    def test_property_matches_build_spec(self) -> None:
        # the contract guard: streams() is exactly [_build_spec(name)]
        b = get_backend("zimage")
        expected = [_build_spec(f"layer_{i:02d}") for i in range(N_LAYERS)]
        expected += [_build_spec("consumed_hidden"), _build_spec("cap_embedded"), _build_spec("refined_context")]
        assert b.streams == expected

    def test_build_spec_layer_name(self) -> None:
        spec = _build_spec("layer_07")
        assert spec.name == "layer_07"
        assert spec.layers == (7,)
        assert spec.pooling.value == "mean"


class TestDitConditioningLoad:
    def _write_dit(self, path: Any, keys: list[str]) -> None:  # noqa: ANN401
        sd = {k: torch.randn(2, 2) for k in keys}
        save_file(sd, str(path))

    def test_valid_keys_load_and_filter(self, tmp_path: Any) -> None:  # noqa: ANN401
        p = tmp_path / "dit.safetensors"
        self._write_dit(
            p,
            [
                "cap_embedder.0.weight",
                "context_refiner.0.attention.qkv.weight",
                "cap_pad_token",
                "layers.0.attention.out.weight",  # non-conditioning DiT key
            ],
        )
        sd = _load_dit_conditioning_sd(p)
        assert set(sd.keys()) == {
            "cap_embedder.0.weight",
            "context_refiner.0.attention.qkv.weight",
            "cap_pad_token",
        }

    def test_missing_keys_raise(self, tmp_path: Any) -> None:  # noqa: ANN401
        p = tmp_path / "dit.safetensors"
        self._write_dit(p, ["cap_embedder.0.weight", "cap_pad_token"])
        with pytest.raises(ValueError, match="missing Z-Image conditioning keys"):
            _load_dit_conditioning_sd(p)

    def test_missing_file_raises(self, tmp_path: Any) -> None:  # noqa: ANN401
        with pytest.raises(FileNotFoundError):
            _load_dit_conditioning_sd(tmp_path / "nope.safetensors")


class TestFakeQuant:
    def test_apply_sets_flag_and_changes_weights(self) -> None:
        b = _make_backend()
        assert b._te_fake_quant is None
        b.apply_fake_quant("int8")
        assert b._te_fake_quant == "int8"

    def test_not_loaded_raises(self) -> None:
        b = ZImageBackend()
        with pytest.raises(RuntimeError, match="not loaded"):
            b.apply_fake_quant("int8")


class TestTokenize:
    def test_ids_aligned(self) -> None:
        b = _make_backend(seq_len=7)
        out = b.tokenize(["a cat"])
        assert out["input_ids"] == [[5, 6, 7, 8, 9, 10, 11]]
        assert out["n_tokens"] == [7]

    def test_not_loaded_raises(self) -> None:
        b = ZImageBackend()
        with pytest.raises(RuntimeError, match="not loaded"):
            b.tokenize(["x"])


class TestEncode:
    def test_encode_all_stages_shapes(self) -> None:
        b = _make_backend(seq_len=9)
        arts = b.encode(["a cat"], ["cat"])
        by_stage = {a.stage: a for a in arts}
        assert set(by_stage) == {Stage.RAW_HIDDEN, Stage.PROJECTED, Stage.CONNECTOR_OUT}

        raw = by_stage[Stage.RAW_HIDDEN]
        assert len(raw.streams) == N_LAYERS + 1
        # pooled streams are canonicalized to [n, 1, d] by Stream.__post_init__
        assert raw.streams["layer_00"].activations.shape == (1, 1, HIDDEN_DIM)
        cons = raw.streams["consumed_hidden"]
        assert cons.activations.shape == (1, 9, HIDDEN_DIM)
        assert cons.mask is not None
        assert cons.mask.shape == (1, 9)
        assert int(cons.mask.sum()) == 9  # no cap-pad at the raw stage

        proj = by_stage[Stage.PROJECTED]
        assert proj.streams["cap_embedded"].activations.shape == (1, 9, CONTEXT_DIM)
        ref = by_stage[Stage.CONNECTOR_OUT]
        # 9 real tokens padded to the 32-multiple with cap_pad_token
        assert ref.streams["refined_context"].activations.shape == (1, PAD_TOKENS_MULTIPLE, CONTEXT_DIM)

    def test_refined_mask_excludes_cap_pad(self) -> None:
        # THE forensic property: cap-pad positions are masked out of metrics
        b = _make_backend(seq_len=9)
        arts = b.encode(["x"], ["cat"])
        ref = next(a for a in arts if a.stage == Stage.CONNECTOR_OUT)
        mask = ref.streams["refined_context"].mask
        assert mask is not None
        assert mask[0, :9].tolist() == [1] * 9
        assert mask[0, 9:].tolist() == [0] * (PAD_TOKENS_MULTIPLE - 9)

    def test_spec_contract_in_artifacts(self) -> None:
        # artifact per-stream spec must equal the declared spec (drift guard)
        b = _make_backend(seq_len=5)
        declared = {s.name: s for s in b.streams}
        for art in b.encode(["x"], ["cat"]):
            for name, stream in art.streams.items():
                assert stream.spec == declared[name]

    def test_stage_filtering(self) -> None:
        b = _make_backend(seq_len=6)
        arts = b.encode(["x"], ["cat"], stages=[Stage.RAW_HIDDEN])
        assert {a.stage for a in arts} == {Stage.RAW_HIDDEN}
        arts = b.encode(["x"], ["cat"], stages=[Stage.PROJECTED])
        assert {a.stage for a in arts} == {Stage.PROJECTED}

    def test_batch_padding_across_prompt_lengths(self) -> None:
        # two prompts of different lengths (7 and 5): fake tokenizer returns
        # per-prompt lengths via distinct backends — use one backend, one
        # length; instead pad check: single prompt of length 1 (minimal)
        b = _make_backend(seq_len=1)
        arts = b.encode(["x", "y"], ["cat", "dog"])
        raw = next(a for a in arts if a.stage == Stage.RAW_HIDDEN)
        assert raw.streams["consumed_hidden"].activations.shape[0] == 2
        ref = next(a for a in arts if a.stage == Stage.CONNECTOR_OUT)
        # 1 real token -> padded to 32; both rows masked identically
        assert ref.streams["refined_context"].mask is not None
        assert int(ref.streams["refined_context"].mask.sum()) == 2

    def test_not_loaded_raises(self) -> None:
        b = ZImageBackend()
        with pytest.raises(RuntimeError, match="not loaded"):
            b.encode(["x"], ["cat"])

    def test_artifact_provenance(self) -> None:
        b = _make_backend(seq_len=4)
        for art in b.encode(["x"], ["cat"]):
            assert art.model == "zimage"
            assert art.backend == "ZImageBackend"
            assert art.ckpt_id == "fake"
            assert art.device == "cpu"
            assert art.dtype == "comfy_bf16"
            assert art.meta["quantized"] is False
            assert art.meta["pad_tokens_multiple"] == PAD_TOKENS_MULTIPLE


class TestDtypeLabelLadder:
    """_make_artifact must use the shared dtype_label precedence (Bugs #2)."""

    @staticmethod
    def _artifact(b: ZImageBackend) -> Any:  # noqa: ANN401
        spec = _build_spec("cap_embedded")
        stream = Stream(
            name="cap_embedded",
            spec=spec,
            activations=torch.zeros(1, 4, CONTEXT_DIM),
            mask=torch.ones(1, 4, dtype=torch.int),
        )
        return b._make_artifact(Stage.PROJECTED, {"cap_embedded": stream}, ["p"], ["cat"])

    def test_default_bf16(self) -> None:
        b = _make_backend()
        assert self._artifact(b).dtype == "comfy_bf16"

    def test_gguf(self) -> None:
        b = _make_backend()
        b._gguf_meta = _fake_gguf_meta("q4_k_m")
        art = self._artifact(b)
        assert art.dtype == "gguf_q4_k_m"
        assert art.meta["gguf"]["quant_label"] == "q4_k_m"  # provenance preserved

    def test_fakequant(self) -> None:
        b = _make_backend()
        b._te_fake_quant = "fp8"
        assert self._artifact(b).dtype == "fakequant_fp8"

    def test_fakequant_outranks_gguf_unified_precedence(self) -> None:
        # regression for audit Bugs #2: z_image used to label gguf_<q> here
        # while krea2 labeled fakequant_<q> for the same arm state
        b = _make_backend()
        b._te_fake_quant = "int8"
        b._gguf_meta = _fake_gguf_meta()
        assert self._artifact(b).dtype == "fakequant_int8"

    def test_comfy_int8_convrot(self) -> None:
        b = _make_backend()
        b._quantized = True
        assert self._artifact(b).dtype == "comfy_int8_convrot"


class TestClose:
    def test_close_frees_modules(self) -> None:
        b = _make_backend()
        b.close()
        assert b.model is None
        assert b.tokenizer is None
        assert b.cap_embedder is None
        assert b.rope_embedder is None
        assert b.context_refiner is None
        assert b.cap_pad_token is None
        assert b._te_sd is None
        assert b._gguf_meta is None
