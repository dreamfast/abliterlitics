"""Tests for the MiniMax H3 backend (no real 32B weights required).

Covers: registry, stream specs, chat-template guard, DiT conditioning
loading (missing/valid keys), and the full encode() path against a fake
model/tokenizer mirroring the comfy structure.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from src.text_encoder.artifact import Stage
from src.text_encoder.backends import get_backend
from src.text_encoder.backends.h3 import (
    CONSUMED_LAYER_IDX,
    DIH_HIDDEN,
    HIDDEN_DIM,
    N_LAYERS,
    H3Backend,
    _assert_no_chat_template,
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


class _FakeTELayer(nn.Module):
    """Mirrors the SD1 'qwen3vl_32b' layer: .transformer.model."""

    def __init__(self, dim: int, n_layers: int) -> None:
        super().__init__()
        self.transformer = _FakeTransformer(dim, n_layers)


class _FakeModel:
    def __init__(self, dim: int, n_layers: int, seq_len: int, tags: torch.Tensor | None) -> None:
        self.qwen3vl_32b = _FakeTELayer(dim, n_layers)
        self._seq_len = seq_len
        self._tags = tags

    def encode_token_weights(self, tokens: Any) -> tuple[torch.Tensor, None, dict]:
        # run a real forward through the fake layers so HookManager fires
        x = torch.randn(1, self._seq_len, HIDDEN_DIM)
        for layer in self.qwen3vl_32b.transformer.model.layers:
            x = layer(x)
        tags = self._tags if self._tags is not None else torch.ones(self._seq_len, dtype=torch.int)
        return x, None, {"minimax_token_tags": tags}


class _FakeTokenizer:
    def __init__(self, n_tokens: int = 7) -> None:
        self.n_tokens = n_tokens

    def tokenize_with_weights(self, text: str) -> dict:
        # ids 5..5+n (never chat markers; first id never 151644)
        entries = [(5 + i, 1.0) for i in range(self.n_tokens)]
        return {"qwen3vl_32b": [entries]}


class _FakeRefiner(nn.Module):
    """Stand-in for comfy's TokenRefiner: accepts transformer_options."""

    def forward(self, x: torch.Tensor, transformer_options: dict | None = None) -> torch.Tensor:  # noqa: ARG002
        return x


def _make_backend(seq_len: int = 9, tags: torch.Tensor | None = None) -> H3Backend:
    b = H3Backend()
    b.model = _FakeModel(HIDDEN_DIM, N_LAYERS, seq_len, tags)
    b.tokenizer = _FakeTokenizer()
    b.condition_proj = nn.Linear(HIDDEN_DIM, DIH_HIDDEN)
    b.token_refiner = _FakeRefiner()
    b.ckpt_id = "fake"
    return b


class TestRegistry:
    def test_h3_registered(self) -> None:
        backend = get_backend("h3")
        assert isinstance(backend, H3Backend)
        assert backend.name == "h3"

    def test_streams_declared_match_constants(self) -> None:
        b = get_backend("h3")
        specs = b.streams
        names = [s.name for s in specs]
        assert names[:N_LAYERS] == [f"layer_{i:02d}" for i in range(N_LAYERS)]
        assert "consumed_hidden" in names
        assert "condition_projected" in names
        assert "refined_context" in names
        by_name = {s.name: s for s in specs}
        assert by_name["consumed_hidden"].layers == (CONSUMED_LAYER_IDX,)
        assert by_name["consumed_hidden"].pooling.value == "none"
        assert by_name["layer_00"].dim == HIDDEN_DIM
        assert by_name["condition_projected"].dim == DIH_HIDDEN
        assert by_name["condition_projected"].stage == Stage.PROJECTED
        assert by_name["refined_context"].stage == Stage.CONNECTOR_OUT
        # streams() must match what encode() produces (compare zips strict)
        assert {s.stage for s in specs} == {Stage.RAW_HIDDEN, Stage.PROJECTED, Stage.CONNECTOR_OUT}

    def test_cuda_refused(self) -> None:
        b = get_backend("h3")
        with pytest.raises(RuntimeError, match="CPU"):
            b.load("x.safetensors", device="cuda", dit_file="y.safetensors")


class TestChatTemplateGuard:
    def test_raw_ids_pass(self) -> None:
        _assert_no_chat_template([100, 200, 300])

    def test_im_start_first_raises(self) -> None:
        with pytest.raises(RuntimeError, match="[Cc]hat-template"):
            _assert_no_chat_template([151644, 200, 300])

    def test_im_end_last_raises(self) -> None:
        with pytest.raises(RuntimeError, match="[Cc]hat-template"):
            _assert_no_chat_template([100, 200, 151645])


class TestDitConditioningLoad:
    def _write_dit(self, path: Any, keys: list[str]) -> None:  # noqa: ANN401
        sd = {}
        for k in keys:
            sd[k] = torch.randn(2, 2)
        save_file(sd, str(path))

    def test_missing_keys_raise(self, tmp_path: Any) -> None:
        p = tmp_path / "dit.safetensors"
        self._write_dit(p, ["condition_proj.weight", "token_refiner.blocks.0.norm1.weight"])
        with pytest.raises(ValueError, match="missing H3 conditioning keys"):
            _load_dit_conditioning_sd(p)

    def test_unrelated_keys_ignored(self, tmp_path: Any) -> None:
        p = tmp_path / "dit.safetensors"
        self._write_dit(
            p,
            ["condition_proj.weight", "condition_proj.bias", "token_refiner.final_norm.weight", "blocks.0.attn.w"],
        )
        sd = _load_dit_conditioning_sd(p)
        assert set(sd.keys()) == {"condition_proj.weight", "condition_proj.bias", "token_refiner.final_norm.weight"}

    def test_missing_file_raises(self, tmp_path: Any) -> None:
        with pytest.raises(FileNotFoundError):
            _load_dit_conditioning_sd(tmp_path / "nope.safetensors")


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
        assert cons.mask is not None and cons.mask.shape == (1, 9)
        assert int(cons.mask.sum()) == 9  # text-only: all positions valid

        proj = by_stage[Stage.PROJECTED]
        assert proj.streams["condition_projected"].activations.shape == (1, 9, DIH_HIDDEN)
        ref = by_stage[Stage.CONNECTOR_OUT]
        assert ref.streams["refined_context"].activations.shape == (1, 9, DIH_HIDDEN)

    def test_token_tags_in_meta_and_mask(self) -> None:
        tags = torch.tensor([1, 1, 1, 0, 0, 1, 1, 1, 1], dtype=torch.int)
        b = _make_backend(seq_len=9, tags=tags)
        arts = b.encode(["prompt"], ["cat"])
        raw = next(a for a in arts if a.stage == Stage.RAW_HIDDEN)
        assert raw.meta["minimax_token_tags"] == [[1, 1, 1, 0, 0, 1, 1, 1, 1]]
        cons = raw.streams["consumed_hidden"]
        assert cons.mask is not None
        assert cons.mask[0].tolist() == [1, 1, 1, 0, 0, 1, 1, 1, 1]
        proj = next(a for a in arts if a.stage == Stage.PROJECTED)
        assert proj.streams["condition_projected"].mask[0].tolist() == [1, 1, 1, 0, 0, 1, 1, 1, 1]

    def test_no_tags_defaults_all_text(self) -> None:
        b = _make_backend(seq_len=5, tags=None)
        arts = b.encode(["x"], ["cat"])
        raw = next(a for a in arts if a.stage == Stage.RAW_HIDDEN)
        assert raw.meta["minimax_token_tags"] == [[1] * 5]

    def test_stage_filtering(self) -> None:
        b = _make_backend(seq_len=6)
        arts = b.encode(["x"], ["cat"], stages=[Stage.RAW_HIDDEN])
        assert {a.stage for a in arts} == {Stage.RAW_HIDDEN}

    def test_artifact_provenance(self) -> None:
        b = _make_backend(seq_len=4)
        arts = b.encode(["x"], ["cat"])
        for a in arts:
            assert a.model == "h3"
            assert a.backend == "H3Backend"
            assert a.ckpt_id == "fake"
            assert a.meta["n_layers"] == N_LAYERS
            assert a.meta["consumed_layer_idx"] == CONSUMED_LAYER_IDX

    def test_tokenize_alignment(self) -> None:
        b = _make_backend()
        out = b.tokenize(["hello", "world"])
        assert out["input_ids"] == [[5, 6, 7, 8, 9, 10, 11], [5, 6, 7, 8, 9, 10, 11]]
        assert out["n_tokens"] == [7, 7]

    def test_encode_requires_load(self) -> None:
        b = H3Backend()
        with pytest.raises(RuntimeError, match="not loaded"):
            b.encode(["x"], ["cat"])

    def test_produced_specs_equal_declared(self) -> None:
        """Cache/compare contract: encode() artifacts carry specs identical
        to streams() declarations (review R2)."""
        b = _make_backend(seq_len=6)
        declared = {s.name: s for s in b.streams}
        arts = b.encode(["x"], ["cat"])
        for art in arts:
            for name, stream in art.streams.items():
                assert stream.spec == declared[name], f"spec mismatch for {name}"

    def test_multi_prompt_heterogeneous_lengths(self) -> None:
        """Right-padding + per-row masks across prompts of different lengths."""

        class _VarLenTokenizer(_FakeTokenizer):
            def tokenize_with_weights(self, text: str) -> dict:
                n = 9 if "long" in text else 4
                entries = [(5 + i, 1.0) for i in range(n)]
                return {"qwen3vl_32b": [entries]}

        class _VarLenModel(_FakeModel):
            def encode_token_weights(self, tokens: Any) -> tuple[torch.Tensor, None, dict]:
                n = len(tokens["qwen3vl_32b"][0])
                x = torch.randn(1, n, HIDDEN_DIM)
                for layer in self.qwen3vl_32b.transformer.model.layers:
                    x = layer(x)
                return x, None, {"minimax_token_tags": torch.ones(n, dtype=torch.int)}

        b = _make_backend(seq_len=4)
        b.tokenizer = _VarLenTokenizer()
        b.model = _VarLenModel(HIDDEN_DIM, N_LAYERS, 4, None)  # type: ignore[arg-type]
        arts = b.encode(["short", "long prompt"], ["a", "a"])
        raw = next(a for a in arts if a.stage == Stage.RAW_HIDDEN)
        cons = raw.streams["consumed_hidden"]
        assert cons.activations.shape == (2, 9, HIDDEN_DIM)
        assert cons.mask is not None
        assert cons.mask[0].tolist() == [1, 1, 1, 1, 0, 0, 0, 0, 0]
        assert cons.mask[1].tolist() == [1] * 9
        # pooled layers keep [n, 1, d] regardless of per-prompt length
        assert raw.streams["layer_07"].activations.shape == (2, 1, HIDDEN_DIM)
