"""Tests for the Krea2 backend (no real weights required).

Covers: registry, stream specs, chat-template guard (krea2 REQUIRES the
template — mirror of the h3 rule), DiT txtfusion loading (int8 dequant +
plain bf16 + missing keys + wrong-TE probe), and the full encode() path
against a fake model/tokenizer mirroring the comfy structure.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from src.text_encoder.artifact import Stage
from src.text_encoder.backends import get_backend
from src.text_encoder.backends.krea2 import (
    DIT_DIM,
    HIDDEN_DIM,
    N_TAPS,
    TAP_LAYERS,
    TAP_STACK_DIM,
    Krea2Backend,
    _assert_chat_template,
    _probe_te,
    load_dit_txtfusion_sd,
)


class _FakeModel:
    """Mirrors the SD1 'qwen3vl_4b' layer + krea2 encode_token_weights."""

    def __init__(self, seq_len: int) -> None:
        self.qwen3vl_4b = nn.Linear(4, 4)  # embed sentinel target (unused here)
        self._seq_len = seq_len

    def encode_token_weights(self, tokens: Any) -> tuple[torch.Tensor, None, dict]:
        # krea2 contract: [1, L, 12*2560] post-template-strip flattened taps
        return torch.randn(1, self._seq_len, TAP_STACK_DIM), None, {}


class _FakeTokenizer:
    def __init__(self, n_tokens: int = 7) -> None:
        self.n_tokens = n_tokens

    def tokenize_with_weights(self, text: str) -> dict:
        # chat-wrapped ids: first is <|im_start|> (151644) as KREA2_TEMPLATE requires
        entries = [(151644, 1.0)] + [(5 + i, 1.0) for i in range(self.n_tokens)]
        return {"qwen3vl_4b": [entries]}


class _FakeTxtfusion(nn.Module):
    """Stand-in for comfy's TextFusionTransformer: accepts mask/options kwargs."""

    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))  # parameters() must be non-empty

    def forward(self, x: torch.Tensor, mask: Any = None, transformer_options: Any = None) -> torch.Tensor:  # noqa: ARG002
        # (B, L, 12, 2560) -> (B, L, 2560): sum taps as a deterministic mix
        return x.sum(dim=2)


class _FakeTxtmlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.pad(x, (0, DIT_DIM - x.shape[-1]))


def _make_backend(seq_len: int = 9) -> Krea2Backend:
    b = Krea2Backend()
    b.model = _FakeModel(seq_len)
    b.tokenizer = _FakeTokenizer()
    b.txtfusion = _FakeTxtfusion()
    b.txtmlp = _FakeTxtmlp()
    b.ckpt_id = "fake"
    return b


class TestRegistry:
    def test_krea2_registered(self) -> None:
        backend = get_backend("krea2")
        assert isinstance(backend, Krea2Backend)
        assert backend.name == "krea2"

    def test_streams_declared_match_constants(self) -> None:
        b = get_backend("krea2")
        specs = b.streams
        names = [s.name for s in specs]
        assert names[:N_TAPS] == [f"layer_{i:02d}" for i in TAP_LAYERS]
        assert "tap_stack" in names
        assert "txtfusion_out" in names
        assert "connector_out" in names
        by_name = {s.name: s for s in specs}
        assert by_name["tap_stack"].layers == TAP_LAYERS
        assert by_name["tap_stack"].pooling.value == "none"
        assert by_name["tap_stack"].dim == TAP_STACK_DIM
        assert by_name["layer_02"].dim == HIDDEN_DIM
        assert by_name["layer_02"].pooling.value == "mean"
        assert by_name["txtfusion_out"].dim == 2560
        assert by_name["txtfusion_out"].stage == Stage.PROJECTED
        assert by_name["connector_out"].dim == DIT_DIM
        assert by_name["connector_out"].stage == Stage.CONNECTOR_OUT
        assert {s.stage for s in specs} == {Stage.RAW_HIDDEN, Stage.PROJECTED, Stage.CONNECTOR_OUT}


class TestChatTemplateGuard:
    def test_templated_ids_pass(self) -> None:
        _assert_chat_template([151644, 872, 198, 100, 200])

    def test_raw_ids_raise(self) -> None:
        with pytest.raises(RuntimeError, match="chat template"):
            _assert_chat_template([100, 200, 300])

    def test_empty_raises(self) -> None:
        with pytest.raises(RuntimeError, match="chat template"):
            _assert_chat_template([])


class TestTeProbe:
    def _write_te(self, path: Any, n_layers: int, visual: bool) -> None:  # noqa: ANN401
        sd: dict[str, torch.Tensor] = {
            "model.embed_tokens.weight": torch.randn(16, 8),
            "model.norm.weight": torch.randn(8),
        }
        for i in range(n_layers):
            sd[f"model.layers.{i}.mlp.down_proj.weight"] = torch.randn(8, 8)
        if visual:
            sd["model.visual.blocks.0.attn.qkv.weight"] = torch.randn(8, 8)
        save_file(sd, str(path))

    def test_valid_te(self, tmp_path: Any) -> None:
        p = tmp_path / "te.safetensors"
        self._write_te(p, 36, visual=True)
        assert _probe_te(p) == (36, True)

    def test_plain_qwen3_rejected_by_probe(self, tmp_path: Any) -> None:
        """A plain Qwen3-4B (z-image TE) has 36 layers but no visual tower."""
        p = tmp_path / "te.safetensors"
        self._write_te(p, 36, visual=False)
        n, has_visual = _probe_te(p)
        assert n == 36 and not has_visual
        b = get_backend("krea2")
        with pytest.raises(ValueError, match="visual"):
            b.load(str(p), dit_file="unused.safetensors")

    def test_wrong_layer_count_rejected(self, tmp_path: Any) -> None:
        p = tmp_path / "te.safetensors"
        self._write_te(p, 50, visual=True)
        b = get_backend("krea2")
        with pytest.raises(ValueError, match="50 layers"):
            b.load(str(p), dit_file="unused.safetensors")


class TestDitTxtfusionLoad:
    def _write_dit(self, path: Any, keys: dict[str, torch.Tensor]) -> None:  # noqa: ANN401
        save_file({f"model.diffusion_model.{k}": v for k, v in keys.items()}, str(path))

    def _base_keys(self) -> dict[str, torch.Tensor]:
        return {
            "txtfusion.projector.weight": torch.randn(1, 12),
            "txtfusion.refiner_blocks.0.attn.wq.weight": torch.randn(16, 16),
            "txtmlp.1.weight": torch.randn(16, 8),
        }

    def test_missing_keys_raise(self, tmp_path: Any) -> None:
        p = tmp_path / "dit.safetensors"
        self._write_dit(p, {"txtfusion.projector.weight": torch.randn(1, 12)})
        with pytest.raises(ValueError, match="missing Krea2 conditioning keys"):
            load_dit_txtfusion_sd(p)  # type: ignore[misc]

    def test_missing_file_raises(self, tmp_path: Any) -> None:
        with pytest.raises(FileNotFoundError):
            load_dit_txtfusion_sd(tmp_path / "nope.safetensors")

    def test_plain_bf16_passthrough(self, tmp_path: Any) -> None:
        p = tmp_path / "dit.safetensors"
        keys = self._base_keys()
        w = torch.randn(16, 16, dtype=torch.bfloat16)
        keys["txtfusion.layerwise_blocks.0.attn.wq.weight"] = w
        self._write_dit(p, keys)
        sd, n_dequant = load_dit_txtfusion_sd(p)
        assert n_dequant == 0  # plain bf16 DiT: nothing dequantized
        assert sd["txtfusion.layerwise_blocks.0.attn.wq.weight"].dtype == torch.float32
        assert torch.equal(sd["txtfusion.layerwise_blocks.0.attn.wq.weight"], w.float())

    def test_int8_tensorwise_dequant(self, tmp_path: Any) -> None:
        p = tmp_path / "dit.safetensors"
        keys = self._base_keys()
        w = torch.randn(16, 16)
        scale = w.abs().max() / 127.0
        q = torch.clamp(torch.round(w / scale), -127, 127).to(torch.int8)
        base = "txtfusion.layerwise_blocks.0.attn.wq"
        keys[base + ".weight"] = q
        keys[base + ".weight_scale"] = scale.reshape(()).float()
        blob = json.dumps({"format": "int8_tensorwise"}).encode()
        keys[base + ".comfy_quant"] = torch.tensor(list(blob), dtype=torch.uint8)
        self._write_dit(p, keys)
        sd, n_dequant = load_dit_txtfusion_sd(p)
        assert n_dequant == 1
        assert base + ".comfy_quant" not in sd
        assert base + ".weight_scale" not in sd
        deq = sd[base + ".weight"]
        assert deq.dtype == torch.float32
        assert torch.allclose(deq, q.float() * scale)
        # RTN error is bounded by scale/2 (half a quantization step)
        assert (deq - w).abs().max() <= scale.item() / 2 + 1e-6

    def test_int8_without_scale_raises(self, tmp_path: Any) -> None:
        p = tmp_path / "dit.safetensors"
        keys = self._base_keys()
        base = "txtfusion.layerwise_blocks.0.attn.wk"
        keys[base + ".weight"] = torch.zeros(16, 16, dtype=torch.int8)
        keys[base + ".comfy_quant"] = torch.zeros(29, dtype=torch.uint8)
        self._write_dit(p, keys)
        with pytest.raises(ValueError, match="lacks weight_scale"):
            load_dit_txtfusion_sd(p)

    def test_unrelated_keys_ignored(self, tmp_path: Any) -> None:
        p = tmp_path / "dit.safetensors"
        keys = self._base_keys()
        keys["blocks.0.attn.wq.weight"] = torch.randn(16, 16)
        self._write_dit(p, keys)
        sd, _n = load_dit_txtfusion_sd(p)
        assert "blocks.0.attn.wq.weight" not in sd


class TestEncode:
    def test_encode_all_stages_shapes(self) -> None:
        b = _make_backend(seq_len=9)
        arts = b.encode(["a cat"], ["cat"])
        by_stage = {a.stage: a for a in arts}
        assert set(by_stage) == {Stage.RAW_HIDDEN, Stage.PROJECTED, Stage.CONNECTOR_OUT}

        raw = by_stage[Stage.RAW_HIDDEN]
        assert len(raw.streams) == N_TAPS + 1
        # pooled tap streams canonicalized to [n, 1, d] by Stream.__post_init__
        assert raw.streams["layer_02"].activations.shape == (1, 1, HIDDEN_DIM)
        stack = raw.streams["tap_stack"]
        assert stack.activations.shape == (1, 9, TAP_STACK_DIM)
        assert stack.mask is not None and stack.mask.shape == (1, 9)
        assert int(stack.mask.sum()) == 9

        proj = by_stage[Stage.PROJECTED]
        assert proj.streams["txtfusion_out"].activations.shape == (1, 9, 2560)
        conn = by_stage[Stage.CONNECTOR_OUT]
        assert conn.streams["connector_out"].activations.shape == (1, 9, DIT_DIM)

    def test_stage_filtering(self) -> None:
        b = _make_backend(seq_len=6)
        arts = b.encode(["x"], ["cat"], stages=[Stage.RAW_HIDDEN])
        assert {a.stage for a in arts} == {Stage.RAW_HIDDEN}

    def test_stage_filtering_projected_only(self) -> None:
        b = _make_backend(seq_len=6)
        arts = b.encode(["x"], ["cat"], stages=[Stage.PROJECTED])
        assert {a.stage for a in arts} == {Stage.PROJECTED}

    def test_stage_filtering_connector_only(self) -> None:
        """CONNECTOR_OUT alone still runs the txtfusion dependency chain."""
        b = _make_backend(seq_len=6)
        arts = b.encode(["x"], ["cat"], stages=[Stage.CONNECTOR_OUT])
        assert {a.stage for a in arts} == {Stage.CONNECTOR_OUT}
        assert arts[0].streams["connector_out"].activations.shape[-1] == DIT_DIM

    def test_missing_txtfusion_modules_raise(self) -> None:
        b = _make_backend(seq_len=4)
        b.txtfusion = None
        with pytest.raises(RuntimeError, match="txtfusion/txtmlp not built"):
            b.encode(["x"], ["cat"], stages=[Stage.PROJECTED])

    def test_artifact_provenance(self) -> None:
        b = _make_backend(seq_len=4)
        b._dit_name = "redcraft.safetensors"
        arts = b.encode(["x"], ["cat"])
        for a in arts:
            assert a.model == "krea2"
            assert a.backend == "Krea2Backend"
            assert a.ckpt_id == "fake"
            assert a.meta["tap_layers"] == list(TAP_LAYERS)
            assert a.meta["dit_file"] == "redcraft.safetensors"

    def test_tokenize_alignment(self) -> None:
        b = _make_backend()
        out = b.tokenize(["hello", "world"])
        assert out["input_ids"][0][0] == 151644
        assert out["n_tokens"] == [8, 8]

    def test_dtype_labels(self) -> None:
        b = _make_backend(seq_len=4)
        arts = b.encode(["x"], ["cat"])
        assert all(a.dtype == "comfy_bf16" for a in arts)
        b2 = _make_backend(seq_len=4)
        b2._quantized = True
        arts2 = b2.encode(["x"], ["cat"])
        assert all(a.dtype == "comfy_int8_tensorwise" for a in arts2)
        b3 = _make_backend(seq_len=4)
        b3._te_fake_quant = "int8"
        arts3 = b3.encode(["x"], ["cat"])
        assert all(a.dtype == "fakequant_int8" for a in arts3)

    def test_encode_requires_load(self) -> None:
        b = Krea2Backend()
        with pytest.raises(RuntimeError, match="not loaded"):
            b.encode(["x"], ["cat"])

    def test_produced_specs_equal_declared(self) -> None:
        """Cache/compare contract: encode() specs identical to streams()."""
        b = _make_backend(seq_len=6)
        declared = {s.name: s for s in b.streams}
        arts = b.encode(["x"], ["cat"])
        for art in arts:
            for name, stream in art.streams.items():
                assert stream.spec == declared[name], f"spec mismatch for {name}"

    def test_untokenized_prompt_rejected(self) -> None:
        """Raw ids (no <|im_start|>) must fail loud, not silently mis-align."""

        class _RawTok(_FakeTokenizer):
            def tokenize_with_weights(self, text: str) -> dict:
                return {"qwen3vl_4b": [[(5 + i, 1.0) for i in range(4)]]}

        b = _make_backend()
        b.tokenizer = _RawTok()
        with pytest.raises(RuntimeError, match="chat template"):
            b.encode(["x"], ["cat"])

    def test_wrong_output_dim_rejected(self) -> None:
        class _BadModel(_FakeModel):
            def encode_token_weights(self, tokens: Any) -> tuple[torch.Tensor, None, dict]:
                return torch.randn(1, 5, 1234), None, {}

        b = _make_backend()
        b.model = _BadModel(5)
        with pytest.raises(RuntimeError, match="30720"):
            b.encode(["x"], ["cat"])

    def test_multi_prompt_heterogeneous_lengths(self) -> None:
        """Right-padding + per-row masks across prompts of different lengths."""

        class _VarLenTok(_FakeTokenizer):
            def tokenize_with_weights(self, text: str) -> dict:
                n = 9 if "long" in text else 4
                entries = [(151644, 1.0)] + [(5 + i, 1.0) for i in range(n)]
                return {"qwen3vl_4b": [entries]}

        class _VarLenModel(_FakeModel):
            def encode_token_weights(self, tokens: Any) -> tuple[torch.Tensor, None, dict]:
                n = len(tokens["qwen3vl_4b"][0]) - 2  # rough strip proxy
                return torch.randn(1, n, TAP_STACK_DIM), None, {}

        b = _make_backend(seq_len=2)
        b.tokenizer = _VarLenTok()
        b.model = _VarLenModel(2)
        arts = b.encode(["short", "long prompt"], ["a", "a"])
        raw = next(a for a in arts if a.stage == Stage.RAW_HIDDEN)
        stack = raw.streams["tap_stack"]
        assert stack.mask is not None
        # short: 1 template token + 4 body -> model returns 3 positions;
        # long: 1 + 9 -> 8 positions (strip proxy = len - 2)
        assert stack.mask[0].tolist() == [1, 1, 1, 0, 0, 0, 0, 0]
        assert stack.mask[1].tolist() == [1] * 8
        assert raw.streams["layer_05"].activations.shape == (2, 1, HIDDEN_DIM)
