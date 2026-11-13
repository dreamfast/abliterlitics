"""Tests for src/kl/kl_divergence.py — tested THROUGH the module, not re-derived.

The compute path (cmd_compute) is FROZEN (AGENTS.md #25/#37/#74); these tests are
its safety net and run entirely on CPU with tiny synthetic tensors:

- lesson-#37 regression: -inf/NaN logprobs must produce a FINITE KL (the
  LP_FLOOR=-100 nan_to_num clamp), pinning the clamp semantics exactly.
- thinkclose response-prefix resolution (CLI-level pure logic, model mocked).
- cmd_collect mocked end-to-end on CPU: metadata, top-5 position logging,
  template override, sanity-abort gate.
- pure helpers: size/loading config, device map split, prefix detection.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F  # noqa: N812  # standard torch convention (per-file-ignore)

import src.kl.kl_divergence as kd

VOCAB = 8
NUM_PROMPTS = 2


# ---------------------------------------------------------------------------
# Fixtures / fakes
# ---------------------------------------------------------------------------


class _FakeBatch(dict):
    """Dict-like tokenizer output with a no-op .to(device)."""

    def to(self, device: Any) -> _FakeBatch:
        return self


class _FakeCollectTokenizer:
    """Stands in for AutoTokenizer inside cmd_collect (CPU, no network)."""

    def __init__(self, render: str) -> None:
        self.render = render
        self.pad_token = "<pad>"
        self.pad_token_id = 0
        self.chat_template: str | None = None
        self.padding_side = "right"
        self.applied_messages: list[list[dict[str, str]]] = []

    def __call__(
        self, text: Any, return_tensors: str | None = None, return_token_type_ids: bool | None = None
    ) -> _FakeBatch:
        assert return_token_type_ids is False
        return _FakeBatch({"input_ids": torch.full((1, 4), 7, dtype=torch.long)})

    def apply_chat_template(
        self, messages: list[dict[str, str]], add_generation_prompt: bool = True, tokenize: bool = False, **kw: Any
    ) -> str:
        assert add_generation_prompt is True and tokenize is False
        self.applied_messages.append(messages)
        return self.render

    def decode(self, ids: Any, skip_special_tokens: bool = False) -> str:
        return "tk"


class _FakeCollectModel:
    """Fake causal LM: CPU params, canned generate() scores."""

    def __init__(self, scores: torch.Tensor | None = None) -> None:
        self.scores = scores if scores is not None else torch.full((1, VOCAB), -1.0)
        self._param = torch.nn.Parameter(torch.randn(2, 2))

    def named_parameters(self) -> Any:
        return iter([("model.embed_tokens.weight", self._param)])

    def parameters(self) -> Any:
        return iter([self._param])

    def buffers(self) -> Any:
        return iter([])

    def modules(self) -> list[_FakeCollectModel]:
        return [self]

    def eval(self) -> _FakeCollectModel:
        return self

    def generate(self, **kw: Any) -> Any:
        return types.SimpleNamespace(scores=[self.scores.clone()])


class _FakeAutoModel:
    instance: _FakeCollectModel | None = None

    @classmethod
    def from_pretrained(cls, path: str, trust_remote_code: bool = True, **cfg: Any) -> _FakeCollectModel:
        cls.instance = _FakeCollectModel()
        return cls.instance


def _install_collect_mocks(monkeypatch: pytest.MonkeyPatch, render: str) -> _FakeCollectTokenizer:
    """Patch AutoTokenizer / auto-class / prompts so cmd_collect runs on CPU."""
    import transformers

    tok = _FakeCollectTokenizer(render)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", classmethod(lambda cls, path, **kw: tok))
    monkeypatch.setattr(kd, "_get_model_auto_class", lambda path: _FakeAutoModel)
    monkeypatch.setattr(kd, "load_prompts", lambda n: ["harmless prompt one", "harmless prompt two"])
    return tok


def _collect_args(tmp_path: Path, **overrides: Any) -> argparse.Namespace:
    defaults: dict[str, Any] = {
        "model": "/fake/model",
        "tokenizer": None,
        "output": tmp_path / "logits_base.pt",
        "num_prompts": NUM_PROMPTS,
        "response_prefix": "thinkclose",
        "save_prefix": None,
        "chat_template_file": None,
        "device_map": None,
        "lora_adapter": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _write_logits(
    path: Path,
    logprobs: torch.Tensor,
    response_prefix: str = "",
    chat_template: str = "(tokenizer default)",
    system_prompt: str = kd.SYSTEM_PROMPT,
) -> None:
    payload = {
        "logprobs": logprobs,
        "prompt_meta": [{"index": i, "preview": f"prompt {i}", "prompt_tokens": 4} for i in range(logprobs.shape[0])],
        "model": "fake/base",
        "num_prompts": logprobs.shape[0],
        "vocab_size": logprobs.shape[-1],
        "system_prompt": system_prompt,
        "chat_template": chat_template,
        "response_prefix": response_prefix,
        "response_prefix_mode": "explicit",
        "split": kd.DEFAULT_SPLIT,
        "column": kd.DEFAULT_COLUMN,
        "methodology_version": kd.METHODOLOGY_VERSION,
        "dtype": str(logprobs.dtype),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _compute_args(tmp_path: Path, base: Path, variant: Path, label: str = "heretic") -> argparse.Namespace:
    return argparse.Namespace(
        base_logits=base,
        variant_logits=variant,
        variant_label=label,
        output=tmp_path / f"kl_{label}.json",
    )


def _run_compute(tmp_path: Path, base_lp: torch.Tensor, var_lp: torch.Tensor, **kw: Any) -> dict[str, Any]:
    base = tmp_path / "logits_base.pt"
    variant = tmp_path / "logits_variant.pt"
    _write_logits(base, base_lp, **kw)
    variant_kw = {k: v for k, v in kw.items() if k in ("response_prefix", "system_prompt")}
    _write_logits(variant, var_lp, **variant_kw)
    args = _compute_args(tmp_path, base, variant)
    kd.cmd_compute(args)
    return json.loads(args.output.read_text())


def _expected_band(mean_kl: float) -> str:
    """The documented interpretation bands of the compute phase."""
    if mean_kl < 0.01:
        return "excellent"
    if mean_kl < 0.1:
        return "very good"
    if mean_kl < 0.5:
        return "moderate"
    if mean_kl < 1.0:
        return "significant"
    return "heavy"


@pytest.fixture()
def clean_logprobs() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    base_lp = torch.log_softmax(torch.randn(4, 32), dim=-1)
    var_lp = torch.log_softmax(torch.randn(4, 32), dim=-1)
    return base_lp, var_lp


# ---------------------------------------------------------------------------
# cmd_compute — the FROZEN path, tested from outside (lesson-#37 regression)
# ---------------------------------------------------------------------------


class TestComputeInfClamp:
    """LP_FLOOR=-100 clamp: -inf/NaN logprobs must never produce NaN KL."""

    def test_kl_finite_with_minus_inf_logprobs(self, tmp_path, clean_logprobs):
        """The lesson-#37 regression: -inf in BOTH target and input positions."""
        base_lp, var_lp = clean_logprobs
        base_lp[0, 0] = float("-inf")  # underflow in the TARGET (base)
        var_lp[1, 5] = float("-inf")  # underflow in the INPUT (variant)
        result = _run_compute(tmp_path, base_lp, var_lp)
        assert math.isfinite(result["kl_divergence_batchmean"])
        assert result["kl_divergence_batchmean"] > 0.0

    def test_clamp_semantics_match_nan_to_num_minus_100(self, tmp_path, clean_logprobs):
        """Pin the exact clamp contract: nan_to_num(nan=LP_FLOOR, posinf=0, neginf=LP_FLOOR)."""
        base_lp, var_lp = clean_logprobs
        base_lp[0, 3] = float("-inf")
        base_lp[2, 7] = float("nan")
        var_lp[0, 1] = float("-inf")
        var_lp[3, 2] = float("inf")  # +inf -> prob 1
        result = _run_compute(tmp_path, base_lp, var_lp)
        lp_floor = -100.0  # the compute phase's LP_FLOOR
        b = torch.nan_to_num(base_lp, nan=lp_floor, posinf=0.0, neginf=lp_floor)
        v = torch.nan_to_num(var_lp, nan=lp_floor, posinf=0.0, neginf=lp_floor)
        expected = F.kl_div(v, b, reduction="batchmean", log_target=True).item()
        assert result["kl_divergence_batchmean"] == pytest.approx(expected, rel=1e-6)
        assert math.isfinite(expected)

    def test_kl_finite_with_nan_logprobs(self, tmp_path, clean_logprobs):
        base_lp, var_lp = clean_logprobs
        var_lp[2, 10] = float("nan")
        result = _run_compute(tmp_path, base_lp, var_lp)
        assert math.isfinite(result["kl_divergence_batchmean"])

    def test_identical_logprobs_with_inf_stay_zero_kl(self, tmp_path, clean_logprobs):
        """KL(P||P)=0 must survive the clamp (a legitimate 0.0 is NOT falsy-dropped)."""
        base_lp, _ = clean_logprobs
        base_lp[1, 9] = float("-inf")
        result = _run_compute(tmp_path, base_lp, base_lp.clone())
        assert result["kl_divergence_batchmean"] == pytest.approx(0.0, abs=1e-6)

    def test_clamping_stats_logged(self, tmp_path, clean_logprobs, caplog):
        base_lp, var_lp = clean_logprobs
        var_lp[0, 4] = float("-inf")
        with caplog.at_level(logging.INFO, logger="src.kl.kl_divergence"):
            result = _run_compute(tmp_path, base_lp, var_lp)
        assert any("Clamping non-finite logprobs" in r.getMessage() for r in caplog.records)
        assert math.isfinite(result["kl_divergence_batchmean"])

    def test_nearly_all_inf_warns_but_computes(self, tmp_path, clean_logprobs, caplog):
        """The >99% -inf warning fires, but the clamp still yields a finite KL."""
        base_lp, _ = clean_logprobs
        var_lp = torch.full((4, 32), float("-inf"))
        with caplog.at_level(logging.INFO, logger="src.kl.kl_divergence"):
            result = _run_compute(tmp_path, base_lp, var_lp)
        assert any("nearly all -inf" in r.getMessage() for r in caplog.records)
        assert math.isfinite(result["kl_divergence_batchmean"])


class TestComputeSummary:
    def test_identical_distributions_excellent_band(self, tmp_path, clean_logprobs):
        base_lp, _ = clean_logprobs
        result = _run_compute(tmp_path, base_lp, base_lp.clone())
        assert result["kl_divergence_batchmean"] == pytest.approx(0.0, abs=1e-6)
        assert result["interpretation"] == "excellent"

    def test_summary_fields_and_internal_consistency(self, tmp_path, clean_logprobs):
        base_lp, var_lp = clean_logprobs
        result = _run_compute(tmp_path, base_lp, var_lp)
        assert result["results_version"] == kd.RESULTS_VERSION
        assert result["variant_label"] == "heretic"
        assert result["base_model"] == "fake/base"
        assert result["variant_model"] == "fake/base"
        assert result["num_prompts"] == 4
        assert result["vocab_size"] == 32
        assert kd.METHODOLOGY_VERSION in result["methodology"]
        # batchmean == per-prompt mean (same denominator)
        assert result["kl_divergence_batchmean"] == pytest.approx(result["kl_divergence_per_prompt_mean"], rel=1e-5)
        # interpretation matches the documented banding of the mean
        assert result["interpretation"] == _expected_band(result["kl_divergence_per_prompt_mean"])
        # per-prompt entries carry base fixture provenance
        per = result["per_prompt_results"]
        assert len(per) == 4
        assert per[0]["prompt_preview"] == "prompt 0"
        assert per[0]["prompt_tokens"] == 4
        assert per[2]["prompt_index"] == 2
        # median/min/max consistent with per-prompt values
        vals = [p["kl_divergence"] for p in per]
        assert result["kl_divergence_per_prompt_min"] == pytest.approx(min(vals))
        assert result["kl_divergence_per_prompt_max"] == pytest.approx(max(vals))
        assert math.isfinite(result["kl_divergence_per_prompt_median"])

    def test_methodology_mentions_dataset_and_prefix(self, tmp_path, clean_logprobs):
        base_lp, var_lp = clean_logprobs
        result = _run_compute(tmp_path, base_lp, var_lp, response_prefix="</think>\n\n")
        assert kd.PROMPT_SOURCE in result["methodology"]
        assert "</think>" in result["methodology"]
        assert kd.SYSTEM_PROMPT in result["methodology"]


class TestComputeValidation:
    def test_prompt_count_mismatch_exits(self, tmp_path, clean_logprobs):
        base_lp, var_lp = clean_logprobs
        base = tmp_path / "b.pt"
        variant = tmp_path / "v.pt"
        _write_logits(base, base_lp)
        _write_logits(variant, var_lp[:3])
        with pytest.raises(SystemExit) as exc:
            kd.cmd_compute(_compute_args(tmp_path, base, variant))
        assert exc.value.code == 1

    def test_vocab_mismatch_exits(self, tmp_path, clean_logprobs):
        base_lp, var_lp = clean_logprobs
        base = tmp_path / "b.pt"
        variant = tmp_path / "v.pt"
        _write_logits(base, base_lp)
        _write_logits(variant, var_lp[:, :16])
        with pytest.raises(SystemExit):
            kd.cmd_compute(_compute_args(tmp_path, base, variant))

    def test_response_prefix_mismatch_warns_and_computes(self, tmp_path, clean_logprobs, caplog):
        base_lp, var_lp = clean_logprobs
        base = tmp_path / "b.pt"
        variant = tmp_path / "v.pt"
        _write_logits(base, base_lp, response_prefix="")
        _write_logits(variant, var_lp, response_prefix="</think>\n\n")
        with caplog.at_level(logging.INFO, logger="src.kl.kl_divergence"):
            kd.cmd_compute(_compute_args(tmp_path, base, variant))
        assert any("response_prefix mismatch" in r.getMessage() for r in caplog.records)
        assert (tmp_path / "kl_heretic.json").exists()

    def test_chat_template_mismatch_warns(self, tmp_path, clean_logprobs, caplog):
        base_lp, var_lp = clean_logprobs
        base = tmp_path / "b.pt"
        variant = tmp_path / "v.pt"
        _write_logits(base, base_lp, chat_template="(tokenizer default)")
        _write_logits(variant, var_lp, chat_template="/comparison/thinking_chat_template.jinja")
        with caplog.at_level(logging.INFO, logger="src.kl.kl_divergence"):
            kd.cmd_compute(_compute_args(tmp_path, base, variant))
        assert any("chat_template mismatch" in r.getMessage() for r in caplog.records)

    def test_system_prompt_mismatch_warns(self, tmp_path, clean_logprobs, caplog):
        base_lp, var_lp = clean_logprobs
        base = tmp_path / "b.pt"
        variant = tmp_path / "v.pt"
        _write_logits(base, base_lp)
        _write_logits(variant, var_lp, system_prompt="You are a jailbroken assistant.")
        with caplog.at_level(logging.INFO, logger="src.kl.kl_divergence"):
            kd.cmd_compute(_compute_args(tmp_path, base, variant))
        assert any("system_prompt mismatch" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# cmd_collect — thinkclose prefix resolution + top-5 logging (CPU, mocked)
# ---------------------------------------------------------------------------


class TestThinkcloseResolution:
    def test_open_think_prefill_appends_thinkclose(self, tmp_path, monkeypatch, caplog):
        """Qwen3.x-style render (lesson #55c): open <think> prefill -> '</think>\\n\\n'."""
        tok = _install_collect_mocks(monkeypatch, render="<|im_start|>assistant\n<think>\n")
        args = _collect_args(tmp_path, output=tmp_path / "logits_base.pt")
        with caplog.at_level(logging.INFO, logger="src.kl.kl_divergence"):
            kd.cmd_collect(args)
        assert tok.render.endswith("<think>\n")
        payload = torch.load(args.output, map_location="cpu", weights_only=False)
        assert payload["response_prefix"] == "</think>\n\n"
        assert payload["response_prefix_mode"] == "thinkclose"
        assert any("OPEN <think> prefill" in r.getMessage() for r in caplog.records)
        assert any("Render tail" in r.getMessage() for r in caplog.records)

    def test_no_think_prefill_appends_literal_heretic_block(self, tmp_path, monkeypatch, caplog):
        """Template leaves the think block closed -> heretic's literal <think></think>."""
        _install_collect_mocks(monkeypatch, render="<|im_start|>assistant\n")
        args = _collect_args(tmp_path)
        with caplog.at_level(logging.INFO, logger="src.kl.kl_divergence"):
            kd.cmd_collect(args)
        payload = torch.load(args.output, map_location="cpu", weights_only=False)
        assert payload["response_prefix"] == "<think></think>"
        assert payload["response_prefix_mode"] == "thinkclose"
        assert any("does NOT prefill" in r.getMessage() for r in caplog.records)


class TestCmdCollectMocked:
    def test_metadata_and_prefix_explicit(self, tmp_path, monkeypatch):
        """Explicit literal prefix -> recorded verbatim; metadata keys complete."""
        _install_collect_mocks(monkeypatch, render="<|assistant|>\n")
        prefix_file = tmp_path / "prefix.txt"
        args = _collect_args(
            tmp_path,
            response_prefix="Answer: ",
            save_prefix=prefix_file,
        )
        kd.cmd_collect(args)
        payload = torch.load(args.output, map_location="cpu", weights_only=False)
        assert payload["response_prefix"] == "Answer: "
        assert payload["response_prefix_mode"] == "explicit"
        assert payload["system_prompt"] == kd.SYSTEM_PROMPT
        assert payload["num_prompts"] == NUM_PROMPTS
        assert payload["vocab_size"] == VOCAB
        assert payload["chat_template"] == "(tokenizer default)"
        assert payload["split"] == kd.DEFAULT_SPLIT
        assert payload["column"] == kd.DEFAULT_COLUMN
        assert payload["methodology_version"] == kd.METHODOLOGY_VERSION
        assert payload["quantization"] == ""
        assert payload["logprobs"].shape == (NUM_PROMPTS, VOCAB)
        assert payload["prompt_meta"][1]["preview"].startswith("harmless prompt")
        assert prefix_file.read_text() == "Answer: "

    def test_none_prefix_yields_empty_metadata(self, tmp_path, monkeypatch):
        _install_collect_mocks(monkeypatch, render="<|assistant|>\n")
        args = _collect_args(tmp_path, response_prefix="none", save_prefix=tmp_path / "p.txt")
        kd.cmd_collect(args)
        payload = torch.load(args.output, map_location="cpu", weights_only=False)
        assert payload["response_prefix"] == ""
        assert (tmp_path / "p.txt").read_text() == ""  # empty file signals "no prefix"

    def test_chat_template_override_applied_and_recorded(self, tmp_path, monkeypatch):
        tok = _install_collect_mocks(monkeypatch, render="<|assistant|>\n")
        tpl = tmp_path / "thinking_chat_template.jinja"
        tpl.write_text("{STOCK TEMPLATE}")
        args = _collect_args(tmp_path, chat_template_file=str(tpl))
        kd.cmd_collect(args)
        assert tok.chat_template == "{STOCK TEMPLATE}"
        payload = torch.load(args.output, map_location="cpu", weights_only=False)
        assert payload["chat_template"] == str(tpl)

    def test_missing_chat_template_file_exits(self, tmp_path, monkeypatch):
        _install_collect_mocks(monkeypatch, render="<|assistant|>\n")
        args = _collect_args(tmp_path, chat_template_file=str(tmp_path / "nope.jinja"))
        with pytest.raises(SystemExit):
            kd.cmd_collect(args)

    def test_top5_first_token_log_line(self, tmp_path, monkeypatch, caplog):
        """The kl-protocol-error detection gate: top-5 candidate tokens are logged."""
        _install_collect_mocks(monkeypatch, render="<|assistant|>\n")
        model = _FakeCollectModel(scores=torch.full((1, VOCAB), -1.0))
        monkeypatch.setattr(_FakeAutoModel, "from_pretrained", classmethod(lambda cls, path, **kw: model))
        args = _collect_args(tmp_path)
        with caplog.at_level(logging.INFO, logger="src.kl.kl_divergence"):
            kd.cmd_collect(args)
        top5 = [r.getMessage() for r in caplog.records if "top-5 first tokens" in r.getMessage()]
        assert len(top5) == 1
        assert "'tk'(0.125)" in top5[0]  # uniform scores -> prob 1/8 after softmax

    def test_prompt0_sanity_stats_logged(self, tmp_path, monkeypatch, caplog):
        _install_collect_mocks(monkeypatch, render="<|assistant|>\n")
        args = _collect_args(tmp_path)
        with caplog.at_level(logging.INFO, logger="src.kl.kl_divergence"):
            kd.cmd_collect(args)
        msgs = [r.getMessage() for r in caplog.records]
        assert any("Prompt 0 sanity" in m and "finite=100.0%" in m for m in msgs)
        assert any("Logits dtype" in m for m in msgs)

    def test_all_inf_logits_abort_before_save(self, tmp_path, monkeypatch, caplog):
        """The fast-fail gate: ~100% -inf logprobs = broken model -> abort."""
        _install_collect_mocks(monkeypatch, render="<|assistant|>\n")
        model = _FakeCollectModel(scores=torch.full((1, VOCAB), float("-inf")))
        monkeypatch.setattr(_FakeAutoModel, "from_pretrained", classmethod(lambda cls, path, **kw: model))
        args = _collect_args(tmp_path)
        with caplog.at_level(logging.INFO, logger="src.kl.kl_divergence"), pytest.raises(SystemExit):
            kd.cmd_collect(args)
        assert any("ABORTING" in r.getMessage() for r in caplog.records)
        assert not args.output.exists()

    def test_system_prompt_passed_to_template(self, tmp_path, monkeypatch):
        """Heretic protocol: the explicit system message overrides template defaults."""
        tok = _install_collect_mocks(monkeypatch, render="<|assistant|>")
        kd.cmd_collect(_collect_args(tmp_path, response_prefix="none"))
        assert len(tok.applied_messages) == NUM_PROMPTS
        for messages in tok.applied_messages:
            assert messages[0] == {"role": "system", "content": kd.SYSTEM_PROMPT}
            assert messages[1]["role"] == "user"


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestGetModelSizeGb:
    def test_sums_safetensors_and_bin_ignoring_other_suffixes(self, tmp_path):
        (tmp_path / "model-00001.safetensors").write_bytes(b"x" * (2 * 1024))
        (tmp_path / "model-00002.safetensors").write_bytes(b"x" * (3 * 1024))
        (tmp_path / "pytorch_model.bin").write_bytes(b"x" * 1024)
        (tmp_path / "model.gguf").write_bytes(b"x" * (10 * 1024))  # not counted
        expected = (2 * 1024 + 3 * 1024 + 1024) / (1024**3)
        assert kd.get_model_size_gb(str(tmp_path)) == pytest.approx(expected)


class _FakeGpuProps:
    def __init__(self, total_memory: int) -> None:
        self.total_memory = total_memory


class _FakeLayersConfig:
    def __init__(self, num_hidden_layers: int) -> None:
        self.num_hidden_layers = num_hidden_layers


@pytest.fixture()
def fake_cuda(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Patch torch.cuda probing + transformers.AutoConfig for loading-config tests."""
    import transformers

    state: dict[str, Any] = {"count": 0, "mem": [32, 24], "layers": 6}

    def _count() -> int:
        return state["count"]

    def _props(i: int) -> _FakeGpuProps:
        return _FakeGpuProps(state["mem"][i] * 1024**3)

    monkeypatch.setattr(torch.cuda, "device_count", _count)
    monkeypatch.setattr(torch.cuda, "get_device_properties", _props)
    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        classmethod(lambda cls, *a, **kw: _FakeLayersConfig(state["layers"])),
    )
    return state


class TestGetLoadingConfig:
    def test_small_model_forced_onto_cuda0(self, fake_cuda):
        cfg = kd.get_loading_config("/m", 20.0)
        assert cfg == {"device_map": {"": "cuda:0"}}

    def test_single_gpu_visible_large_model_cpu_offload(self, fake_cuda):
        fake_cuda["count"] = 0
        cfg = kd.get_loading_config("/m", 40.0)
        assert cfg["device_map"] == "auto"
        assert cfg["max_memory"] == {0: kd.GPU0_BUDGET, "cpu": kd.CPU_BUDGET}

    def test_dual_gpu_split_within_vram(self, fake_cuda):
        fake_cuda["count"] = 2  # 56 GiB total; 40 <= 56*0.85
        cfg = kd.get_loading_config("/m", 40.0)
        dm = cfg["device_map"]
        assert isinstance(dm, dict)
        assert dm["model.embed_tokens"] == 0
        assert set(v for k, v in dm.items() if k.startswith("model.layers.")) <= {0, 1}
        assert dm["lm_head"] == 1  # last GPU
        assert sum(1 for v in dm.values() if v == 0) + sum(1 for v in dm.values() if v == 1) >= fake_cuda["layers"]

    def test_dual_gpu_exceeding_vram_sequential_cpu_offload(self, fake_cuda):
        fake_cuda["count"] = 2
        cfg = kd.get_loading_config("/m", 60.0)
        assert cfg["device_map"] == "sequential"
        assert set(cfg["max_memory"]) == {0, 1, "cpu"}

    def test_split_device_map_distributes_all_layers_and_remainder(self, fake_cuda):
        fake_cuda["count"] = 2
        dm = kd.build_split_device_map("/m")
        layers = {int(k.rsplit(".", 1)[1]): v for k, v in dm.items() if k.startswith("model.layers.")}
        assert sorted(layers) == list(range(fake_cuda["layers"]))  # no layer lost, no dup
        assert dm["model.embed_tokens"] == 0

    def test_split_device_map_cpu_offload_tail(self, fake_cuda, monkeypatch):
        fake_cuda["count"] = 2
        # build_split_device_map derives size from disk; patch the size probe
        # so the 50 GB model trips the >85%-of-VRAM CPU-offload branch
        monkeypatch.setattr(kd, "get_model_size_gb", lambda p: 50.0)
        dm = kd.build_split_device_map("/m")
        layers = {int(k.rsplit(".", 1)[1]): v for k, v in dm.items() if k.startswith("model.layers.")}
        assert "cpu" in set(layers.values())
        assert dm["model.norm"] == "cpu"
        # GPU-assigned layers come first, CPU layers are the tail
        assigned = [layers[i] for i in sorted(layers)]
        assert assigned[-1] == "cpu"


class TestGetInputDevice:
    def test_prefers_embed_param(self):
        embed = torch.nn.Parameter(torch.randn(2, 2))
        head = torch.nn.Parameter(torch.randn(2, 2))

        class M:
            def named_parameters(self):
                return iter([("lm_head.weight", head), ("model.embed_tokens.weight", embed)])

        assert kd.get_input_device(M()) == embed.device

    def test_falls_back_to_first_non_meta_param(self):
        meta = torch.nn.Parameter(torch.randn(2, 2, device="meta"))
        cpu = torch.nn.Parameter(torch.randn(2, 2))

        class M:
            def named_parameters(self):
                return iter([("layer.0.weight", meta), ("layer.1.weight", cpu)])

            def parameters(self):
                return iter([meta, cpu])

        assert kd.get_input_device(M()) == cpu.device


class TestLoadPrompts:
    def test_filters_empty_and_strips(self, monkeypatch):
        fake_datasets = types.SimpleNamespace(
            load_dataset=lambda name, split=None: [
                {"text": "  harmless one  "},
                {"text": "   "},
                {"text": "harmless two"},
            ]
        )
        monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
        assert kd.load_prompts(5) == ["harmless one", "harmless two"]

    def test_warns_when_fewer_available(self, monkeypatch, caplog):
        fake_datasets = types.SimpleNamespace(load_dataset=lambda name, split=None: [{"text": "only"}])
        monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
        with caplog.at_level(logging.INFO, logger="src.kl.kl_divergence"):
            prompts = kd.load_prompts(10)
        assert prompts == ["only"]
        assert any("requested 10 prompts but only 1 available" in r.getMessage() for r in caplog.records)


class _FakeBatchTokenizer:
    """Batch tokenizer/generate pair for detect_response_prefix."""

    def __init__(self, batch_decode_rounds: list[list[str]]) -> None:
        self._rounds = list(batch_decode_rounds)
        self.pad_token = "<pad>"
        self.pad_token_id = 0

    def __call__(
        self,
        batch: Any,
        return_tensors: str | None = None,
        padding: bool | None = None,
        return_token_type_ids: bool | None = None,
    ) -> _FakeBatch:
        b = len(batch)
        return _FakeBatch({"input_ids": torch.ones((b, 4), dtype=torch.long)})

    def apply_chat_template(self, messages: Any, add_generation_prompt: bool = True, tokenize: bool = False) -> str:
        return "<rendered>"

    def batch_decode(self, sequences: Any, skip_special_tokens: bool = False) -> list[str]:
        return self._rounds.pop(0)


class _FakePrefixModel:
    def generate(self, **kw: Any) -> torch.Tensor:
        b = kw["input_ids"].shape[0]
        return torch.zeros((b, 7), dtype=torch.long)


class TestDetectResponsePrefix:
    def test_cot_initializer_replaced_and_extended(self, monkeypatch):
        """Heretic main.py:396-431: <think> prefix -> closed block, then extend."""
        tok = _FakeBatchTokenizer(
            batch_decode_rounds=[
                ["<think>Plan.</think>Answer", "<think>Plan.</think>Answer"],
                ["Yes, sure.", "Yes, sure."],
            ]
        )
        prefix = kd.detect_response_prefix(_FakePrefixModel(), tok, ["p1", "p2"], kd.SYSTEM_PROMPT, torch.device("cpu"))
        assert prefix == "<think></think>Yes, sure."

    def test_no_common_prefix_returns_empty(self):
        tok = _FakeBatchTokenizer(batch_decode_rounds=[["completely", "different"]])
        prefix = kd.detect_response_prefix(_FakePrefixModel(), tok, ["p1", "p2"], kd.SYSTEM_PROMPT, torch.device("cpu"))
        assert prefix == ""

    def test_plain_prefix_without_cot_untouched(self):
        tok = _FakeBatchTokenizer(batch_decode_rounds=[["Sure", "Sure"]])
        prefix = kd.detect_response_prefix(_FakePrefixModel(), tok, ["p1"], kd.SYSTEM_PROMPT, torch.device("cpu"))
        assert prefix == "Sure"


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


class TestCli:
    def test_main_compute_dispatch(self, tmp_path, monkeypatch, clean_logprobs):
        base_lp, var_lp = clean_logprobs
        base = tmp_path / "b.pt"
        variant = tmp_path / "v.pt"
        _write_logits(base, base_lp)
        _write_logits(variant, var_lp)
        out = tmp_path / "kl_cli.json"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "kl_divergence.py",
                "compute",
                "--base-logits",
                str(base),
                "--variant-logits",
                str(variant),
                "--variant-label",
                "cli",
                "--output",
                str(out),
            ],
        )
        kd.main()
        summary = json.loads(out.read_text())
        assert summary["variant_label"] == "cli"
        assert math.isfinite(summary["kl_divergence_batchmean"])

    def test_num_prompts_must_be_positive(self, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            ["kl_divergence.py", "collect", "--model", "/m", "--output", "/tmp/x.pt", "--num-prompts", "0"],
        )
        with pytest.raises(SystemExit) as exc:
            kd.main()
        assert exc.value.code == 2

    def test_subcommand_required(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["kl_divergence.py"])
        with pytest.raises(SystemExit):
            kd.main()

    def test_thinkclose_help_advertises_deterministic_mode(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["kl_divergence.py", "collect", "--help"])
        with pytest.raises(SystemExit) as exc:
            kd.main()
        assert exc.value.code == 0
        assert "thinkclose" in capsys.readouterr().out

    def test_comparison_arg_removed(self, monkeypatch, capsys):
        """The audit's dead --comparison arg must not resurface (silent no-op)."""
        monkeypatch.setattr(
            sys,
            "argv",
            ["kl_divergence.py", "collect", "--model", "/m", "--output", "/tmp/x.pt", "--comparison", "/c"],
        )
        with pytest.raises(SystemExit) as exc:
            kd.main()
        # unknown-argument error -> exit code 2 (previously it parsed silently)
        assert exc.value.code == 2
        assert "unrecognized arguments: --comparison" in capsys.readouterr().err
