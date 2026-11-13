"""Tests for src/kl/kl_experiment.py (robustness harness — NOT frozen).

Covers: _spearman tie semantics (average ranks + NaN guards), the meta-device
guard ported from kl_divergence (journal #266), build_chat_text, cmd_report
merge logic, and a full CPU collect-variant pass with a fake model whose
outputs are cross-checked against the manual KL formula.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F  # noqa: N812  # standard torch convention

import src.kl.kl_experiment as kx

VOCAB = 8
T_CAP = 8


# ---------------------------------------------------------------------------
# _spearman — tie semantics + guards
# ---------------------------------------------------------------------------


class TestSpearman:
    def test_perfect_monotonic_is_one(self):
        assert kx._spearman([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]) == pytest.approx(1.0)

    def test_inverse_is_minus_one(self):
        assert kx._spearman([1.0, 2.0, 3.0], [30.0, 20.0, 10.0]) == pytest.approx(-1.0)

    def test_ties_use_average_ranks(self):
        """kl_methods' ordinal-rank copy returned 1.0 here; average ranks give 0.866."""
        a = [1.0, 1.0, 2.0]
        b = [1.0, 2.0, 3.0]
        # a ranks (1-based, averaged): [1.5, 1.5, 3.0]; b ranks: [1, 2, 3]
        # Pearson: 1.5 / (sqrt(1.5) * sqrt(2)) = 0.8660254
        assert kx._spearman(a, b) == pytest.approx(0.8660254, rel=1e-6)

    def test_fewer_than_two_items_is_nan(self):
        assert math.isnan(kx._spearman([1.0], [2.0]))
        assert math.isnan(kx._spearman([], []))

    def test_constant_input_is_nan(self):
        assert math.isnan(kx._spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))
        assert math.isnan(kx._spearman([1.0, 2.0, 3.0], [5.0, 5.0, 5.0]))


# ---------------------------------------------------------------------------
# build_chat_text
# ---------------------------------------------------------------------------


class _RecordingTokenizer:
    def __init__(self, render: str = "RENDER") -> None:
        self.render = render
        self.calls: list[dict[str, Any]] = []

    def apply_chat_template(
        self, messages: Any, add_generation_prompt: bool = True, tokenize: bool = False, **kw: Any
    ) -> str:
        self.calls.append(
            {"messages": messages, "add_generation_prompt": add_generation_prompt, "tokenize": tokenize, **kw}
        )
        return self.render


class TestBuildChatText:
    def test_appends_response_prefix(self):
        tok = _RecordingTokenizer()
        out = kx.build_chat_text(tok, "what is 2+2", "<think></think>")
        assert out == "RENDER<think></think>"
        # heretic protocol: explicit system prompt + user message
        assert tok.calls[0]["messages"][0] == {"role": "system", "content": kx.SYSTEM_PROMPT}
        assert tok.calls[0]["messages"][1] == {"role": "user", "content": "what is 2+2"}
        assert tok.calls[0]["add_generation_prompt"] is True

    def test_empty_prefix_appends_nothing(self):
        tok = _RecordingTokenizer(render="CHAT")
        assert kx.build_chat_text(tok, "q", "") == "CHAT"

    def test_enable_thinking_passed_through(self):
        tok = _RecordingTokenizer()
        kx.build_chat_text(tok, "q", "", enable_thinking=False)
        assert tok.calls[0]["enable_thinking"] is False


# ---------------------------------------------------------------------------
# load_model_and_tokenizer meta-device guard (ported from kl_divergence, #266)
# ---------------------------------------------------------------------------


class _GuardModel:
    def __init__(self, param_specs: list[str], hf_device_map: Any = "unset") -> None:
        self._specs = param_specs
        self._tensors = {
            "meta": torch.nn.Parameter(torch.randn(2, 2, device="meta")),
            "cpu": torch.nn.Parameter(torch.randn(2, 2)),
        }
        if hf_device_map != "unset":
            self.hf_device_map = hf_device_map

    def named_parameters(self) -> Any:
        return iter([(f"model.{s}", self._tensors[s]) for s in self._specs])

    def parameters(self) -> Any:
        return iter([self._tensors[s] for s in self._specs])

    def eval(self) -> _GuardModel:
        return self


class _GuardAutoModel:
    model: _GuardModel | None = None

    @classmethod
    def from_pretrained(cls, path: str, trust_remote_code: bool = True, **cfg: Any) -> _GuardModel:
        assert cls.model is not None
        return cls.model


class _GuardTokenizer:
    pad_token: str | None = None
    eos_token = "<eos>"


@pytest.fixture()
def guard_env(monkeypatch: pytest.MonkeyPatch) -> list[_GuardModel]:
    import transformers

    created: list[_GuardModel] = []

    def _auto(path: str) -> type[_GuardAutoModel]:
        return _GuardAutoModel

    def _from_pretrained(cls: Any, path: str, **kw: Any) -> _GuardModel:
        assert created, "test must seed _GuardAutoModel.model first"
        return created[0]

    monkeypatch.setattr(kx, "get_model_size_gb", lambda p: 20.0)
    monkeypatch.setattr(kx, "get_loading_config", lambda p, s: {"device_map": {"": "cpu"}})
    monkeypatch.setattr(kx, "_get_model_auto_class", _auto)
    monkeypatch.setattr(_GuardAutoModel, "from_pretrained", classmethod(_from_pretrained))
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", classmethod(lambda cls, path, **kw: _GuardTokenizer())
    )
    return created


class TestMetaDeviceGuard:
    def test_meta_with_accel_offload_proceeds(self, guard_env, caplog):
        """transformers 5.x >VRAM path: meta params + cpu/disk offload hooks = OK."""
        guard_env.append(_GuardModel(["meta", "cpu"], hf_device_map={"": "cpu"}))
        with caplog.at_level(logging.INFO, logger="src.kl.kl_experiment"):
            model, tok, device = kx.load_model_and_tokenizer("/m", "/t")
        assert device.type == "cpu"
        assert any("accelerate offload hooks" in r.getMessage() for r in caplog.records)
        assert any("Proceeding" in r.getMessage() for r in caplog.records)

    def test_meta_without_offload_evidence_aborts(self, guard_env, caplog):
        """Broken load: meta params with NO hf_device_map offload -> hard abort."""
        guard_env.append(_GuardModel(["meta", "cpu"]))  # no hf_device_map attribute
        with caplog.at_level(logging.ERROR, logger="src.kl.kl_experiment"), pytest.raises(SystemExit) as exc:
            kx.load_model_and_tokenizer("/m", "/t")
        assert exc.value.code == 1
        assert any("ABORTING" in r.getMessage() for r in caplog.records)
        assert any("logits would be garbage" in r.getMessage() for r in caplog.records)

    def test_meta_with_disk_offload_proceeds(self, guard_env):
        guard_env.append(_GuardModel(["meta", "cpu"], hf_device_map={0: 0, 1: "disk"}))
        _, _, device = kx.load_model_and_tokenizer("/m", "/t")
        assert device.type == "cpu"

    def test_no_meta_params_proceeds(self, guard_env, caplog):
        guard_env.append(_GuardModel(["cpu", "cpu"], hf_device_map={"": 0}))
        with caplog.at_level(logging.INFO, logger="src.kl.kl_experiment"):
            model, _, _ = kx.load_model_and_tokenizer("/m", "/t")
        assert not any("ABORTING" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# cmd_report — merge + rank correlations
# ---------------------------------------------------------------------------


def _write_exp(path: Path, label: str, kl_1t: float, kl_32t: float, dataset: str) -> None:
    path.write_text(
        json.dumps(
            {
                "variant_label": label,
                "dataset": dataset,
                "dolly_1t_kl": kl_1t,
                "dolly_32t_kl": kl_32t,
            }
        )
    )


class TestCmdReport:
    def test_merge_with_partial_production_coverage(self, tmp_path):
        exp_dir = tmp_path / "exp"
        exp_dir.mkdir()
        kl_dir = tmp_path / "kl"
        kl_dir.mkdir()
        _write_exp(exp_dir / "exp_heretic.json", "heretic", 0.02, 0.10, "databricks/databricks-dolly-15k")
        _write_exp(exp_dir / "exp_huihui.json", "huihui", 0.05, 0.20, "databricks/databricks-dolly-15k")
        (kl_dir / "kl_heretic.json").write_text(json.dumps({"kl_divergence_batchmean": 0.0536}))
        # huihui has NO production file -> alpaca_1t None -> excluded from rho
        out = tmp_path / "comparison.json"
        kx.cmd_report(argparse.Namespace(experiment_dir=str(exp_dir), existing_kl_dir=str(kl_dir), output=str(out)))
        report = json.loads(out.read_text())
        assert report["prompt_source"] == "dolly"
        assert len(report["variants"]) == 2
        by_label = {r["variant"]: r for r in report["variants"]}
        assert by_label["heretic"]["alpaca_1t"] == pytest.approx(0.0536)
        assert by_label["heretic"]["dolly_1t"] == pytest.approx(0.02)
        assert by_label["huihui"]["alpaca_1t"] is None
        assert report["rank_correlations"]["n"] == 1

    def test_spearman_over_complete_rows(self, tmp_path):
        exp_dir = tmp_path / "exp"
        exp_dir.mkdir()
        kl_dir = tmp_path / "kl"
        kl_dir.mkdir()
        # perfect rank agreement between production and experiment
        for label, prod, d1, d32 in [("a", 0.01, 0.05, 0.10), ("b", 0.05, 0.20, 0.30), ("c", 0.10, 0.40, 0.60)]:
            _write_exp(exp_dir / f"exp_{label}.json", label, d1, d32, "databricks/databricks-dolly-15k")
            (kl_dir / f"kl_{label}.json").write_text(json.dumps({"kl_divergence_batchmean": prod}))
        out = tmp_path / "comparison.json"
        kx.cmd_report(argparse.Namespace(experiment_dir=str(exp_dir), existing_kl_dir=str(kl_dir), output=str(out)))
        report = json.loads(out.read_text())
        rc = report["rank_correlations"]
        assert rc["n"] == 3
        assert rc["alpaca_1t_vs_dolly_1t"] == pytest.approx(1.0)
        assert rc["alpaca_1t_vs_dolly_32t"] == pytest.approx(1.0)
        assert rc["dolly_1t_vs_dolly_32t"] == pytest.approx(1.0)

    def test_no_production_numbers_yields_nan_correlations(self, tmp_path):
        exp_dir = tmp_path / "exp"
        exp_dir.mkdir()
        kl_dir = tmp_path / "kl"
        kl_dir.mkdir()
        _write_exp(exp_dir / "exp_x.json", "x", 0.02, 0.10, "databricks/databricks-dolly-15k")
        out = tmp_path / "comparison.json"
        kx.cmd_report(argparse.Namespace(experiment_dir=str(exp_dir), existing_kl_dir=str(kl_dir), output=str(out)))
        report = json.loads(out.read_text())  # json round-trips NaN
        assert report["rank_correlations"]["n"] == 0
        assert math.isnan(report["rank_correlations"]["alpaca_1t_vs_dolly_1t"])

    def test_gpqa_dataset_sets_source_label(self, tmp_path):
        exp_dir = tmp_path / "exp"
        exp_dir.mkdir()
        kl_dir = tmp_path / "kl"
        kl_dir.mkdir()
        _write_exp(exp_dir / "exp_q.json", "q", 0.02, 0.10, "Idavidrein/gpqa (gpqa_diamond)")
        out = tmp_path / "comparison.json"
        kx.cmd_report(argparse.Namespace(experiment_dir=str(exp_dir), existing_kl_dir=str(kl_dir), output=str(out)))
        report = json.loads(out.read_text())
        assert report["prompt_source"] == "gpqa"
        assert report["variants"][0]["gpqa_1t"] == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# cmd_collect_variant — full CPU pass with cross-checked math
# ---------------------------------------------------------------------------


class _FakeVariantBatch(dict):
    def to(self, device: Any) -> _FakeVariantBatch:
        return self


class _FakeVariantTokenizer:
    pad_token_id = 0

    def __call__(
        self, text: str, return_tensors: str | None = None, return_token_type_ids: bool | None = None
    ) -> _FakeVariantBatch:
        return _FakeVariantBatch({"input_ids": torch.full((1, 4), 7, dtype=torch.long)})

    def apply_chat_template(
        self, messages: Any, add_generation_prompt: bool = True, tokenize: bool = False, **kw: Any
    ) -> str:
        return "<chat>"


class _FakeVariantModel:
    """Deterministic fake: fixed generate() scores, position-independent forward."""

    PROMPT_LEN = 4

    def __init__(self, gen_seed: int) -> None:
        g = torch.Generator().manual_seed(gen_seed)
        self._scores = torch.randn(1, VOCAB, generator=g)
        self._fwd_logits = torch.linspace(-1.0, 1.0, VOCAB).repeat(1, 1, 1)  # [1, 1, V] broadcastable

    def named_parameters(self) -> Any:
        return iter([("model.embed_tokens.weight", torch.nn.Parameter(torch.randn(2, 2)))])

    def eval(self) -> _FakeVariantModel:
        return self

    def generate(self, **kw: Any) -> Any:
        return types.SimpleNamespace(scores=[self._scores.clone()])

    def __call__(self, input_ids: torch.Tensor | None = None, attention_mask: torch.Tensor | None = None) -> Any:
        length = input_ids.shape[1]
        logits = self._fwd_logits.expand(1, length, VOCAB)
        return types.SimpleNamespace(logits=logits.clone())


@pytest.fixture()
def base_data(tmp_path: Path) -> Path:
    """Synthetic base .pt as cmd_collect_base would write (prompt_ids omitted)."""
    torch.manual_seed(123)
    cont_lengths = [2, 3]
    payload = {
        "prompts": ["prompt one", "prompt two"],
        "continuation_ids": [torch.randint(0, 5, (t,)) for t in cont_lengths],
        "cont_lengths": cont_lengths,
        "base_1t_logprobs": torch.log_softmax(torch.randn(2, VOCAB), dim=-1),
        "base_32t_logprobs": [torch.log_softmax(torch.randn(t, VOCAB), dim=-1) for t in cont_lengths],
        "response_prefix": "",
        "enable_thinking": False,
        "system_prompt": kx.SYSTEM_PROMPT,
        "num_prompts": 2,
        "num_tokens_max": T_CAP,
        "prompt_source": "dolly",
        "dataset": "databricks/databricks-dolly-15k",
        "seed": 42,
    }
    out = tmp_path / "exp_base.pt"
    torch.save(payload, out)
    return out


class TestCmdCollectVariant:
    def _run(self, tmp_path: Path, base_data: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        model = _FakeVariantModel(gen_seed=7)
        monkeypatch.setattr(
            kx, "load_model_and_tokenizer", lambda m, t: (model, _FakeVariantTokenizer(), torch.device("cpu"))
        )
        out = tmp_path / "exp_heretic.json"
        kx.cmd_collect_variant(
            argparse.Namespace(
                model="/m", tokenizer="/t", base_data=str(base_data), variant_label="heretic", output=str(out)
            )
        )
        return json.loads(out.read_text())

    def test_summary_fields(self, tmp_path, base_data, monkeypatch):
        result = self._run(tmp_path, base_data, monkeypatch)
        assert result["variant_label"] == "heretic"
        assert result["num_prompts"] == 2
        assert result["avg_continuation_tokens"] == pytest.approx(2.5)
        assert len(result["dolly_1t_per_prompt"]) == 2
        assert len(result["dolly_32t_per_prompt"]) == 2
        # trajectory accumulators sized by num_tokens_max
        assert len(result["dolly_32t_per_position"]) == T_CAP
        assert set(result["dolly_32t_cumulative"]) == {"1", "2", "4", "8"}
        assert "production method" in result["methodology_dolly_1t"]
        assert "teacher-forced" in result["methodology_dolly_32t"] or "greedy" in result["methodology_dolly_32t"]
        for key in ("dolly_1t_kl", "dolly_32t_kl", "dolly_1t_kl_std", "dolly_32t_kl_std"):
            assert math.isfinite(result[key])

    def test_1token_kl_matches_manual_formula(self, tmp_path, base_data, monkeypatch):
        """Cross-check per-prompt 1-token KL against the exact production formula."""
        base = torch.load(base_data, map_location="cpu", weights_only=False)
        model = _FakeVariantModel(gen_seed=7)
        v1t = F.log_softmax(model._scores.float(), dim=-1)[0]  # same for every prompt
        result = self._run(tmp_path, base_data, monkeypatch)
        for i in range(2):
            expected = F.kl_div(
                v1t.unsqueeze(0),
                base["base_1t_logprobs"][i].unsqueeze(0),
                reduction="batchmean",
                log_target=True,
            ).item()
            assert result["dolly_1t_per_prompt"][i] == pytest.approx(expected, rel=1e-6)

    def test_32token_kl_matches_manual_formula(self, tmp_path, base_data, monkeypatch):
        """Cross-check the teacher-forced per-position KL against manual slicing."""
        base = torch.load(base_data, map_location="cpu", weights_only=False)
        result = self._run(tmp_path, base_data, monkeypatch)
        plen = _FakeVariantModel.PROMPT_LEN
        for i, t_i in enumerate(base["cont_lengths"]):
            # fake forward is position-independent -> every row identical
            v32t = F.log_softmax(torch.linspace(-1.0, 1.0, VOCAB), dim=-1).expand(t_i, VOCAB)
            expected = (
                F.kl_div(v32t, base["base_32t_logprobs"][i].float(), reduction="none", log_target=True)
                .sum(dim=-1)
                .mean()
                .item()
            )
            assert result["dolly_32t_per_prompt"][i] == pytest.approx(expected, rel=1e-5), (
                f"prompt {i}: slice [{plen - 1}:{plen - 1 + t_i}] must be scored"
            )

    def test_cumulative_trajectory_consistent_with_pointwise(self, tmp_path, base_data, monkeypatch):
        result = self._run(tmp_path, base_data, monkeypatch)
        pointwise = result["dolly_32t_per_position"]
        for n in (1, 2, 4, 8):
            assert result["dolly_32t_cumulative"][str(n)] == pytest.approx(sum(pointwise[:n]) / n, rel=1e-6)

    def test_batchmean_equals_per_prompt_mean(self, tmp_path, base_data, monkeypatch):
        result = self._run(tmp_path, base_data, monkeypatch)
        assert result["dolly_1t_kl"] == pytest.approx(sum(result["dolly_1t_per_prompt"]) / 2, rel=1e-6)
        assert result["dolly_32t_kl"] == pytest.approx(sum(result["dolly_32t_per_prompt"]) / 2, rel=1e-6)


# ---------------------------------------------------------------------------
# dataset loaders + dispatch
# ---------------------------------------------------------------------------


class _FakeDataset:
    """Iterable/indexable row store mimicking the datasets.Dataset surface used."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def shuffle(self, seed: int) -> _FakeDataset:
        rng = random.Random(seed)
        rows = list(self._rows)
        rng.shuffle(rows)
        return _FakeDataset(rows)

    def select(self, indices: Any) -> _FakeDataset:
        return _FakeDataset([self._rows[i] for i in indices])

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, i: int) -> dict[str, Any]:
        return self._rows[i]


_DOLLY_ROWS = [
    {"instruction": "  answer me  ", "context": ""},
    {"instruction": " use ctx ", "context": " some context "},
    {"instruction": "", "context": ""},
]
_GPQA_ROWS = [
    {
        "Question": f"Q{i}?",
        "Correct Answer": "right",
        "Incorrect Answer 1": "w1",
        "Incorrect Answer 2": "w2",
        "Incorrect Answer 3": "w3",
    }
    for i in range(3)
]


def _fake_load_dataset(name: str, config: str | None = None, split: str | None = None) -> _FakeDataset:
    if "dolly" in name:
        return _FakeDataset(_DOLLY_ROWS)
    if "gpqa" in name:
        return _FakeDataset(_GPQA_ROWS)
    if "no_robots" in name:
        return _FakeDataset([{"prompt": f" nr{i} "} for i in range(4)] + [{"prompt": "  "}])
    if "gsm8k" in name:
        return _FakeDataset([{"question": f"math{i}"} for i in range(4)])
    if "truthful" in name:
        return _FakeDataset([{"question": f"fact{i}"} for i in range(4)])
    if "MMLU" in name:
        return _FakeDataset([{"question": f"mc{i}", "options": ["a", "", "b ", None]} for i in range(3)])
    if "abliterix" in name:
        return _FakeDataset([{"prompt": f"good{i}"} for i in range(510)])
    raise AssertionError(f"unexpected dataset {name}")


@pytest.fixture()
def fake_datasets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=_fake_load_dataset))


class TestDatasetLoaders:
    def test_dolly_joins_context_and_strips(self, fake_datasets):
        prompts = kx.load_dolly_prompts(10)
        # seeded shuffle reorders rows; content semantics are what we pin
        assert sorted(prompts) == sorted(["answer me", "some context\n\nuse ctx"])

    def test_gpqa_builds_mc_block_with_all_four_options(self, fake_datasets):
        prompts = kx.load_gpqa_prompts(3)
        assert len(prompts) == 3
        for i, p in enumerate(prompts):
            assert p.startswith(f"Question: Q{i}?\n\n")
            assert p.endswith("\n\nAnswer:")
            assert all(f"({c}) " in p for c in "ABCD")

    def test_no_robots_drops_blank(self, fake_datasets):
        assert sorted(kx.load_no_robots_prompts(10)) == sorted(f"nr{i}" for i in range(4))

    def test_gsm8k_and_truthfulqa_extract_question_column(self, fake_datasets):
        assert kx.load_gsm8k_prompts(2) == ["math0", "math1"]
        assert kx.load_truthfulqa_prompts(4) == [f"fact{i}" for i in range(4)]

    def test_mmlu_pro_filters_empty_options(self, fake_datasets):
        prompts = kx.load_mmlu_pro_prompts(3)
        assert len(prompts) == 3
        assert "(A) a" in prompts[0] and "(B) b" in prompts[0]
        assert "(C)" not in prompts[0]  # blank/None options filtered

    def test_good1000_reads_held_out_slice_400_500(self, fake_datasets):
        prompts = kx.load_good1000_prompts(100)
        assert prompts == [f"good{i}" for i in range(400, 500)]  # Abliterix held-out train[400:500]

    def test_dispatch_routes_to_named_loader(self, monkeypatch, fake_datasets):
        seen: list[str] = []
        for tag, fn in [
            ("gpqa", "load_gpqa_prompts"),
            ("no_robots", "load_no_robots_prompts"),
            ("gsm8k", "load_gsm8k_prompts"),
            ("truthfulqa", "load_truthfulqa_prompts"),
            ("mmlu_pro", "load_mmlu_pro_prompts"),
            ("good1000", "load_good1000_prompts"),
        ]:
            monkeypatch.setattr(kx, fn, lambda n, tag=tag: seen.append(tag) or [tag])
            assert kx.load_prompts(tag, 5) == [tag]
        assert seen == ["gpqa", "no_robots", "gsm8k", "truthfulqa", "mmlu_pro", "good1000"]

    def test_unknown_source_defaults_to_dolly(self, fake_datasets):
        prompts = kx.load_prompts("whatever", 10)
        assert sorted(prompts) == sorted(["answer me", "some context\n\nuse ctx"])

    def test_dataset_names_cover_all_sources(self):
        assert set(kx.DATASET_NAMES) == {"dolly", "gpqa", "no_robots", "gsm8k", "truthfulqa", "mmlu_pro", "good1000"}
        assert "databricks-dolly-15k" in kx.DATASET_NAMES["dolly"]


class TestKlExperimentCli:
    def test_main_report_dispatch(self, tmp_path, monkeypatch):
        exp_dir = tmp_path / "exp"
        exp_dir.mkdir()
        kl_dir = tmp_path / "kl"
        kl_dir.mkdir()
        _write_exp(exp_dir / "exp_x.json", "x", 0.02, 0.10, "databricks/databricks-dolly-15k")
        out = tmp_path / "comparison.json"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "kl_experiment.py",
                "report",
                "--experiment-dir",
                str(exp_dir),
                "--existing-kl-dir",
                str(kl_dir),
                "--output",
                str(out),
            ],
        )
        kx.main()
        assert json.loads(out.read_text())["prompt_source"] == "dolly"

    def test_subcommand_required(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["kl_experiment.py"])
        with pytest.raises(SystemExit):
            kx.main()
