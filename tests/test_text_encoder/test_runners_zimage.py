"""Tests for the z-image TE runner scripts (audit .cache/audit/te-runners-zimage.md).

Locks the argparse surface (7 of 9 runners were at 0% coverage) and the fixed
behaviors:
- run_attention --cache-device keying (audit bug #1: runtime --device was the
  cache key while the analysis cache is cpu-keyed)
- run_controls --merge pure-relative dose tolerance (audit bug #3)
- run_img_metrics manifest-derived base seed + zero-pair fail-loud (bug #6)
- run_breadth anchor-seed derivation (bug #7)
- run_directional transform-aware cache index (bug #2)
- run_ab_test $COMFYUI_MODELS default + workflow patching

Heavy dependencies (ComfyUI import, DiT loading, LPIPS/CLIP meters) are
stubbed; the numeric cores run on tiny CPU tensors.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from PIL import Image

from src.text_encoder.run_ab_test import load_prompt_suite, patch_workflow
from src.text_encoder.run_ab_test import parse_args as parse_ab_args
from src.text_encoder.run_analysis import parse_args as parse_analysis_args
from src.text_encoder.run_attention import parse_args as parse_attention_args
from src.text_encoder.run_breadth import _anchor_seed_pair, load_records
from src.text_encoder.run_breadth import parse_args as parse_breadth_args
from src.text_encoder.run_controls import dose_mse_matches
from src.text_encoder.run_controls import parse_args as parse_controls_args
from src.text_encoder.run_directional import _load_raw_hidden, _resolve_payload
from src.text_encoder.run_directional import parse_args as parse_directional_args
from src.text_encoder.run_discriminability import parse_args as parse_discriminability_args
from src.text_encoder.run_dit_divergence import parse_args as parse_divergence_args
from src.text_encoder.run_img_metrics import (
    _assert_pairs_mined,
    _resolve_base_seed,
    _resolve_path,
)
from src.text_encoder.run_img_metrics import (
    parse_args as parse_img_args,
)

# ---------------------------------------------------------------------------
# Arg-parse smoke tests (the 0%-coverage surface)
# ---------------------------------------------------------------------------


class TestParseArgsSmoke:
    def test_controls_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            sys, "argv", ["prog", "--base", "b.safetensors", "--heretic", "h.safetensors", "--dit", "d.safetensors"]
        )
        args = parse_controls_args()
        assert args.cache_dir == "/te_cache"
        assert args.results_dir == "/results"
        assert args.device == "cuda"
        assert args.cache_device == "cpu"  # analysis cache is cpu-keyed
        assert args.backend == "zimage"
        assert args.n_prompts == 6
        assert args.seed == 42

    def test_discriminability_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "prog",
                "--base",
                "b",
                "--variants",
                "v1,v2",
                "--dit",
                "d",
                "--cache-device",
                "cuda",
                "--backend",
                "krea2",
            ],
        )
        args = parse_discriminability_args()
        assert args.cache_device == "cuda"
        assert args.backend == "krea2"
        assert args.variant_names == ""

    def test_dit_divergence_cache_device_default_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["prog", "--base", "b", "--variant", "v", "--dit", "d"])
        args = parse_divergence_args()
        assert args.cache_device == ""  # resolved to cpu inside main
        assert args.variant_name == "variant"

    def test_attention_cache_device_defaults_cpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Audit bug #1: the cache key must default to cpu, NOT the runtime device."""
        monkeypatch.setattr(sys, "argv", ["prog", "--base", "b", "--variants", "v", "--dit", "d"])
        args = parse_attention_args()
        assert args.device == "cuda"  # runtime device unchanged
        assert args.cache_device == "cpu"  # NEW flag, cpu-keyed cache
        monkeypatch.setattr(
            sys, "argv", ["prog", "--base", "b", "--variants", "v", "--dit", "d", "--cache-device", "cuda"]
        )
        assert parse_attention_args().cache_device == "cuda"

    def test_analysis_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["prog", "--base", "b", "--variant", "v", "--dit", "d"])
        args = parse_analysis_args()
        assert args.device == "cpu"
        assert args.backend == "zimage"
        assert args.n_boot == 1000

    def test_img_metrics_base_seed_default_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["prog", "--manifest", "m.json", "--out-dir", "o"])
        args = parse_img_args()
        assert args.base_seed is None  # resolved from the manifest in main

    def test_breadth_anchor_seeds_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["prog", "--manifest", "m.json", "--suite", "s.yaml", "--out", "o.json"])
        args = parse_breadth_args()
        assert args.anchor_seeds == ""
        assert args.device == "cuda"

    def test_directional_pairs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["prog", "--cache-dir", "c", "--results-dir", "r", "--pairs", "h=/m/h:/m/b"])
        args = parse_directional_args()
        assert args.pairs == ["h=/m/h:/m/b"]
        assert args.n_perm == 200

    def test_ab_test_models_root_env_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("COMFYUI_MODELS", raising=False)
        with pytest.raises(SystemExit) as exc:
            parse_ab_args(["--workflow", "w.json", "--comparison", "c.json", "--out-dir", "o"])
        assert exc.value.code == 2  # argparse error, not a silent absolute default

        monkeypatch.setenv("COMFYUI_MODELS", "/models")
        args = parse_ab_args(["--workflow", "w.json", "--comparison", "c.json", "--out-dir", "o"])
        assert args.models_root == "/models"

        args = parse_ab_args(
            ["--workflow", "w.json", "--comparison", "c.json", "--out-dir", "o", "--models-root", "/other"]
        )
        assert args.models_root == "/other"  # explicit flag wins over env


# ---------------------------------------------------------------------------
# run_controls: --merge relative tolerance (audit bug #3)
# ---------------------------------------------------------------------------


class TestDoseMseMatches:
    def test_identical(self) -> None:
        assert dose_mse_matches([0.02, 0.03], [0.02, 0.03])

    def test_small_recapture_drift_passes(self) -> None:
        assert dose_mse_matches([0.0200], [0.0205])  # 2.5% drift

    def test_sub_unit_wrong_variant_fails(self) -> None:
        """The regression: 0.02 vs 0.052 (2.6x) passed the old max(1.0,..) floor."""
        assert not dose_mse_matches([0.020], [0.052])

    def test_large_value_relative_band(self) -> None:
        assert dose_mse_matches([10.0], [10.29])  # within 5%
        assert not dose_mse_matches([10.0], [10.7])  # 7% apart

    def test_zero_exact_match(self) -> None:
        assert dose_mse_matches([0.0], [0.0])
        assert not dose_mse_matches([0.0], [0.001])


# ---------------------------------------------------------------------------
# run_directional: transform-aware cache index (audit bug #2)
# ---------------------------------------------------------------------------


def _raw_hidden_payload(ckpt_id: str, transform: str, device: str = "cpu") -> dict[str, Any]:
    return {
        "stage": "raw_hidden",
        "ckpt_id": ckpt_id,
        "device": device,
        "meta": {"transform": transform},
        "streams": {},
        "categories": ["innocent"],
        "prompts": ["p"],
    }


class TestDirectionalCacheIndex:
    def test_index_keys_include_transform_and_device(self, tmp_path: Path) -> None:
        torch.save(_raw_hidden_payload("/m/base.safetensors", "orig"), tmp_path / "a.pt")
        torch.save(_raw_hidden_payload("/m/base.safetensors", "fakequant_int8"), tmp_path / "b.pt")
        torch.save({"stage": "projected", "ckpt_id": "/m/x", "meta": {}}, tmp_path / "c.pt")  # skipped
        (tmp_path / "corrupt.pt").write_bytes(b"not-a-payload")  # skipped with warning
        index = _load_raw_hidden(tmp_path)
        assert set(index) == {
            ("/m/base.safetensors", "orig", "cpu"),
            ("/m/base.safetensors", "fakequant_int8", "cpu"),
        }

    def test_resolve_prefers_orig_over_last_written(self, tmp_path: Path) -> None:
        """Old last-write-wins resolved the fakequant noise-floor payload."""
        torch.save(_raw_hidden_payload("/m/base.safetensors", "orig"), tmp_path / "zzz_orig.pt")
        torch.save(_raw_hidden_payload("/m/base.safetensors", "fakequant_int8"), tmp_path / "aaa_fq.pt")
        index = _load_raw_hidden(tmp_path)
        payload = _resolve_payload(index, "/m/base.safetensors")
        assert payload is not None
        assert payload["meta"]["transform"] == "orig"

    def test_resolve_falls_back_with_warning(self, tmp_path: Path) -> None:
        torch.save(_raw_hidden_payload("/m/base.safetensors", "fakequant_int8"), tmp_path / "a.pt")
        index = _load_raw_hidden(tmp_path)
        payload = _resolve_payload(index, "/m/base.safetensors")
        assert payload is not None
        assert payload["meta"]["transform"] == "fakequant_int8"

    def test_resolve_unknown_ckpt(self, tmp_path: Path) -> None:
        torch.save(_raw_hidden_payload("/m/base.safetensors", "orig"), tmp_path / "a.pt")
        index = _load_raw_hidden(tmp_path)
        assert _resolve_payload(index, "/m/missing.safetensors") is None

    def test_resolve_multi_device_deterministic(self, tmp_path: Path) -> None:
        torch.save(_raw_hidden_payload("/m/v.safetensors", "orig", device="cuda"), tmp_path / "a.pt")
        torch.save(_raw_hidden_payload("/m/v.safetensors", "orig", device="cpu"), tmp_path / "b.pt")
        p1 = _resolve_payload(_load_raw_hidden(tmp_path), "/m/v.safetensors")
        p2 = _resolve_payload(_load_raw_hidden(tmp_path), "/m/v.safetensors")
        assert p1 is not None and p2 is not None
        assert p1["device"] == p2["device"]  # deterministic pick (sorted keys)


# ---------------------------------------------------------------------------
# run_breadth: anchor-seed derivation (audit bug #7)
# ---------------------------------------------------------------------------


def _rec(variant: str, label: str, seed: int) -> dict[str, Any]:
    return {"variant": variant, "label": label, "seed": seed, "pass": "warm", "path": f"{variant}__{label}__{seed}.png"}


class TestBreadthAnchorSeeds:
    def test_derived_from_base_records(self) -> None:
        records = {
            ("base", "l1", 42): _rec("base", "l1", 42),
            ("base", "l1", 43): _rec("base", "l1", 43),
            ("base", "l2", 42): _rec("base", "l2", 42),
            ("base", "l2", 43): _rec("base", "l2", 43),
            ("heretic", "l1", 42): _rec("heretic", "l1", 42),
        }
        assert _anchor_seed_pair(records, "base", "") == (42, 43)

    def test_derived_uses_lowest_two_when_more(self) -> None:
        records = {("base", "l1", s): _rec("base", "l1", s) for s in (44, 42, 43)}
        assert _anchor_seed_pair(records, "base", "") == (42, 43)

    def test_single_seed_returns_none(self) -> None:
        records = {("base", "l1", 7): _rec("base", "l1", 7)}
        assert _anchor_seed_pair(records, "base", "") is None

    def test_explicit_override(self) -> None:
        records = {("base", "l1", 42): _rec("base", "l1", 42)}
        assert _anchor_seed_pair(records, "base", "5,9") == (5, 9)

    def test_explicit_malformed(self) -> None:
        with pytest.raises(SystemExit):
            _anchor_seed_pair({}, "base", "5")

    def test_load_records_filters_pass(self, tmp_path: Path) -> None:
        manifest = tmp_path / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "images": [
                        {"variant": "base", "label": "l1", "seed": 42, "pass": "warm"},
                        {"variant": "base", "label": "l1", "seed": 42, "pass": "cold"},
                    ]
                }
            )
        )
        records = load_records(manifest, "warm")
        assert list(records) == [("base", "l1", 42)]


# ---------------------------------------------------------------------------
# run_img_metrics: seed derivation + zero-pair fail-loud (audit bug #6)
# ---------------------------------------------------------------------------


class TestImgMetricsSeeds:
    def test_explicit_wins(self) -> None:
        assert _resolve_base_seed(5, {"seed": 7}) == 5

    def test_manifest_seed_used(self) -> None:
        assert _resolve_base_seed(None, {"seed": 7}) == 7

    def test_fallback_42_when_manifest_silent(self) -> None:
        assert _resolve_base_seed(None, {}) == 42

    def test_assert_pairs_mined_all_zero_raises(self) -> None:
        with pytest.raises(SystemExit, match="no pairs mined"):
            _assert_pairs_mined({"te_swap": 0, "cross_prompt": 0, "seed_change": 0, "floor_band": 0})

    def test_assert_pairs_mined_any_nonzero_ok(self) -> None:
        _assert_pairs_mined({"te_swap": 0, "cross_prompt": 0, "seed_change": 1, "floor_band": 0})

    def test_resolve_path_ab_images_tail(self, tmp_path: Path) -> None:
        img = tmp_path / "ab_images" / "x.png"
        img.parent.mkdir()
        img.write_bytes(b"png")
        assert _resolve_path("ab_images/x.png", tmp_path) == img

    def test_resolve_path_missing_returns_candidate(self, tmp_path: Path) -> None:
        cand = _resolve_path("/nowhere/x.png", tmp_path)
        assert str(cand) == "/nowhere/x.png"


# ---------------------------------------------------------------------------
# run_ab_test: workflow patching + prompt suite (audit coverage item #5)
# ---------------------------------------------------------------------------


def _ab_workflow(loader_class: str = "CLIPLoader", node_id: str = "35") -> dict[str, Any]:
    return {
        "34": {"class_type": "KSampler", "inputs": {"seed": 1}},
        node_id: {"class_type": loader_class, "inputs": {"clip_name": "old.safetensors", "type": "lumina2"}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
        "36": {"class_type": "SaveImage", "inputs": {"filename_prefix": "old"}},
    }


class TestPatchWorkflow:
    def test_safetensors_patch(self) -> None:
        wf = _ab_workflow()
        out = patch_workflow(wf, "35", "qwen/base.safetensors", "pos", "neg", seed=7, filename_prefix="p/v")
        assert out["35"]["class_type"] == "CLIPLoader"
        assert out["35"]["inputs"]["clip_name"] == "qwen/base.safetensors"
        assert out["6"]["inputs"]["text"] == "pos"
        assert out["7"]["inputs"]["text"] == "neg"
        assert out["34"]["inputs"]["seed"] == 7
        assert out["36"]["inputs"]["filename_prefix"] == "p/v"

    def test_gguf_swap_preserves_shared_inputs(self) -> None:
        wf = _ab_workflow("CLIPLoader", "35")
        out = patch_workflow(wf, "35", "qwen/base.gguf", "pos", "neg", seed=1, filename_prefix="p")
        assert out["35"]["class_type"] == "CLIPLoaderGGUF"
        assert out["35"]["inputs"]["clip_name"] == "qwen/base.gguf"
        assert out["35"]["inputs"]["type"] == "lumina2"  # preserved from the old node

    def test_input_workflow_not_mutated(self) -> None:
        wf = _ab_workflow()
        original = json.dumps(wf, sort_keys=True)
        patch_workflow(wf, "35", "new.safetensors", "a", "b", seed=2, filename_prefix="x")
        assert json.dumps(wf, sort_keys=True) == original


class TestLoadPromptSuite:
    def test_flat_layout_per_category(self, tmp_path: Path) -> None:
        suite = tmp_path / "s.yaml"
        suite.write_text(
            yaml.safe_dump(
                [
                    {"text": "a cat", "category": "innocent"},
                    {"text": "a dog", "category": "innocent"},
                    {"text": "fight", "category": "violent"},
                ]
            )
        )
        prompts = load_prompt_suite(suite, per_category=1)
        assert [p["category"] for p in prompts] == ["innocent", "violent"]  # first per sorted category

    def test_stress_layout_zero_takes_all(self, tmp_path: Path) -> None:
        suite = tmp_path / "s.yaml"
        suite.write_text(
            yaml.safe_dump(
                {
                    "prompts": [
                        {"text": "x", "category": "c2", "label": "lab-x", "checks": ["k"]},
                        {"text": "y", "category": "c1", "label": "lab-y"},
                    ]
                }
            )
        )
        prompts = load_prompt_suite(suite, per_category=0)
        assert len(prompts) == 2


# ---------------------------------------------------------------------------
# Main() integration smokes — heavy deps stubbed
# ---------------------------------------------------------------------------


def _write_suite(path: Path, prompts: list[tuple[str, str]]) -> None:
    path.write_text(yaml.safe_dump([{"text": t, "category": c} for t, c in prompts]))


def _stub_import_comfy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.text_encoder.backends.comfyui_loader.import_comfy", lambda *a, **k: None)


def _stub_zimage_dit_loader(monkeypatch: pytest.MonkeyPatch, dit: Any) -> None:
    monkeypatch.setattr("src.text_encoder.backends.z_image_dit.load_zimage_dit", lambda *a, **k: dit)


def _fake_divergence_result(base: list[torch.Tensor]) -> dict[str, Any]:
    return {
        "per_prompt": [{"final_latent_div": 0.3, "final_velocity_div": 0.1} for _ in base],
        "noise_control_final": 0.9,
        "identical_control_final": 0.0,
    }


def _taboo_artifact() -> Any:
    """Full-suite raw_hidden stand-in: layer_34 pooled acts differing per category."""
    pooled = torch.zeros(2, 1, 8)
    pooled[1] = 1.0  # the 'violent' prompt rows differ from 'innocent'
    from types import SimpleNamespace

    return SimpleNamespace(
        streams={"layer_34": SimpleNamespace(activations=pooled)},
        categories=["innocent", "violent"],
    )


def _cache_ctxs(te: str, prompts: list[str], fill: float) -> list[torch.Tensor]:
    return [torch.full((4, 8), fill) for _ in prompts]


class TestRunDitDivergenceMain:
    def test_writes_result(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.text_encoder.run_dit_divergence import main as div_main

        suite = tmp_path / "suite.yaml"
        _write_suite(suite, [("a dog", "innocent"), ("a fight", "violent")])
        _stub_import_comfy(monkeypatch)
        _stub_zimage_dit_loader(monkeypatch, object())

        loaded: list[tuple[str, str]] = []

        def fake_load(
            cache_dir: Path, te: str, device: str, ctx: str, prompts: list[str], cats: list[str], **kw: Any
        ) -> list[torch.Tensor]:
            loaded.append((te, device))
            return _cache_ctxs(te, prompts, 0.0)

        monkeypatch.setattr("src.text_encoder.run_dit_divergence._load_consumed_from_cache", fake_load)

        def fake_div(dit: Any, base: list[torch.Tensor], var: list[torch.Tensor], **kw: Any) -> dict[str, Any]:
            return {
                "per_prompt": [
                    {
                        "velocity_div": [0.1, 0.2],
                        "latent_div": [0.01, 0.02],
                        "final_latent_div": 0.02,
                        "final_velocity_div": 0.2,
                    }
                ],
                "noise_control_final": 0.9,
                "identical_control_final": 0.0,
                "n_steps": kw["n_steps"],
                "seed": kw["seed"],
                "latent_hw": kw["latent_hw"],
            }

        monkeypatch.setattr("src.text_encoder.backends.z_image_dit.run_divergence", fake_div)

        results_dir = tmp_path / "results"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "prog",
                "--base",
                "/m/base.safetensors",
                "--variant",
                "/m/heretic.safetensors",
                "--variant-name",
                "heretic",
                "--dit",
                "/m/dit.safetensors",
                "--cache-dir",
                str(tmp_path / "cache"),
                "--results-dir",
                str(results_dir),
                "--suite",
                str(suite),
                "--n-prompts",
                "2",
                "--n-steps",
                "2",
                "--latent-hw",
                "8",
                "--device",
                "cpu",
            ],
        )
        assert div_main() == 0
        # cache loads must be keyed on the RESOLVED cache device (cpu), not --device
        assert {te for te, _dev in loaded} == {"/m/base.safetensors", "/m/heretic.safetensors"}
        assert all(dev == "cpu" for _te, dev in loaded)
        out = json.loads((results_dir / "te_dit_divergence_heretic.json").read_text())
        assert out["summary"]["mean_final_latent_div"] == pytest.approx(0.02)
        assert out["summary"]["div_control_ratio"] == pytest.approx(0.02 / 0.9)


class TestRunDiscriminabilityMain:
    def test_writes_per_variant(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.text_encoder.run_discriminability import main as disc_main

        suite = tmp_path / "suite.yaml"
        _write_suite(suite, [("a dog", "innocent"), ("a fight", "violent")])
        _stub_import_comfy(monkeypatch)

        # DiT responds to the conditioning magnitude so base/variant trajectories differ
        def dit(x: torch.Tensor, t: torch.Tensor, c: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
            return torch.full_like(x, 0.01 * float(c.abs().sum()))

        _stub_zimage_dit_loader(monkeypatch, dit)

        def fake_load(
            cache_dir: Path, te: str, device: str, ctx: str, prompts: list[str], cats: list[str], **kw: Any
        ) -> list[torch.Tensor]:
            return _cache_ctxs(te, prompts, 0.1 if "heretic" in te else 0.0)

        monkeypatch.setattr("src.text_encoder.run_dit_divergence._load_consumed_from_cache", fake_load)

        results_dir = tmp_path / "results"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "prog",
                "--base",
                "/m/base.safetensors",
                "--variants",
                "/m/heretic.safetensors",
                "--variant-names",
                "heretic",
                "--dit",
                "/m/dit.safetensors",
                "--cache-dir",
                str(tmp_path / "cache"),
                "--results-dir",
                str(results_dir),
                "--suite",
                str(suite),
                "--n-prompts",
                "2",
                "--n-steps",
                "2",
                "--latent-hw",
                "8",
                "--device",
                "cpu",
                "--cache-device",
                "cpu",
            ],
        )
        assert disc_main() == 0
        out = json.loads((results_dir / "tebench_discriminability_heretic.json").read_text())
        assert out["base_spread"] > 0  # different per-prompt noise -> nonzero spread
        assert out["spread_ratio"] == pytest.approx(out["variant_spread"] / out["base_spread"])


class TestRunControlsMain:
    def test_writes_battery(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.text_encoder.run_controls import main as controls_main

        suite = tmp_path / "suite.yaml"
        _write_suite(suite, [("a dog", "innocent"), ("a fight", "violent")])
        _stub_import_comfy(monkeypatch)
        _stub_zimage_dit_loader(monkeypatch, object())

        def fake_load(
            cache_dir: Path, te: str, device: str, ctx: str, prompts: list[str], cats: list[str], **kw: Any
        ) -> list[torch.Tensor]:
            return _cache_ctxs(te, prompts, 0.1 if "heretic" in te else 0.0)

        monkeypatch.setattr("src.text_encoder.run_dit_divergence._load_consumed_from_cache", fake_load)
        monkeypatch.setattr("src.text_encoder.cache.load_artifact", lambda cache_dir, key: _taboo_artifact())
        monkeypatch.setattr(
            "src.text_encoder.backends.z_image_dit.run_divergence",
            lambda dit, base, var, **kw: _fake_divergence_result(base),
        )

        results_dir = tmp_path / "results"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "prog",
                "--base",
                "/m/base.safetensors",
                "--heretic",
                "/m/heretic.safetensors",
                "--dit",
                "/m/dit.safetensors",
                "--cache-dir",
                str(tmp_path / "cache"),
                "--results-dir",
                str(results_dir),
                "--suite",
                str(suite),
                "--n-prompts",
                "2",
                "--n-steps",
                "2",
                "--latent-hw",
                "8",
                "--device",
                "cpu",
                "--dose-steps",
                "1.0",
                "--alphas",
                "1.0",
                "--conditions",
                "noise,null",
            ],
        )
        assert controls_main() == 0
        out = json.loads((results_dir / "tebench_controls.json").read_text())
        assert set(out["conditions"]) == {"noise_1x", "null"}
        for cond in out["conditions"].values():
            assert cond["summary"]["div_control_ratio"] == pytest.approx(0.3 / 0.9)
        # max dose = heretic-vs-base MSE = 0.1^2 per prompt
        assert out["max_dose_mse"] == [pytest.approx(0.01), pytest.approx(0.01)]

    def test_merge_refuses_wrong_dose_variant(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """End-to-end --merge guard: a 2x-off sub-1.0 dose reference must refuse (bug #3)."""
        from src.text_encoder.run_controls import main as controls_main

        suite = tmp_path / "suite.yaml"
        _write_suite(suite, [("a dog", "innocent"), ("a fight", "violent")])
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        prior = {
            "backend": "zimage",
            "base": "/m/base.safetensors",
            "heretic": "/m/heretic.safetensors",
            "dit": "/m/dit.safetensors",
            "max_dose_mse": [0.02, 0.02],
            "conditions": {"noise_1x": {}},
            "dose_steps": [1.0],
            "alphas": [1.0],
            "heretic_mse": [0.01, 0.01],
        }
        (results_dir / "tebench_controls.json").write_text(json.dumps(prior))
        _stub_import_comfy(monkeypatch)
        _stub_zimage_dit_loader(monkeypatch, object())

        # recaptured cache yields max dose 0.1^2=0.01, 2x below the prior 0.02
        def fake_load(
            cache_dir: Path, te: str, device: str, ctx: str, prompts: list[str], cats: list[str], **kw: Any
        ) -> list[torch.Tensor]:
            return _cache_ctxs(te, prompts, 0.1 if "heretic" in te else 0.0)

        monkeypatch.setattr("src.text_encoder.run_dit_divergence._load_consumed_from_cache", fake_load)
        monkeypatch.setattr("src.text_encoder.cache.load_artifact", lambda cache_dir, key: _taboo_artifact())
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "prog",
                "--base",
                "/m/base.safetensors",
                "--heretic",
                "/m/heretic.safetensors",
                "--dit",
                "/m/dit.safetensors",
                "--cache-dir",
                str(tmp_path / "cache"),
                "--results-dir",
                str(results_dir),
                "--suite",
                str(suite),
                "--n-prompts",
                "2",
                "--dose-steps",
                "1.0",
                "--alphas",
                "1.0",
                "--conditions",
                "noise",
                "--merge",
            ],
        )
        with pytest.raises(SystemExit, match="merge refused"):
            controls_main()
        # prior file untouched
        assert json.loads((results_dir / "tebench_controls.json").read_text())["conditions"] == {"noise_1x": {}}

    def test_merge_accepts_small_drift(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.text_encoder.run_controls import main as controls_main

        suite = tmp_path / "suite.yaml"
        _write_suite(suite, [("a dog", "innocent"), ("a fight", "violent")])
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        prior = {
            "backend": "zimage",
            "base": "/m/base.safetensors",
            "heretic": "/m/heretic.safetensors",
            "dit": "/m/dit.safetensors",
            "max_dose_mse": [0.02, 0.02],
            "conditions": {},
            "dose_steps": [1.0],
            "alphas": [1.0],
            "heretic_mse": [0.01, 0.01],
        }
        (results_dir / "tebench_controls.json").write_text(json.dumps(prior))
        _stub_import_comfy(monkeypatch)
        _stub_zimage_dit_loader(monkeypatch, object())

        # 1.5% recapture drift (heretic fill 0.1425 -> MSE 0.02030625) passes the guard
        def fake_load(
            cache_dir: Path, te: str, device: str, ctx: str, prompts: list[str], cats: list[str], **kw: Any
        ) -> list[torch.Tensor]:
            return _cache_ctxs(te, prompts, 0.1425 if "heretic" in te else 0.0)

        monkeypatch.setattr("src.text_encoder.run_dit_divergence._load_consumed_from_cache", fake_load)
        monkeypatch.setattr("src.text_encoder.cache.load_artifact", lambda cache_dir, key: _taboo_artifact())
        monkeypatch.setattr(
            "src.text_encoder.backends.z_image_dit.run_divergence",
            lambda dit, base, var, **kw: _fake_divergence_result(base),
        )
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "prog",
                "--base",
                "/m/base.safetensors",
                "--heretic",
                "/m/heretic.safetensors",
                "--dit",
                "/m/dit.safetensors",
                "--cache-dir",
                str(tmp_path / "cache"),
                "--results-dir",
                str(results_dir),
                "--suite",
                str(suite),
                "--n-prompts",
                "2",
                "--dose-steps",
                "1.0",
                "--alphas",
                "1.0",
                "--conditions",
                "noise",
                "--merge",
            ],
        )
        assert controls_main() == 0
        out = json.loads((results_dir / "tebench_controls.json").read_text())
        assert "noise_1x" in out["conditions"]  # merged in
        assert out["max_dose_mse"] == [pytest.approx(0.0203, rel=2e-3)] * 2


class TestRunAttentionMain:
    def test_uses_cache_device_not_runtime_device(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Audit bug #1 regression: the cache loader is keyed on --cache-device."""
        from src.text_encoder.run_attention import main as attn_main

        suite = tmp_path / "suite.yaml"
        _write_suite(suite, [("a dog", "innocent")])
        _stub_import_comfy(monkeypatch)

        def dit(
            x: torch.Tensor, t: torch.Tensor, c: torch.Tensor, n: torch.Tensor, transformer_options: Any = None
        ) -> torch.Tensor:
            return torch.zeros_like(x)

        _stub_zimage_dit_loader(monkeypatch, dit)

        cache_devices: list[str] = []

        def fake_load(
            cache_dir: Path, te: str, device: str, ctx: str, prompts: list[str], cats: list[str], **kw: Any
        ) -> list[torch.Tensor]:
            cache_devices.append(device)
            return [torch.zeros(5, 8) for _ in prompts]

        monkeypatch.setattr("src.text_encoder.run_dit_divergence._load_consumed_from_cache", fake_load)

        results_dir = tmp_path / "results"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "prog",
                "--base",
                "/m/base.safetensors",
                "--variants",
                "/m/heretic.safetensors",
                "--variant-names",
                "heretic",
                "--dit",
                "/m/dit.safetensors",
                "--cache-dir",
                str(tmp_path / "cache"),
                "--results-dir",
                str(results_dir),
                "--suite",
                str(suite),
                "--n-prompts",
                "1",
                "--n-steps",
                "2",
                "--latent-hw",
                "8",
                "--device",
                "cpu",
            ],
        )
        assert attn_main() == 0
        assert cache_devices == ["cpu", "cpu"]  # base + variant, both keyed on cache-device
        out = json.loads((results_dir / "tebench_attention_heretic.json").read_text())
        assert len(out["per_prompt"]) == 1


class TestRunBreadthMain:
    def _setup(self, tmp_path: Path, base_seeds: list[int]) -> tuple[Path, Path]:
        suite = tmp_path / "suite.yaml"
        suite.write_text(
            yaml.safe_dump(
                [
                    {"text": "a dog", "category": "innocent", "label": "l1"},
                    {"text": "a fight", "category": "violent", "label": "l2"},
                ]
            )
        )
        images = tmp_path / "ab_images"
        images.mkdir()
        entries = []
        for seed in base_seeds:
            for var in ("base", "heretic"):
                for lab in ("l1", "l2"):
                    p = images / _png_name(var, lab, seed)
                    _write_png(p, color=(seed, len(var), len(lab)))
                    entries.append({"variant": var, "label": lab, "seed": seed, "pass": "warm", "path": str(p)})
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({"images": entries, "seed": base_seeds[0]}))
        return suite, manifest

    def test_derives_anchor_seeds(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.text_encoder.run_breadth import main as breadth_main

        suite, manifest = self._setup(tmp_path, base_seeds=[42, 43])
        _stub_meters(monkeypatch)
        out_path = tmp_path / "tebench_breadth.json"
        monkeypatch.setattr(
            sys,
            "argv",
            ["prog", "--manifest", str(manifest), "--suite", str(suite), "--out", str(out_path), "--device", "cpu"],
        )
        assert breadth_main() == 0
        out = json.loads(out_path.read_text())
        assert out["anchor_seeds"] == [42, 43]  # derived from base records, not hardcoded
        for cat in ("innocent", "violent"):
            assert out["seed_anchor_per_category"][cat]["n"] == 1  # one label per category
            assert out["per_category"]["heretic"][cat]["n"] == 2  # 2 seeds x 1 label

    def test_single_seed_skips_anchor_loudly(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.text_encoder.run_breadth import main as breadth_main

        suite, manifest = self._setup(tmp_path, base_seeds=[42])
        _stub_meters(monkeypatch)
        out_path = tmp_path / "tebench_breadth.json"
        monkeypatch.setattr(
            sys,
            "argv",
            ["prog", "--manifest", str(manifest), "--suite", str(suite), "--out", str(out_path), "--device", "cpu"],
        )
        assert breadth_main() == 0
        out = json.loads(out_path.read_text())
        assert out["anchor_seeds"] == []
        assert out["seed_anchor_per_category"] == {}


class TestRunImgMetricsMain:
    def _setup(self, tmp_path: Path, seed: int, extra_seed: int | None = None) -> tuple[Path, Path]:
        images = tmp_path / "ab_images"
        images.mkdir(exist_ok=True)
        entries = []
        seeds = [seed] + ([extra_seed] if extra_seed is not None else [])
        for s in seeds:
            for var in ("base_int8", "heretic", "base_fp8"):
                for lab in ("l1", "l2"):
                    fname = f"{var}__{lab}__s{s}.png"
                    _write_png(images / fname, color=(s, len(var), len(lab)))
                    entries.append(
                        {
                            "variant": var,
                            "label": lab,
                            "seed": s,
                            "pass": "warm",
                            "path": f"ab_images/{fname}",
                            "prompt": f"prompt {lab}",
                        }
                    )
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({"images": entries, "seed": seed}))
        return manifest, tmp_path

    def test_manifest_seed_derived_and_files_written(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.text_encoder.run_img_metrics import main as img_main

        manifest, work = self._setup(tmp_path, seed=7, extra_seed=8)
        monkeypatch.setattr("src.text_encoder.run_img_metrics.LPIPSMeter", _StubLPIPS)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "prog",
                "--manifest",
                str(manifest),
                "--out-dir",
                str(work),
                "--device",
                "cpu",
                "--lpips-only",
                "--n-boot",
                "10",
            ],
        )
        assert img_main() == 0
        anchors = json.loads((work / "tebench_img_anchors.json").read_text())
        assert anchors["env"]["base_seed"] == 7  # derived from the manifest, not hardcoded
        assert anchors["seed_change"]["n_pairs"] == 6  # 3 variants x 2 labels @ seed 8 vs 7
        assert anchors["floor_band"]["n_pairs"] == 2
        per_var = json.loads((work / "tebench_img_heretic.json").read_text())
        assert per_var["te_swap"]["n_pairs"] == 2
        assert per_var["te_swap"]["anchor_ratios"]["lpips/seed_change"] == pytest.approx(1.0)

    def test_wrong_base_seed_fails_loud(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """--base-seed not matching the manifest mines 0 pairs -> hard fail (bug #6)."""
        from src.text_encoder.run_img_metrics import main as img_main

        manifest, work = self._setup(tmp_path, seed=7)
        monkeypatch.setattr("src.text_encoder.run_img_metrics.LPIPSMeter", _StubLPIPS)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "prog",
                "--manifest",
                str(manifest),
                "--out-dir",
                str(work),
                "--device",
                "cpu",
                "--lpips-only",
                "--base-seed",
                "42",
            ],
        )
        with pytest.raises(SystemExit, match="no pairs mined"):
            img_main()
        assert not (work / "tebench_img_anchors.json").exists()


class TestRunAnalysisMain:
    def test_end_to_end_with_fake_backend(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from types import SimpleNamespace

        from src.text_encoder.artifact import ConditioningArtifact, Pooling, Stage, Stream, StreamSpec
        from src.text_encoder.run_analysis import main as analysis_main

        suite = tmp_path / "suite.yaml"
        _write_suite(
            suite,
            [("a dog", "innocent"), ("a fight", "violent"), ("a cat", "innocent"), ("a crash", "violent")],
        )
        g = torch.Generator().manual_seed(0)

        def make_artifact(te_file: str, prompts: list[str], categories: list[str]) -> ConditioningArtifact:
            n = len(prompts)
            offset = float(sum(map(ord, te_file)) % 7) * 0.01
            acts = torch.randn(n, 2, 8, generator=g) + offset
            spec = StreamSpec(
                name="consumed_hidden",
                source_encoder="qwen3",
                layers=(0,),
                pooling=Pooling.NONE,
                stage=Stage.RAW_HIDDEN,
                dim=8,
            )
            stream = Stream(name="consumed_hidden", spec=spec, activations=acts, mask=torch.ones(n, 2, dtype=torch.int))
            return ConditioningArtifact(
                model="fake",
                backend="FakeBackend",
                ckpt_id=te_file,
                stage=Stage.RAW_HIDDEN,
                streams={"consumed_hidden": stream},
                prompts=prompts,
                categories=categories,
                dtype="bf16",
                device="cpu",
            )

        encoded: list[str] = []

        class FakeBackend:
            current = ""
            streams = [SimpleNamespace(stage=Stage.RAW_HIDDEN)]

            def load(self, te_file: str, device: str, dit_file: str) -> None:
                FakeBackend.current = te_file
                encoded.append(te_file)

            def apply_fake_quant(self, quant: str) -> None:
                assert quant == "int8"

            def encode(self, prompts: list[str], categories: list[str]) -> list[ConditioningArtifact]:
                return [make_artifact(FakeBackend.current, prompts, categories)]

            def close(self) -> None:
                pass

        def fake_get_backend(name: str) -> FakeBackend:
            assert name == "zimage"
            return FakeBackend()

        monkeypatch.setattr("src.text_encoder.backends.get_backend", fake_get_backend)

        results_dir = tmp_path / "results"
        cache_dir = tmp_path / "cache"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "prog",
                "--base",
                "/m/base.safetensors",
                "--variant",
                "/m/heretic.safetensors",
                "--variant-name",
                "heretic",
                "--dit",
                "/m/dit.safetensors",
                "--backend",
                "zimage",
                "--prompts",
                str(suite),
                "--results-dir",
                str(results_dir),
                "--cache-dir",
                str(cache_dir),
                "--device",
                "cpu",
                "--n-boot",
                "5",
            ],
        )
        assert analysis_main() == 0
        # base (orig) + base (fakequant noise floor) + variant (orig)
        assert [Path(f).name for f in encoded] == [
            "base.safetensors",
            "base.safetensors",
            "heretic.safetensors",
        ]
        assert (results_dir / "te_heretic.json").exists()
        assert (results_dir / "te_heretic.md").exists()
        assert len(list(cache_dir.glob("*.pt"))) == 3  # every encode wrote a cache artifact


# ---------------------------------------------------------------------------
# Shared stubs
# ---------------------------------------------------------------------------


class _StubLPIPS:
    """LPIPSMeter stand-in: deterministic score, no model download."""

    def __init__(self, net: str = "alex", device: str = "cpu") -> None:
        self.net = net
        self.device = device

    def score(self, a: torch.Tensor, b: torch.Tensor) -> float:
        return 0.25


class _StubCLIP:
    def __init__(self, model_name: str = "ViT-L-14", pretrained: str = "openai", device: str = "cpu") -> None:
        self.model_name = model_name
        self.pretrained = pretrained

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        return torch.ones(1, 4)

    def preprocess_pil(self, pil: Any) -> Any:
        return "preprocessed"

    def encode_image(self, x: Any) -> torch.Tensor:
        return torch.ones(1, 4)

    def clip_score(self, x: Any, prompt: str) -> float:
        return 20.0


def _stub_meters(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.text_encoder.run_breadth.LPIPSMeter", _StubLPIPS)
    monkeypatch.setattr("src.text_encoder.run_breadth.CLIPMeter", _StubCLIP)


def _png_name(var: str, lab: str, seed: int) -> str:
    return f"{var}__{lab}__{seed}.png"


def _write_png(path: Path, size: int = 16, color: tuple[int, int, int] = (120, 30, 90)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size), color=color).save(path)
