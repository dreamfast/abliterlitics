"""Tests for the Krea2 A/B workflow patching + prompt suite.

Locks the patch contract against the two-stage sampler graph (BOTH KSampler
seeds patched — each stage derives noise from its seed), the krea2
CLIPLoader-type guard (wrong workflow file fails loud), and suite loading
with targets passthrough.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.text_encoder.run_krea2_ab import krea2_patch_workflow, load_krea2_suite


def _krea2_workflow() -> dict[str, Any]:
    """Minimal krea2-shaped two-stage API workflow (same class types as krea2_t2i_api.json)."""
    return {
        "49": {
            "class_type": "ResolutionSelector",
            "inputs": {"aspect_ratio": "4:3 (Standard)", "megapixels": 1.0, "multiple": 8},
        },
        "53": {"class_type": "EmptyLatentImage", "inputs": {"width": ["49", 0], "height": ["49", 1], "batch_size": 1}},
        "54": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": "base.safetensors", "type": "krea2", "device": "default"},
        },
        "55": {"class_type": "VAELoader", "inputs": {"vae_name": "wan.safetensors"}},
        "69": {"class_type": "UNETLoader", "inputs": {"unet_name": "dit.safetensors", "weight_dtype": "default"}},
        "70": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["54", 0]}},
        "71": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["70", 0]}},
        "72": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["69", 0],
                "positive": ["70", 0],
                "negative": ["71", 0],
                "latent_image": ["53", 0],
                "seed": 111,
                "steps": 6,
                "cfg": 1.0,
                "sampler_name": "res_2s",
                "scheduler": "beta",
                "denoise": 1.0,
            },
        },
        "73": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["69", 0],
                "positive": ["70", 0],
                "negative": ["71", 0],
                "latent_image": ["72", 0],
                "seed": 222,
                "steps": 2,
                "cfg": 1.0,
                "sampler_name": "deis_3m",
                "scheduler": "bong_tangent",
                "denoise": 0.2,
            },
        },
        "68": {"class_type": "VAEDecode", "inputs": {"samples": ["73", 0], "vae": ["55", 0]}},
        "74": {"class_type": "SaveImageWebP", "inputs": {"images": ["68", 0], "filename_prefix": "ComfyUI"}},
    }


class TestKrea2Patch:
    def test_patches_te_prompt_seeds_prefix(self) -> None:
        wf = krea2_patch_workflow(
            _krea2_workflow(),
            te_name="heretic_int8.safetensors",
            prompt="a sign reading COFFEE",
            seed=4242,
            filename_prefix="abl-krea2/heretic_int8",
        )
        assert wf["54"]["inputs"]["clip_name"] == "heretic_int8.safetensors"
        assert wf["70"]["inputs"]["text"] == "a sign reading COFFEE"
        # BOTH sampler stages share the A/B seed (each stage derives noise from it)
        assert wf["72"]["inputs"]["seed"] == 4242
        assert wf["73"]["inputs"]["seed"] == 4242
        assert wf["74"]["inputs"]["filename_prefix"] == "abl-krea2/heretic_int8"

    def test_input_workflow_untouched(self) -> None:
        original = _krea2_workflow()
        snapshot = json.loads(json.dumps(original))
        krea2_patch_workflow(original, "x.safetensors", "p", 1, "pre")
        assert original == snapshot

    def test_wrong_clip_type_fails_loud(self) -> None:
        wf = _krea2_workflow()
        wf["54"]["inputs"]["type"] = "qwen_image"
        with pytest.raises(ValueError, match="krea2"):
            krea2_patch_workflow(wf, "x.safetensors", "p", 1, "pre")

    def test_missing_clip_loader_fails_loud(self) -> None:
        wf = _krea2_workflow()
        del wf["54"]
        with pytest.raises(ValueError, match="CLIPLoader"):
            krea2_patch_workflow(wf, "x.safetensors", "p", 1, "pre")

    def test_two_clip_text_encodes_fail_loud(self) -> None:
        wf = _krea2_workflow()
        wf["70b"] = {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["54", 0]}}
        with pytest.raises(ValueError, match="CLIPTextEncode"):
            krea2_patch_workflow(wf, "x.safetensors", "p", 1, "pre")

    def test_no_sampler_fails_loud(self) -> None:
        wf = _krea2_workflow()
        del wf["72"]
        del wf["73"]
        with pytest.raises(ValueError, match="KSampler"):
            krea2_patch_workflow(wf, "x.safetensors", "p", 1, "pre")

    def test_no_prefix_node_fails(self) -> None:
        wf = _krea2_workflow()
        wf["74"]["inputs"].pop("filename_prefix")
        with pytest.raises(ValueError, match="filename_prefix"):
            krea2_patch_workflow(wf, "x.safetensors", "p", 1, "pre")

    def test_real_workflow_file_patches(self) -> None:
        """The committed workflow file stays patch-compatible (guards re-exports)."""
        wf_path = Path(__file__).parents[2] / "comparisons/te-krea2/workflows/krea2_t2i_api.json"
        if not wf_path.exists():
            pytest.skip("workflow file not present (comparisons/ untracked in some checkouts)")
        wf = krea2_patch_workflow(json.loads(wf_path.read_text()), "heretic.safetensors", "p", 7, "pre")
        clip = next(n for n in wf.values() if n.get("class_type") == "CLIPLoader")
        assert clip["inputs"]["clip_name"] == "heretic.safetensors"


class TestSuite:
    def test_dict_layout_with_targets(self, tmp_path: Path) -> None:
        f = tmp_path / "suite.yaml"
        f.write_text(
            "prompts:\n"
            "  - label: sign_coffee\n"
            "    category: signage\n"
            '    text: "a sign reading COFFEE"\n'
            '    targets: ["COFFEE"]\n'
            "  - label: control_x\n"
            "    category: control\n"
            '    text: "a landscape"\n'
        )
        suite = load_krea2_suite(f)
        assert suite[0]["targets"] == ["COFFEE"]
        assert suite[1].get("targets", []) == []

    def test_missing_text_rejected(self, tmp_path: Path) -> None:
        f = tmp_path / "suite.yaml"
        f.write_text("prompts:\n  - label: x\n    category: signage\n")
        with pytest.raises(ValueError, match="text/category"):
            load_krea2_suite(f)

    def test_empty_suite_rejected(self, tmp_path: Path) -> None:
        f = tmp_path / "suite.yaml"
        f.write_text("prompts: []\n")
        with pytest.raises(ValueError, match="No prompts"):
            load_krea2_suite(f)
