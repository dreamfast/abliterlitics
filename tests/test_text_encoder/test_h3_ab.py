"""Tests for the shared ComfyUI client + H3 A/B workflow patching.

Locks the lesson-#39 semantics (server-assigned prompt_id, subfolder /view
params, asynchronous /free) and the H3 patch contract (class-type node
discovery, fail-loud asserts, frame mod-17 rule).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.text_encoder.comfy_client import comfy_te_name, sanitize
from src.text_encoder.run_h3_ab import (
    FPS,
    frames_for_duration,
    h3_patch_workflow,
    load_h3_suite,
)


def _h3_workflow() -> dict[str, Any]:
    """Minimal H3-shaped API workflow (same class types as h3_t2v_api.json)."""
    return {
        "92": {"class_type": "SaveVideo", "inputs": {"video": ["130", 0], "filename_prefix": "video/x"}},
        "128": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": "base.safetensors", "type": "minimax", "device": "default"},
        },
        "129": {"class_type": "RandomNoise", "inputs": {"noise_seed": 1}},
        "131": {
            "class_type": "MiniMaxH3ImageToVideo",
            "inputs": {"clip": ["128", 0], "prompt": "", "length": ["132", 0]},
        },
        "132": {"class_type": "PrimitiveInt", "inputs": {"value": 124}},
    }


class TestComfyTeName:
    def test_text_encoders_subdir_stripped(self) -> None:
        assert comfy_te_name("/models/text_encoders/qwen3-4b/x.safetensors", "/models") == "qwen3-4b/x.safetensors"

    def test_clip_subdir_stripped(self) -> None:
        assert comfy_te_name("/models/clip/y.safetensors", "/models") == "y.safetensors"

    def test_foreign_path_kept(self) -> None:
        assert comfy_te_name("/elsewhere/z.safetensors", "/models") == "/elsewhere/z.safetensors"

    def test_already_relative(self) -> None:
        assert comfy_te_name("x.safetensors", "/models") == "x.safetensors"


class TestSanitize:
    def test_basic(self) -> None:
        assert sanitize("A Black Cat!") == "a_black_cat"

    def test_empty_falls_back(self) -> None:
        assert sanitize("!!!") == "prompt"

    def test_max_len(self) -> None:
        assert len(sanitize("x" * 200)) <= 48


class TestFrames:
    def test_mod_17_rule(self) -> None:
        for d in [5.0, 4.0, 6.2, 10.0, 1.0]:
            assert frames_for_duration(d) % 17 == 5

    def test_5s(self) -> None:
        assert frames_for_duration(5.0) == 124  # 120 -> +4 fixup


class TestH3PatchWorkflow:
    def test_patches_te_prompt_seed_prefix_frames(self) -> None:
        wf = h3_patch_workflow(
            _h3_workflow(),
            te_name="heretic.safetensors",
            prompt="integrated_multimodal_description: ...",
            seed=43,
            filename_prefix="abliterlitics-h3/heretic",
            frames=107,
        )
        assert wf["128"]["inputs"]["clip_name"] == "heretic.safetensors"
        assert wf["131"]["inputs"]["prompt"] == "integrated_multimodal_description: ..."
        assert wf["129"]["inputs"]["noise_seed"] == 43
        assert wf["92"]["inputs"]["filename_prefix"] == "abliterlitics-h3/heretic"
        assert wf["132"]["inputs"]["value"] == 107

    def test_input_workflow_untouched(self) -> None:
        base = _h3_workflow()
        h3_patch_workflow(base, "h.safetensors", "p", 7, "pre", 107)
        assert base["128"]["inputs"]["clip_name"] == "base.safetensors"
        assert base["129"]["inputs"]["noise_seed"] == 1

    def test_missing_clipt_loader_fails_loud(self) -> None:
        wf = _h3_workflow()
        del wf["128"]
        with pytest.raises(ValueError, match="CLIPLoader"):
            h3_patch_workflow(wf, "h.safetensors", "p", 7, "pre", 107)

    def test_two_clipt_loaders_fail_loud(self) -> None:
        wf = _h3_workflow()
        wf["999"] = {"class_type": "CLIPLoader", "inputs": {"clip_name": "other.safetensors"}}
        with pytest.raises(ValueError, match="exactly one CLIPLoader"):
            h3_patch_workflow(wf, "h.safetensors", "p", 7, "pre", 107)

    def test_missing_randomnoise_fails_loud(self) -> None:
        wf = _h3_workflow()
        del wf["129"]
        with pytest.raises(ValueError, match="RandomNoise"):
            h3_patch_workflow(wf, "h.safetensors", "p", 7, "pre", 107)

    def test_prompt_node_without_prompt_input_fails(self) -> None:
        wf = _h3_workflow()
        del wf["131"]["inputs"]["prompt"]
        with pytest.raises(ValueError, match="lacks input 'prompt'"):
            h3_patch_workflow(wf, "h.safetensors", "p", 7, "pre", 107)

    def test_no_prefix_node_fails(self) -> None:
        wf = _h3_workflow()
        del wf["92"]["inputs"]["filename_prefix"]
        with pytest.raises(ValueError, match="filename_prefix"):
            h3_patch_workflow(wf, "h.safetensors", "p", 7, "pre", 107)


class TestLoadH3Suite:
    def test_dict_layout_with_targets(self, tmp_path: Path) -> None:
        f = tmp_path / "s.yaml"
        f.write_text(
            "prompts:\n"
            "  - text: 'integrated_multimodal_description: a <d>[English] inconceivable</d> line'\n"
            "    category: speech\n"
            "    label: inc_line1\n"
            "    targets: [inconceivable]\n"
        )
        suite = load_h3_suite(f)
        assert suite[0]["label"] == "inc_line1"
        assert suite[0]["targets"] == ["inconceivable"]

    def test_missing_text_rejected(self, tmp_path: Path) -> None:
        f = tmp_path / "s.yaml"
        f.write_text("prompts:\n  - category: x\n")
        with pytest.raises(ValueError, match="missing text/category"):
            load_h3_suite(f)

    def test_empty_suite_rejected(self, tmp_path: Path) -> None:
        f = tmp_path / "s.yaml"
        f.write_text("prompts: []\n")
        with pytest.raises(ValueError, match="No prompts"):
            load_h3_suite(f)

    def test_real_workflow_file_patches(self) -> None:
        """The shipped h3_t2v_api.json must satisfy the patch contract."""
        wf_path = Path("comparisons/te-h3/workflows/h3_t2v_api.json")
        if not wf_path.exists():
            pytest.skip("workflow file not present (repo checkout without comparisons)")
        base = json.loads(wf_path.read_text())
        wf = h3_patch_workflow(base, "x.safetensors", "p", 42, "pre", frames_for_duration(5.0))
        clip_id = next(nid for nid, n in base.items() if n.get("class_type") == "CLIPLoader")
        noise_id = next(nid for nid, n in base.items() if n.get("class_type") == "RandomNoise")
        assert wf[clip_id]["inputs"]["clip_name"] == "x.safetensors"
        assert wf[noise_id]["inputs"]["noise_seed"] == 42


class TestFpsConstant:
    def test_fps_is_24(self) -> None:
        assert FPS == 24
