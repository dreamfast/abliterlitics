"""Tests for the shared h3/krea2 A/B scaffolding (common_ab.py).

Locks: fail-loud node discovery, the --labels filter contract (label OR
sanitized-text match, fail loud on zero matches), arm-list building (base
first, order preserved, pass-through naming when models root unset), the
manifest UPSERT-by-key contract (reruns never duplicate entries), and the
shared generation loop (skip-existing / dry-run / rerun semantics) with a
faked ComfyUI client.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from src.text_encoder import common_ab


def _suite() -> list[dict[str, Any]]:
    return [
        {"text": "a red cat", "category": "mono", "label": "cat"},
        {"text": "utterly inconceivable", "category": "rare"},  # unlabeled -> sanitize fallback
    ]


def _comparison() -> dict[str, Any]:
    return {
        "settings": {"tebench": {"base_te_file": "/models/text_encoders/base.safetensors"}},
        "variants": {
            "heretic": {"te_file": "/models/text_encoders/heretic.safetensors"},
            "no_te": {"other": 1},
            "aaa_first": {"te_file": "/models/text_encoders/aaa.safetensors"},
        },
    }


class TestFindUniqueNode:
    def test_unique_match(self) -> None:
        wf = {"1": {"class_type": "CLIPLoader", "inputs": {"clip_name": "x"}}}
        nid, node = common_ab.find_unique_node(wf, "CLIPLoader", input_name="clip_name")
        assert nid == "1"
        assert node["inputs"]["clip_name"] == "x"

    def test_zero_matches_raises(self) -> None:
        with pytest.raises(ValueError, match="found 0"):
            common_ab.find_unique_node({}, "CLIPLoader")

    def test_two_matches_raises(self) -> None:
        wf = {"1": {"class_type": "A", "inputs": {}}, "2": {"class_type": "A", "inputs": {}}}
        with pytest.raises(ValueError, match="found 2"):
            common_ab.find_unique_node(wf, "A")

    def test_missing_input_raises(self) -> None:
        wf = {"1": {"class_type": "CLIPLoader", "inputs": {}}}
        with pytest.raises(ValueError, match="lacks input"):
            common_ab.find_unique_node(wf, "CLIPLoader", input_name="clip_name")


class TestFilterSuiteLabels:
    def test_empty_filter_returns_full_suite(self) -> None:
        suite = _suite()
        assert common_ab.filter_suite_labels(suite, "") is suite

    def test_filter_by_label(self) -> None:
        out = common_ab.filter_suite_labels(_suite(), "cat")
        assert [p["label"] for p in out] == ["cat"]

    def test_unlabeled_matches_via_sanitize_fallback(self) -> None:
        out = common_ab.filter_suite_labels(_suite(), "utterly_inconceivable")
        assert [p["text"] for p in out] == ["utterly inconceivable"]

    def test_zero_matches_fails_loud(self) -> None:
        with pytest.raises(SystemExit, match="matched no prompts"):
            common_ab.filter_suite_labels(_suite(), "nope")


class TestBuildArmJobs:
    def test_base_first_then_variant_order_preserved(self) -> None:
        jobs = common_ab.build_arm_jobs(_comparison(), "/models", "")
        assert jobs == [
            ("base", "base.safetensors"),  # text_encoders subdir stripped
            ("heretic", "heretic.safetensors"),
            ("aaa_first", "aaa.safetensors"),  # variants without te_file skipped, order kept
        ]

    def test_variant_filter(self) -> None:
        assert common_ab.build_arm_jobs(_comparison(), "/models", "aaa_first") == [("aaa_first", "aaa.safetensors")]

    def test_base_filter(self) -> None:
        assert common_ab.build_arm_jobs(_comparison(), "/models", "base") == [("base", "base.safetensors")]

    def test_no_arms_fails_loud(self) -> None:
        comp = _comparison()
        comp["settings"]["tebench"]["base_te_file"] = ""
        comp["variants"] = {"no_te": {}}
        with pytest.raises(SystemExit, match="No arms to generate"):
            common_ab.build_arm_jobs(comp, "/models", "")

    def test_unset_models_root_warns_and_passes_through(self, caplog: Any) -> None:
        with caplog.at_level(logging.WARNING):
            jobs = common_ab.build_arm_jobs(_comparison(), "", "")
        assert jobs == [
            ("base", "/models/text_encoders/base.safetensors"),
            ("heretic", "/models/text_encoders/heretic.safetensors"),
            ("aaa_first", "/models/text_encoders/aaa.safetensors"),
        ]
        assert "models root unset" in caplog.text


class TestUpsertManifestEntry:
    def test_append_then_replace_in_place(self) -> None:
        m: dict[str, Any] = {}
        assert common_ab.upsert_manifest_entry(m, "clips", {"key": "k1", "v": 1}) is False
        assert common_ab.upsert_manifest_entry(m, "clips", {"key": "k1", "v": 2}) is True
        assert common_ab.upsert_manifest_entry(m, "clips", {"key": "k2", "v": 3}) is False
        # replaced entry keeps its position; no duplicates ever accumulate
        assert m["clips"] == [{"key": "k1", "v": 2}, {"key": "k2", "v": 3}]


class TestRunAbLoop:
    def _invoke(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        manifest: dict[str, Any],
        calls: list[tuple[str, str, int]],
        *,
        skip_existing: bool = False,
        dry_run: bool = False,
        list_key: str = "clips",
        extra_entry: dict[str, Any] | None = None,
        out_subdir: str | None = None,
    ) -> tuple[Path, Path]:
        def fake_fetch(api_url: str, wf: dict[str, Any], te_node_id: str) -> tuple[bytes, dict[str, Any]]:
            calls.append((wf["te"], wf["prompt"], wf["seed"]))
            return b"DATA", {"server_gen_ms": 5, "client_wall_ms": 10}

        monkeypatch.setattr(common_ab, "queue_and_fetch", fake_fetch)
        monkeypatch.setattr(common_ab, "query_vram", lambda api: 4096)

        def patch_and_node(
            workflow: dict[str, Any], name: str, te_name: str, prompt: str, seed: int
        ) -> tuple[dict[str, Any], str]:
            return {"te": te_name, "prompt": prompt, "seed": seed}, "nodeX"

        tag = out_subdir or f"{list_key}_{dry_run}_{skip_existing}_{len(calls)}"
        out_dir = tmp_path / f"ab_{tag}"
        manifest_path = tmp_path / f"manifest_{tag}.json"
        common_ab.run_ab_loop(
            manifest=manifest,
            manifest_path=manifest_path,
            out_dir=out_dir,
            jobs=[("base", "te_base.safetensors"), ("heretic", "te_her.safetensors")],
            suite=_suite(),
            seeds=[42],
            workflow={},
            api_url="http://x",
            skip_existing=skip_existing,
            dry_run=dry_run,
            list_key=list_key,
            ext=".mp4" if list_key == "clips" else ".webp",
            patch_and_node=patch_and_node,
            extra_entry=extra_entry,
        )
        return out_dir, manifest_path

    def test_generate_write_and_vram_capture(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[tuple[str, str, int]] = []
        manifest: dict[str, Any] = {"clips": []}
        out_dir, manifest_path = self._invoke(tmp_path, monkeypatch, manifest, calls)

        assert len(calls) == 4  # 2 arms x 2 prompts x 1 seed
        assert len(manifest["clips"]) == 4
        # on-disk manifest matches in-memory (rewritten after every clip)
        assert json.loads(manifest_path.read_text())["clips"] == manifest["clips"]
        # vram captured on the first clip of EACH arm only
        assert [c["vram"] for c in manifest["clips"]] == [4096, None, 4096, None]
        # clip files written under <out>/<arm>/<stem>__seed<seed>.mp4
        assert (out_dir / "base" / "cat__seed42.mp4").exists()
        assert (out_dir / "heretic" / "utterly_inconceivable__seed42.mp4").exists()

    def test_rerun_without_skip_existing_upserts_not_duplicates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[tuple[str, str, int]] = []
        manifest: dict[str, Any] = {"clips": []}
        _, _ = self._invoke(tmp_path, monkeypatch, manifest, calls)
        assert len(manifest["clips"]) == 4

        calls.clear()
        _, _ = self._invoke(tmp_path, monkeypatch, manifest, calls)
        # regeneration happened (4 new fetches) but the manifest did NOT grow
        assert len(calls) == 4
        assert len(manifest["clips"]) == 4
        keys = [c["key"] for c in manifest["clips"]]
        assert len(keys) == len(set(keys))

    def test_skip_existing_skips_completed_clips(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[tuple[str, str, int]] = []
        manifest: dict[str, Any] = {"clips": []}
        _, _ = self._invoke(tmp_path, monkeypatch, manifest, calls, out_subdir="shared")

        calls.clear()
        # SAME out_dir as run 1: every clip file exists -> nothing re-fetched
        _, _ = self._invoke(tmp_path, monkeypatch, manifest, calls, skip_existing=True, out_subdir="shared")
        assert calls == []
        assert len(manifest["clips"]) == 4

    def test_dry_run_generates_nothing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[tuple[str, str, int]] = []
        manifest: dict[str, Any] = {"clips": []}
        out_dir, _ = self._invoke(tmp_path, monkeypatch, manifest, calls, dry_run=True)
        assert calls == []
        assert manifest["clips"] == []
        assert not out_dir.exists()

    def test_extra_entry_passthrough(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[tuple[str, str, int]] = []
        manifest: dict[str, Any] = {"images": []}
        _, _ = self._invoke(tmp_path, monkeypatch, manifest, calls, list_key="images", extra_entry={"pass": "warm"})
        assert all(c["pass"] == "warm" for c in manifest["images"])


def test_suite_loader_delegates_share_contract(tmp_path: Path) -> None:
    """load_h3_suite/load_krea2_suite are documented delegates of common_ab."""
    from src.text_encoder.run_h3_ab import load_h3_suite
    from src.text_encoder.run_krea2_ab import load_krea2_suite

    f = tmp_path / "suite.yaml"
    f.write_text("prompts:\n  - text: hello\n    category: c\n")
    assert load_h3_suite(f) == load_krea2_suite(f) == common_ab.load_prompt_suite(f)
    assert load_h3_suite(f)[0]["text"] == "hello"
