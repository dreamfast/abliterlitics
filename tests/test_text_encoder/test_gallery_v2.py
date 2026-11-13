"""Tests for the Phase-6 gallery v2 — strict same-seed pairing, seed slider,
word-readout strips (P12/P13/P14).

The seed-pairing regression: the Phase-4 top-up appended seeds 43+ for a subset
of variants; the old last-record-wins selection cross-paired seeds across
variants (seed-change LPIPS 0.516 = "a different picture").
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from src.text_encoder.generate_report import _ab_section


def _webp(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=(120, 120, 120)).save(path, "WEBP")


def _manifest(tmp: Path, seeds_by_variant: dict[str, list[int]]) -> Path:
    """Manifest with 1 label x given variants x given seeds (warm pass only)."""
    images: list[dict[str, Any]] = []
    for var, seeds in seeds_by_variant.items():
        for s in seeds:
            name = f"graffiti__pass-warm__seed{s}.webp"
            _webp(tmp / "ab_images" / var / name)
            images.append(
                {
                    "key": f"{var}::abliterlitics_graffiti::prompt text",
                    "variant": var,
                    "label": "abliterlitics_graffiti",
                    "prompt": "a graffiti alley with a black cat on a windowsill",
                    "checks": ["cat visible"],
                    "pass": "warm",
                    "seed": s,
                    "path": str((tmp / "ab_images" / var / name).relative_to(tmp.parent)),
                    "perf": {},
                }
            )
    mdir = tmp / "ab_images"
    (mdir / "manifest.json").write_text(json.dumps({"seed": 42, "images": images}))
    return mdir / "manifest.json"


def _results_dir(tmp: Path) -> Path:
    return tmp  # _ab_section reads results_dir/ab_images/manifest.json


class TestGalleryPairing:
    def test_same_seed_pairing_despite_later_records(self, tmp_path: Path) -> None:
        """base has seeds [42, 49]; gguf_f16 only [42] — both panes must use 42."""
        results = _results_dir(tmp_path)
        _manifest(results, {"base": [42, 49], "base_gguf_f16": [42]})
        html = _ab_section([], {}, results, split_images=False)
        assert 'id="sec-ab"' in html
        import re

        m = re.search(r"const IMG = (\{.*?\});\n", html, re.S)
        assert m, "IMG payload missing"
        img = json.loads(m.group(1))
        g = img["abliterlitics_graffiti"]
        # base: only the DEFAULT seed entry is embedded when no common slider
        # seeds exist beyond it; and it must be the seed-42 file, never 49
        assert sorted(g["base"].keys()) == ["42"]
        assert sorted(g["base_gguf_f16"].keys()) == ["42"]
        assert g["base"]["42"].endswith("seed42.webp") or ": seed" not in g["base"]["42"]
        assert "seed49" not in json.dumps(img)

    def test_slider_seeds_are_the_intersection(self, tmp_path: Path) -> None:
        results = _results_dir(tmp_path)
        _manifest(results, {"base": [42, 43, 44], "heretic": [42, 43, 44]})
        html = _ab_section([], {}, results, split_images=False)
        import re

        assert re.search(r"const SEEDS = \[42,\s*43,\s*44\]", html)
        img = json.loads(re.search(r"const IMG = (\{.*?\});\n", html, re.S).group(1))
        assert sorted(img["abliterlitics_graffiti"]["base"].keys()) == ["42", "43", "44"]

    def test_slider_excludes_partial_seeds(self, tmp_path: Path) -> None:
        """One variant missing seed 44 -> slider is the intersection [42,43]."""
        results = _results_dir(tmp_path)
        _manifest(results, {"base": [42, 43, 44], "heretic": [42, 43]})
        html = _ab_section([], {}, results, split_images=False)
        import re

        assert re.search(r"const SEEDS = \[42,\s*43\]", html)
        assert "44" not in re.search(r"const SEEDS = (\[[^\]]*\])", html).group(1)
        img44 = json.loads(re.search(r"const IMG = (\{.*?\});\n", html, re.S).group(1))
        assert "44" not in json.dumps(img44)  # no seed-44 image smuggled in either

    def test_split_images_mode_uses_relative_paths(self, tmp_path: Path) -> None:
        results = _results_dir(tmp_path)
        _manifest(results, {"base": [42], "heretic": [42]})
        graphs = tmp_path.parent / "graphs"
        html = _ab_section([], {}, results, split_images=True, graphs_dir=graphs)
        assert "data:image/webp" not in html
        assert "ab_images/abliterlitics_graffiti/base_s42.webp" in html
        assert (graphs / "ab_images" / "abliterlitics_graffiti" / "base_s42.webp").exists()


class TestGalleryWordPanel:
    def _doc(self) -> dict[str, Any]:
        toks = ["<|im_start|>", "user", "a", " black", " cat", " on", " a", " windowsill", "<|im_end|>"]
        base = [0.30, 0.10, 0.02, 0.06, 0.05, 0.02, 0.01, 0.05, 0.10]
        var = [0.30, 0.10, 0.02, 0.06, 0.01, 0.02, 0.01, 0.01, 0.10]
        return {
            "variant_name": "base_int4",
            "has_word_readout": True,
            "jsd_mean_all": 0.0074,
            "per_prompt": [
                {
                    "prompt": "a black cat on a windowsill",
                    "category": "innocent",
                    "label": "abliterlitics_graffiti",
                    "tokens": toks,
                    "base_mean_readout": base,
                    "variant_mean_readout": var,
                }
            ],
        }

    def test_rows_compact_data(self) -> None:
        from src.text_encoder.phase4_report import gallery_word_rows

        rows = gallery_word_rows(self._doc(), "abliterlitics_graffiti")
        assert rows is not None
        words = [r[0] for r in rows if r[3] == "word"]
        assert "cat" in words and "windowsill" in words
        # repeated words kept separate (positional pairing, no summing)
        assert words.count("a") == 2
        # [word, mb, mv, kind] quadruples, floats rounded
        cat = next(r for r in rows if r[0] == "cat")
        assert len(cat) == 4 and isinstance(cat[1], float) and isinstance(cat[2], float)
        assert cat[2] < cat[1]  # mass drops under this variant

    def test_rows_none_for_unknown_label(self) -> None:
        from src.text_encoder.phase4_report import gallery_word_rows

        assert gallery_word_rows(self._doc(), "no_such_label") is None

    def test_build_gallery_word_data(self) -> None:
        from src.text_encoder.phase4_report import build_gallery_word_data

        data = build_gallery_word_data({"base_int4": self._doc()})
        assert "abliterlitics_graffiti" in data
        entry = data["abliterlitics_graffiti"]["base_int4"]
        assert set(entry.keys()) == {"prompt", "rows"}
        assert "black cat" in entry["prompt"]

    def test_ab_section_includes_words_payload(self, tmp_path: Path) -> None:
        results = _results_dir(tmp_path)
        _manifest(results, {"base": [42], "heretic": [42]})
        (results / "tebench_attention_base_int4.json").write_text(json.dumps(self._doc()))
        # base_int4 has no gallery images; the strip still lands in WORDS
        html = _ab_section([], {}, results, split_images=False)
        assert "const WORDS = " in html
        import re

        words = json.loads(re.search(r"const WORDS = (\{.*?\});\n", html, re.S).group(1))
        assert "base_int4" in words["abliterlitics_graffiti"]
        # compact rows payload — NO pre-rendered table HTML in the report
        assert "rows" in words["abliterlitics_graffiti"]["base_int4"]
        assert "<table" not in json.dumps(words)
        # client-side paragraph renderer present
        assert "ab-words-para" in html and "renderWord" not in html


class TestSuiteLabels:
    def test_labels_loaded_from_stress_suite(self) -> None:
        from src.text_encoder.run_attention import _suite_labels

        labels = _suite_labels("prompts/ab_stress_prompts.yaml")
        assert len(labels) >= 18
        assert "abliterlitics_graffiti" in labels.values()

    def test_missing_file_returns_empty(self) -> None:
        from src.text_encoder.run_attention import _suite_labels

        assert _suite_labels("/nonexistent.yaml") == {}

    def test_malformed_entry_skipped(self, tmp_path: Path) -> None:
        from src.text_encoder.run_attention import _suite_labels

        f = tmp_path / "suite.yaml"
        f.write_text(
            "prompts:\n  - label: has_no_text\n    category: x\n  - text: hello\n    label: kept\n    category: x\n"
        )
        labels = _suite_labels(str(f))
        assert labels == {"hello": "kept"}


class TestWordContentMerge:
    _doc = TestGalleryWordPanel._doc  # reuse the attention-doc fixture

    def _content(self) -> dict[str, dict[str, list[list[Any]]]]:
        # must mirror _sample_attn_doc's token grouping exactly (8 rows):
        # [chat], a, black, cat, on, a, windowsill, [chat]
        rows = [
            ["[chat template]", 0.99, "chat"],
            ["a", 0.99, "word"],
            ["black", 0.95, "word"],
            ["cat", 0.96, "word"],
            ["on", 0.97, "word"],
            ["a", 0.99, "word"],
            ["windowsill", 0.93, "word"],
            ["[chat template]", 0.99, "chat"],
        ]
        return {"base_int4": {"abliterlitics_graffiti": rows}}

    def test_cos_merged_index_aligned(self) -> None:
        from src.text_encoder.phase4_report import build_gallery_word_data

        data = build_gallery_word_data({"base_int4": self._doc()}, self._content())
        rows = data["abliterlitics_graffiti"]["base_int4"]["rows"]
        assert all(len(r) == 5 for r in rows)
        cat = next(r for r in rows if r[0] == "cat")
        assert cat[4] == pytest.approx(0.96)

    def test_mismatched_content_skipped(self) -> None:
        from src.text_encoder.phase4_report import build_gallery_word_data

        bad = {"base_int4": {"abliterlitics_graffiti": [["[chat template]", 0.99, "chat"], ["a", 0.99, "word"]]}}
        data = build_gallery_word_data({"base_int4": self._doc()}, bad)
        rows = data["abliterlitics_graffiti"]["base_int4"]["rows"]
        assert all(len(r) == 4 for r in rows)  # no cos, no None padding

    def test_token_groups_tracks_indices(self) -> None:
        from src.text_encoder.attention_metrics import token_groups, word_rows_ordered

        toks = ["<|im_start|>", "user", "a", " black", " cat", "<pad>", "<pad>"]
        groups = token_groups(toks)
        assert [g[0] for g in groups] == ["[chat template]", "a", "black", "cat", "[pad]"]
        assert groups[3][2] == [4]  # "cat" is its own space-prefixed word
        # grouping equivalence with word_rows_ordered labels/kinds
        rows = word_rows_ordered(toks, [0.3, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05])
        assert [r[0] for r in rows] == [g[0] for g in groups]
        assert [r[2] for r in rows] == [g[1] for g in groups]
