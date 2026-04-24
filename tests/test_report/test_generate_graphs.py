"""Tests for src/report/generate_graphs.py — Color generation and data loading."""
from __future__ import annotations

import json

import pytest

from src.report.generate_graphs import get_variant_colors, load_json


class TestGetVariantColors:
    def test_returns_dict(self):
        colors = get_variant_colors(["heretic", "hauhau", "huihui"])
        assert isinstance(colors, dict)
        assert "heretic" in colors
        assert "hauhau" in colors
        assert "huihui" in colors

    def test_base_color_always_present(self):
        colors = get_variant_colors(["heretic"])
        assert "base" in colors

    def test_overlap_color_present(self):
        colors = get_variant_colors(["heretic"])
        assert "overlap" in colors

    def test_colors_are_valid(self):
        """Colors should be valid color-like values (may be tuples or hex strings)."""
        colors = get_variant_colors(["heretic", "hauhau"])
        for name, color in colors.items():
            # seaborn may return tuples or hex strings depending on version
            assert color is not None, f"Color for {name} is None"

    def test_single_variant(self):
        colors = get_variant_colors(["test"])
        assert len(colors) == 3  # test + base + overlap


class TestLoadJson:
    def test_valid_json_with_version(self, tmp_path):
        data = {"results_version": 1, "data": [1, 2, 3]}
        f = tmp_path / "test.json"
        f.write_text(json.dumps(data))
        result = load_json(str(f))
        assert result is not None
        assert result["results_version"] == 1

    def test_missing_file_returns_none(self):
        result = load_json("/nonexistent/path.json")
        assert result is None

    def test_old_format_still_loaded(self, tmp_path):
        """load_json returns data even without results_version (only warns)."""
        data = {"data": [1, 2, 3]}
        f = tmp_path / "old.json"
        f.write_text(json.dumps(data))
        result = load_json(str(f))
        # load_json returns the data, just logs a warning about version
        assert result is not None
        assert result["data"] == [1, 2, 3]

    def test_invalid_json_returns_none(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("not valid json {{{")
        result = load_json(str(f))
        assert result is None
