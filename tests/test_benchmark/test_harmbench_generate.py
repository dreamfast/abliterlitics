"""Tests for src/benchmark/harmbench_generate.py — Constants and structure."""
from __future__ import annotations

import pytest


class TestHarmbenchGenerateConstants:
    def test_csv_url_importable(self):
        from src.benchmark.harmbench_generate import HARMBENCH_BEHAVIORS_CSV
        assert isinstance(HARMBENCH_BEHAVIORS_CSV, str)
        assert "githubusercontent.com" in HARMBENCH_BEHAVIORS_CSV
        assert "HarmBench" in HARMBENCH_BEHAVIORS_CSV

    def test_refusal_substrings_importable(self):
        from src.benchmark.harmbench_generate import REFUSAL_SUBSTRINGS
        assert isinstance(REFUSAL_SUBSTRINGS, list)
        assert len(REFUSAL_SUBSTRINGS) > 5
        assert "I cannot" in REFUSAL_SUBSTRINGS
