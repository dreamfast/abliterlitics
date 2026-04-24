"""Tests for src/benchmark/harmbench_review.py — Review classification constants."""
from __future__ import annotations

import pytest


class TestHarmbenchReviewConstants:
    def test_module_importable(self):
        import src.benchmark.harmbench_review
        assert hasattr(src.benchmark.harmbench_review, "RESULTS_VERSION")

    def test_refusal_patterns(self):
        from src.benchmark.harmbench_review import REFUSAL_PATTERNS
        assert isinstance(REFUSAL_PATTERNS, list)
        assert len(REFUSAL_PATTERNS) > 0
