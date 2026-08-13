"""Tests for prompt suite loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.text_encoder.suite import count_per_category, load_suite, suite_hash


class TestLoadSuite:
    def test_loads_default_suite(self) -> None:
        suite_path = Path(__file__).parent.parent.parent / "prompts" / "te_suite.yaml"
        prompts, categories = load_suite(suite_path)
        assert len(prompts) == len(categories)
        assert len(prompts) > 0

    def test_counts_per_category(self) -> None:
        suite_path = Path(__file__).parent.parent.parent / "prompts" / "te_suite.yaml"
        _, categories = load_suite(suite_path)
        counts = count_per_category(categories)
        assert "innocent" in counts
        assert "extreme" in counts
        assert "multilingual" in counts
        assert all(c >= 5 for c in counts.values()), f"Need >=5 per category, got {counts}"

    def test_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_suite("/nonexistent/path.yaml")

    def test_suite_hash_deterministic(self) -> None:
        prompts = ["a", "b", "c"]
        cats = ["x", "y", "z"]
        assert suite_hash(prompts, cats) == suite_hash(prompts, cats)

    def test_suite_hash_different_content(self) -> None:
        assert suite_hash(["a"], ["x"]) != suite_hash(["b"], ["x"])

    def test_suite_hash_order_independent(self) -> None:
        """Sorted hash — reordered content with same items = same hash."""
        assert suite_hash(["a", "b"], ["x", "y"]) == suite_hash(["b", "a"], ["y", "x"])
