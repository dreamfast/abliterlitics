"""Tests for src/weight/technique_fingerprint.py — Layer entropy and fingerprint math."""
from __future__ import annotations

import math

import pytest

from src.weight.technique_fingerprint import _layer_entropy


class TestLayerEntropy:
    def test_uniform_distribution_max_entropy(self):
        """Equal counts across all layers -> 100% normalized entropy."""
        counts = {0: 10, 1: 10, 2: 10, 3: 10}
        result = _layer_entropy(counts, total_layers=4)
        assert result == pytest.approx(100.0, abs=0.1)

    def test_single_layer_zero_entropy(self):
        """All changes in one layer -> 0% normalized entropy."""
        counts = {0: 100, 1: 0, 2: 0, 3: 0}
        result = _layer_entropy(counts, total_layers=4)
        assert result == pytest.approx(0.0, abs=0.1)

    def test_empty_counts(self):
        counts = {}
        result = _layer_entropy(counts, total_layers=4)
        assert result == 0.0

    def test_zero_total_layers(self):
        counts = {0: 5}
        result = _layer_entropy(counts, total_layers=0)
        assert result == 0.0

    def test_single_layer_total(self):
        """Single layer with all changes — entropy formula returns 0 when max_entropy=0."""
        counts = {0: 50}
        result = _layer_entropy(counts, total_layers=1)
        # With 1 layer, max_entropy = log2(1) = 0, so division guard returns 0
        assert result == 0.0
