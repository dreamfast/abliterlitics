"""Tests for src/weight/cross_arch_comparison.py — Module importability."""
from __future__ import annotations


class TestCrossArchImportable:
    def test_module_imports(self):
        import src.weight.cross_arch_comparison
        assert hasattr(src.weight.cross_arch_comparison, "main")
