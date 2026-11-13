"""Tests for the shared TE runner CLI helpers (src/text_encoder/_cli.py).

The audit (te-runners-zimage, Duplication #1) consolidated only the
most-duplicated argparse declarations; these tests pin the flag names,
defaults, and required-ness each helper installs.
"""

from __future__ import annotations

import argparse

import pytest

from src.text_encoder._cli import add_backend, add_cache_dir, add_comparison, add_device, add_results_dir


def _parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser()


def _parse(p: argparse.ArgumentParser, argv: list[str]) -> argparse.Namespace:
    return p.parse_args(argv)


class TestAddDevice:
    def test_default_transplanted(self) -> None:
        p = _parser()
        add_device(p, default="cuda")
        assert _parse(p, []).device == "cuda"

    def test_override(self) -> None:
        p = _parser()
        add_device(p, default="cpu")
        assert _parse(p, ["--device", "cuda"]).device == "cuda"


class TestAddCacheDir:
    def test_default_and_override(self) -> None:
        p = _parser()
        add_cache_dir(p, default="/te_cache")
        assert _parse(p, []).cache_dir == "/te_cache"
        assert _parse(p, ["--cache-dir", "/tmp/c"]).cache_dir == "/tmp/c"


class TestAddResultsDir:
    def test_default_and_override(self) -> None:
        p = _parser()
        add_results_dir(p, default="/results")
        assert _parse(p, []).results_dir == "/results"
        assert _parse(p, ["--results-dir", "/tmp/r"]).results_dir == "/tmp/r"


class TestAddBackend:
    def test_default_and_choices(self) -> None:
        p = _parser()
        add_backend(p, default="zimage", choices=("zimage", "krea2"))
        assert _parse(p, []).backend == "zimage"
        assert _parse(p, ["--backend", "krea2"]).backend == "krea2"
        with pytest.raises(SystemExit):
            _parse(p, ["--backend", "h3"])

    def test_wider_choices(self) -> None:
        p = _parser()
        add_backend(p, default="zimage", choices=("zimage", "ltx2", "h3", "krea2"))
        assert _parse(p, ["--backend", "h3"]).backend == "h3"


class TestAddComparison:
    def test_required(self) -> None:
        p = _parser()
        add_comparison(p)
        with pytest.raises(SystemExit):
            _parse(p, [])
        assert _parse(p, ["--comparison", "c/comparison.json"]).comparison == "c/comparison.json"
