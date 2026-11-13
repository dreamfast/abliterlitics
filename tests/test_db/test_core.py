"""Tests for src/db/core.py — load_behaviors offline safety.

``db init`` crashed hard offline (uncaught URLError from the HarmBench CSV
fetch). The fix caches the CSV under ``<repo>/.cache/`` (or ``ABL_BEHAVIORS_CACHE``)
and raises a clear RuntimeError when both cache and network are unavailable.
"""

from __future__ import annotations

import urllib.error
from unittest.mock import patch

import pytest

from src.db.core import load_behaviors

_CSV = "BehaviorID,Behavior,SemanticCategory,Tags\nb1,Do a thing,chemical,\n"


class TestLoadBehaviors:
    def test_cache_hit_no_network(self, tmp_path, monkeypatch):
        cache = tmp_path / "behaviors.csv"
        cache.write_text(_CSV)
        monkeypatch.setenv("ABL_BEHAVIORS_CACHE", str(cache))

        def _fail(*a, **k):  # any network attempt is a test failure
            raise AssertionError("network fetch attempted despite cache hit")

        with patch("urllib.request.urlopen", _fail):
            behaviors = load_behaviors()
        assert len(behaviors) == 1
        assert behaviors[0]["BehaviorID"] == "b1"
        assert behaviors[0]["SemanticCategory"] == "chemical"

    def test_network_success_populates_cache(self, tmp_path, monkeypatch):
        cache = tmp_path / "behaviors.csv"
        monkeypatch.setenv("ABL_BEHAVIORS_CACHE", str(cache))
        assert not cache.exists()

        resp = type("R", (), {"read": staticmethod(lambda: _CSV.encode())})()
        with patch("urllib.request.urlopen", return_value=resp):
            behaviors = load_behaviors()
        assert len(behaviors) == 1
        assert cache.exists()  # cached for offline re-runs

        # Second load served from cache (network would fail now)
        with patch("urllib.request.urlopen", side_effect=AssertionError("network used twice")):
            assert len(load_behaviors()) == 1

    def test_offline_no_cache_raises_clear_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ABL_BEHAVIORS_CACHE", str(tmp_path / "missing.csv"))

        err = urllib.error.URLError("connection refused")
        with patch("urllib.request.urlopen", side_effect=err), pytest.raises(RuntimeError) as excinfo:
            load_behaviors()
        msg = str(excinfo.value)
        assert "network fetch failed" in msg
        assert str(tmp_path / "missing.csv") in msg  # cache path in remediation hint

    def test_force_download_refreshes_cache(self, tmp_path, monkeypatch):
        cache = tmp_path / "behaviors.csv"
        cache.write_text(_CSV)
        monkeypatch.setenv("ABL_BEHAVIORS_CACHE", str(cache))
        fresh = _CSV.replace("b1", "b2")

        resp = type("R", (), {"read": staticmethod(lambda: fresh.encode())})()
        with patch("urllib.request.urlopen", return_value=resp):
            behaviors = load_behaviors(force_download=True)
        assert behaviors[0]["BehaviorID"] == "b2"
        assert cache.read_text().startswith("BehaviorID,Behavior,SemanticCategory,Tags\nb2")
