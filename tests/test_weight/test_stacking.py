"""Tests for src/weight/stacking_analysis.py — Math helper functions."""

from __future__ import annotations

import pytest
import torch

from src.weight.stacking_analysis import (
    cosine_sim,
    explained_variance,
    regression_slope,
)


class TestCosineSim:
    def test_identical(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        assert cosine_sim(a, a) == pytest.approx(1.0, abs=1e-6)

    def test_opposite(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        assert cosine_sim(a, -a) == pytest.approx(-1.0, abs=1e-6)


class TestRegressionSlope:
    def test_perfect_positive(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        y = 2.0 * x + 1.0
        assert regression_slope(x, y) == pytest.approx(2.0, abs=1e-4)

    def test_perfect_negative(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        y = -3.0 * x
        assert regression_slope(x, y) == pytest.approx(-3.0, abs=1e-4)

    def test_constant_x(self):
        x = torch.tensor([5.0, 5.0, 5.0])
        y = torch.tensor([1.0, 2.0, 3.0])
        assert regression_slope(x, y) == 0.0


class TestExplainedVariance:
    def test_perfect_linear(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 3.0 * x + 7.0
        assert explained_variance(x, y) == pytest.approx(1.0, abs=1e-4)

    def test_poor_fit(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([10.0, -5.0, 20.0])
        r2 = explained_variance(x, y)
        assert r2 < 0.5


class TestStackingShapeGuardAndReport:
    """Lesson #79 guard + audit deletions (dup ratio write, empty metadata)."""

    EXTENDED_KEY = "model.layers.0.mlp.gate_proj.weight"

    def _run_main(self, tmp_path, monkeypatch, base, va, vb, output, extra_args=()):
        import sys

        from src.weight.stacking_analysis import main

        argv = [
            "stacking_analysis.py",
            "--base",
            str(base),
            "--variant-a",
            str(va),
            "--variant-b",
            str(vb),
            "--output",
            str(output),
            *extra_args,
        ]
        monkeypatch.setattr(sys, "argv", argv)
        main()
        return output

    def test_arch_extending_variant_b_skipped_not_crashed(self, tmp_path, monkeypatch):
        import json

        from tests.conftest import create_mock_model
        from tests.test_weight._arch_extending import make_arch_extending_variant

        base = create_mock_model(tmp_path / "base", arch="qwen3", seed=42)
        va = create_mock_model(tmp_path / "va", arch="qwen3", seed=43)
        vb_ext = make_arch_extending_variant(
            create_mock_model(tmp_path / "vb", arch="qwen3", seed=44),
            tmp_path / "vb_ext",
            self.EXTENDED_KEY,
        )
        output = tmp_path / "stacking.json"

        self._run_main(tmp_path, monkeypatch, base, va, vb_ext, output)

        data = json.loads(output.read_text())
        assert data["shape_skipped"] == 1

    def test_extra_variant_shape_mismatch_counted(self, tmp_path, monkeypatch):
        import json

        from tests.conftest import create_mock_model
        from tests.test_weight._arch_extending import make_arch_extending_variant

        base = create_mock_model(tmp_path / "base", arch="qwen3", seed=42)
        va = create_mock_model(tmp_path / "va", arch="qwen3", seed=43)
        vb = create_mock_model(tmp_path / "vb", arch="qwen3", seed=44)
        ve_ext = make_arch_extending_variant(
            create_mock_model(tmp_path / "ve", arch="qwen3", seed=45),
            tmp_path / "ve_ext",
            self.EXTENDED_KEY,
        )
        output = tmp_path / "stacking.json"

        self._run_main(
            tmp_path,
            monkeypatch,
            base,
            va,
            vb,
            output,
            extra_args=("--extra-variants", str(ve_ext), "--extra-labels", "fresh"),
        )

        data = json.loads(output.read_text())
        # The extended key is skipped once for the extra variant's delta.
        assert data["shape_skipped"] == 1
        assert "fresh" in data["extra_variant_baselines"]

    def test_report_has_no_empty_metadata_key(self, tmp_path, monkeypatch):
        import json

        from tests.conftest import create_mock_model

        base = create_mock_model(tmp_path / "base", arch="qwen3", seed=42)
        va = create_mock_model(tmp_path / "va", arch="qwen3", seed=43)
        vb = create_mock_model(tmp_path / "vb", arch="qwen3", seed=44)
        output = tmp_path / "stacking.json"

        self._run_main(tmp_path, monkeypatch, base, va, vb, output)

        data = json.loads(output.read_text())
        assert "metadata" not in data
        # The single ratio write survives (consumed by report generation).
        assert "ratio_Dr_Db" in data["stacking_metrics"]
