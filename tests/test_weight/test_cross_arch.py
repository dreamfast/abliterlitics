"""Tests for src/weight/cross_arch_comparison.py — Module importability."""

from __future__ import annotations


class TestCrossArchImportable:
    def test_module_imports(self):
        import src.weight.cross_arch_comparison

        assert hasattr(src.weight.cross_arch_comparison, "main")


class TestComparisonMode:
    """Pipeline wiring: run_weights.sh/sequencer pass --comparison/--results-dir."""

    def test_discovers_panel_and_writes_report(self, tmp_path, monkeypatch, tmp_comparison_dir):
        import json
        import sys

        from src.config import ComparisonConfig
        from src.weight.cross_arch_comparison import main
        from src.weight.panel_comparison import run_analysis as run_panel

        config = ComparisonConfig.from_dir(tmp_comparison_dir)
        results_dir = tmp_path / "results" / "test-comp" / "weight"
        run_panel(
            base_path=str(config.base_path),
            variant_paths={v.name: str(v.path) for v in config.variants},
            output_dir=results_dir,
            config=config,
        )
        out = tmp_path / "cross_arch.json"

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "cross_arch_comparison.py",
                "--comparison",
                str(tmp_comparison_dir),
                "--results-dir",
                str(results_dir),
                "--output",
                str(out),
            ],
        )
        main()

        data = json.loads(out.read_text())
        assert data["results_version"] == 1
        # Label comes from the panel's metadata comparison_name.
        assert "test-comp" in data["models"]
        assert data["models"]["test-comp"]["family"]

    def test_default_output_lands_in_results_dir(self, tmp_path, monkeypatch, tmp_comparison_dir):
        import sys

        from src.config import ComparisonConfig
        from src.weight.cross_arch_comparison import main
        from src.weight.panel_comparison import run_analysis as run_panel

        config = ComparisonConfig.from_dir(tmp_comparison_dir)
        results_dir = tmp_path / "results" / "test-comp" / "weight"
        run_panel(
            base_path=str(config.base_path),
            variant_paths={v.name: str(v.path) for v in config.variants},
            output_dir=results_dir,
            config=config,
        )

        monkeypatch.setattr(
            sys,
            "argv",
            ["cross_arch_comparison.py", "--comparison", str(tmp_comparison_dir), "--results-dir", str(results_dir)],
        )
        main()

        assert (results_dir / "cross_arch.json").exists()

    def test_no_panel_exits_1(self, tmp_path, monkeypatch, tmp_comparison_dir):
        import sys

        import pytest

        from src.weight.cross_arch_comparison import main

        empty_results = tmp_path / "empty"
        empty_results.mkdir()
        monkeypatch.setattr(
            sys,
            "argv",
            ["cross_arch_comparison.py", "--comparison", str(tmp_comparison_dir), "--results-dir", str(empty_results)],
        )
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 1

    def test_no_args_errors(self, monkeypatch):
        import sys

        import pytest

        from src.weight.cross_arch_comparison import main

        monkeypatch.setattr(sys, "argv", ["cross_arch_comparison.py"])
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 2  # argparse error


class TestExplicitPanelsMode:
    """The original --panels/--labels contract must keep working."""

    def _write_panel(self, path, family="qwen3"):
        import json

        panel = {
            "architecture": {"family": family, "layer_count": 4, "total_keys": 100},
            "pairwise_changed_counts": {"base->v1": 3},
            "per_layer_changes": {"base->v1": {"0": {"count": 3, "types": ["attn", "mlp"]}}},
        }
        path.write_text(json.dumps(panel))
        return path

    def test_panels_mode_report(self, tmp_path, monkeypatch):
        import json
        import sys

        from src.weight.cross_arch_comparison import main

        p1 = self._write_panel(tmp_path / "p1.json", family="qwen3")
        p2 = self._write_panel(tmp_path / "p2.json", family="gemma4")
        out = tmp_path / "cross_arch.json"

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "cross_arch_comparison.py",
                "--panels",
                str(p1),
                str(p2),
                "--labels",
                "small",
                "large",
                "--output",
                str(out),
            ],
        )
        main()

        data = json.loads(out.read_text())
        assert set(data["models"]) == {"small", "large"}
        assert data["models"]["small"]["family"] == "qwen3"
        assert data["models"]["large"]["family"] == "gemma4"
        assert data["technique_edit_counts_matrix"]["base->v1"] == {"small": 3, "large": 3}
        assert data["type_comparison"]["attn"] == {"small": 1, "large": 1}

    def test_panels_mode_requires_output(self, tmp_path, monkeypatch):
        import sys

        import pytest

        from src.weight.cross_arch_comparison import main

        p1 = self._write_panel(tmp_path / "p1.json")
        monkeypatch.setattr(
            sys,
            "argv",
            ["cross_arch_comparison.py", "--panels", str(p1), "--labels", "l1"],
        )
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 2
