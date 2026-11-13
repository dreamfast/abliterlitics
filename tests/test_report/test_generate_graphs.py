"""Tests for src/report/generate_graphs.py.

Covers: color generation (CVD-safe palette contract), load_json error paths
(warning-not-silent-skip + re-raise), venn subset math (2/3-way + 4+ pairwise
fallback + missing-library warning), RESULTS_VERSION SVG stamping, and main()
auto-discovery (rglob patterns, stem parsing, dense-model expert skip).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest

from src import RESULTS_VERSION
from src.report import generate_graphs as gg
from src.report.generate_graphs import (
    OKABE_ITO,
    gen_aggressiveness_bar,
    gen_expert_heatmap,
    gen_layer_heatmap,
    gen_svd_plots,
    gen_venn,
    get_color,
    get_variant_colors,
    load_json,
)

LOGGER = "src.report.generate_graphs"


def _panel(keys_by_variant: dict[str, set[str]]) -> dict:
    """Build a minimal multi_model_panel-shaped dict for venn/aggressiveness tests."""
    panel: dict = {"pairwise_changed_counts": {}}
    for v, keys in keys_by_variant.items():
        pk = f"base->{v}"
        panel["pairwise_changed_counts"][pk] = len(keys)
        panel[f"{pk}_keys"] = sorted(keys)
    return panel


def _svg_names(out_dir: Path) -> set[str]:
    return {p.name for p in out_dir.glob("*.svg")}


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

    def test_colors_are_hex_strings(self):
        colors = get_variant_colors(["heretic", "hauhau"])
        for name, color in colors.items():
            assert isinstance(color, str) and color.startswith("#"), f"bad color for {name}: {color}"

    def test_palette_is_okabe_ito(self):
        """Variant colors must come from the CVD-safe Okabe-Ito palette (not husl)."""
        names = [f"v{i}" for i in range(len(OKABE_ITO))]
        colors = get_variant_colors(names)
        for name in names:
            assert colors[name] in OKABE_ITO

    def test_palette_entries_are_distinct_hex(self):
        assert len(set(OKABE_ITO)) == len(OKABE_ITO)
        for c in OKABE_ITO:
            assert c.startswith("#") and len(c) == 7

    def test_more_variants_than_palette_wraps(self):
        n = len(OKABE_ITO) + 2
        colors = get_variant_colors([f"v{i}" for i in range(n)])
        assert all(colors[f"v{i}"].startswith("#") for i in range(n))

    def test_single_variant(self):
        colors = get_variant_colors(["test"])
        assert len(colors) == 3  # test + base + overlap


class TestGetColor:
    def test_known_name(self):
        assert get_color({"heretic": "#E69F00"}, "heretic") == "#E69F00"

    def test_unknown_name_falls_back_to_neutral(self):
        assert get_color({}, "ghost") == gg.DEFAULT_COLOR


class TestLoadJson:
    def test_valid_json_with_version(self, tmp_path):
        data = {"results_version": RESULTS_VERSION, "data": [1, 2, 3]}
        f = tmp_path / "test.json"
        f.write_text(json.dumps(data))
        result = load_json(str(f))
        assert result is not None
        assert result["results_version"] == RESULTS_VERSION

    def test_old_format_still_loaded(self, tmp_path):
        """load_json returns data even without results_version (only warns)."""
        data = {"data": [1, 2, 3]}
        f = tmp_path / "old.json"
        f.write_text(json.dumps(data))
        result = load_json(str(f))
        assert result is not None
        assert result["data"] == [1, 2, 3]

    def test_missing_file_warns_and_returns_none(self, caplog):
        """Regression: missing files must not be swallowed silently (audit bug)."""
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            result = load_json("/nonexistent/path.json")
        assert result is None
        assert any(r.levelno == logging.WARNING and "/nonexistent/path.json" in r.getMessage() for r in caplog.records)

    def test_corrupt_json_warns_and_returns_none(self, tmp_path, caplog):
        """Regression: corrupt JSON must not be swallowed silently (audit bug)."""
        f = tmp_path / "bad.json"
        f.write_text("not valid json {{{")
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            result = load_json(f)
        assert result is None
        assert any("corrupt JSON" in r.getMessage() and str(f) in r.getMessage() for r in caplog.records)

    def test_version_mismatch_warns(self, tmp_path, caplog):
        f = tmp_path / "future.json"
        f.write_text(json.dumps({"results_version": RESULTS_VERSION + 1}))
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            result = load_json(f)
        assert result is not None
        assert any("results_version mismatch" in r.getMessage() for r in caplog.records)

    def test_matching_version_no_mismatch_warning(self, tmp_path, caplog):
        f = tmp_path / "cur.json"
        f.write_text(json.dumps({"results_version": RESULTS_VERSION}))
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            result = load_json(f)
        assert result is not None
        assert "mismatch" not in caplog.text

    def test_other_os_errors_reraise_with_path(self, tmp_path):
        """Read failures that are not missing/corrupt files must not be swallowed."""
        with pytest.raises(RuntimeError, match="Failed to read results file"):
            load_json(tmp_path)  # a directory → IsADirectoryError (an OSError)


class TestGenVenn:
    def test_two_way_subset_math(self, tmp_path, monkeypatch):
        captured = {}

        def fake_venn2(*, subsets, set_labels, set_colors, alpha, ax):
            captured["subsets"] = subsets
            captured["labels"] = set_labels

        monkeypatch.setattr(gg, "venn2", fake_venn2)
        # heretic edits {t1,t2,t3}, huihui edits {t2,t3,t4} → only 1 / only 1 / overlap 2
        panel = _panel({"heretic": {"t1", "t2", "t3"}, "huihui": {"t2", "t3", "t4"}})
        gen_venn(panel, tmp_path, "Label", get_variant_colors(["heretic", "huihui"]))
        assert captured["subsets"] == (1, 1, 2)
        assert captured["labels"] == ("Heretic", "Huihui")
        assert "venn_Label.svg" in _svg_names(tmp_path)

    def test_two_way_real_render(self, tmp_path):
        panel = _panel({"heretic": {"t1", "t2", "t3"}, "huihui": {"t2", "t3", "t4"}})
        gen_venn(panel, tmp_path, "Label", get_variant_colors(["heretic", "huihui"]))
        assert "venn_Label.svg" in _svg_names(tmp_path)

    def test_three_way_subset_math(self, tmp_path, monkeypatch):
        captured = {}

        def fake_venn3(*, subsets, set_labels, set_colors, alpha, ax):
            captured["subsets"] = subsets

        monkeypatch.setattr(gg, "venn3", fake_venn3)
        # a={1,2,3,4} b={3,4,5,6} c={4,5,7,8}
        panel = _panel({"a": {"1", "2", "3", "4"}, "b": {"3", "4", "5", "6"}, "c": {"4", "5", "7", "8"}})
        gen_venn(panel, tmp_path, "Label", get_variant_colors(["a", "b", "c"]))
        # venn3 subset order: (only_a, only_b, ab, only_c, ac, bc, all3)
        assert captured["subsets"] == (2, 1, 1, 2, 0, 1, 1)
        assert "venn_Label.svg" in _svg_names(tmp_path)

    def test_three_way_real_render(self, tmp_path):
        panel = _panel({"a": {"1", "2", "3", "4"}, "b": {"3", "4", "5", "6"}, "c": {"4", "5", "7", "8"}})
        gen_venn(panel, tmp_path, "Label", get_variant_colors(["a", "b", "c"]))
        assert "venn_Label.svg" in _svg_names(tmp_path)

    def test_four_variants_produce_pairwise_venns(self, tmp_path, monkeypatch):
        calls = []

        def fake_venn2(*, subsets, set_labels, set_colors, alpha, ax):
            calls.append(set_labels)

        monkeypatch.setattr(gg, "venn2", fake_venn2)
        names = ["ara", "heretic", "heretic2", "huihui"]
        panel = _panel({v: {f"{v}_k1", "shared"} for v in names})
        gen_venn(panel, tmp_path, "Label", get_variant_colors(names))
        assert len(calls) == 6  # C(4, 2) pairwise fallback
        expected = {f"venn_Label_{a}_vs_{b}.svg" for i, a in enumerate(names) for b in names[i + 1 :]}
        assert expected <= _svg_names(tmp_path)

    def test_variant_to_variant_pairs_are_ignored(self, tmp_path, monkeypatch):
        """Only base->X pairs participate in the venn (variant->variant skipped)."""
        monkeypatch.setattr(gg, "venn2", lambda **kwargs: None)
        panel = _panel({"a": {"x"}, "b": {"y"}})
        panel["pairwise_changed_counts"]["a->b"] = 1
        panel["a->b_keys"] = ["z"]
        gen_venn(panel, tmp_path, "Label", get_variant_colors(["a", "b"]))
        # exactly one venn file (a vs b); the a->b pair contributed none
        assert _svg_names(tmp_path) == {"venn_Label.svg"}

    def test_missing_venn_lib_warns_and_skips(self, tmp_path, monkeypatch, caplog):
        """Regression: no matplotlib_venn → a WARNING (was misleading info), no files."""
        monkeypatch.setattr(gg, "HAS_VENN", False)
        panel = _panel({"a": {"x"}, "b": {"y"}})
        with caplog.at_level(logging.INFO, logger=LOGGER):
            gen_venn(panel, tmp_path, "Label", get_variant_colors(["a", "b"]))
        assert _svg_names(tmp_path) == set()
        assert any(
            r.levelno == logging.WARNING and "matplotlib_venn unavailable" in r.getMessage() for r in caplog.records
        )

    def test_single_variant_no_venn(self, tmp_path, monkeypatch):
        def boom(**kwargs):
            raise AssertionError("venn functions must not be called for a single variant")

        monkeypatch.setattr(gg, "venn2", boom)
        monkeypatch.setattr(gg, "venn3", boom)
        panel = _panel({"a": {"x"}})
        gen_venn(panel, tmp_path, "Label", get_variant_colors(["a"]))
        assert _svg_names(tmp_path) == set()

    def test_empty_panel_noop(self, tmp_path):
        gen_venn(None, tmp_path, "Label", {})
        gen_venn({"pairwise_changed_counts": {}}, tmp_path, "Label", {})
        assert _svg_names(tmp_path) == set()


class _FakeAx:
    """Records bar() calls so data-flow assertions avoid brittle image checks."""

    def __init__(self):
        self.bar_calls = []

    def bar(self, *args, **kwargs):
        self.bar_calls.append(args)
        return []

    def text(self, *args, **kwargs):
        pass

    def set_ylabel(self, *args, **kwargs):
        pass

    def set_title(self, *args, **kwargs):
        pass

    def set_xlabel(self, *args, **kwargs):
        pass

    def tick_params(self, *args, **kwargs):
        pass


class _FakeFig:
    def __init__(self):
        self.saved = []

    def savefig(self, path, **kwargs):
        self.saved.append((Path(path), kwargs))

    def suptitle(self, *args, **kwargs):
        pass


class TestGenAggressivenessBar:
    def test_bar_counts_in_sorted_variant_order(self, tmp_path, monkeypatch):
        fake_ax, fake_fig = _FakeAx(), _FakeFig()
        monkeypatch.setattr(gg.plt, "subplots", lambda *a, **k: (fake_fig, fake_ax))
        monkeypatch.setattr(gg.plt, "close", lambda *a, **k: None)
        panel = _panel({"heretic": {"t1", "t2"}, "huihui": {"t3"}})
        gen_aggressiveness_bar(panel, tmp_path, "Label", get_variant_colors(["heretic", "huihui"]))
        # variants iterated in sorted-name order: heretic (2 edits) before huihui (1)
        assert fake_ax.bar_calls[0][0] == ["Heretic", "Huihui"]
        assert fake_ax.bar_calls[0][1] == [2, 1]
        path, kwargs = fake_fig.saved[0]
        assert path.name == "aggressiveness_Label.svg"
        assert kwargs["format"] == "svg"

    def test_real_render(self, tmp_path):
        panel = _panel({"heretic": {"t1", "t2"}, "huihui": {"t3"}})
        gen_aggressiveness_bar(panel, tmp_path, "Label", get_variant_colors(["heretic", "huihui"]))
        assert "aggressiveness_Label.svg" in _svg_names(tmp_path)

    def test_no_counts_no_file(self, tmp_path):
        gen_aggressiveness_bar(None, tmp_path, "Label", {})
        gen_aggressiveness_bar({"pairwise_changed_counts": {}}, tmp_path, "Label", get_variant_colors(["v"]))
        assert _svg_names(tmp_path) == set()


class TestResultsVersionStamping:
    """Regression: SVG outputs must carry the producing RESULTS_VERSION (audit)."""

    def test_svg_metadata_contains_results_version(self, tmp_path):
        panel = _panel({"heretic": {"t1", "t2"}, "huihui": {"t3"}})
        gen_aggressiveness_bar(panel, tmp_path, "Label", get_variant_colors(["heretic", "huihui"]))
        svg = (tmp_path / "aggressiveness_Label.svg").read_text()
        assert f"results_version={RESULTS_VERSION}" in svg

    def test_venn_svg_metadata_contains_results_version(self, tmp_path):
        panel = _panel({"heretic": {"t1", "t2"}, "huihui": {"t3"}})
        gen_venn(panel, tmp_path, "Label", get_variant_colors(["heretic", "huihui"]))
        svg = (tmp_path / "venn_Label.svg").read_text()
        assert f"results_version={RESULTS_VERSION}" in svg

    def test_save_fig_passes_metadata_kwarg(self, tmp_path, monkeypatch):
        fake_ax, fake_fig = _FakeAx(), _FakeFig()
        monkeypatch.setattr(gg.plt, "subplots", lambda *a, **k: (fake_fig, fake_ax))
        monkeypatch.setattr(gg.plt, "close", lambda *a, **k: None)
        panel = _panel({"v": {"t1"}})
        gen_aggressiveness_bar(panel, tmp_path, "Label", get_variant_colors(["v"]))
        _, kwargs = fake_fig.saved[0]
        assert kwargs["metadata"]["Description"] == f"results_version={RESULTS_VERSION}"
        assert kwargs["metadata"]["Creator"] == "abliterlitics generate_graphs"


class TestGenLayerHeatmap:
    METRICS = ("edit_density", "mean_edit_norm", "mean_norm_shift")

    @staticmethod
    def _layer_data(n=3):
        return {
            str(i): {
                "edit_density": float(i + 1),
                "mean_edit_norm": 0.01 * (i + 1),
                "mean_norm_shift": 0.001 * (i + 1),
            }
            for i in range(n)
        }

    def test_renders_one_file_per_metric_with_variant_suffix(self, tmp_path):
        gen_layer_heatmap(self._layer_data(), tmp_path, "Label", variant="heretic", colors={"heretic": "#E69F00"})
        for metric in self.METRICS:
            assert f"layer_{metric}_Label_heretic.svg" in _svg_names(tmp_path)

    def test_no_variant_suffix_when_unnamed(self, tmp_path):
        gen_layer_heatmap(self._layer_data(1), tmp_path, "Label", variant="")
        assert "layer_edit_density_Label.svg" in _svg_names(tmp_path)

    def test_all_zero_metric_skipped(self, tmp_path):
        data = {"0": {"edit_density": 0.0, "mean_edit_norm": 0.0, "mean_norm_shift": 0.0}}
        gen_layer_heatmap(data, tmp_path, "Label", variant="v")
        assert _svg_names(tmp_path) == set()

    def test_non_numeric_layer_key_skipped_with_warning(self, tmp_path, monkeypatch, caplog):
        """Regression (audit, latent): non-numeric layer keys must not crash int() sorting."""
        fake_ax = _FakeAx()
        monkeypatch.setattr(gg.plt, "subplots", lambda *a, **k: (_FakeFig(), fake_ax))
        monkeypatch.setattr(gg.plt, "close", lambda *a, **k: None)
        data = self._layer_data(2)
        data["bogus"] = {"edit_density": 5.0}
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            gen_layer_heatmap(data, tmp_path, "Label", variant="v", colors={})
        assert any("non-numeric layer key" in r.getMessage() for r in caplog.records)
        assert fake_ax.bar_calls[0][0] == ["L0", "L1"]  # bogus layer excluded from bars

    def test_numeric_string_layers_sorted_numerically(self, tmp_path, monkeypatch):
        fake_ax = _FakeAx()
        monkeypatch.setattr(gg.plt, "subplots", lambda *a, **k: (_FakeFig(), fake_ax))
        monkeypatch.setattr(gg.plt, "close", lambda *a, **k: None)
        data = {"10": {"edit_density": 3.0}, "2": {"edit_density": 1.0}}
        gen_layer_heatmap(data, tmp_path, "Label", variant="v", colors={})
        assert fake_ax.bar_calls[0][0] == ["L2", "L10"]  # numeric order, not lexicographic

    def test_empty_noop(self, tmp_path):
        gen_layer_heatmap(None, tmp_path, "Label")
        assert _svg_names(tmp_path) == set()


class TestGenSvdPlots:
    @staticmethod
    def _svd():
        return {
            "tensor_results": [
                {"tensor_type": "attn", "heretic_vs_base": {"effective_rank_90pct_energy": 3, "energy_top1_pct": 90.0}},
                {"tensor_type": "attn", "heretic_vs_base": {"effective_rank_90pct_energy": 5, "energy_top1_pct": 70.0}},
                {"tensor_type": "mlp", "heretic_vs_base": {"effective_rank_90pct_energy": 1, "energy_top1_pct": 95.0}},
            ]
        }

    def test_renders_summary_with_variant_suffix(self, tmp_path):
        gen_svd_plots(self._svd(), tmp_path, "Label", variant="heretic")
        assert "svd_summary_Label_heretic.svg" in _svg_names(tmp_path)

    def test_picks_first_vs_base_key_per_result(self, tmp_path, monkeypatch):
        captured = {}

        class FakeAx:
            def bar(self, *args, **kwargs):
                captured.setdefault("bars", []).append(args)
                return []

            def set_xticks(self, *a, **k):
                pass

            def set_xticklabels(self, *a, **k):
                pass

            def set_ylabel(self, *a, **k):
                pass

            def set_title(self, *a, **k):
                pass

            def axhline(self, *a, **k):
                pass

        monkeypatch.setattr(gg.plt, "subplots", lambda *a, **k: (_FakeFig(), (FakeAx(), FakeAx())))
        monkeypatch.setattr(gg.plt, "close", lambda *a, **k: None)
        data = {
            "tensor_results": [
                {"tensor_type": "attn", "heretic_vs_base": {"effective_rank_90pct_energy": 4, "energy_top1_pct": 80.0}}
            ]
        }
        gen_svd_plots(data, tmp_path, "Label", variant="")
        assert captured["bars"][0][1] == [4.0]  # rank from the *_vs_base dict

    def test_empty_and_missing_results_noop(self, tmp_path):
        gen_svd_plots(None, tmp_path, "Label")
        gen_svd_plots({"tensor_results": []}, tmp_path, "Label")
        # result with no variant dict and no dict fallback → no plottable labels
        gen_svd_plots({"tensor_results": [{"tensor_type": "attn"}]}, tmp_path, "Label")
        assert _svg_names(tmp_path) == set()


class TestGenExpertHeatmap:
    @staticmethod
    def _expert():
        return {
            "per_expert_details": {
                "heretic": {
                    "l0_e0": {"expert_id": 0, "layer": 0, "total_edit_norm": 1.5},
                    "l0_e1": {"expert_id": 1, "layer": 0, "total_edit_norm": 0.5},
                    "l1_e0": {"expert_id": 0, "layer": 1, "total_edit_norm": 0.2},
                }
            }
        }

    def test_renders_one_heatmap_per_variant(self, tmp_path):
        gen_expert_heatmap(self._expert(), tmp_path, "Label")
        assert "expert_heatmap_heretic_Label.svg" in _svg_names(tmp_path)

    def test_empty_and_missing_noop(self, tmp_path):
        gen_expert_heatmap(None, tmp_path, "Label")
        gen_expert_heatmap({"per_expert_details": {}}, tmp_path, "Label")
        gen_expert_heatmap({"per_expert_details": {"v": {}}}, tmp_path, "Label")  # empty details skipped
        assert _svg_names(tmp_path) == set()


class TestMainDiscovery:
    """main() auto-discovery: rglob patterns, stem parsing, dense-model skip."""

    @staticmethod
    def _make_tree(root: Path) -> Path:
        r = root / "results"
        r.mkdir(parents=True)
        panel = _panel({"heretic": {"t1", "t2"}, "huihui": {"t2", "t3"}})
        (r / "multi_model_panel.json").write_text(json.dumps(panel))
        for v in ("heretic", "huihui"):
            vd = r / v
            vd.mkdir()
            layer = {
                "results_version": RESULTS_VERSION,
                "layer_progression": {
                    str(i): {
                        "edit_density": float(i + 1),
                        "mean_edit_norm": 0.01 * (i + 1),
                        "mean_norm_shift": 0.001 * (i + 1),
                    }
                    for i in range(3)
                },
            }
            (vd / f"layer_analysis_{v}.json").write_text(json.dumps(layer))
            svd = {
                "results_version": RESULTS_VERSION,
                "tensor_results": [
                    {
                        "tensor_type": "attn",
                        f"{v}_vs_base": {"effective_rank_90pct_energy": 2, "energy_top1_pct": 88.0},
                    }
                ],
            }
            (vd / f"svd_{v}.json").write_text(json.dumps(svd))
        # dense model → expert heatmap skipped (audit test gap #4)
        (r / "expert_analysis.json").write_text(json.dumps({"architecture": {"experts": 0}}))
        return r

    @staticmethod
    def _run_main(monkeypatch, results_dir: Path, out_dir: Path) -> None:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "generate_graphs.py",
                "--results-dir",
                str(results_dir),
                "--output-dir",
                str(out_dir),
                "--label",
                "TestModel",
            ],
        )
        gg.main()

    def test_main_generates_expected_graphs(self, tmp_path, monkeypatch):
        tree = self._make_tree(tmp_path)
        out = tmp_path / "graphs"
        self._run_main(monkeypatch, tree, out)
        names = _svg_names(out)
        for expected in (
            "venn_TestModel.svg",
            "aggressiveness_TestModel.svg",
            "layer_edit_density_TestModel_heretic.svg",
            "layer_mean_norm_shift_TestModel_huihui.svg",
            "svd_summary_TestModel_heretic.svg",
            "svd_summary_TestModel_huihui.svg",
        ):
            assert expected in names, f"missing {expected}; got {sorted(names)}"
        # dense-model expert files skipped
        assert not any(n.startswith("expert_") for n in names)

    def test_main_dense_expert_logged(self, tmp_path, monkeypatch, caplog):
        tree = self._make_tree(tmp_path)
        with caplog.at_level(logging.INFO, logger=LOGGER):
            self._run_main(monkeypatch, tree, tmp_path / "graphs")
        assert any("dense model" in rec.getMessage() for rec in caplog.records)

    def test_main_survives_corrupt_file(self, tmp_path, monkeypatch, caplog):
        """One corrupt file warns and skips that graph family; others still render."""
        tree = self._make_tree(tmp_path)
        (tree / "heretic" / "svd_heretic.json").write_text("{corrupt")
        out = tmp_path / "graphs"
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            self._run_main(monkeypatch, tree, out)
        names = _svg_names(out)
        assert names, "other graph families must still render"
        assert "svd_summary_TestModel_heretic.svg" not in names
        assert "svd_summary_TestModel_huihui.svg" in names
        assert any("corrupt JSON" in rec.getMessage() for rec in caplog.records)

    def test_main_expert_error_file_warns(self, tmp_path, monkeypatch, caplog):
        """expert_analysis.py writes {'error': ...} for dense models — warn, don't crash."""
        tree = self._make_tree(tmp_path)
        (tree / "expert_analysis.json").write_text(json.dumps({"error": "not an MoE"}))
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            self._run_main(monkeypatch, tree, tmp_path / "graphs")
        assert any("expert analysis failed" in rec.getMessage() for rec in caplog.records)
