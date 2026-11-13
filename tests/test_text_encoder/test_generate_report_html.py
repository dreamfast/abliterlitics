"""Smoke tests for generate_report.generate_html + shared report_style.

generate_html (the ~300-statement production HTML assembly: figure embedding,
TOC, section wiring, A/B gallery emission) previously had zero pytest
coverage — it only ever ran via runners/run_te_report.sh. These tests drive
it end-to-end on a minimal fixture results dir: real te_*.json loading via
load_results, placeholder SVGs for the embedded figures, and empty results
for every section loader (each must warn-and-skip, never raise).

report_style holds the CSS constants + verdict-class mapping shared by the
report generations (deduplicated from generate_report / h3_report).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.text_encoder.generate_report import (
    _directional_section,
    _load_divergence_steps,
    generate_html,
    graph_divergence_steps,
    load_results,
)
from src.text_encoder.report_style import verdict_css_class

SVG_BYTES = b"<svg xmlns='http://www.w3.org/2000/svg'/>"

EMBEDDED_FIGURES = (
    "layer_profile",
    "damage_ranking",
    "category_gradient",
    "layer_heatmap",
    "gguf_ladder",
    "divergence_steps",
)


def _stream_result(stream: str = "refined_context", dmg: float | None = 10.0) -> dict[str, Any]:
    """One stream-report dict, same shape compare.Report.to_json emits."""
    return {
        "stream": stream,
        "stage": "CONNECTOR_OUT",
        "n_prompts": 60,
        "metrics": {"cka_linear": 0.99, "cos_all": 0.999, "mse": 1e-4},
        "damage_quant_units": dmg,
        "damage_quant_ci": None,
        "damage_semantic_units": 0.01,
        "per_category": {},
    }


def _variant_json(name: str, backend: str = "ZImageBackend") -> dict[str, Any]:
    """te_<name>.json body, mirroring compare.Report.to_json (run_analysis output)."""
    return {
        "base_id": "/models/base.safetensors",
        "backend": backend,
        "reference_floors": {"quant": "fakequant_int8"},
        "prompt_set_hash": "abc123",
        "variants": [
            {
                "name": name,
                "variant_id": name,
                "dtype": "comfy_bf16",
                "streams": [_stream_result()],
            }
        ],
    }


def _fixture_results_dir(tmp_path: Path, with_svgs: bool = True) -> tuple[Path, list[dict[str, Any]]]:
    """Minimal results dir: two te_*.json variants + placeholder SVGs."""
    for name in ("base_int8", "heretic"):
        (tmp_path / f"te_{name}.json").write_text(json.dumps(_variant_json(name)))
    if with_svgs:
        for fig in EMBEDDED_FIGURES:
            (tmp_path / f"te_{fig}.svg").write_bytes(SVG_BYTES)
        (tmp_path / "te_response_curve.svg").write_bytes(SVG_BYTES)
    variants = load_results(tmp_path)
    assert len(variants) == 2
    return tmp_path, variants


class TestGenerateHtmlSmoke:
    def test_full_report_with_figures(self, tmp_path: Path) -> None:
        results_dir, variants = _fixture_results_dir(tmp_path, with_svgs=True)
        out = results_dir / "te_report.html"
        generate_html(variants, results_dir, out, split_images=False)

        text = out.read_text()
        assert text.startswith("<!DOCTYPE html>")
        assert text.rstrip().endswith("</html>")
        # core sections wired
        assert "Pipeline divergence table" in text
        assert "Visualizations" in text
        assert "Summary of findings" in text
        assert "Methodology" in text
        # variant data flowed through load_results into the tables
        assert "heretic" in text
        assert "base_int8" in text
        # all six embedded figures present as data URIs (self-contained HTML)
        assert text.count("data:image/svg+xml;base64,") == 7  # 6 figures + response curve
        assert 'id="fig-layer_profile"' in text
        assert "Response curve: trajectory divergence vs image response" in text
        # shared style constants interpolated (report_style wiring)
        assert "font-family: -apple-system" in text
        assert ".dmg-high { color: #c0392b; font-weight: bold; }" in text

    def test_report_without_svg_files(self, tmp_path: Path) -> None:
        """Missing graph SVGs must not break assembly — figures simply omitted."""
        results_dir, variants = _fixture_results_dir(tmp_path, with_svgs=False)
        out = results_dir / "te_report.html"
        generate_html(variants, results_dir, out)
        text = out.read_text()
        assert text.startswith("<!DOCTYPE html>")
        assert "data:image/svg+xml;base64," not in text
        assert 'id="fig-layer_profile"' not in text

    def test_empty_results_dir_missing_variants_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No te_\\*.json variant results"):
            load_results(tmp_path)


class TestVerdictCssClass:
    """The verdict-text -> cell-class mapping (shared by the report generations)."""

    @pytest.mark.parametrize(
        ("verdict", "expected"),
        [
            ("SUPPORTED", "ok"),
            ("SUPPORTED (base clean)", "ok"),
            ("PARTIAL (interface measured, directional deferred)", "mid"),
            ("REFUTED (no damage)", "mid"),  # benign null — not a failure color
            ("REFUTED (no structure)", "mid"),
            ("REFUTED", "bad"),
            ("NOT RUN", "mid"),
            ("VACUOUS (no failures)", "mid"),
        ],
    )
    def test_mapping(self, verdict: str, expected: str) -> None:
        assert verdict_css_class(verdict) == expected


class TestGraphDivergenceSteps:
    """Per-step divergence chart (te_dit_divergence_*.json -> SVG)."""

    @staticmethod
    def _steps_json(variant_name: str, n_steps: int = 4) -> dict[str, Any]:
        return {
            "variant_name": variant_name,
            "n_steps": n_steps,
            "noise_control_final": 0.42,
            "per_prompt": [
                {"latent_div": [0.001, 0.01, 0.1, 0.9]},
                {"latent_div": [0.002, 0.02, 0.2, 1.1]},
            ],
        }

    def test_load_and_graph(self, tmp_path: Path) -> None:
        for name in ("base_int8", "heretic"):
            (tmp_path / f"te_dit_divergence_{name}.json").write_text(json.dumps(self._steps_json(name)))
        steps = _load_divergence_steps(tmp_path)
        assert set(steps) == {"base_int8", "heretic"}
        assert steps["heretic"]["noise_control_final"] == 0.42

        out = tmp_path / "te_divergence_steps.svg"
        graph_divergence_steps(steps, out)
        svg = out.read_text()
        assert "<svg" in svg
        # both series rendered, family-colored, anchor line annotated
        assert "heretic" in svg and "base_int8" in svg
        assert "#4c72b0" in svg and "#b8b8b8" in svg
        assert "different-seed anchor" in svg

    def test_empty_data_skips_output(self, tmp_path: Path) -> None:
        out = tmp_path / "te_divergence_steps.svg"
        graph_divergence_steps({}, out)
        assert not out.exists()  # early return, no crash

    def test_corrupt_json_warn_and_skip(self, tmp_path: Path) -> None:
        (tmp_path / "te_dit_divergence_bad.json").write_text("{not json")
        assert _load_divergence_steps(tmp_path) == {}

    def test_curves_without_latent_div_skipped(self, tmp_path: Path) -> None:
        d = self._steps_json("heretic")
        d["per_prompt"] = [{"latent_div": []}]
        (tmp_path / "te_dit_divergence_heretic.json").write_text(json.dumps(d))
        out = tmp_path / "te_divergence_steps.svg"
        graph_divergence_steps(_load_divergence_steps(tmp_path), out)
        assert out.exists()  # anchor still drawn, no series crash


class TestDirectionalSection:
    """Directional analysis section (te_directional_*.json rows)."""

    @staticmethod
    def _directional_json(variant_name: str) -> dict[str, Any]:
        return {
            "variant_name": variant_name,
            "backend": "zimage",
            "layers": [
                {
                    "layer": 34,
                    "cos_taboo_mean": 0.91,
                    "cos_z": 6.1,
                    "orth_fraction": 0.02,
                    "toward_innocent_mean": 1e-3,
                    "toward_innocent_z": 0.5,
                },
                {
                    "layer": 41,
                    "cos_taboo_mean": 0.95,
                    "cos_z": 8.3,
                    "orth_fraction": 0.01,
                    "toward_innocent_mean": 2e-3,
                    "toward_innocent_z": 0.9,
                },
            ],
        }

    def test_renders_rows(self, tmp_path: Path) -> None:
        (tmp_path / "te_directional_heretic.json").write_text(json.dumps(self._directional_json("heretic")))
        (tmp_path / "te_directional_base_int8.json").write_text(json.dumps(self._directional_json("base_int8")))
        section = _directional_section(tmp_path)
        assert "heretic" in section and "base_int8" in section
        # consumed layer (34) + peak-z layer (41) cells present
        assert ">34<" in section and ">41<" in section
        assert "ref-var" in section  # base_* rendered as reference variant

    def test_no_data_returns_empty(self, tmp_path: Path) -> None:
        assert _directional_section(tmp_path) == ""

    def test_missing_consumed_layer_skipped(self, tmp_path: Path) -> None:
        d = self._directional_json("heretic")
        d["layers"] = [{**d["layers"][0], "layer": 12}]  # consumed layer 34 absent
        (tmp_path / "te_directional_heretic.json").write_text(json.dumps(d))
        assert _directional_section(tmp_path) == ""
