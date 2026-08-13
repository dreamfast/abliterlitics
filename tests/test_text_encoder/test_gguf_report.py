"""Tests for the GGUF report sections (generate_report.py).

Covers: _gguf_section ladder table, GGUF suffixes in _decomposition_section
(_s vs _m resolution), _key_findings GGUF aggregation, and the ladder graph's
data extraction (graph_gguf_ladder smoke via matplotlib Agg).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import pytest

matplotlib.use("Agg")

from src.text_encoder.generate_report import (  # noqa: E402
    _decomposition_section,
    _gguf_qtype_mix,
    _gguf_section,
    _key_findings,
    graph_gguf_ladder,
)


def _sr(stream: str = "refined_context", damage: float | None = 1.0, ci: list[float] | None = None) -> dict[str, Any]:
    return {
        "stream": stream,
        "stage": "CONNECTOR_OUT",
        "n_prompts": 60,
        "metrics": {
            "cka_linear": 0.99,
            "cka_rbf": 0.99,
            "cos_all": 0.999,
            "cos_last": 0.999,
            "mse": 1e-4,
            "mse_ci": None,
            "l2": 1e-2,
            "rank_corr": 0.99,
        },
        "damage_quant_units": damage,
        "damage_quant_ci": ci,
        "semantic_ceiling_mse": 0.01,
        "damage_semantic_units": 0.01,
        "per_category": {"innocent": {"mse": 5e-5}, "extreme": {"mse": 2e-4}},
    }


def _gguf_meta(bpw: float, label: str, hist: dict[str, int]) -> dict[str, Any]:
    return {
        "quant_label": label,
        "bpw": bpw,
        "qtype_histogram": hist,
        "arch": "qwen3",
        "n_tensors": 398,
        "n_params": 4_022_468_096,
    }


HIST = {"Q4_K": 216, "Q6_K": 37, "F32": 145}


def _variants() -> list[dict[str, Any]]:
    return [
        {
            "name": "base_gguf_f16",
            "variant_id": "f16.gguf",
            "dtype": "gguf_f16",
            "gguf": _gguf_meta(16.01, "f16", {"F16": 253, "F32": 145}),
            "backend": "ZImageBackend",
            "reference": "base.safetensors",
            "floor": "real_int8",
            "streams": {"refined_context": _sr(damage=0.01)},
        },
        {
            "name": "base_gguf_q4_k_m",
            "variant_id": "q4.gguf",
            "dtype": "gguf_q4_k_m",
            "gguf": _gguf_meta(4.97, "q4_k_m", HIST),
            "backend": "ZImageBackend",
            "reference": "base.safetensors",
            "floor": "real_int8",
            "streams": {"refined_context": _sr(damage=120.0)},
        },
        {
            "name": "base_gguf_q4_k_s",
            "variant_id": "q4s.gguf",
            "dtype": "gguf_q4_k_s",
            "gguf": _gguf_meta(4.74, "q4_k_s", HIST),
            "backend": "ZImageBackend",
            "reference": "base.safetensors",
            "floor": "real_int8",
            "streams": {"refined_context": _sr(damage=140.0)},
        },
        {
            "name": "heretic_gguf_q4_k_m",
            "variant_id": "hq4.gguf",
            "dtype": "gguf_q4_k_m",
            "gguf": _gguf_meta(4.97, "q4_k_m", HIST),
            "backend": "ZImageBackend",
            "reference": "base_gguf_q4_k_m",
            "floor": "real_int8",
            "streams": {"refined_context": _sr(damage=55.0, ci=[40.0, 70.0])},
        },
        {
            "name": "heretic_gguf_q4_k_m__vs_base",
            "variant_id": "hq4.gguf",
            "dtype": "gguf_q4_k_m",
            "gguf": _gguf_meta(4.97, "q4_k_m", HIST),
            "backend": "ZImageBackend",
            "reference": "base.safetensors",
            "floor": "real_int8",
            "streams": {"refined_context": _sr(damage=200.0)},
        },
    ]


class TestGgufSection:
    def test_ladder_table_renders_sorted_by_bpw(self) -> None:
        out = _gguf_section(
            _variants(),
            divergence={
                "base_gguf_q4_k_m": {
                    "div_control_ratio": 0.016,
                    "mean_final_latent_div": 0.02,
                    "noise_control_final": 1.27,
                }
            },
        )
        assert "GGUF quantization ladder" in out
        assert "Q4_K + Q6_K" in out
        assert "4.97" in out and "16.01" in out
        # f16 row (bpw 16.01) before q4 rows
        assert out.index("16.01") < out.index("4.97")
        # heretic row annotated with its matched reference - as LIVE markup
        # (the span must not be double-escaped into visible text)
        assert "vs base_gguf_q4_k_m" in out
        assert "&lt;span" not in out
        # divergence column populated only where data exists
        assert "1.6%" in out
        # dual-run entries excluded
        assert "(vs base)" not in out.replace("vs base_gguf_q4_k_m", "")

    def test_empty_when_no_gguf(self) -> None:
        assert _gguf_section([], divergence={}) == ""

    def test_qtype_mix_labels(self) -> None:
        assert _gguf_qtype_mix(HIST) == "Q4_K + Q6_K"
        assert _gguf_qtype_mix({"F16": 253, "F32": 145}) == "F16"
        assert _gguf_qtype_mix(None) == "-"
        mixed3 = {"Q3_K": 144, "Q4_K": 104, "Q5_K": 4, "Q6_K": 1, "F32": 145}
        assert _gguf_qtype_mix(mixed3) == "Q3_K + Q4_K + Q5_K…"


class TestDecompositionSuffixes:
    def test_s_vs_m_resolution(self) -> None:
        variants = _variants()
        out = _decomposition_section(variants)
        # heretic_gguf_q4_k_m must resolve base_gguf_q4_k_m (not _s)
        assert "heretic_gguf_q4_k_m" in out
        row_m = out.split("<tr><td>heretic_gguf_q4_k_m</td>")[1].split("</tr>")[0]
        # pure ablation 55x (vs matched base), combined 200x (dual entry;
        # >=100 renders integer-per the _cell convention)
        assert "55.00×" in row_m and "200×" in row_m

    def test_mxfp8_resolves_before_fp8(self) -> None:
        """Regression: 'heretic_mxfp8'.endswith('fp8') is True - the suffix
        tuple must try mxfp8 first or the row resolves the WRONG base and
        drops its pure-ablation cell."""
        variants = [
            {
                "name": "base_mxfp8",
                "variant_id": "b",
                "dtype": "comfy_bf16",
                "gguf": None,
                "backend": "ZImageBackend",
                "reference": "base.safetensors",
                "floor": "real_int8",
                "streams": {"refined_context": _sr(damage=4.4)},
            },
            {
                "name": "heretic_mxfp8",
                "variant_id": "h",
                "dtype": "comfy_bf16",
                "gguf": None,
                "backend": "ZImageBackend",
                "reference": "base_mxfp8",
                "floor": "real_int8",
                "streams": {"refined_context": _sr(damage=34.8)},
            },
        ]
        out = _decomposition_section(variants)
        row = out.split("<tr><td>heretic_mxfp8</td>")[1].split("</tr>")[0]
        assert "34.80×" in row, "pure-ablation cell must be present (reference matched)"

    def test_gguf_f16_suffix_resolves(self) -> None:
        # Build a heretic_gguf_f16 with matched base present
        variants = _variants() + [
            {
                "name": "heretic_gguf_f16",
                "variant_id": "hf16.gguf",
                "dtype": "gguf_f16",
                "gguf": _gguf_meta(16.01, "f16", {"F16": 253}),
                "backend": "ZImageBackend",
                "reference": "base_gguf_f16",
                "floor": "real_int8",
                "streams": {"refined_context": _sr(damage=33.0)},
            },
        ]
        out = _decomposition_section(variants)
        assert "heretic_gguf_f16" in out


class TestKeyFindingsAggregation:
    def test_gguf_grouped_not_individual(self) -> None:
        findings = " ".join(_key_findings(_variants(), "zimage"))
        assert "GGUF ladder (base containers, 3 rungs)" in findings
        assert "GGUF × heretic" in findings
        # individual gguf bullets suppressed
        assert "base_gguf_q4_k_s</b> (pure quantization)" not in findings

    def test_no_gguf_findings_when_absent(self) -> None:
        variants = [
            {
                "name": "heretic_int8",
                "variant_id": "x",
                "dtype": "comfy_bf16",
                "gguf": None,
                "backend": "ZImageBackend",
                "reference": "base_int8",
                "floor": "real_int8",
                "streams": {"refined_context": _sr(damage=34.0)},
            },
        ]
        findings = " ".join(_key_findings(variants, "zimage"))
        assert "GGUF" not in findings


class TestLadderGraph:
    def test_graph_writes_svg(self, tmp_path: Path) -> None:
        out = tmp_path / "te_gguf_ladder.svg"
        graph_gguf_ladder(_variants(), out)
        assert out.exists()
        content = out.read_text()
        assert "svg" in content

    def test_graph_skips_without_gguf(self, tmp_path: Path) -> None:
        out = tmp_path / "te_gguf_ladder.svg"
        graph_gguf_ladder([], out)
        assert not out.exists()

    def test_graph_tolerates_none_gguf_mixins(self, tmp_path: Path) -> None:
        """Regression: safetensors variants carry gguf=None (key present) -
        graph must not crash mixing them with GGUF variants."""
        mixed = _variants() + [
            {
                "name": "heretic_int8",
                "variant_id": "x",
                "dtype": "comfy_bf16",
                "gguf": None,
                "backend": "ZImageBackend",
                "reference": "base_int8",
                "floor": "real_int8",
                "streams": {"refined_context": _sr(damage=34.0)},
            },
            {
                "name": "base_int8",
                "variant_id": "x",
                "dtype": "comfy_bf16",
                "gguf": None,
                "backend": "ZImageBackend",
                "reference": "base.safetensors",
                "floor": "real_int8",
                "streams": {"refined_context": _sr(damage=1.0)},
            },
        ]
        out = tmp_path / "te_gguf_ladder.svg"
        graph_gguf_ladder(mixed, out)
        assert out.exists()


if __name__ == "__main__":
    pytest.main([__file__])
