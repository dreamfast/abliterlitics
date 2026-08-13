"""Tests for report rendering."""

from __future__ import annotations

import json

from src.text_encoder.compare import VariantReport
from src.text_encoder.metrics import MetricResult
from src.text_encoder.report import Report


def _make_variant_report(stream_name: str = "test") -> VariantReport:
    sr_cls = type(
        "SR",
        (),
        {
            "stream": stream_name,
            "stage": "raw_hidden",
            "n_prompts": 10,
            "cka_linear": MetricResult(value=0.95),
            "cka_rbf": MetricResult(value=0.93),
            "cos_all": MetricResult(value=0.88, ci_low=0.85, ci_high=0.91),
            "cos_last": MetricResult(value=0.86),
            "mse": MetricResult(value=0.001, ci_low=0.0008, ci_high=0.0012),
            "l2": MetricResult(value=0.03),
            "rank_corr": MetricResult(value=0.99),
            "damage_quant_units": MetricResult(value=0.5, ci_low=0.4, ci_high=0.6),
            "semantic_ceiling_mse": 0.02,
            "damage_semantic_units": MetricResult(value=0.05),
            "per_category": {"innocent": {"cka_linear": 0.97}, "extreme": {"cka_linear": 0.91}},
        },
    )
    return VariantReport(variant_id="/fake/heretic", dtype="bf16", streams=[sr_cls])


class TestReport:
    def test_to_json_serializable(self) -> None:
        report = Report(
            base_id="/fake/base",
            backend="LTX2Backend",
            reference_floors={"quant": "fakequant_fp8"},
            variants=[_make_variant_report()],
            prompt_set_hash="abc123",
        )
        data = report.to_json()
        # Must be JSON serializable
        s = json.dumps(data)
        assert "te_results_version" in s
        assert "variants" in data
        assert len(data["variants"]) == 1

    def test_to_markdown_contains_streams(self) -> None:
        report = Report(
            base_id="/fake/base",
            backend="LTX2Backend",
            reference_floors={},
            variants=[_make_variant_report(stream_name="gemma_hidden")],
            prompt_set_hash="abc123",
        )
        md = report.to_markdown()
        assert "gemma_hidden" in md
        assert "CKA" in md
        assert "Damage" in md

    def test_to_markdown_per_category(self) -> None:
        report = Report(
            base_id="/fake/base",
            backend="LTX2Backend",
            reference_floors={},
            variants=[_make_variant_report()],
            prompt_set_hash="abc123",
        )
        md = report.to_markdown()
        assert "innocent" in md
        assert "extreme" in md
