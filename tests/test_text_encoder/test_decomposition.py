"""Tests for the report-layer decomposition logic.

Locks in the quantization-vs-abliteration decomposition contract:
- matched-quant detection by variant NAME (stable namespace)
- combined read from the __vs_base dual-run entry (distinct measurement)
- missing runs render as None (not duplicated numbers)
- __vs_base display naming
"""

from __future__ import annotations

from src.text_encoder.generate_report import _decomposition_section, _display, _is_reference_variant


def _variant(name: str, reference: str, damage: float | None, stream: str = "refined_context") -> dict:
    return {
        "name": name,
        "variant_id": f"/models/{name}.safetensors",
        "dtype": "comfy_bf16",
        "backend": "zimage",
        "reference": reference,
        "floor": "fakequant_int8",
        "streams": (
            {stream: {"stream": stream, "stage": "connector_out", "damage_quant_units": damage}}
            if damage is not None
            else {stream: {"stream": stream, "stage": "connector_out", "damage_quant_units": None}}
        ),
    }


class TestDisplayNames:
    def test_vs_base_suffix(self) -> None:
        assert _display("heretic_int8__vs_base") == "heretic_int8 (vs base)"

    def test_plain_name_unchanged(self) -> None:
        assert _display("heretic_int8") == "heretic_int8"


class TestIsReferenceVariant:
    def test_base_prefix(self) -> None:
        assert _is_reference_variant({"name": "base_int8"}) is True

    def test_heretic_not_reference(self) -> None:
        assert _is_reference_variant({"name": "heretic_int8"}) is False


class TestDecomposition:
    def test_full_decomposition(self) -> None:
        """All three numbers from three DISTINCT measurements."""
        variants = [
            _variant("base_int8", "qwen_3_4b.safetensors", 0.99),  # pure quant
            _variant("heretic_int8", "base_int8", 0.42),  # matched-quant (pure ablation)
            _variant("heretic_int8__vs_base", "qwen_3_4b.safetensors", 1.30),  # combined
        ]
        html_out = _decomposition_section(variants)
        assert "0.99" in html_out  # pure quant
        assert "0.42" in html_out  # pure ablation
        assert "1.30" in html_out  # combined (DISTINCT from pure ablation)
        assert "heretic_int8" in html_out

    def test_combined_not_duplicated_when_no_dual_run(self) -> None:
        """Without the __vs_base entry, combined must be missing (—), not a copy."""
        variants = [
            _variant("base_int8", "qwen_3_4b.safetensors", 0.99),
            _variant("heretic_int8", "base_int8", 0.42),
        ]
        html_out = _decomposition_section(variants)
        assert "0.42" in html_out  # pure ablation
        # combined cell is "—" — 0.42 must NOT appear twice as combined
        cells = [ln for ln in html_out.splitlines() if "heretic_int8</td>" in ln]
        assert len(cells) == 1
        row = cells[0]
        assert row.count("0.42") == 1  # appears once (pure ablation), not as combined

    def test_unmatched_reference_no_pure_ablation(self) -> None:
        """Fallback run (reference = global base) -> pure ablation is —."""
        variants = [
            _variant("base_int8", "qwen_3_4b.safetensors", 0.99),
            _variant("heretic_int8", "qwen_3_4b.safetensors", 1.30),  # fallback combined
        ]
        html_out = _decomposition_section(variants)
        row = [ln for ln in html_out.splitlines() if "heretic_int8</td>" in ln][0]
        assert "—" in row  # pure ablation missing

    def test_no_base_quant_no_section(self) -> None:
        variants = [_variant("heretic_int8", "qwen_3_4b.safetensors", 1.30)]
        assert _decomposition_section(variants) == ""

    def test_vs_base_entries_skipped_as_rows(self) -> None:
        """__vs_base entries must not generate their own decomposition rows."""
        variants = [
            _variant("base_int8", "qwen_3_4b.safetensors", 0.99),
            _variant("heretic_int8", "base_int8", 0.42),
            _variant("heretic_int8__vs_base", "qwen_3_4b.safetensors", 1.30),
        ]
        html_out = _decomposition_section(variants)
        rows = [ln for ln in html_out.splitlines() if ln.startswith("<tr><td>")]
        assert len(rows) == 1  # only heretic_int8, not the __vs_base sibling
