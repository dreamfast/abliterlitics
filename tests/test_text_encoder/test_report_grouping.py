"""Tests for report variant grouping (family × container × quant tier)."""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")

from src.text_encoder.generate_report import (  # noqa: E402
    _gguf_section,
    _group_ordered,
    _group_row,
    _group_sort_key,
    _table_rows,
    _variant_group,
    graph_layer_heatmap,
    graph_layer_profile,
)

GGUF_META = lambda bpw: {"quant_label": "x", "bpw": bpw, "qtype_histogram": {"Q4_K": 1}}  # noqa: E731


def _var(name: str, gguf: dict | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "variant_id": name,
        "dtype": "comfy_bf16",
        "gguf": gguf,
        "backend": "ZImageBackend",
        "reference": "base",
        "floor": "real_int8",
        "streams": {
            "refined_context": {
                "stream": "refined_context",
                "stage": "CONNECTOR_OUT",
                "n_prompts": 60,
                "metrics": {"cka_linear": 0.99, "cos_all": 0.999, "mse": 1e-4},
                "damage_quant_units": 10.0,
                "damage_quant_ci": None,
                "damage_semantic_units": 0.01,
                "per_category": {},
            }
        },
    }


class TestVariantGroup:
    def test_five_groups(self) -> None:
        assert _variant_group(_var("base_int8"))[1].startswith("Base model — safetensors")
        assert _variant_group(_var("base_gguf_q4_k_m", GGUF_META(4.9)))[1].startswith("Base model — GGUF")
        assert _variant_group(_var("heretic"))[1].startswith("Abliterated (heretic) — safetensors")
        assert _variant_group(_var("heretic_gguf_q8_0", GGUF_META(8.5)))[1].startswith("Abliterated (heretic) — GGUF")
        assert _variant_group(_var("heretic_int8__vs_base"))[1].startswith("Abliterated quant — combined")

    def test_group_order_monotonic(self) -> None:
        order = [
            _variant_group(_var("base_int8"))[0],
            _variant_group(_var("base_gguf_f16", GGUF_META(16)))[0],
            _variant_group(_var("heretic"))[0],
            _variant_group(_var("heretic_gguf_f16", GGUF_META(16)))[0],
            _variant_group(_var("heretic_int8__vs_base"))[0],
        ]
        assert order == sorted(order) == [0, 1, 2, 3, 4]

    def test_name_only_gguf_inference(self) -> None:
        """Divergence summaries carry names only — container inferred from name."""
        assert _variant_group({"name": "base_gguf_q6_k"})[0] == 1
        assert _variant_group({"name": "heretic_gguf_q6_k"})[0] == 3
        assert _variant_group({"name": "base_int8"})[0] == 0


class TestGroupSortKey:
    def test_safetensors_tier_order(self) -> None:
        names = ("heretic", "heretic_int8", "heretic_fp8", "heretic_mxfp8", "heretic_nvfp4", "heretic_int4")
        keys = [_group_sort_key(_var(n))[0] for n in names]
        assert keys == sorted(keys)

    def test_gguf_bpw_descending(self) -> None:
        keys = [_group_sort_key(_var("base_gguf_q4_k_m", GGUF_META(b)))[0] for b in (16.0, 8.5, 4.9, 4.1)]
        assert keys == sorted(keys)

    def test_name_only_gguf_ladder_order(self) -> None:
        # Expected ladder: f16 (bpw 16) -> q8_0 -> q6_k -> ... -> q3_k_m
        ladder = ["heretic_gguf_f16", "heretic_gguf_q8_0", "heretic_gguf_q6_k", "heretic_gguf_q3_k_m"]
        keys = [_group_sort_key({"name": n, "gguf": None})[0] for n in ladder]
        assert keys == sorted(keys)

    def test_group_ordered_full_matrix(self) -> None:
        variants = [
            _var("heretic_gguf_q8_0", GGUF_META(8.5)),
            _var("base_int8"),
            _var("heretic"),
            _var("heretic_int8__vs_base"),
            _var("base_gguf_q4_k_m", GGUF_META(4.9)),
            _var("heretic_int4"),
        ]
        ordered = [v["name"] for v in _group_ordered(variants)]
        expected_seq = ["base_int8", "base_gguf_q4_k_m", "heretic", "heretic_gguf_q8_0", "heretic_int8__vs_base"]
        pos = [ordered.index(n) for n in expected_seq]
        assert pos == sorted(pos)


class TestGroupedTables:
    def test_table_rows_group_headers(self) -> None:
        variants = [
            _var("heretic_gguf_q8_0", GGUF_META(8.5)),
            _var("base_int8"),
            _var("heretic"),
        ]
        rows = _table_rows(variants)
        i_base = rows.index("Base model — safetensors quants")
        i_heretic = rows.index("Abliterated (heretic) — safetensors")
        i_gguf = rows.index("Abliterated (heretic) — GGUF containers")
        assert i_base < i_heretic < i_gguf
        assert 'class="group-row"' in rows

    def test_ladder_split_into_blocks(self) -> None:
        variants = [
            _var("heretic_gguf_q4_k_m", GGUF_META(4.9)),
            _var("base_gguf_f16", GGUF_META(16.0)),
            _var("base_gguf_q4_k_m", GGUF_META(4.9)),
        ]
        out = _gguf_section(variants, divergence={})
        i_base_block = out.index("Base containers — pure GGUF quantization")
        i_heretic_block = out.index("Heretic containers — pure ablation")
        assert i_base_block < i_heretic_block
        # base_f16 row inside base block, before heretic block
        assert out.index("base_gguf_f16") < i_heretic_block

    def test_group_row_colspan(self) -> None:
        assert 'colspan="4"' in _group_row("x", colspan=4)
        assert 'colspan="9"' in _group_row("x")


class TestGroupedGraphs:
    def test_heatmap_grouped_ordering(self, tmp_path: Any) -> None:
        variants = [
            _var("heretic_gguf_q8_0", GGUF_META(8.5)),
            _var("base_int8"),
            _var("heretic"),
        ]
        # add a layer stream so the heatmap has data
        for v in variants:
            v["streams"]["layer_35"] = {
                "stream": "layer_35",
                "stage": "RAW_HIDDEN",
                "n_prompts": 60,
                "metrics": {"mse": 1.0},
            }
        out = tmp_path / "hm.svg"
        graph_layer_heatmap(variants, out)
        assert out.exists()

    def test_layer_profile_filters_gguf_and_duals(self, tmp_path: Any) -> None:
        variants = [
            _var("base_int8"),
            _var("heretic"),
            _var("heretic_gguf_q8_0", GGUF_META(8.5)),
            _var("heretic_int8__vs_base"),
        ]
        for v in variants:
            v["streams"]["layer_00"] = {
                "stream": "layer_00",
                "stage": "RAW_HIDDEN",
                "n_prompts": 60,
                "metrics": {"mse": 1.0},
            }
        out = tmp_path / "lp.svg"
        graph_layer_profile(variants, out)
        assert out.exists()
        content = out.read_text()
        assert "base_int8" in content and "heretic" in content
        assert "heretic_gguf_q8_0" not in content
        assert "heretic_int8__vs_base" not in content


class TestGroupedGraphsR1:
    def _with_streams(self, name: str, gguf: dict | None = None, cats: bool = True) -> dict[str, Any]:
        v = _var(name, gguf)
        per_cat = {c: {"mse": 1e-4 * (10**i)} for i, c in enumerate(["innocent", "mild", "extreme"])} if cats else {}
        v["streams"]["refined_context"]["per_category"] = per_cat
        v["streams"]["cap_embedded"] = {
            "stream": "cap_embedded",
            "stage": "PROJECTED",
            "n_prompts": 60,
            "metrics": {"cka_linear": 0.99, "cos_all": 0.999, "mse": 1e-4},
            "damage_quant_units": 99.0,
            "damage_quant_ci": None,
            "damage_semantic_units": None,
            "per_category": {},
        }
        return v

    def test_damage_ranking_dit_input_only_grouped(self, tmp_path: Any) -> None:
        """One bar per variant (headline stream), NOT one per pipeline stage."""
        from src.text_encoder.generate_report import graph_damage_ranking

        variants = [self._with_streams("heretic"), self._with_streams("base_int8")]
        out = tmp_path / "dr.svg"
        graph_damage_ranking(variants, out)
        content = out.read_text()
        # headline stream only — cap_embedded rows must NOT appear even
        # though the fixture provides a damaged cap_embedded stream (99x)
        assert "cap_embedded" not in content
        # group annotations present
        assert "base quants" in content and "heretic" in content

    def test_damage_ranking_log_axis(self, tmp_path: Any) -> None:
        from src.text_encoder.generate_report import graph_damage_ranking

        variants = [self._with_streams("base_gguf_f16", GGUF_META(16.0))]
        variants[0]["streams"]["refined_context"]["damage_quant_units"] = 0.001
        out = tmp_path / "dr.svg"
        graph_damage_ranking(variants, out)
        assert out.exists()

    def test_category_gradient_two_panels(self, tmp_path: Any) -> None:
        from src.text_encoder.generate_report import graph_category_gradient

        variants = [
            self._with_streams("base_int8"),
            self._with_streams("heretic"),
            self._with_streams("heretic_gguf_q4_k_m", GGUF_META(4.9)),
        ]
        out = tmp_path / "cg.svg"
        graph_category_gradient(variants, out)
        content = out.read_text()
        assert "Base quants (noise floor)" in content
        assert "Abliterated (heretic)" in content

    def test_category_gradient_skips_no_category_data(self, tmp_path: Any) -> None:
        from src.text_encoder.generate_report import graph_category_gradient

        variants = [self._with_streams("heretic", cats=False)]
        out = tmp_path / "cg.svg"
        graph_category_gradient(variants, out)
        assert not out.exists()
