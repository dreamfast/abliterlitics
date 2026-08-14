"""Tests for Phase-4 report section builders — verdicts, ladders, controls.

Synthetic tebench_*.json fixtures in tmp dirs; verifies section HTML renders,
verdicts compute against pre-registered thresholds, and terminology guards
(no "Damage" column labels in Phase-4 sections).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.text_encoder.phase4_report import (
    _controls_section,
    _img_section,
    _verdict_section,
    build_phase4_sections,
    load_img_metrics,
)


def _mk_img_doc(variant: str, lpips: float, clip_delta: float = -0.1) -> dict:
    return {
        "variant": variant,
        "reference": "base_int8",
        "te_swap": {
            "n_pairs": 20,
            "summaries": [
                {"metric": "lpips", "n": 20, "mean": lpips, "ci_low": lpips * 0.9, "ci_high": lpips * 1.1},
                {"metric": "ssim", "n": 20, "mean": 0.8, "ci_low": 0.7, "ci_high": 0.9},
            ],
            "anchor_ratios": {"lpips/seed_change": lpips / 0.5},
        },
        "cross_prompt": {"n_pairs": 30, "summaries": [{"metric": "lpips", "n": 30, "mean": 0.6}]},
        "clip_score_delta_mean": clip_delta,
    }


def _mk_anchors(seed_lpips: float = 0.5, floor_lpips: float = 0.19) -> dict:
    return {
        "seed_change": {
            "n_pairs": 168,
            "summaries": [{"metric": "lpips", "n": 168, "mean": seed_lpips, "ci_low": 0.48, "ci_high": 0.53}],
        },
        "floor_band": {
            "n_pairs": 20,
            "summaries": [{"metric": "lpips", "n": 20, "mean": floor_lpips, "ci_low": 0.17, "ci_high": 0.21}],
        },
    }


class TestImgSection:
    def test_renders_with_anchors(self, tmp_path) -> None:
        anchors = _mk_anchors()
        html = _img_section(anchors, {"heretic": _mk_img_doc("heretic", 0.15)}, tmp_path)
        assert "sec-img" in html
        assert "0.500" in html and "0.190" in html  # anchors
        assert "0.150" in html  # heretic lpips
        assert "Damage" not in html

    def test_empty_variants(self, tmp_path) -> None:
        assert _img_section(_mk_anchors(), {}, tmp_path) == ""


class TestControlsSection:
    def test_ladder_order_and_notes(self) -> None:
        controls = {
            "conditions": {
                "null": {
                    "summary": {"mean_final_latent_div": 0.4, "noise_control_final": 1.2, "div_control_ratio": 0.33}
                },
                "order_shuffle": {
                    "summary": {"mean_final_latent_div": 0.1, "noise_control_final": 1.2, "div_control_ratio": 0.08}
                },
                "embed_shuffle": {
                    "summary": {"mean_final_latent_div": 0.9, "noise_control_final": 1.2, "div_control_ratio": 0.75}
                },
                "noise_0.25x": {
                    "summary": {"mean_final_latent_div": 0.02, "noise_control_final": 1.2, "div_control_ratio": 0.017}
                },
                "noise_1x": {
                    "summary": {"mean_final_latent_div": 0.08, "noise_control_final": 1.2, "div_control_ratio": 0.067}
                },
                "noise_2x": {
                    "summary": {"mean_final_latent_div": 0.15, "noise_control_final": 1.2, "div_control_ratio": 0.125}
                },
                "heretic_a1": {
                    "summary": {"mean_final_latent_div": 0.05, "noise_control_final": 1.2, "div_control_ratio": 0.042}
                },
            }
        }
        html = _controls_section(controls)
        assert "sec-controls" in html
        assert "null" in html and "heretic_a1" in html
        assert "monotonic as pre-registered" in html  # P4 pass
        assert "meaning destruction dominates" in html  # P6 (0.9 > 1.5*0.1)
        # ladder ordering: null row before noise rows
        assert html.index(">null<") < html.index(">noise_0.25x<")

    def test_none(self) -> None:
        assert _controls_section(None) == ""


class TestVerdictSection:
    def _ctx(self, heretic_lp: float, int4_lp: float, q8_lp: float) -> tuple:
        anchors = _mk_anchors()
        img_vars = {
            "heretic": _mk_img_doc("heretic", heretic_lp),
            "heretic_int4": _mk_img_doc("heretic_int4", int4_lp),
            "base_gguf_q8_0": _mk_img_doc("base_gguf_q8_0", q8_lp),
            # hy3-C1 regression: abliterated quants must NOT leak into the
            # base-family clean tier via substring matching
            "heretic_gguf_q8_0": _mk_img_doc("heretic_gguf_q8_0", 0.30),
            "heretic_int8": _mk_img_doc("heretic_int8", 0.29),
            "base_nvfp4": _mk_img_doc("base_nvfp4", 0.28),
            # round-2 fixture: matched pairs for per-tier deltas — including
            # a QUANTIZATION-DOMINATED heavy tier (|heretic_int4 - base_int4|
            # within the ±0.02 guard, mirroring real data -0.005)
            "base_int8": _mk_img_doc("base_int8", 0.15),
            "base_int4": _mk_img_doc("base_int4", 0.455),
        }
        controls = {
            "conditions": {
                "order_shuffle": {
                    "summary": {"mean_final_latent_div": 0.1, "noise_control_final": 1.2, "div_control_ratio": 0.08}
                },
                "embed_shuffle": {
                    "summary": {"mean_final_latent_div": 0.9, "noise_control_final": 1.2, "div_control_ratio": 0.75}
                },
            }
        }
        return anchors, img_vars, controls

    def test_h1_supported_within_band(self) -> None:
        anchors, img_vars, controls = self._ctx(heretic_lp=0.15, int4_lp=0.45, q8_lp=0.19)
        html = _verdict_section(anchors, img_vars, controls, {}, {}, {})
        assert "H1" in html and "SUPPORTED" in html
        assert "DEFERRED" in html  # H6 always present
        # column labels clean; "damage" only in the policy explainer of the
        # labeled interpretation block (allowed by #117/#118)
        assert "<th>Damage" not in html and "Damage (quant" not in html

    def test_h1_refuted_above_band(self) -> None:
        anchors, img_vars, controls = self._ctx(heretic_lp=0.50, int4_lp=0.45, q8_lp=0.19)
        html = _verdict_section(anchors, img_vars, controls, {}, {}, {})
        h1_row = html[html.index("H1") : html.index("H2")]
        assert "REFUTED" in h1_row

    def test_h2_threshold_base_only_tier(self) -> None:
        anchors, img_vars, controls = self._ctx(heretic_lp=0.15, int4_lp=0.45, q8_lp=0.19)
        html = _verdict_section(anchors, img_vars, controls, {}, {}, {})
        h2_row = html[html.index("H2") : html.index("H3")]
        assert "SUPPORTED" in h2_row  # base q8 0.19 <= band; int4 0.45 > band
        # hy3-C1: heretic rows must NOT flip the verdict; hy3-R2 (round 2):
        # attribution must be PER-TIER DELTA, never the global heretic max
        # (int4 is quantization-dominated and must not be credited to ablation)
        assert "abliteration adds up to +" in h2_row
        assert "not via quantization" in h2_row
        assert "quantization-dominated" in h2_row

    def test_h2_refuted_when_base_tier_fails(self) -> None:
        anchors, img_vars, controls = self._ctx(heretic_lp=0.15, int4_lp=0.45, q8_lp=0.45)
        html = _verdict_section(anchors, img_vars, controls, {}, {}, {})
        h2_row = html[html.index("H2") : html.index("H3")]
        assert "REFUTED" in h2_row  # base q8 0.45 above band -> P2 genuinely fails

    def test_exec_summary_no_invisible_claim_when_above_band(self) -> None:
        from src.text_encoder.phase4_report import _executive_summary

        anchors = _mk_anchors()
        img_vars = {
            "heretic": _mk_img_doc("heretic", 0.50),
            "base_nvfp4": _mk_img_doc("base_nvfp4", 0.28),
        }
        controls = {
            "conditions": {
                "noise_1x": {"summary": {"mean_final_latent_div": 0.03}},
                "heretic_a10": {"summary": {"mean_final_latent_div": 0.033}},
            }
        }
        html = _executive_summary(anchors, img_vars, controls, {}, {})
        assert "invisible" not in html  # deepseek-C1: no unqualified invisibility claim
        assert "small-edit effect" in html

    def test_h7_not_run(self) -> None:
        html = _verdict_section(None, {}, None, {}, {}, {})
        h7_row = html[html.index("H7") :]
        assert "NOT RUN" in h7_row

    def test_h6_always_deferred(self) -> None:
        html = _verdict_section(None, {}, None, {}, {}, {})
        assert "H6" in html and "DEFERRED" in html


class TestBuildPhase4:
    def test_empty_dir_renders_only_verdicts(self, tmp_path: Path) -> None:
        # data-driven sections absent; the verdict table (with static H3/H6/H7
        # rows) still renders — it is the policy table
        html = build_phase4_sections(tmp_path)
        assert "sec-img" not in html
        assert "sec-controls" not in html
        assert "sec-attention" not in html
        assert "sec-verdicts" in html and "H6" in html

    def test_full_load(self, tmp_path: Path) -> None:
        (tmp_path / "tebench_img_anchors.json").write_text(json.dumps(_mk_anchors()))
        (tmp_path / "tebench_img_heretic.json").write_text(json.dumps(_mk_img_doc("heretic", 0.15)))
        (tmp_path / "tebench_attention_heretic.json").write_text(
            json.dumps({"variant_name": "heretic", "jsd_mean_all": 0.01, "jsd_per_layer": {"3": 0.02, "7": 0.05}})
        )
        (tmp_path / "tebench_discriminability_heretic.json").write_text(
            json.dumps({"variant_name": "heretic", "base_spread": 0.9, "variant_spread": 0.89, "spread_ratio": 0.99})
        )
        html = build_phase4_sections(tmp_path)
        assert "sec-img" in html
        assert "sec-attention" in html
        assert "sec-discriminability" in html
        assert "sec-verdicts" in html

    def test_loaders(self, tmp_path: Path) -> None:
        (tmp_path / "tebench_img_anchors.json").write_text(json.dumps(_mk_anchors()))
        (tmp_path / "tebench_img_heretic.json").write_text(json.dumps(_mk_img_doc("heretic", 0.15)))
        anchors, variants = load_img_metrics(tmp_path)
        assert anchors is not None and "heretic" in variants


class TestPhase5ClipAdherence:
    def _img_vars(self) -> dict[str, dict[str, Any]]:
        def doc(delta: float) -> dict[str, Any]:
            return {
                "reference": "base_int8",
                "clip_score_delta_mean": delta,
                "clip_scores": {
                    f"p{i}": {"variant": 28.0 + delta, "reference": 28.0, "delta": delta} for i in range(5)
                },
            }

        return {"base_int8": doc(0.0), "heretic": doc(-0.4), "base_int4": doc(-3.1)}

    def test_p8_section_renders_and_classifies(self) -> None:
        from src.text_encoder.phase4_report import _clip_adherence_section

        html = _clip_adherence_section(self._img_vars())
        assert "sec-clip" in html
        assert "P8 partially holds" in html  # only int4 is visible, flagged as tier
        assert "in band" in html and "visibly degraded" in html

    def test_p8_holds_when_all_in_band(self) -> None:
        from src.text_encoder.phase4_report import _clip_adherence_section

        vars_ = self._img_vars()
        del vars_["base_int4"]
        html = _clip_adherence_section(vars_)
        assert "P8 holds" in html and "partially" not in html

    def test_empty_when_no_clip(self) -> None:
        from src.text_encoder.phase4_report import _clip_adherence_section

        assert _clip_adherence_section({"base": {"te_swap": {}}}) == ""


class TestNoiseEquivalentDose:
    def test_interpolates_between_rungs(self) -> None:
        from src.text_encoder.phase4_report import _noise_equivalent_dose

        ladder = [(0.05, 0.010), (0.1, 0.014), (0.15, 0.018), (0.25, 0.0237)]
        dose = _noise_equivalent_dose(ladder, 0.014)
        assert dose == pytest.approx(0.1)
        mid = _noise_equivalent_dose(ladder, 0.016)
        assert 0.1 < mid < 0.15

    def test_none_outside_bracket(self) -> None:
        from src.text_encoder.phase4_report import _noise_equivalent_dose

        ladder = [(0.05, 0.010), (0.1, 0.014), (0.25, 0.0237)]
        assert _noise_equivalent_dose(ladder, 0.005) is None  # below
        assert _noise_equivalent_dose(ladder, 0.05) is None  # above

    def test_p9_note_gated_on_bracket(self) -> None:
        from src.text_encoder.phase4_report import _controls_section

        def cond(div: float) -> dict[str, Any]:
            return {
                "summary": {
                    "mean_final_latent_div": div,
                    "noise_control_final": 0.152,
                    "div_control_ratio": div / 0.152,
                }
            }

        base_conds = {
            "noise_0.05x": cond(0.012),
            "noise_0.1x": cond(0.014),
            "noise_0.15x": cond(0.017),
            "noise_1x": cond(0.032),
            "heretic_a1": cond(0.0155),
            "heretic_a10": cond(0.0324),
        }
        doc = {
            "conditions": base_conds,
            "heretic_mse": [1.34] * 6,
            "max_dose_mse": [95.0] * 6,
        }
        html = _controls_section(doc)
        assert "P9 NOISE-EQUIVALENT DOSE" in html
        assert "× its own conditioning MSE" in html
        # max_dose ~71x heretic MSE -> interpolated dose lands far above 0.5x
        # heretic-MSE -> pre-registered P9 REFUTED wording
        assert "P9 REFUTED" in html
        # without a bracket the note must be absent
        no_bracket = {
            **doc,
            "conditions": {
                k: v for k, v in base_conds.items() if k not in ("noise_0.05x", "noise_0.1x", "noise_0.15x")
            },
        }
        html2 = _controls_section(no_bracket)
        assert "P9 NOISE-EQUIVALENT DOSE" not in html2


class TestPhase5Breadth:
    def _breadth(self, heretic_shifts: list[float]) -> dict[str, Any]:
        cats = ["landscape", "multi_subject", "portrait", "text_render"]
        base_overall = sum(heretic_shifts) / len(heretic_shifts)
        per_pair = [{"variant": "heretic", "lpips": s} for s in heretic_shifts * 4] + [
            {"variant": "base_int4", "lpips": 0.4} for _ in range(16)
        ]
        per_cat = {
            "heretic": {c: {"n": 8, "lpips_mean": s} for c, s in zip(cats, heretic_shifts, strict=True)},
            "base_int4": {c: {"n": 8, "lpips_mean": 0.4} for c in cats},
        }
        return {
            "base": "base",
            "categories": cats,
            "per_category": per_cat,
            "seed_anchor_per_category": {c: {"n": 8, "lpips_mean": 0.5} for c in cats},
            "per_pair": per_pair,
            "overall_h": base_overall,
        }

    def test_p10_uniform(self) -> None:
        from src.text_encoder.phase4_report import _breadth_section

        html = _breadth_section(self._breadth([0.28, 0.30, 0.29, 0.31]))
        assert "sec-breadth" in html
        assert "P10 holds" in html

    def test_p10_concentrated(self) -> None:
        from src.text_encoder.phase4_report import _breadth_section

        html = _breadth_section(self._breadth([0.29, 0.30, 0.28, 0.60]))
        assert "P10 fails" in html
        assert "text_render" in html  # names the concentrating category

    def test_none_category_does_not_mislabel_worst(self) -> None:
        from src.text_encoder.phase4_report import _breadth_section

        doc = self._breadth([0.29, 0.30, 0.28, 0.60])
        doc["per_category"]["heretic"]["portrait"]["lpips_mean"] = None  # missing category
        html = _breadth_section(doc)
        assert "P10 fails" in html
        assert "text_render" in html  # the TRUE worst, not the None slot

    def test_heretic_missing_deferred_note(self) -> None:
        from src.text_encoder.phase4_report import _breadth_section

        doc = self._breadth([0.29, 0.30, 0.28, 0.31])
        del doc["per_category"]["heretic"]
        html = _breadth_section(doc)
        assert "P10 deferred" in html

    def test_empty(self) -> None:
        from src.text_encoder.phase4_report import _breadth_section

        assert _breadth_section(None) == ""


class TestPhase5H6Verdict:
    def _taboo(self, heretic_deltas: dict[str, float], int8_deltas: dict[str, float]) -> dict[str, Any]:
        tiers = ["tier1_nudity", "tier2_explicit", "tier3_gore"]
        base_gap = {"tier1_nudity": 0.10, "tier2_explicit": -0.05, "tier3_gore": 0.08}
        per_tier: dict[str, dict[str, dict[str, Any]]] = {"base": {}, "heretic": {}, "base_int8": {}}
        for t in tiers:
            per_tier["base"][t] = {"n": 12, "gap_mean": base_gap[t], "nsfw_rate": 0.5, "lpips_mean": 0.0}
            per_tier["heretic"][t] = {
                "n": 12,
                "gap_mean": base_gap[t] + heretic_deltas.get(t, 0.0),
                "nsfw_rate": 0.5,
                "lpips_mean": 0.1,
            }
            per_tier["base_int8"][t] = {
                "n": 12,
                "gap_mean": base_gap[t] + int8_deltas.get(t, 0.0),
                "nsfw_rate": 0.5,
                "lpips_mean": 0.05,
            }
        return {"tiers": tiers, "per_tier": per_tier}

    def test_supported_two_tiers_outside_band(self) -> None:
        from src.text_encoder.phase4_report import _h6_verdict

        label, evidence, cls = _h6_verdict(
            self._taboo(
                {"tier1_nudity": 0.06, "tier2_explicit": 0.08, "tier3_gore": 0.0},
                {"tier1_nudity": 0.01, "tier2_explicit": -0.01, "tier3_gore": 0.01},
            )
        )
        assert label == "SUPPORTED" and cls == "yes"
        assert "±0.01" in evidence

    def test_refuted_within_control_band(self) -> None:
        from src.text_encoder.phase4_report import _h6_verdict

        label, _e, cls = _h6_verdict(
            self._taboo(
                {"tier1_nudity": 0.008, "tier2_explicit": -0.006, "tier3_gore": 0.0},
                {"tier1_nudity": 0.01, "tier2_explicit": -0.01, "tier3_gore": 0.01},
            )
        )
        assert label == "REFUTED" and cls == ""

    def test_mixed_one_tier_only(self) -> None:
        from src.text_encoder.phase4_report import _h6_verdict

        label, _e, cls = _h6_verdict(
            self._taboo(
                {"tier1_nudity": 0.06, "tier2_explicit": 0.0, "tier3_gore": 0.0},
                {"tier1_nudity": 0.0, "tier2_explicit": 0.0, "tier3_gore": 0.0},
            )
        )
        assert label == "MIXED" and cls == "part"

    def test_deferred_when_none(self) -> None:
        from src.text_encoder.phase4_report import _h6_verdict

        assert _h6_verdict(None)[0] == "DEFERRED"

    def test_section_no_images_note(self) -> None:
        from src.text_encoder.phase4_report import _taboo_section

        html = _taboo_section(
            self._taboo(
                {"tier1_nudity": 0.06, "tier2_explicit": 0.08},
                {"tier1_nudity": 0.01, "tier2_explicit": -0.01},
            )
        )
        assert "sec-taboo" in html
        assert "No explicit imagery is embedded" in html
        assert "H6 verdict: SUPPORTED" in html

    def test_no_int8_control_is_inconclusive_not_supported(self) -> None:
        from src.text_encoder.phase4_report import _h6_verdict

        taboo = self._taboo({"tier1_nudity": 0.06, "tier2_explicit": 0.08}, {})
        del taboo["per_tier"]["base_int8"]
        label, evidence, cls = _h6_verdict(taboo)
        assert label == "INCONCLUSIVE"
        assert "negative control" in evidence
        assert cls == ""

    def test_partial_gap_rows_inconclusive(self) -> None:
        from src.text_encoder.phase4_report import _h6_verdict

        taboo = self._taboo({"tier1_nudity": 0.06}, {})
        taboo["per_tier"]["base"]["tier2_explicit"]["gap_mean"] = None
        taboo["per_tier"]["heretic"]["tier2_explicit"]["gap_mean"] = None
        taboo["per_tier"]["base"]["tier3_gore"]["gap_mean"] = None
        taboo["per_tier"]["heretic"]["tier3_gore"]["gap_mean"] = None
        label, _e, _c = _h6_verdict(taboo)
        assert label == "INCONCLUSIVE"

    def test_gore_tier_nsfw_rendered_na(self) -> None:
        from src.text_encoder.phase4_report import _taboo_section

        html = _taboo_section(self._taboo({"tier1_nudity": 0.06}, {"tier1_nudity": 0.01}))
        # tier3_gore column shows n/a for the sexual NSFW rate on every row
        assert "n/a" in html
