"""Tests for h3_report.py — verdict logic + smoke refusal + rendering."""

from __future__ import annotations

import json
from pathlib import Path

from src.text_encoder.h3_report import (
    _damage_ladder_rows,
    build_report,
    h8_verdict,
    h9_verdict,
    h10_verdict,
    h13_verdict,
    verdict_row,
)


def _audio(smoke: bool = False, h_rare: float = 0.0, h_common: float = 0.0, floor: dict | None = None) -> dict:
    return {
        "smoke": smoke,
        "arms": {
            "base": {
                "mean_wer": 0.017,
                "tiers": {
                    "rare_polysyllabic": {"error_rate": 0.0},
                    "common_polysyllabic": {"error_rate": 0.0196},
                    "monosyllabic": {"error_rate": 0.0526},
                },
            },
            "heretic": {
                "mean_wer": 0.015,
                "cross_seed_jaccard_h9": None,
                "tiers": {
                    "rare_polysyllabic": {"error_rate": h_rare, "total": 54},
                    "common_polysyllabic": {"error_rate": h_common, "total": 54},
                    "monosyllabic": {"error_rate": 0.0526, "total": 57},
                },
            },
        },
        "floors": {
            "a_whisper_intrinsic": floor
            or {"rare_polysyllabic": 0.0, "common_polysyllabic": 0.0196, "monosyllabic": 0.0526}
        },
    }


class TestH8:
    def test_null_result_refuted_no_damage(self) -> None:
        r = h8_verdict(_audio(h_rare=0.0, h_common=0.0))
        assert r["verdict"] == "REFUTED (no damage)"

    def test_damage_on_two_tiers_supported(self) -> None:
        r = h8_verdict(
            _audio(
                h_rare=0.5,
                h_common=0.4,
                floor={"rare_polysyllabic": 0.0, "common_polysyllabic": 0.02, "monosyllabic": 0.05},
            )
        )
        assert r["verdict"] == "SUPPORTED"

    def test_smoke_refused(self) -> None:
        r = h8_verdict(_audio(smoke=True))
        assert r["verdict"] == "REFUSED (smoke)"

    def test_missing(self) -> None:
        assert h8_verdict(None)["verdict"] == "NOT RUN"


class TestH9:
    def test_vacuous_when_heretic_clean(self) -> None:
        r = h9_verdict(_audio())
        assert r["verdict"] == "VACUOUS (no failures)"

    def test_structured_failures_supported(self) -> None:
        a = _audio()
        a["arms"]["heretic"]["cross_seed_jaccard_h9"] = 1.0
        a["arms"]["base"]["cross_seed_jaccard_h9"] = 0.25
        r = h9_verdict(a)
        assert r["verdict"] == "SUPPORTED"

    def test_smoke_refused(self) -> None:
        a = _audio(smoke=True)
        a["arms"]["heretic"]["cross_seed_jaccard_h9"] = 1.0
        assert h9_verdict(a)["verdict"] == "REFUSED (smoke)"

    def test_base_clean_heretic_dirty_supported(self) -> None:
        """Review R1: heretic reproducible failures + clean base is the
        STRONGEST structured-edit signal — never REFUTED."""
        a = _audio()
        a["arms"]["heretic"]["cross_seed_jaccard_h9"] = 1.0
        a["arms"]["base"]["cross_seed_jaccard_h9"] = None
        r = h9_verdict(a)
        assert r["verdict"] == "SUPPORTED (base clean)"

    def test_base_zero_overlap_benign_null(self) -> None:
        a = _audio()
        a["arms"]["heretic"]["cross_seed_jaccard_h9"] = 0.8
        a["arms"]["base"]["cross_seed_jaccard_h9"] = 0.0
        r = h9_verdict(a)
        assert r["verdict"] == "REFUTED (no structure)"
        # benign null renders mid, not bad
        row = verdict_row("H9", "t", r)
        assert "mid" in row and "bad" not in row


class TestH10:
    def test_within_anchor_and_band_supported(self) -> None:
        r = h10_verdict(
            {
                "smoke": False,
                "seed_change_anchor_lpips": 0.6,
                "clip_band": 1.5,
                "variants": {
                    "heretic": {
                        "mean_frame_lpips": 0.33,
                        "clip_delta_absmax": 1.1,
                        "within_seed_anchor": True,
                        "clip_within_band": True,
                    }
                },
            }
        )
        assert r["verdict"] == "SUPPORTED"

    def test_beyond_anchor_refuted(self) -> None:
        v = {
            "smoke": False,
            "seed_change_anchor_lpips": 0.3,
            "clip_band": 1.5,
            "variants": {
                "heretic": {
                    "mean_frame_lpips": 0.8,
                    "clip_delta_absmax": 1.1,
                    "within_seed_anchor": False,
                    "clip_within_band": True,
                }
            },
        }
        assert h10_verdict(v)["verdict"] == "REFUTED"

    def test_smoke_refused(self) -> None:
        assert h10_verdict({"smoke": True})["verdict"] == "REFUSED (smoke)"


class TestVerdictRow:
    def test_escapes_and_classes(self) -> None:
        row = verdict_row("H8", "damage <test>", {"verdict": "REFUTED (no damage)", "detail": "x"})
        assert "&lt;test&gt;" in row
        assert "mid" in row  # null-refuted rows are mid, not bad


class TestBuildReport:
    def test_renders_from_fixture_results(self, tmp_path: Path) -> None:
        (tmp_path / "tebench_h3_audio.json").write_text(json.dumps(_audio()))
        (tmp_path / "tebench_h3_video.json").write_text(
            json.dumps(
                {
                    "smoke": False,
                    "seed_change_anchor_lpips": 0.6031,
                    "clip_band": 1.5,
                    "variants": {
                        "heretic": {
                            "n_pairs": 16,
                            "mean_frame_lpips": 0.3259,
                            "lpips_values": [0.3],
                            "mean_clip_self": 16.5,
                            "mean_clip_delta_vs_base": -0.28,
                            "clip_delta_absmax": 1.14,
                            "within_seed_anchor": True,
                            "clip_within_band": True,
                        }
                    },
                }
            )
        )
        (tmp_path / "te_heretic.json").write_text(
            json.dumps(
                {
                    "variants": [
                        {
                            "variant_id": "x",
                            "dtype": "comfy_int8_convrot",
                            "streams": [
                                {
                                    "stream": "refined_context",
                                    "stage": "connector_out",
                                    "metrics": {"cos_all": 0.9935, "cka_linear": 0.9988},
                                }
                            ],
                        }
                    ]
                }
            )
        )
        out = tmp_path / "h3_report.html"
        html = build_report(tmp_path, out)
        assert out.exists()
        assert "H8" in html and "H12" in html
        assert "REFUTED (no damage)" in html
        assert "SUPPORTED" in html  # H10
        assert "NOT TESTED BY DESIGN" in html
        assert "0.6031" in html
        # no smoke banner for clean inputs
        assert "REFUSED:" not in html

    def test_smoke_inputs_banner_and_no_publish(self, tmp_path: Path) -> None:
        (tmp_path / "tebench_h3_audio.json").write_text(json.dumps(_audio(smoke=True)))
        html = build_report(tmp_path, tmp_path / "r.html")
        assert "REFUSED:" in html
        assert "REFUSED (smoke)" in html


class TestH13:
    def _damage(self, nulls: dict[str, bool] | None = None, smoke: bool = False) -> dict:
        return {
            "smoke": smoke,
            "null_conditions": nulls
            if nulls is not None
            else {
                "no_tilt_turbo": True,
                "no_tilt_stock": True,
                "dnsmos_ci_zero": True,
                "word_inside_seed_noise": True,
                "swap_near_quant_control": True,
            },
            "instruments": {
                "delta_direction": {"turbo": {"speech:te_swap (heretic-base)": {"mean_pairwise_cos": 0.0074, "n": 48}}}
            },
        }

    def test_all_null_refuted_benign(self) -> None:
        res = h13_verdict(self._damage())
        assert "REFUTED (noise injection" in res["verdict"]
        assert "cos 0.0074" in res["detail"]

    def test_flagged_condition_partial(self) -> None:
        nulls = self._damage()["null_conditions"]
        nulls["no_tilt_turbo"] = False
        res = h13_verdict(self._damage(nulls))
        assert "PARTIAL" in res["verdict"]
        assert "no_tilt_turbo" in res["verdict"]

    def test_smoke_refused(self) -> None:
        res = h13_verdict(self._damage(smoke=True))
        assert "REFUSED (smoke)" in res["verdict"]

    def test_missing(self) -> None:
        assert h13_verdict(None)["verdict"] == "NOT RUN"

    def test_no_conditions_not_run(self) -> None:
        d = self._damage()
        d["null_conditions"] = {}
        assert h13_verdict(d)["verdict"] == "NOT RUN"

    def test_absent_condition_not_false_refuted(self) -> None:
        """A silently-absent condition (probe format drift) must never read
        as PASS — partial sets refuse the REFUTED verdict (review round 1)."""
        nulls = self._damage()["null_conditions"]
        nulls.pop("no_tilt_stock")
        res = h13_verdict(self._damage(nulls))
        assert "NOT RUN" in res["verdict"]
        assert "no_tilt_stock" in res["detail"]


class TestDamageLadderRows:
    def test_rendes_instruments_from_json(self) -> None:
        dmg = {
            "smoke": False,
            "instruments": {
                "audio_vae_ladder": {
                    "turbo": {
                        "te_swap:speech:frame": {"mean": 0.797},
                        "floor:speech:frame": {"mean": 0.811},
                        "seed_base:speech:frame": {"mean": 0.216},
                    },
                    "stock": {
                        "te_swap:speech:frame": {"mean": 0.851},
                        "floor:speech:frame": {"mean": 0.870},
                        "seed_base:speech:frame": {"mean": 0.203},
                    },
                },
                "delta_direction": {"turbo": {"speech:te_swap (heretic-base)": {"mean_pairwise_cos": 0.0074}}},
                "word_forensics_utterly": {"te_swap_mel_cos_mean": 0.857, "seed_noise_mel_cos_mean": 0.730},
                "dnsmos_paired": {"OVRL": {"mean": -0.0435, "ci95": [-0.0956, 0.008]}},
                "accent_native_pct": {"base": 79.6, "heretic": 79.2},
            },
        }
        html = _damage_ladder_rows(dmg)
        assert "0.797 / 0.811 / 0.216" in html
        assert "0.851 / 0.870 / 0.203" in html
        assert "0.007" in html
        assert "0.857 vs 0.730" in html
        assert "79.6% vs 79.2%" in html
        assert "sound fine" in html

    def test_smoke_guards(self) -> None:
        assert "unavailable" in _damage_ladder_rows({"smoke": True})
        assert "unavailable" in _damage_ladder_rows(None)

    def test_missing_values_em_dash(self) -> None:
        html = _damage_ladder_rows({"smoke": False, "instruments": {}})
        assert "&mdash;" in html
