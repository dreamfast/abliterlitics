"""Tests for ladder_summary — reroll-fraction metric + ranked ladder output."""

from __future__ import annotations

import json
from pathlib import Path

from src.text_encoder.ladder_summary import anchor_lpips, build_summary, load_ladder, render_markdown


def _write_arm(results_dir: Path, name: str, vals: list[float]) -> None:
    doc = {"te_swap": {"values": {"lpips": vals}}}
    (results_dir / f"tebench_img_{name}.json").write_text(json.dumps(doc))


def _write_anchor(results_dir: Path, mean: float, n: int) -> None:
    (results_dir / "tebench_img_anchors.json").write_text(
        json.dumps(
            {
                "seed_change": {
                    "n_pairs": n,
                    "summaries": [
                        {"metric": "clip_img_dist", "mean": 0.9},
                        {"metric": "lpips", "mean": mean, "n": n},
                    ],
                }
            }
        )
    )


class TestLoadLadder:
    def test_loads_arms_and_anchors(self, tmp_path: Path) -> None:
        _write_arm(tmp_path, "heretic", [0.2, 0.3])
        _write_arm(tmp_path, "base_int4", [0.5, 0.5])
        _write_anchor(tmp_path, 0.516, 168)
        arms, anchors = load_ladder(tmp_path)
        assert set(arms) == {"heretic", "base_int4"}
        assert arms["heretic"]["lpips_mean"] == 0.25
        assert anchors.get("seed_change") is not None

    def test_skips_empty_arms(self, tmp_path: Path) -> None:
        _write_arm(tmp_path, "empty", [])
        arms, _ = load_ladder(tmp_path)
        assert arms == {}


class TestAnchorLpips:
    def test_metric_match_skips_other_summaries(self, tmp_path: Path) -> None:
        _write_anchor(tmp_path, 0.516, 168)
        _, anchors = load_ladder(tmp_path)
        mean, n = anchor_lpips(anchors)
        assert mean == 0.516
        assert n == 168

    def test_absent_anchor(self) -> None:
        assert anchor_lpips({}) == (None, None)

    def test_lpips_without_mean_skipped(self) -> None:
        anchors = {"seed_change": {"summaries": [{"metric": "lpips"}]}}
        assert anchor_lpips(anchors) == (None, None)


class TestBuildSummary:
    def test_reroll_fraction_and_ranking(self) -> None:
        arms = {"heretic": {"lpips_mean": 0.29, "n": 20}, "base_int4": {"lpips_mean": 0.49, "n": 20}}
        s = build_summary(arms, 0.516, seed_anchor_n=168)
        assert [r["arm"] for r in s["ladder"]] == ["heretic", "base_int4"]  # ascending
        assert s["ladder"][0]["reroll_fraction_pct"] == 56.2
        assert s["reference_reroll_pct"] == 56.2
        assert s["seed_change_anchor_lpips"] == 0.516
        assert s["seed_change_anchor_n"] == 168

    def test_null_anchor_gives_null_fraction(self) -> None:
        s = build_summary({"heretic": {"lpips_mean": 0.29, "n": 5}}, None)
        assert s["ladder"][0]["reroll_fraction_pct"] is None
        assert s["seed_change_anchor_lpips"] is None

    def test_custom_reference_arm(self) -> None:
        arms = {"heretic": {"lpips_mean": 0.29, "n": 5}, "abliterix": {"lpips_mean": 0.4, "n": 5}}
        s = build_summary(arms, 0.5, reference_arm="abliterix")
        assert s["reference_reroll_pct"] == 80.0


class TestMarkdown:
    def test_table_anchor_row_uses_true_anchor_n(self) -> None:
        s = build_summary({"heretic": {"lpips_mean": 0.29, "n": 20}}, 0.516, seed_anchor_n=168)
        md = render_markdown(s, "Krea2 ladder")
        assert "| **SEED CHANGE (anchor)** | **0.5160** | 168 | **100.0%** |" in md
        assert "56.2% of a full seed change" in md

    def test_no_anchor_row_when_null(self) -> None:
        s = build_summary({"heretic": {"lpips_mean": 0.29, "n": 5}}, None)
        md = render_markdown(s, "X")
        assert "SEED CHANGE" not in md
        assert "full seed change" not in md
