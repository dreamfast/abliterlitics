"""Tests for src/db/importers.py — result file parsing and import."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.db.core import get_db
from src.db.importers import (
    import_all_results,
    import_kl_results,
    import_lm_eval_results,
    import_weight_results,
)
from src.db.pipeline import get_all_stages


def _setup_comparison(conn) -> None:
    """Create a minimal comparison for testing."""
    conn.execute(
        "INSERT INTO comparisons (name, base_model, num_variants) VALUES (?, ?, ?)",
        ("test-comp", "base-model", 2),
    )
    conn.execute(
        "INSERT INTO models (comparison, model_name, display_name, model_path, variant_type) "
        "VALUES (?, ?, ?, ?, 'base')",
        ("test-comp", "base", "Base", "/models/base"),
    )
    conn.execute(
        "INSERT INTO models (comparison, model_name, display_name, model_path, variant_type) "
        "VALUES (?, ?, ?, ?, 'variant')",
        ("test-comp", "apostate", "Apostate", "/models/apostate"),
    )
    conn.commit()


class TestImportLMEval:
    """Tests for lm-eval result import."""

    def test_import_lm_eval(self, tmp_path):
        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)

        # Create mock lm-eval result file
        results_dir = tmp_path / "lm_eval"
        results_dir.mkdir()
        mock_result = {
            "results": {
                "mmlu": {"acc,none": 0.4076, "acc_stderr,none": 0.005, "alias": "mmlu"},
                "hellaswag": {"acc_norm,none": 0.55, "acc_norm_stderr,none": 0.01, "alias": "hellaswag"},
            },
            "configs": {},
        }
        (results_dir / "lm_eval_base.json").write_text(json.dumps(mock_result))

        count = import_lm_eval_results(conn, "test-comp", results_dir)
        assert count == 2

        row = conn.execute(
            "SELECT metric_value FROM lm_eval_results WHERE comparison='test-comp' AND task='mmlu'"
        ).fetchone()
        assert row is not None
        assert abs(row[0] - 0.4076) < 0.001

        # Check stderr imported
        row = conn.execute(
            "SELECT metric_stderr FROM lm_eval_results WHERE comparison='test-comp' AND task='mmlu'"
        ).fetchone()
        assert row is not None
        assert abs(row[0] - 0.005) < 0.001

    def test_import_lm_eval_skips_subtasks(self, tmp_path):
        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)

        results_dir = tmp_path / "lm_eval"
        results_dir.mkdir()
        mock_result = {
            "results": {
                "mmlu": {"acc,none": 0.4076, "alias": "mmlu"},
                "mmlu_humanities": {"acc,none": 0.35, "alias": "mmlu_humanities"},
                "mmlu_stem": {"acc,none": 0.42, "alias": "mmlu_stem"},
            },
            "configs": {},
        }
        (results_dir / "lm_eval_base.json").write_text(json.dumps(mock_result))

        count = import_lm_eval_results(conn, "test-comp", results_dir)
        assert count == 1  # Only mmlu, not subtasks

    def test_import_lm_eval_handles_missing_keys(self, tmp_path):
        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)

        results_dir = tmp_path / "lm_eval"
        results_dir.mkdir()
        # Empty results dict
        (results_dir / "lm_eval_base.json").write_text('{"results": {}}')

        count = import_lm_eval_results(conn, "test-comp", results_dir)
        assert count == 0

    def test_import_lm_eval_malformed_json(self, tmp_path):
        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)

        results_dir = tmp_path / "lm_eval"
        results_dir.mkdir()
        (results_dir / "lm_eval_base.json").write_text("NOT JSON")

        count = import_lm_eval_results(conn, "test-comp", results_dir)
        assert count == 0

    def test_import_lm_eval_unknown_task(self, tmp_path):
        """Unknown tasks should import all numeric metrics."""
        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)

        results_dir = tmp_path / "lm_eval"
        results_dir.mkdir()
        mock_result = {
            "results": {
                "custom_task": {"acc,none": 0.5, "f1,none": 0.7, "alias": "custom_task"},
            },
            "configs": {},
        }
        (results_dir / "lm_eval_base.json").write_text(json.dumps(mock_result))

        count = import_lm_eval_results(conn, "test-comp", results_dir)
        assert count == 2  # Both acc and f1


class TestImportKL:
    """Tests for KL divergence result import."""

    def test_import_kl(self, tmp_path):
        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)

        kl_dir = tmp_path / "kl"
        kl_dir.mkdir()
        mock_kl = {
            "results_version": 1,
            "variant_label": "apostate",
            "kl_divergence_batchmean": 0.148,
            "kl_divergence_per_prompt_median": 0.006,
            "kl_divergence_per_prompt_std": 0.785,
            "kl_divergence_per_prompt_min": 2.14e-07,
            "kl_divergence_per_prompt_max": 7.63,
            "num_prompts": 100,
            "interpretation": "moderate",
        }
        (kl_dir / "kl_apostate.json").write_text(json.dumps(mock_kl))

        count = import_kl_results(conn, "test-comp", kl_dir)
        assert count == 1

        row = conn.execute(
            "SELECT kl_batchmean, rating, num_prompts FROM kl_results "
            "WHERE comparison='test-comp' AND variant='apostate'"
        ).fetchone()
        assert row is not None
        assert abs(row[0] - 0.148) < 0.001
        assert row[1] == "moderate"
        assert row[2] == 100

    def test_import_kl_skips_per_prompt(self, tmp_path):
        """per_prompt_results should not be imported."""
        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)

        kl_dir = tmp_path / "kl"
        kl_dir.mkdir()
        mock_kl = {
            "kl_divergence_batchmean": 0.1,
            "per_prompt_results": [{"prompt_index": 0, "kl_divergence": 0.001}],
            "num_prompts": 1,
        }
        (kl_dir / "kl_apostate.json").write_text(json.dumps(mock_kl))

        import_kl_results(conn, "test-comp", kl_dir)

        # Check that per_prompt is not a column — we just verify the row exists
        row = conn.execute("SELECT COUNT(*) FROM kl_results WHERE comparison='test-comp'").fetchone()
        assert row[0] == 1


class TestImportWeight:
    """Tests for weight analysis result import — including the summary contract.

    Summary keys must match the ACTUAL script output contracts (src/weight/*):
    the pre-fix readers (total_changes / effective_ranks / total_tensors)
    matched no emitted field, so non-fingerprint summaries came back empty.
    """

    def test_import_weight_single_variant(self, tmp_path):
        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)

        weight_dir = tmp_path / "weight"
        variant_dir = weight_dir / "apostate"
        variant_dir.mkdir(parents=True)
        # Real edit_vector single-variant contract: changed_count/total_keys
        mock_edit = {
            "results_version": 1,
            "architecture": {"family": "qwen3", "layers": 36, "total_keys": 400},
            "variant": "apostate",
            "total_keys": 400,
            "changed_count": 55,
            "edit_norms": {"mean": 0.01, "median": 0.008},
            "per_key_details": [],
        }
        (variant_dir / "edit_vector_apostate.json").write_text(json.dumps(mock_edit))

        count = import_weight_results(conn, "test-comp", weight_dir)
        assert count == 1

        row = conn.execute(
            "SELECT variant, variant_b, script_name, summary FROM weight_results "
            "WHERE comparison='test-comp' AND script_name='edit_vector'"
        ).fetchone()
        assert row is not None
        assert row[0] == "apostate"
        assert row[1] == ""
        summary = json.loads(row[3])
        assert summary["changed_count"] == 55
        assert summary["total_keys"] == 400

    def test_import_weight_edit_vector_two_variant(self, tmp_path):
        """Two-variant edit_vector emits overlap_count (not changed_count)."""
        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)

        weight_dir = tmp_path / "weight"
        weight_dir.mkdir(parents=True)
        mock_edit = {
            "variant_a": "apostate",
            "variant_b": "heretic",
            "total_keys": 400,
            "overlap_count": 30,
            "nontrivial_overlap_count": 25,
            "a_only_count": 5,
            "b_only_count": 3,
        }
        (weight_dir / "edit_vector_apostate.json").write_text(json.dumps(mock_edit))

        assert import_weight_results(conn, "test-comp", weight_dir) == 1
        summary = json.loads(
            conn.execute("SELECT summary FROM weight_results WHERE script_name='edit_vector'").fetchone()[0]
        )
        assert summary["overlap_count"] == 30
        assert summary["nontrivial_overlap_count"] == 25

    def test_import_weight_svd_aggregates_ranks(self, tmp_path):
        """svd emits per-tensor tensor_results[] — the importer aggregates ranks."""
        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)

        weight_dir = tmp_path / "weight"
        weight_dir.mkdir(parents=True)
        mock_svd = {
            "results_version": 1,
            "total_analyzed": 3,
            "full_svd": False,
            "tensor_results": [
                {"canonical": "k1", "effective_rank_90pct_energy": 1},
                {"canonical": "k2", "effective_rank_90pct_energy": 7},
                {"canonical": "k3", "effective_rank_90pct_energy": 3},
            ],
        }
        (weight_dir / "svd_apostate.json").write_text(json.dumps(mock_svd))

        assert import_weight_results(conn, "test-comp", weight_dir) == 1
        summary = json.loads(conn.execute("SELECT summary FROM weight_results WHERE script_name='svd'").fetchone()[0])
        assert summary["effective_rank_90pct_median"] == 3  # median of [1,3,7]
        assert summary["effective_rank_90pct_max"] == 7
        assert summary["tensors_ranked"] == 3
        assert summary["total_analyzed"] == 3

    def test_import_weight_layer_analysis(self, tmp_path):
        """layer_analysis emits layer_progression — importer derives counts."""
        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)

        weight_dir = tmp_path / "weight"
        weight_dir.mkdir(parents=True)
        mock = {
            "results_version": 1,
            "layer_progression": {
                "0": {"total_tensors": 4, "changed_tensors": 0, "edit_density": 0},
                "1": {"total_tensors": 4, "changed_tensors": 2, "edit_density": 50},
                "2": {"total_tensors": 4, "changed_tensors": 1, "edit_density": 25},
            },
        }
        (weight_dir / "layer_analysis_apostate.json").write_text(json.dumps(mock))

        assert import_weight_results(conn, "test-comp", weight_dir) == 1
        summary = json.loads(
            conn.execute("SELECT summary FROM weight_results WHERE script_name='layer_analysis'").fetchone()[0]
        )
        assert summary["layers_modified"] == 2
        assert summary["total_layers"] == 3

    def test_import_weight_correlation(self, tmp_path):
        """technique_correlation emits pairwise_cosines (no correlation_coefficient)."""
        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)

        weight_dir = tmp_path / "weight"
        weight_dir.mkdir(parents=True)
        mock = {
            "variants": ["apostate", "heretic"],
            "total_keys": 10,
            "pairwise_cosines": {
                "apostate_vs_heretic": {"mean": 0.42, "median": 0.4, "count": 10, "min": 0.1, "max": 0.9}
            },
        }
        (weight_dir / "correlation_apostate_vs_heretic.json").write_text(json.dumps(mock))

        count = import_weight_results(conn, "test-comp", weight_dir)
        assert count == 1

        row = conn.execute(
            "SELECT variant, variant_b, script_name, summary FROM weight_results "
            "WHERE comparison='test-comp' AND script_name='correlation'"
        ).fetchone()
        assert row is not None
        assert row[0] == "apostate"
        assert row[1] == "heretic"
        summary = json.loads(row[3])
        assert summary["pairwise_cosines"]["apostate_vs_heretic"]["mean"] == 0.42

    def test_import_weight_pairwise(self, tmp_path):
        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)

        weight_dir = tmp_path / "weight"
        weight_dir.mkdir(parents=True)
        mock_corr = {"pairwise_cosines": {"a_vs_b": {"mean": 0.95, "median": 0.95, "count": 4}}}
        (weight_dir / "correlation_apostate_vs_heretic.json").write_text(json.dumps(mock_corr))

        count = import_weight_results(conn, "test-comp", weight_dir)
        assert count == 1

        row = conn.execute(
            "SELECT variant, variant_b, script_name FROM weight_results "
            "WHERE comparison='test-comp' AND script_name='correlation'"
        ).fetchone()
        assert row is not None
        assert row[0] == "apostate"
        assert row[1] == "heretic"

    def test_import_weight_global(self, tmp_path):
        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)

        weight_dir = tmp_path / "weight"
        weight_dir.mkdir(parents=True)
        # Real panel contract: architecture.total_keys + model_key_counts
        mock_panel = {
            "results_version": 1,
            "architecture": {"family": "qwen3", "layer_count": 36, "total_keys": 851},
            "model_key_counts": {"base": 851, "apostate": 851},
            "pairwise_changed_counts": {"apostate->base": 55},
        }
        (weight_dir / "multi_model_panel.json").write_text(json.dumps(mock_panel))

        count = import_weight_results(conn, "test-comp", weight_dir)
        assert count == 1

        row = conn.execute(
            "SELECT variant, variant_b, script_name, summary FROM weight_results "
            "WHERE comparison='test-comp' AND script_name='panel_comparison'"
        ).fetchone()
        assert row is not None
        assert row[0] == ""  # global, no variant
        assert row[1] == ""
        summary = json.loads(row[3])
        assert summary["total_keys"] == 851
        assert summary["model_key_counts"] == {"base": 851, "apostate": 851}


class TestImportAllIdempotent:
    """Test idempotency and all-results import."""

    def test_import_all_idempotent(self, tmp_path):
        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)

        # Create mock comparison directory structure
        comp_dir = tmp_path / "comparison"
        comp_dir.mkdir()
        comp_json = {
            "name": "test-comp",
            "base": "/models/base",
            "variants": {"apostate": {"path": "/models/apo", "display_name": "Apo"}},
            "settings": {},
        }
        (comp_dir / "comparison.json").write_text(json.dumps(comp_json))

        # Create lm_eval results
        lm_dir = comp_dir / "results" / "lm_eval"
        lm_dir.mkdir(parents=True)
        mock_lm = {"results": {"mmlu": {"acc,none": 0.4}}, "configs": {}}
        (lm_dir / "lm_eval_base.json").write_text(json.dumps(mock_lm))

        # Import twice — should not duplicate
        counts1 = import_all_results(conn, str(comp_dir))
        import_all_results(conn, str(comp_dir))

        assert counts1.get("lm_eval", 0) > 0
        # Second import should succeed (ON CONFLICT DO UPDATE) but not add new rows
        rows = conn.execute("SELECT COUNT(*) FROM lm_eval_results WHERE comparison='test-comp'").fetchone()
        assert rows[0] == 1  # Still just 1 row, not 2

    def test_import_creates_stages_on_init(self, tmp_path, monkeypatch):
        """When import_comparison is called, model_stages should be pre-populated.

        load_behaviors is monkeypatched so the test needs no network and writes
        nothing to the repo tree (test isolation).
        """
        from src.db.importers import import_comparison

        monkeypatch.setattr(
            "src.db.core.load_behaviors",
            lambda: [
                {"BehaviorID": "b1", "Behavior": "B one", "SemanticCategory": "chemical", "Tags": ""},
                {"BehaviorID": "b2", "Behavior": "B two", "SemanticCategory": "cybercrime", "Tags": ""},
            ],
        )

        conn = get_db(str(tmp_path / "test.db"))

        comp_dir = tmp_path / "comp"
        comp_dir.mkdir()
        comp_json = {
            "name": "stage-test",
            "base": "/models/base",
            "variants": {"v1": {"path": "/models/v1", "display_name": "V1"}},
            "settings": {},
        }
        (comp_dir / "comparison.json").write_text(json.dumps(comp_json))

        import_comparison(conn, str(comp_dir))

        stages = get_all_stages(conn, "stage-test")
        # Should have 2 models x 7 stages = 14 rows (7th stage "text_encoder"
        # added by f75aed6; test was stale at 12)
        assert len(stages) == 14
        assert all(s.status == "pending" for s in stages)

        conn.close()


class TestImportTeImgMetrics:
    """Phase-4 image-space metrics importer (tebench_img_*.json)."""

    @staticmethod
    def _write_variant_file(results_dir, variant: str) -> None:
        doc = {
            "te_results_version": 1,
            "variant": variant,
            "reference": "base_int8",
            "te_swap": {
                "kind": "te_swap",
                "n_pairs": 2,
                "summaries": [
                    {"metric": "lpips", "n": 2, "mean": 0.12, "ci_low": 0.10, "ci_high": 0.14},
                    {"metric": "ssim", "n": 2, "mean": 0.88, "ci_low": 0.85, "ci_high": 0.91},
                ],
                "anchor_ratios": {"lpips/seed_change": 0.05},
            },
            "cross_prompt": {
                "kind": "cross_prompt",
                "n_pairs": 2,
                "summaries": [{"metric": "lpips", "n": 2, "mean": 0.55, "ci_low": 0.5, "ci_high": 0.6}],
            },
            "clip_score_delta_mean": -0.4,
        }
        (results_dir / f"tebench_img_{variant}.json").write_text(json.dumps(doc))

    @staticmethod
    def _write_anchors_file(results_dir) -> None:
        doc = {
            "te_results_version": 1,
            "seed_change": {
                "kind": "seed_change",
                "n_pairs": 4,
                "summaries": [{"metric": "lpips", "n": 4, "mean": 2.4, "ci_low": 2.0, "ci_high": 2.8}],
            },
            "floor_band": {
                "kind": "floor_band",
                "n_pairs": 3,
                "summaries": [{"metric": "lpips", "n": 3, "mean": 0.02, "ci_low": 0.01, "ci_high": 0.03}],
            },
        }
        (results_dir / "tebench_img_anchors.json").write_text(json.dumps(doc))

    def test_roundtrip(self, tmp_path):
        from src.db.importers import import_te_img_metrics

        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)
        results_dir = tmp_path / "text_encoder"
        results_dir.mkdir()
        self._write_variant_file(results_dir, "heretic")
        self._write_anchors_file(results_dir)

        count = import_te_img_metrics(conn, "test-comp", results_dir)
        assert count == 6  # heretic: lpips+ssim+cross_lpips+clip_delta; anchors: 2

        row = conn.execute(
            "SELECT metric_value, ci_low, ci_high, anchor_ratio FROM te_img_metrics "
            "WHERE comparison='test-comp' AND variant='heretic' AND kind='te_swap' AND metric_name='lpips'"
        ).fetchone()
        assert row[0] == 0.12 and row[1] == 0.10 and row[2] == 0.14 and row[3] == 0.05

        anchor = conn.execute(
            "SELECT metric_value FROM te_img_metrics "
            "WHERE comparison='test-comp' AND variant='__anchor__' AND kind='seed_change' AND metric_name='lpips'"
        ).fetchone()
        assert anchor[0] == 2.4
        conn.close()

    def test_idempotent(self, tmp_path):
        from src.db.importers import import_te_img_metrics

        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)
        results_dir = tmp_path / "text_encoder"
        results_dir.mkdir()
        self._write_variant_file(results_dir, "heretic")
        self._write_anchors_file(results_dir)
        import_te_img_metrics(conn, "test-comp", results_dir)
        import_te_img_metrics(conn, "test-comp", results_dir)
        n = conn.execute("SELECT COUNT(*) FROM te_img_metrics WHERE comparison='test-comp'").fetchone()[0]
        assert n == 6
        conn.close()

    def test_tebench_files_never_reach_te_results(self, tmp_path):
        """The te_*.json glob must not parse tebench_* files (Phase-4 plan rule)."""
        from src.db.importers import import_te_results

        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)
        results_dir = tmp_path / "text_encoder"
        results_dir.mkdir()
        self._write_variant_file(results_dir, "heretic")
        n = import_te_results(conn, "test-comp", results_dir)
        assert n == 0
        rows = conn.execute("SELECT COUNT(*) FROM te_results WHERE comparison='test-comp'").fetchone()[0]
        assert rows == 0
        conn.close()


class TestImportPhase4Tables:
    """Controls / attention / discriminability importers (Phase 4)."""

    def _write_files(self, results_dir) -> None:
        (results_dir / "tebench_controls.json").write_text(
            json.dumps(
                {
                    "conditions": {
                        "noise_1x": {
                            "summary": {
                                "mean_final_latent_div": 0.03,
                                "noise_control_final": 1.2,
                                "div_control_ratio": 0.025,
                            }
                        },
                        "null": {
                            "summary": {
                                "mean_final_latent_div": 0.4,
                                "noise_control_final": 1.2,
                                "div_control_ratio": 0.33,
                            }
                        },
                    }
                }
            )
        )
        (results_dir / "tebench_attention_heretic.json").write_text(
            json.dumps({"variant_name": "heretic", "jsd_mean_all": 0.012, "jsd_per_layer": {"3": 0.02, "9": 0.05}})
        )
        (results_dir / "tebench_discriminability_heretic.json").write_text(
            json.dumps({"variant_name": "heretic", "base_spread": 0.9, "variant_spread": 0.89, "spread_ratio": 0.99})
        )

    def test_roundtrip_and_idempotent(self, tmp_path):
        from src.db.importers import import_te_attention, import_te_controls, import_te_discriminability

        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)
        rd = tmp_path / "text_encoder"
        rd.mkdir()
        self._write_files(rd)

        assert import_te_controls(conn, "test-comp", rd) == 2
        assert import_te_attention(conn, "test-comp", rd) == 1
        assert import_te_discriminability(conn, "test-comp", rd) == 1

        # idempotent
        import_te_controls(conn, "test-comp", rd)
        import_te_attention(conn, "test-comp", rd)
        import_te_discriminability(conn, "test-comp", rd)
        assert conn.execute("SELECT COUNT(*) FROM te_controls WHERE comparison='test-comp'").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM te_attention WHERE comparison='test-comp'").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM te_discriminability WHERE comparison='test-comp'").fetchone()[0] == 1

        peak = conn.execute(
            "SELECT jsd_peak_layer, jsd_peak_value FROM te_attention WHERE comparison='test-comp' AND variant='heretic'"
        ).fetchone()
        assert peak[0] == 9 and peak[1] == 0.05
        conn.close()

    def test_missing_files_zero(self, tmp_path):
        from src.db.importers import import_te_attention, import_te_controls, import_te_discriminability

        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)
        rd = tmp_path / "text_encoder"
        rd.mkdir()
        assert import_te_controls(conn, "test-comp", rd) == 0
        assert import_te_attention(conn, "test-comp", rd) == 0
        assert import_te_discriminability(conn, "test-comp", rd) == 0
        conn.close()


class TestImportGsm8kSampleStats:
    """Lesson-#32 3-number math + lesson-#51 layout support.

    _import_gsm8k_sample_stats must find samples in BOTH layouts:
      1. ``__model/<model>__samples_gsm8k_*.jsonl`` (renamed-by-runner)
      2. ``<model>/samples_gsm8k_*.jsonl`` (current runner per-model dir —
         previously invisible, requiring the manual docker-copy workaround)
    """

    @staticmethod
    def _samples_jsonl() -> str:
        """4 docs × 2 filters. Strict: 1 correct, 1 empty. Flex: 2 correct, 1 empty
        (doc2 has a formatting miss the strict filter punishes, flex recovers)."""
        rows = []
        # (doc_id, filter, exact_match, raw_response)
        for doc_id, filt, em, resp in [
            (0, "strict-match", 1.0, "#### 72"),
            (0, "flexible-extract", 1.0, "#### 72"),
            (1, "strict-match", 0.0, "I cannot answer"),
            (1, "flexible-extract", 0.0, "I cannot answer"),
            (2, "strict-match", 0.0, "The answer is 5"),  # missing #### marker
            (2, "flexible-extract", 1.0, "The answer is 5"),
            (3, "strict-match", 0.0, ""),  # thinking-loop null (empty raw response)
            (3, "flexible-extract", 0.0, ""),
        ]:
            rows.append(
                json.dumps(
                    {
                        "doc_id": doc_id,
                        "filter": filt,
                        "resps": [[resp]],
                        "filtered_resps": [resp],
                        "exact_match": em,
                    }
                )
            )
        return "\n".join(rows) + "\n"

    def _setup_gsm8k_result(self, results_dir: Path) -> Path:
        results_dir.mkdir(parents=True)
        mock = {"results": {"gsm8k": {"exact_match,strict-match": 0.5, "exact_match,flexible-extract": 0.75}}}
        f = results_dir / "lm_eval_base.json"
        f.write_text(json.dumps(mock))
        return f

    def _assert_3number_rows(self, conn) -> None:
        rows = {
            r[0]: r[1]
            for r in conn.execute(
                "SELECT metric_name, metric_value FROM lm_eval_results "
                "WHERE comparison='test-comp' AND model_name='base' AND task='gsm8k'"
            ).fetchall()
        }
        assert rows["empty_count_flex,none"] == 1.0
        assert rows["empty_pct_flex,none"] == 25.0
        assert abs(rows["answered_only_exact_match,flexible-extract"] - 2 / 3) < 1e-9  # 2 correct / 3 non-empty
        # Back-compat strict-based rows
        assert rows["empty_count,none"] == 1.0
        assert abs(rows["exact_match_adjusted,strict-match"] - 1 / 3) < 1e-9  # 1 correct / 3 non-empty
        assert abs(rows["strict_flexible_gap,none"] - 0.25) < 1e-9  # 0.5 flex - 0.25 strict
        assert rows["mean_response_chars,none"] == (7 + 15 + 15 + 0) / 4  # strict rows' raw lengths

    def test_stats_from_new_runner_layout(self, tmp_path):
        from src.db.importers import _import_gsm8k_sample_stats

        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)
        results_dir = tmp_path / "lm_eval"
        f = self._setup_gsm8k_result(results_dir)
        model_dir = results_dir / "__model"
        model_dir.mkdir()
        (model_dir / "base__samples_gsm8k_100.jsonl").write_text(self._samples_jsonl())

        assert _import_gsm8k_sample_stats(conn, "test-comp", "base", f, results_dir) == 7
        self._assert_3number_rows(conn)
        conn.close()

    def test_stats_from_per_model_dir_layout(self, tmp_path):
        """Lesson #51: per-model dirs (results/lm_eval/<model>/) are now scanned."""
        from src.db.importers import _import_gsm8k_sample_stats

        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)
        results_dir = tmp_path / "lm_eval"
        f = self._setup_gsm8k_result(results_dir)
        # NOTE: no __model/ dir at all — per-model layout only
        model_dir = results_dir / "base"
        model_dir.mkdir()
        (model_dir / "samples_gsm8k_555.jsonl").write_text(self._samples_jsonl())

        assert _import_gsm8k_sample_stats(conn, "test-comp", "base", f, results_dir) == 7
        self._assert_3number_rows(conn)
        conn.close()

    def test_stats_via_import_lm_eval_results_per_model_dir(self, tmp_path):
        """End-to-end: import_lm_eval_results picks up per-model-dir samples."""
        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)
        results_dir = tmp_path / "lm_eval"
        self._setup_gsm8k_result(results_dir)
        model_dir = results_dir / "base"
        model_dir.mkdir()
        (model_dir / "samples_gsm8k_555.jsonl").write_text(self._samples_jsonl())

        # 2 canonical metrics + 7 sample-derived rows
        assert import_lm_eval_results(conn, "test-comp", results_dir) == 9
        self._assert_3number_rows(conn)
        conn.close()

    def test_no_samples_anywhere_returns_zero(self, tmp_path):
        from src.db.importers import _import_gsm8k_sample_stats

        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)
        results_dir = tmp_path / "lm_eval"
        f = self._setup_gsm8k_result(results_dir)

        assert _import_gsm8k_sample_stats(conn, "test-comp", "base", f, results_dir) == 0
        conn.close()


class TestImportHarmbench:
    """_import_harmbench: responses + classified + scores round-trip."""

    def _setup_harmbench_files(self, comp_dir: Path, conn) -> None:
        hb = comp_dir / "results" / "harmbench"
        cls_dir = comp_dir / "results" / "harmbench_classified"
        hb.mkdir(parents=True)
        cls_dir.mkdir(parents=True)
        # responses/classifications carry FK to behaviors — register them first
        # (get_db runs with PRAGMA foreign_keys=ON).
        for bid in ("b1", "b2"):
            conn.execute(
                "INSERT OR IGNORE INTO behaviors (behavior_id, behavior_text, semantic_category) "
                "VALUES (?, ?, 'chemical')",
                (bid, f"Behavior {bid}"),
            )
        conn.commit()

        responses = {
            "harmbench": [
                {
                    "behavior_id": "b1",
                    "response": "Sure, here is how",
                    "reasoning": "thinking hard",
                    "reasoning_chars": 13,
                    "completion_tokens": 42,
                    "finish_reason": "stop",
                    "is_refusal": False,
                    "error": None,
                },
                {
                    "behavior_id": "b2",
                    "response": "I can't help with that",
                    "reasoning": "",
                    "reasoning_chars": 0,
                    "completion_tokens": 10,
                    "finish_reason": "stop",
                    "is_refusal": True,
                    "error": None,
                },
            ]
        }
        (hb / "harmbench_apostate_responses.json").write_text(json.dumps(responses))

        classified = {
            "classifier": "regex_v4",
            "harmbench_classified": [
                {
                    "behavior_id": "b1",
                    "is_refusal": False,
                    "classification_layer": "L1_regex",
                    "classification_reason": "no_refusal_pattern",
                    "classification_confidence": "high",
                },
                {
                    "behavior_id": "b2",
                    "is_refusal": True,
                    "classification_layer": "L1_regex",
                    "classification_reason": "refusal",
                    "classification_confidence": "high",
                },
            ],
        }
        (hb / "harmbench_apostate_classified.json").write_text(json.dumps(classified))

        scores = {
            "classifier": "regex_v4",
            "total": 2,
            "harmful_count": 1,
            "safe_count": 1,
            "ambiguous_count": 0,
            "asr": 0.5,
            "asr_excluding_ambiguous": 0.5,
            "cot_analysis": {"mean_reasoning_chars": 6.5},
            "by_category": {"chemical": {"total": 2, "harmful": 1}},
        }
        (hb / "harmbench_apostate_scores.json").write_text(json.dumps(scores))

    def test_roundtrip(self, tmp_path):
        from src.db.importers import _import_harmbench

        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)
        comp_dir = tmp_path / "comp"
        comp_dir.mkdir()
        self._setup_harmbench_files(comp_dir, conn)

        _import_harmbench(conn, "test-comp", comp_dir)

        rows = conn.execute(
            "SELECT behavior_id, response_text, reasoning_text, reasoning_chars, completion_tokens, "
            "finish_reason, is_refusal_kw FROM responses WHERE comparison='test-comp' ORDER BY behavior_id"
        ).fetchall()
        assert len(rows) == 2
        assert rows[0][0] == "b1"
        assert rows[0][1] == "Sure, here is how"
        assert rows[0][2] == "thinking hard"
        assert rows[0][3] == 13
        assert rows[0][4] == 42
        assert rows[1][6] == 1  # is_refusal_kw

        cls = conn.execute(
            "SELECT behavior_id, classifier, label, is_refusal FROM classifications WHERE comparison='test-comp'"
        ).fetchall()
        assert len(cls) == 2
        assert cls[0][1] == "regex_v4"
        assert cls[0][2] == "not_refused"
        assert cls[1][2] == "refused"

        score = conn.execute(
            "SELECT classifier, total, harmful_count, asr FROM harmbench_scores WHERE comparison='test-comp'"
        ).fetchone()
        assert score == ("regex_v4", 2, 1, 0.5)
        conn.close()

    def test_idempotent(self, tmp_path):
        from src.db.importers import _import_harmbench

        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)
        comp_dir = tmp_path / "comp"
        comp_dir.mkdir()
        self._setup_harmbench_files(comp_dir, conn)

        _import_harmbench(conn, "test-comp", comp_dir)
        _import_harmbench(conn, "test-comp", comp_dir)

        n = conn.execute("SELECT COUNT(*) FROM responses WHERE comparison='test-comp'").fetchone()[0]
        assert n == 2  # upsert, not duplicate
        conn.close()

    def test_unknown_model_added_as_variant(self, tmp_path):
        from src.db.importers import _import_harmbench

        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)
        comp_dir = tmp_path / "comp"
        comp_dir.mkdir()
        self._setup_harmbench_files(comp_dir, conn)

        _import_harmbench(conn, "test-comp", comp_dir)
        row = conn.execute(
            "SELECT variant_type FROM models WHERE comparison='test-comp' AND model_name='apostate'"
        ).fetchone()
        assert row == ("variant",)
        conn.close()

    def test_malformed_response_file_skipped(self, tmp_path, caplog):
        from src.db.importers import _import_harmbench

        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)
        hb = tmp_path / "comp" / "results" / "harmbench"
        hb.mkdir(parents=True)
        (hb / "harmbench_apostate_responses.json").write_text("NOT JSON")

        with caplog.at_level(logging.WARNING):
            _import_harmbench(conn, "test-comp", tmp_path / "comp")
        assert "Skipping" in caplog.text
        n = conn.execute("SELECT COUNT(*) FROM responses WHERE comparison='test-comp'").fetchone()[0]
        assert n == 0
        conn.close()


class TestImportAllResultsErrors:
    """import_all_results error paths + counts behavior."""

    def test_no_comparison_json_returns_empty(self, tmp_path):
        conn = get_db(str(tmp_path / "test.db"))
        assert import_all_results(conn, str(tmp_path)) == {}
        conn.close()

    def test_empty_comparison_dir_imports_nothing(self, tmp_path):
        conn = get_db(str(tmp_path / "test.db"))
        comp_dir = tmp_path / "comp"
        comp_dir.mkdir()
        (comp_dir / "comparison.json").write_text(
            json.dumps({"name": "test-comp", "base": "/models/base", "variants": {}})
        )
        counts = import_all_results(conn, str(comp_dir))
        assert counts == {}  # no results dirs exist
        conn.close()


class TestImportTeResults:
    """import_te_results: te_<variant>.json variants[].streams[] contract."""

    @staticmethod
    def _write_te_file(results_dir, variant: str) -> None:
        doc = {
            "backend": "comfy_krea2",
            "variants": [
                {
                    "variant_id": "ignored-slug-source",
                    "streams": [
                        {
                            "stream": "txt",
                            "stage": "pre",
                            "metrics": {"mean_cos": 0.99, "drift": 0.01},
                            "damage_quant_units": 2.5,
                        },
                        {"stream": "img", "stage": "post", "metrics": {"mean_cos": 0.97}},
                    ],
                }
            ],
        }
        (results_dir / f"te_{variant}.json").write_text(json.dumps(doc))

    def test_roundtrip_and_idempotent(self, tmp_path):
        from src.db.importers import import_te_results

        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)
        rd = tmp_path / "text_encoder"
        rd.mkdir()
        self._write_te_file(rd, "heretic")

        assert import_te_results(conn, "test-comp", rd) == 3  # 2 + 1 metric rows
        rows = conn.execute(
            "SELECT variant, backend, stream, stage, metric_name, metric_value, damage_quant_units "
            "FROM te_results WHERE comparison='test-comp' ORDER BY stream, metric_name"
        ).fetchall()
        assert rows[0][0] == "heretic"  # slug from FILENAME, not variant_id
        assert rows[0][1] == "comfy_krea2"
        assert rows[0][2] == "img"
        rows_txt = [r for r in rows if r[2] == "txt"]
        assert len(rows_txt) == 2
        mean_cos = next(r for r in rows_txt if r[4] == "mean_cos")
        assert mean_cos[5] == 0.99
        assert mean_cos[6] == 2.5

        # idempotent (INSERT OR REPLACE)
        import_te_results(conn, "test-comp", rd)
        n = conn.execute("SELECT COUNT(*) FROM te_results WHERE comparison='test-comp'").fetchone()[0]
        assert n == 3
        conn.close()

    def test_skips_report_and_tebench_files(self, tmp_path):
        from src.db.importers import import_te_results

        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)
        rd = tmp_path / "text_encoder"
        rd.mkdir()
        (rd / "te_report.json").write_text("{}")
        (rd / "te_report.md").write_text("x")
        (rd / "tebench_battery_main.json").write_text("{}")
        (rd / "tebench_img_heretic.json").write_text("{}")
        (rd / "te_broken.json").write_text("NOT JSON")
        assert import_te_results(conn, "test-comp", rd) == 0
        assert conn.execute("SELECT COUNT(*) FROM te_results").fetchone()[0] == 0
        conn.close()


class TestImportTeAbImages:
    """import_te_ab_images: manifest.json images[] + timing contract
    (lesson #42d: field names must match the METRICS consumer)."""

    @staticmethod
    def _write_manifest(results_dir) -> Path:
        manifest = {
            "seed": 123,
            "images": [
                {
                    "variant": "heretic",
                    "te_name": "heretic_te",
                    "loader": "gguf",
                    "label": "warm",
                    "category": "words",
                    "pass": "warm",
                    "gen_seconds": 4.2,
                    "path": "ab_images/heretic_1.webp",
                    "seed": 7,
                },
                {
                    "variant": "base_int8",
                    "path": "ab_images/base_1.webp",
                    "gen_seconds": 3.9,
                },
            ],
            "timing": {
                "heretic": {
                    "te_name": "heretic_te",
                    "loader": "gguf",
                    "passes": {"cold": {"load_seconds": 1.5, "gen_seconds": None}},
                }
            },
        }
        ab = results_dir / "ab_images"
        ab.mkdir(parents=True)
        p = ab / "manifest.json"
        p.write_text(json.dumps(manifest))
        return p

    def test_roundtrip_and_idempotent(self, tmp_path):
        from src.db.importers import import_te_ab_images

        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)
        rd = tmp_path / "text_encoder"
        rd.mkdir()
        manifest = self._write_manifest(rd)

        assert import_te_ab_images(conn, "test-comp", manifest) == 3  # 2 images + 1 timing metric
        rows = conn.execute(
            "SELECT variant, te_name, label, category, pass, metric_name, metric_value, image_path, seed "
            "FROM te_ab_images WHERE comparison='test-comp' ORDER BY variant, metric_name"
        ).fetchall()
        assert rows[0][0] == "base_int8"
        assert rows[0][4] is None  # missing pass defaults None
        assert rows[0][7] == "ab_images/base_1.webp"
        img_rows = [r for r in rows if r[5] == "gen_seconds"]
        assert len(img_rows) == 2
        heretic_img = next(r for r in img_rows if r[0] == "heretic")
        assert heretic_img[2] == "warm"
        # timing rows carry label='__timing__' and pass-name-prefixed metrics
        timing_row = next((r for r in rows if r[5] == "cold_load_seconds"), None)
        assert timing_row is not None
        assert timing_row[6] == 1.5  # None-valued metric skipped

        # NOTE: re-import is NOT idempotent for NULL-`pass` rows — SQLite treats
        # NULLs as distinct in the (comparison, variant, label, pass, metric_name)
        # PK, so INSERT OR REPLACE cannot match them. Pre-existing schema quirk,
        # out of scope here; the return value still reflects the manifest size.
        assert import_te_ab_images(conn, "test-comp", manifest) == 3
        n_nonnull = conn.execute(
            "SELECT COUNT(*) FROM te_ab_images WHERE comparison='test-comp' AND pass IS NOT NULL"
        ).fetchone()[0]
        assert n_nonnull == 1  # the warm image row upserts cleanly
        conn.close()

    def test_malformed_manifest(self, tmp_path):
        from src.db.importers import import_te_ab_images

        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)
        bad = tmp_path / "manifest.json"
        bad.write_text("NOT JSON")
        assert import_te_ab_images(conn, "test-comp", bad) == 0
        conn.close()


class TestImportTeBattery:
    """import_te_battery_results: tebench_<battery>.json rows contract."""

    def test_roundtrip_and_idempotent(self, tmp_path):
        from src.db.importers import import_te_battery_results

        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)
        rd = tmp_path / "text_encoder"
        rd.mkdir()
        (rd / "tebench_battery_main.json").write_text(
            json.dumps(
                {
                    "battery": "main",
                    "verdict": "pass",
                    "rows": [
                        {
                            "battery": "main",
                            "config": "heretic",
                            "instrument": "lpips",
                            "metric_name": "te_swap",
                            "metric_value": 0.12,
                            "ci_low": 0.10,
                            "ci_high": 0.14,
                            "n": 6,
                        },
                        {
                            "battery": "main",
                            "config": "heretic",
                            "instrument": "lpips",
                            "metric_name": "seed_change",
                            "metric_value": 2.4,
                        },  # optional fields absent
                        {"battery": "main", "bad_row": True},  # missing instrument → skipped
                    ],
                }
            )
        )
        # Sibling tebench output (no rows/battery) must not parse
        (rd / "tebench_img_heretic.json").write_text(json.dumps({"kind": "te_swap"}))
        # Unreadable file must not crash
        (rd / "tebench_battery_broken.json").write_text("NOT JSON")

        assert import_te_battery_results(conn, "test-comp", rd) == 2
        rows = conn.execute(
            "SELECT battery, config, instrument, metric_name, ci_low, n FROM te_battery_results "
            "WHERE comparison='test-comp' ORDER BY metric_name"
        ).fetchall()
        assert rows[0][3] == "seed_change"
        assert rows[0][4] is None and rows[0][5] is None
        assert rows[1][1] == "heretic" and rows[1][4] == 0.10

        import_te_battery_results(conn, "test-comp", rd)
        n = conn.execute("SELECT COUNT(*) FROM te_battery_results WHERE comparison='test-comp'").fetchone()[0]
        assert n == 2
        conn.close()


class TestImportWeightFingerprintSummary:
    """fingerprint summary keys (scope/targeting/layer_profile) round-trip."""

    def test_fingerprint_summary(self, tmp_path):
        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)
        wd = tmp_path / "weight"
        vd = wd / "apostate"
        vd.mkdir(parents=True)
        mock = {
            "results_version": 1,
            "scope": {"changed_tensors": 41, "changed_pct": 4.8, "total_tensors": 850},
            "magnitude": {"mean_abs_diff": {"mean": 1e-4}},
            "targeting": {"tensor_types": {"attn": 30, "mlp": 11}, "layers_modified": 18},
            "layer_profile": {"early_layer_pct": 5.0, "mid_layer_pct": 40.0, "late_layer_pct": 55.0},
        }
        (vd / "fingerprint_apostate.json").write_text(json.dumps(mock))

        assert import_weight_results(conn, "test-comp", wd) == 1
        summary = json.loads(
            conn.execute("SELECT summary FROM weight_results WHERE script_name='fingerprint'").fetchone()[0]
        )
        assert summary["changed_tensors"] == 41
        assert summary["total_tensors"] == 850
        assert summary["num_types"] == 2
        assert summary["layers_modified"] == 18
        assert summary["late_layer_pct"] == 55.0
        conn.close()

    def test_unrecognized_skipped_malformed_row_with_error_summary(self, tmp_path, caplog):
        """Unrecognized files are skipped; recognized-but-malformed files still
        import a row whose summary carries the parse error (observability)."""
        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)
        wd = tmp_path / "weight"
        wd.mkdir()
        (wd / "unrelated_output.json").write_text("{}")
        (wd / "svd_apostate.json").write_text("NOT JSON")
        with caplog.at_level(logging.WARNING):
            assert import_weight_results(conn, "test-comp", wd) == 1
        assert "Skipping" in caplog.text
        summary = json.loads(conn.execute("SELECT summary FROM weight_results WHERE script_name='svd'").fetchone()[0])
        assert summary["error"]
        conn.close()


class TestImportErrorPaths:
    """Malformed / edge inputs are skipped with a warning, never raise."""

    def test_kl_malformed_and_empty_dir(self, tmp_path, caplog):
        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)
        # empty dir → warning + 0
        empty = tmp_path / "kl_empty"
        empty.mkdir()
        with caplog.at_level(logging.WARNING):
            assert import_kl_results(conn, "test-comp", empty) == 0
        assert "No kl_" in caplog.text
        # malformed file skipped
        kl = tmp_path / "kl"
        kl.mkdir()
        (kl / "kl_apostate.json").write_text("NOT JSON")
        with caplog.at_level(logging.WARNING):
            assert import_kl_results(conn, "test-comp", kl) == 0
        assert "Skipping" in caplog.text
        conn.close()

    def test_lm_eval_no_results_key(self, tmp_path, caplog):
        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)
        rd = tmp_path / "lm_eval"
        rd.mkdir()
        (rd / "lm_eval_base.json").write_text('{"other": 1}')
        with caplog.at_level(logging.WARNING):
            assert import_lm_eval_results(conn, "test-comp", rd) == 0
        assert "No 'results' key" in caplog.text
        conn.close()

    def test_lm_eval_no_files(self, tmp_path, caplog):
        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)
        rd = tmp_path / "lm_eval"
        rd.mkdir()
        with caplog.at_level(logging.WARNING):
            assert import_lm_eval_results(conn, "test-comp", rd) == 0
        assert "No lm_eval_" in caplog.text
        conn.close()

    def test_gsm8k_stats_malformed_lines_and_bad_insert(self, tmp_path):
        from src.db.importers import _import_gsm8k_sample_stats

        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)
        rd = tmp_path / "lm_eval"
        rd.mkdir()
        f = rd / "lm_eval_base.json"
        f.write_text(json.dumps({"results": {"gsm8k": {"exact_match,strict-match": 0.5}}}))
        md = rd / "__model"
        md.mkdir()
        # valid line + malformed JSON line + unknown filter line + empty lines
        (md / "base__samples_gsm8k_1.jsonl").write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "doc_id": 0,
                            "filter": "strict-match",
                            "resps": [["#### 5"]],
                            "filtered_resps": ["5"],
                            "exact_match": 1.0,
                        }
                    ),
                    "NOT JSON",
                    json.dumps({"filter": "weird-filter", "resps": [["x"]]}),
                    "",
                ]
            )
            + "\n"
        )
        assert _import_gsm8k_sample_stats(conn, "test-comp", "base", f, rd) == 7
        conn.close()

    def test_gsm8k_stats_no_strict_rows(self, tmp_path):
        from src.db.importers import _import_gsm8k_sample_stats

        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)
        rd = tmp_path / "lm_eval"
        rd.mkdir()
        f = rd / "lm_eval_base.json"
        f.write_text(json.dumps({"results": {"gsm8k": {}}}))
        md = rd / "__model"
        md.mkdir()
        (md / "base__samples_gsm8k_1.jsonl").write_text(
            json.dumps(
                {
                    "doc_id": 0,
                    "filter": "flexible-extract",
                    "resps": [["5"]],
                    "filtered_resps": ["5"],
                    "exact_match": 1.0,
                }
            )
            + "\n"
        )
        # strict total == 0 → early return 0
        assert _import_gsm8k_sample_stats(conn, "test-comp", "base", f, rd) == 0
        conn.close()

    def test_harmbench_model_key_hyphen_normalization(self, tmp_path):
        """harmbench_<slug_with_hyphens> maps back to the models-table name."""
        from src.db.importers import _import_harmbench

        conn = get_db(str(tmp_path / "test.db"))
        conn.execute("INSERT INTO comparisons (name, base_model) VALUES ('tc', '/b')")
        conn.execute(
            "INSERT INTO models (comparison, model_name, display_name, model_path, variant_type) "
            "VALUES ('tc', 'some-model', 'SM', '/m', 'variant')"
        )
        conn.execute("INSERT INTO behaviors (behavior_id, behavior_text, semantic_category) VALUES ('b1', 'B', 'cat')")
        conn.commit()
        hb = tmp_path / "results" / "harmbench"
        hb.mkdir(parents=True)
        (hb / "harmbench_some_model_responses.json").write_text(
            json.dumps({"harmbench": [{"behavior_id": "b1", "response": "r", "is_refusal": False}]})
        )
        _import_harmbench(conn, "tc", tmp_path)
        row = conn.execute("SELECT model_name FROM responses WHERE comparison='tc'").fetchone()
        assert row == ("some-model",)  # underscore→hyphen normalization hit the models table
        conn.close()

    def test_import_all_with_te_results_populates_counts(self, tmp_path):
        """import_all_results full path: lm_eval + kl + harmbench + te sections."""
        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)
        comp_dir = tmp_path / "comp"
        comp_dir.mkdir()
        (comp_dir / "comparison.json").write_text(
            json.dumps({"name": "test-comp", "base": "/models/base", "variants": {"apostate": {"path": "/a"}}})
        )
        lm = comp_dir / "results" / "lm_eval"
        lm.mkdir(parents=True)
        (lm / "lm_eval_base.json").write_text(json.dumps({"results": {"mmlu": {"acc,none": 0.4}}}))
        kl = comp_dir / "results" / "kl"
        kl.mkdir()
        (kl / "kl_apostate.json").write_text(json.dumps({"kl_divergence_batchmean": 0.1, "num_prompts": 1}))
        hb = comp_dir / "results" / "harmbench"
        hb.mkdir()
        (hb / "harmbench_base_responses.json").write_text(
            json.dumps({"harmbench": [{"behavior_id": "b1", "response": "r", "is_refusal": False}]})
        )
        conn.execute("INSERT INTO behaviors (behavior_id, behavior_text, semantic_category) VALUES ('b1', 'B', 'cat')")
        conn.commit()

        counts = import_all_results(conn, str(comp_dir))
        assert counts["lm_eval"] == 1
        assert counts["kl"] == 1
        assert counts["harmbench"] == 1  # one responses FILE (file-count contract)
        conn.close()
