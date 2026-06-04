"""Tests for src/db/importers.py — result file parsing and import."""

from __future__ import annotations

import json

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
    """Tests for weight analysis result import."""

    def test_import_weight_single_variant(self, tmp_path):
        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)

        weight_dir = tmp_path / "weight"
        variant_dir = weight_dir / "apostate"
        variant_dir.mkdir(parents=True)
        mock_edit = {
            "total_changes": 55,
            "scope": {"changed_tensors": 55, "changed_pct": 2.3},
        }
        (variant_dir / "edit_vector_apostate.json").write_text(json.dumps(mock_edit))

        count = import_weight_results(conn, "test-comp", weight_dir)
        assert count == 1

        row = conn.execute(
            "SELECT variant, variant_b, script_name FROM weight_results "
            "WHERE comparison='test-comp' AND script_name='edit_vector'"
        ).fetchone()
        assert row is not None
        assert row[0] == "apostate"
        assert row[1] == ""

    def test_import_weight_pairwise(self, tmp_path):
        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)

        weight_dir = tmp_path / "weight"
        weight_dir.mkdir(parents=True)
        mock_corr = {"correlation_coefficient": 0.95}
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
        mock_panel = {"pairwise_changed_counts": {"apostate": 55}, "total_tensors": 2400}
        (weight_dir / "multi_model_panel.json").write_text(json.dumps(mock_panel))

        count = import_weight_results(conn, "test-comp", weight_dir)
        assert count == 1

        row = conn.execute(
            "SELECT variant, variant_b, script_name FROM weight_results "
            "WHERE comparison='test-comp' AND script_name='panel_comparison'"
        ).fetchone()
        assert row is not None
        assert row[0] == ""  # global, no variant
        assert row[1] == ""


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

    def test_import_creates_stages_on_init(self, tmp_path):
        """When import_comparison is called, model_stages should be pre-populated."""
        from src.db.importers import import_comparison

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
        # Should have 2 models × 6 stages = 12 rows
        assert len(stages) == 12
        assert all(s.status == "pending" for s in stages)

        conn.close()
