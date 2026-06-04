"""Tests for src/db_tables.py — markdown table generation."""

from __future__ import annotations

from src.db.core import get_db
from src.db_tables import (
    _bold_best,
    generate_all,
    generate_benchmark_delta_table,
    generate_benchmark_table,
    generate_combined_summary_table,
    generate_harmbench_category_table,
    generate_harmbench_table,
    generate_kl_table,
    generate_weight_summary_table,
)


def _setup_comparison(conn) -> None:
    """Create a comparison with base + 2 variants and sample data."""
    conn.execute("INSERT INTO comparisons (name, base_model, num_variants) VALUES ('tc', 'base', 2)")
    conn.execute(
        "INSERT INTO models (comparison, model_name, display_name, model_path, variant_type) "
        "VALUES ('tc', 'base', 'Base', '/m/base', 'base')"
    )
    conn.execute(
        "INSERT INTO models (comparison, model_name, display_name, model_path, variant_type) "
        "VALUES ('tc', 'v1', 'Variant1', '/m/v1', 'variant')"
    )
    conn.execute(
        "INSERT INTO models (comparison, model_name, display_name, model_path, variant_type) "
        "VALUES ('tc', 'v2', 'Variant2', '/m/v2', 'variant')"
    )

    # lm-eval results
    for model, mmlu, hellaswag in [("base", 0.40, 0.55), ("v1", 0.42, 0.56), ("v2", 0.38, 0.54)]:
        conn.execute(
            "INSERT INTO lm_eval_results (comparison, model_name, task, metric_name, metric_value) "
            "VALUES ('tc', ?, 'mmlu', 'acc,none', ?)",
            (model, mmlu),
        )
        conn.execute(
            "INSERT INTO lm_eval_results (comparison, model_name, task, metric_name, metric_value) "
            "VALUES ('tc', ?, 'hellaswag', 'acc_norm,none', ?)",
            (model, hellaswag),
        )

    # KL results
    conn.execute(
        "INSERT INTO kl_results (comparison, variant, kl_batchmean, rating) VALUES ('tc', 'v1', 0.134, 'very good')"
    )
    conn.execute(
        "INSERT INTO kl_results (comparison, variant, kl_batchmean, rating) VALUES ('tc', 'v2', 0.245, 'moderate')"
    )

    conn.commit()


def _setup_harmbench(conn) -> None:
    """Add behaviors, responses, and classifications for ASR view testing."""
    # Behaviors
    for bid, cat in [("b1", "chemical"), ("b2", "cybercrime"), ("b3", "harassment")]:
        conn.execute(
            "INSERT OR IGNORE INTO behaviors (behavior_id, behavior_text, semantic_category) VALUES (?, ?, ?)",
            (bid, f"Behavior {bid}", cat),
        )

    # Responses + classifications for each model
    for model in ["base", "v1", "v2"]:
        for bid in ["b1", "b2", "b3"]:
            is_ref = 1 if model == "base" else 0
            conn.execute(
                "INSERT INTO responses (comparison, model_name, behavior_id, is_refusal_kw) "
                "VALUES ('tc', ?, ?, ?) "
                "ON CONFLICT(comparison, model_name, behavior_id) DO UPDATE SET is_refusal_kw = excluded.is_refusal_kw",
                (model, bid, is_ref),
            )
            conn.execute(
                "INSERT INTO classifications "
                "(comparison, model_name, behavior_id, classifier, label, is_refusal, confidence) "
                "VALUES ('tc', ?, ?, 'test_cls', ?, ?, 'high') "
                "ON CONFLICT(comparison, model_name, behavior_id, classifier) "
                "DO UPDATE SET is_refusal = excluded.is_refusal",
                (model, bid, "refused" if is_ref else "not_refused", is_ref),
            )
    conn.commit()


class TestBoldBest:
    """Tests for the _bold_best helper."""

    def test_higher_bolds_max(self):
        result = _bold_best(["0.40", "0.42", "0.38"], "higher")
        assert result[1] == "**0.42**"

    def test_lower_bolds_min(self):
        result = _bold_best(["5.1", "2.3", "8.7"], "lower")
        assert result[1] == "**2.3**"

    def test_ties_no_bold(self):
        result = _bold_best(["0.42", "0.42", "0.38"], "higher")
        assert "**" not in str(result)

    def test_single_value_no_bold(self):
        result = _bold_best(["0.42"], "higher")
        assert result == ["0.42"]

    def test_empty_list(self):
        assert _bold_best([], "higher") == []


class TestBenchmarkTable:
    """Tests for benchmark table generation."""

    def test_benchmark_table_formatting(self, tmp_path):
        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)
        table = generate_benchmark_table(conn, "tc")
        assert "|" in table
        assert "MMLU" in table
        assert "HellaSwag" in table
        conn.close()

    def test_benchmark_bold_highest_acc(self, tmp_path):
        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)
        table = generate_benchmark_table(conn, "tc")
        # v1 has 0.42 MMLU — should be bolded
        assert "**0.4200**" in table
        conn.close()

    def test_benchmark_perplexity_bold_lowest(self, tmp_path):
        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)
        # Add lambada with lower-is-better
        conn.execute(
            "INSERT INTO lm_eval_results (comparison, model_name, task, metric_name, metric_value) "
            "VALUES ('tc', 'base', 'lambada_openai', 'perplexity,none', 5.0)"
        )
        conn.execute(
            "INSERT INTO lm_eval_results (comparison, model_name, task, metric_name, metric_value) "
            "VALUES ('tc', 'v1', 'lambada_openai', 'perplexity,none', 3.2)"
        )
        conn.execute(
            "INSERT INTO lm_eval_results (comparison, model_name, task, metric_name, metric_value) "
            "VALUES ('tc', 'v2', 'lambada_openai', 'perplexity,none', 7.8)"
        )
        conn.commit()
        table = generate_benchmark_table(conn, "tc")
        assert "**3.2**" in table
        conn.close()

    def test_benchmark_ties_no_bold(self, tmp_path):
        conn = get_db(str(tmp_path / "test.db"))
        conn.execute("INSERT INTO comparisons (name, base_model, num_variants) VALUES ('tc2', 'b', 1)")
        conn.execute(
            "INSERT INTO models (comparison, model_name, display_name, model_path, variant_type) "
            "VALUES ('tc2', 'b', 'Base', '/b', 'base')"
        )
        conn.execute(
            "INSERT INTO models (comparison, model_name, display_name, model_path, variant_type) "
            "VALUES ('tc2', 'v', 'V', '/v', 'variant')"
        )
        for m in ["b", "v"]:
            conn.execute(
                "INSERT INTO lm_eval_results (comparison, model_name, task, metric_name, metric_value) "
                "VALUES ('tc2', ?, 'mmlu', 'acc,none', 0.42)",
                (m,),
            )
        conn.commit()
        table = generate_benchmark_table(conn, "tc2")
        assert "**" not in table
        conn.close()

    def test_benchmark_delta_table(self, tmp_path):
        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)
        table = generate_benchmark_delta_table(conn, "tc")
        assert "+0.0200" in table  # v1 mmlu: 0.42 - 0.40 = +0.02
        assert "-0.0200" in table  # v2 mmlu: 0.38 - 0.40 = -0.02
        conn.close()


class TestHarmbenchTable:
    """Tests for HarmBench ASR table."""

    def test_harmbench_asr_table(self, tmp_path):
        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)
        _setup_harmbench(conn)
        table = generate_harmbench_table(conn, "tc")
        assert "ASR" in table
        assert "base" in table
        conn.close()

    def test_harmbench_asr_base_not_bolded(self, tmp_path):
        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)
        _setup_harmbench(conn)
        table = generate_harmbench_table(conn, "tc")
        # Base should not be bolded even though its ASR is lower (0%)
        # Variants have 100% ASR — should be bolded
        lines = table.split("\n")
        base_line = [line for line in lines if "base" in line and "Model" not in line][0]
        assert "**" not in base_line or "100.0%" in base_line
        conn.close()

    def test_harmbench_category_table(self, tmp_path):
        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)
        _setup_harmbench(conn)
        table = generate_harmbench_category_table(conn, "tc")
        assert "chemical" in table
        conn.close()


class TestKLTable:
    """Tests for KL divergence table."""

    def test_kl_table_with_ratings(self, tmp_path):
        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)
        table = generate_kl_table(conn, "tc")
        assert "very good" in table
        assert "**0.1340**" in table  # Lowest KL bolded
        conn.close()


class TestWeightSummaryTable:
    """Tests for weight modification summary table."""

    def test_weight_summary_with_fingerprint(self, tmp_path):
        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)
        # Insert weight results with fingerprint summaries
        import json

        for variant, pct in [("v1", 2.3), ("v2", 5.1)]:
            summary = json.dumps(
                {
                    "script": "fingerprint",
                    "changed_pct": pct,
                    "changed_tensors": int(pct * 24),
                }
            )
            conn.execute(
                "INSERT INTO weight_results "
                "(comparison, variant, variant_b, script_name, result_path, summary) "
                "VALUES ('tc', ?, '', 'fingerprint', ?, ?)",
                (variant, f"/results/weight/{variant}/fingerprint.json", summary),
            )
        conn.commit()

        table = generate_weight_summary_table(conn, "tc")
        assert "Tensors changed" in table
        # v1 has lower percentage — should be bolded
        assert "**2.30%**" in table
        conn.close()

    def test_weight_summary_no_data(self, tmp_path):
        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)
        table = generate_weight_summary_table(conn, "tc")
        assert "No weight" in table
        conn.close()


class TestSummaryTable:
    """Tests for combined summary table."""

    def test_summary_table(self, tmp_path):
        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)
        _setup_harmbench(conn)
        table = generate_combined_summary_table(conn, "tc")
        assert "HarmBench ASR" in table
        assert "MMLU" in table
        assert "KL Divergence" in table
        conn.close()

    def test_empty_comparison(self, tmp_path):
        conn = get_db(str(tmp_path / "test.db"))
        table = generate_benchmark_table(conn, "nonexistent")
        assert "No models" in table or "no" in table.lower()
        conn.close()

    def test_single_variant(self, tmp_path):
        conn = get_db(str(tmp_path / "test.db"))
        conn.execute("INSERT INTO comparisons (name, base_model, num_variants) VALUES ('sv', 'b', 1)")
        conn.execute(
            "INSERT INTO models (comparison, model_name, display_name, model_path, variant_type) "
            "VALUES ('sv', 'b', 'Base', '/b', 'base')"
        )
        conn.execute(
            "INSERT INTO models (comparison, model_name, display_name, model_path, variant_type) "
            "VALUES ('sv', 'v', 'V', '/v', 'variant')"
        )
        for m in ["b", "v"]:
            conn.execute(
                "INSERT INTO lm_eval_results (comparison, model_name, task, metric_name, metric_value) "
                "VALUES ('sv', ?, 'mmlu', 'acc,none', 0.42)",
                (m,),
            )
        conn.commit()
        table = generate_benchmark_table(conn, "sv")
        assert "MMLU" in table
        # Tied values — should not bold
        assert "**" not in table
        conn.close()

    def test_generate_all(self, tmp_path):
        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)
        _setup_harmbench(conn)
        result = generate_all(conn, "tc")
        assert "## Benchmarks" in result
        assert "## HarmBench" in result
        assert "## KL Divergence" in result
        conn.close()
