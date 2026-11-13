"""Tests for src/db_tables.py — markdown table generation."""

from __future__ import annotations

from src.db.core import get_db
from src.db_tables import (
    _bold_best,
    CARD_KL_CLAIMS,
    generate_all,
    generate_benchmark_delta_table,
    generate_benchmark_table,
    generate_combined_summary_table,
    generate_harmbench_category_table,
    generate_harmbench_judge_table,
    generate_harmbench_table,
    generate_kl_calibration_table,
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
        # Accuracy is rendered as a percentage; v1 has 42% MMLU — highest, bolded
        assert "**42.00%**" in table
        conn.close()

    def test_benchmark_perplexity_omitted_when_dead_task(self, tmp_path):
        """lambada_openai is in DEAD_TASKS and is never rendered in the main table.

        Lower-is-better bolding is covered by ``TestBoldBest::test_lower_bolds_min``
        at the unit level. No live OLL v2 task is lower-is-better, so there is no
        end-to-end path to exercise here.
        """
        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)
        for model, pp in [("base", 5.0), ("v1", 3.2), ("v2", 7.8)]:
            conn.execute(
                "INSERT INTO lm_eval_results (comparison, model_name, task, metric_name, metric_value) "
                "VALUES ('tc', ?, 'lambada_openai', 'perplexity,none', ?)",
                (model, pp),
            )
        conn.commit()
        table = generate_benchmark_table(conn, "tc")
        assert "lambada" not in table.lower()
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
        # Deltas are rendered as percentage-point change from base (see db_tables.py)
        # v1 mmlu: (0.42-0.40)/0.40*100 = +5.00% ; v2 mmlu: -5.00%
        assert "+5.00%" in table  # v1 mmlu delta
        assert "-5.00%" in table  # v2 mmlu delta
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
        # Header column is now "Changed" (count of changed tensors).
        assert "Changed" in table
        # Bolding is on the lowest *changed count* (least modified = best).
        # v1 = 55 changed tensors (int(2.3*24)), v2 = 122 — v1 is bolded.
        assert "**55**" in table
        # Percent renders to one decimal.
        assert "2.3%" in table
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
        # Combined summary header uses short column names: "ASR" and "KL".
        assert "ASR" in table
        assert "MMLU" in table  # appears in "MMLU-Pro Δ"
        assert "KL" in table
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


class TestHarmbenchJudgeTable:
    """Tests for the 4-way rubric judge table (no_refusal/refusal/soft_refusal/degenerate)."""

    def _setup_judge(self, conn) -> None:
        """base + 2 variants; llm_reviews mixing LEGACY and NEW verdict vocabulary.

        Expected under the 4-way rubric:
        - base: b1/b2 refused (legacy), b3 soft_refusal (new) -> 0 no_refusal,
          ASR 0/(3-0) = 0.0%, Soft-Refusal col = 1
        - v1: b1/b2 complied (legacy -> no_refusal), b3 refused
          -> ASR 2/3 = 66.7%
        - v2: b1 legacy refused (judge-a wins over judge-old's older
          'complied'), b2 legacy refused, b3 'degenerate' (new vocab)
          -> ASR 0/(3-1) = 0.0%, Degenerate col = 1
        """
        _setup_comparison(conn)
        _setup_harmbench(conn)
        verdicts = {("base", "b1"): "refused", ("base", "b2"): "refused", ("base", "b3"): "soft_refusal",
                    ("v1", "b1"): "complied", ("v1", "b2"): "complied", ("v1", "b3"): "refused",
                    ("v2", "b1"): "refused", ("v2", "b2"): "refused", ("v2", "b3"): "degenerate"}
        for (model, bid), verdict in verdicts.items():
            conn.execute(
                "INSERT INTO llm_reviews (comparison, model_name, behavior_id, reviewer_model, verdict) "
                "VALUES ('tc', ?, ?, 'judge-a', ?)",
                (model, bid, verdict),
            )
        # v2: an older second reviewer on b1 (superseded by judge-a's later review)
        conn.execute(
            "INSERT INTO llm_reviews (comparison, model_name, behavior_id, reviewer_model, verdict, reviewed_at) "
            "VALUES ('tc', 'v2', 'b1', 'judge-old', 'complied', '2026-01-01T00:00:00Z')"
        )
        conn.commit()

    def test_judge_asr_and_4way_counts(self, tmp_path):
        conn = get_db(str(tmp_path / "test.db"))
        self._setup_judge(conn)
        table = generate_harmbench_judge_table(conn, "tc")
        # v1: 2 no_refusal / 3 judged = 66.7%
        assert "66.7%" in table
        # v2: b3 degenerate excluded from denominator -> 0/2 = 0.0%
        assert "0.0%" in table
        # 4-way columns present + rubric footnote
        assert "Soft-Refusal" in table
        assert "Degenerate" in table
        assert "Rubric:" in table
        conn.close()

    def test_degenerate_verdict_excluded_from_denominator(self, tmp_path):
        conn = get_db(str(tmp_path / "test.db"))
        self._setup_judge(conn)
        # v1 b3 flips from refusal to degenerate verdict -> ASR 2/(3-1) = 100%
        conn.execute(
            "UPDATE llm_reviews SET verdict = 'degenerate' "
            "WHERE comparison = 'tc' AND model_name = 'v1' AND behavior_id = 'b3'"
        )
        conn.commit()
        table = generate_harmbench_judge_table(conn, "tc")
        assert "100.0%" in table
        conn.close()

    def test_base_never_bolded(self, tmp_path):
        conn = get_db(str(tmp_path / "test.db"))
        self._setup_judge(conn)
        table = generate_harmbench_judge_table(conn, "tc")
        base_line = next(line for line in table.splitlines() if line.startswith("| Base |"))
        assert "**" not in base_line
        conn.close()

    def test_all_degenerate_arm_shows_dash(self, tmp_path):
        """An arm whose every verdict is degenerate has no determinate items."""
        conn = get_db(str(tmp_path / "test.db"))
        self._setup_judge(conn)
        conn.execute("UPDATE llm_reviews SET verdict = 'degenerate' WHERE comparison = 'tc' AND model_name = 'v1'")
        conn.commit()
        table = generate_harmbench_judge_table(conn, "tc")
        v1_line = next(line for line in table.splitlines() if "Variant1" in line)
        assert "| - |" in v1_line  # ASR placeholder, denominator = 0
        conn.close()

    def test_arm_without_judge_coverage(self, tmp_path):
        conn = get_db(str(tmp_path / "test.db"))
        self._setup_judge(conn)
        conn.execute("DELETE FROM llm_reviews WHERE comparison = 'tc' AND model_name = 'v1'")
        conn.commit()
        table = generate_harmbench_judge_table(conn, "tc")
        assert "No LLM-judge review coverage for: Variant1" in table
        assert "- | - | - | - | - | -" in table  # placeholder cells for the uncovered arm
        conn.close()


class TestCardKlClaims:
    """CARD_KL_CLAIMS is per-comparison (stale cross-comparison claims = bug #167-class)."""

    def test_per_comparison_structure(self):
        assert isinstance(CARD_KL_CLAIMS, dict)
        for cmp_name, claims in CARD_KL_CLAIMS.items():
            assert isinstance(claims, dict), f"{cmp_name} claims must be a dict"
            for variant, value in claims.items():
                assert isinstance(value, float), f"{cmp_name}/{variant} claim must be float"

    def test_qwen38_claims_do_not_leak_into_gemma4(self):
        q = CARD_KL_CLAIMS.get("qwen38-27b", {})
        # The qwen38 apostate claim (kcrn_report) must not overwrite the gemma one
        assert q.get("apostate") == 0.0032
        gemma = CARD_KL_CLAIMS["gemma4-e4b"]
        assert gemma["apostate"] == 0.119
        # qwen38-only arms are absent from gemma dicts
        assert "ultra_heretic" not in gemma
        assert "trohrbaugh" not in gemma

    def test_calibration_table_scoped_to_comparison(self, tmp_path):
        conn = get_db(str(tmp_path / "test.db"))
        conn.execute("INSERT INTO comparisons (name, base_model, num_variants) VALUES ('tc', 'b', 0)")
        conn.execute(
            "INSERT INTO kl_results (comparison, variant, kl_batchmean) VALUES ('tc', 'apostate', 0.004)"
        )
        conn.commit()
        # Comparison with no claims entry -> no data (does not use gemma claims)
        table = generate_kl_calibration_table(conn, "tc")
        assert "No KL calibration data available" in table
        conn.close()
