"""Tests for src/db/schema.py — verify all tables, views, indexes exist."""

from __future__ import annotations

import sqlite3

from src.db.core import get_db


def _get_tables(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
    return {r[0] for r in rows}


def _get_views(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='view' ORDER BY name").fetchall()
    return {r[0] for r in rows}


def _get_indexes(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    return {r[0] for r in rows}


class TestSchema:
    """Verify all tables, views, and indexes are created by get_db()."""

    def test_creates_all_tables(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        conn = get_db(db_path)
        tables = _get_tables(conn)
        conn.close()

        expected = {
            "behaviors",
            "comparisons",
            "models",
            "model_stages",
            "responses",
            "classifications",
            "llm_reviews",
            "human_overrides",
            "weight_results",
            "kl_results",
            "lm_eval_results",
        }
        assert expected.issubset(tables), f"Missing tables: {expected - tables}"

    def test_creates_all_views(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        conn = get_db(db_path)
        views = _get_views(conn)
        conn.close()

        expected = {
            "v_latest_classification",
            "v_review_status",
            "v_harmbench_summary",
            "v_harmbench_by_category",
            "v_pipeline_status",
        }
        assert expected.issubset(views), f"Missing views: {expected - views}"

    def test_creates_indexes(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        conn = get_db(db_path)
        indexes = _get_indexes(conn)
        conn.close()

        expected = {
            "idx_model_stages_comp",
            "idx_model_stages_status",
            "idx_lm_eval_comp_model",
            "idx_lm_eval_task",
            "idx_weight_comp",
            "idx_responses_comp",
            "idx_classifications_comp",
        }
        assert expected.issubset(indexes), f"Missing indexes: {expected - indexes}"

    def test_models_composite_pk(self, tmp_path):
        """Verify models table accepts two 'base' models with different comparisons."""
        db_path = str(tmp_path / "test.db")
        conn = get_db(db_path)

        conn.execute(
            "INSERT INTO comparisons (name, base_model) VALUES (?, ?)",
            ("comp-a", "/models/a"),
        )
        conn.execute(
            "INSERT INTO comparisons (name, base_model) VALUES (?, ?)",
            ("comp-b", "/models/b"),
        )
        conn.execute(
            "INSERT INTO models "
            "(comparison, model_name, display_name, model_path, variant_type) "
            "VALUES (?, ?, ?, ?, 'base')",
            ("comp-a", "base", "Base A", "/models/a"),
        )
        conn.execute(
            "INSERT INTO models "
            "(comparison, model_name, display_name, model_path, variant_type) "
            "VALUES (?, ?, ?, ?, 'base')",
            ("comp-b", "base", "Base B", "/models/b"),
        )
        conn.commit()

        count = conn.execute("SELECT COUNT(*) FROM models").fetchone()[0]
        assert count == 2

        # Verify we can distinguish them
        rows = conn.execute(
            "SELECT comparison, display_name FROM models WHERE model_name = 'base' ORDER BY comparison"
        ).fetchall()
        assert rows[0] == ("comp-a", "Base A")
        assert rows[1] == ("comp-b", "Base B")
        conn.close()

    def test_responses_composite_fk(self, tmp_path):
        """Verify responses FK works with (comparison, model_name) composite."""
        db_path = str(tmp_path / "test.db")
        conn = get_db(db_path)

        conn.execute("INSERT INTO comparisons (name, base_model) VALUES ('test', '/m')")
        conn.execute(
            "INSERT INTO models "
            "(comparison, model_name, display_name, model_path, variant_type) "
            "VALUES ('test', 'base', 'Base', '/m', 'base')"
        )
        conn.execute(
            "INSERT INTO behaviors (behavior_id, behavior_text, semantic_category) VALUES ('b1', 'test', 'test')"
        )
        conn.execute(
            """INSERT INTO responses (comparison, model_name, behavior_id, response_text, is_refusal_kw)
            VALUES ('test', 'base', 'b1', 'I cannot help', 1)"""
        )
        conn.commit()

        count = conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
        assert count == 1
        conn.close()

    def test_idempotent_schema(self, tmp_path):
        """Calling get_db() twice should not raise."""
        db_path = str(tmp_path / "test.db")
        conn1 = get_db(db_path)
        conn1.close()
        conn2 = get_db(db_path)
        tables = _get_tables(conn2)
        assert "comparisons" in tables
        conn2.close()
