"""Tests for src/db/schema.py — verify all tables, views, indexes exist."""

from __future__ import annotations

import sqlite3

import pytest

from src.db.core import get_db
from src.db.schema import SCHEMA


def _register_fixture(conn: sqlite3.Connection) -> None:
    """Minimal comparison/model/behavior/response rows satisfying FKs."""
    conn.execute("INSERT INTO comparisons (name, base_model) VALUES ('c', '/b')")
    conn.execute("INSERT INTO models (comparison, model_name, display_name, model_path) VALUES ('c', 'm', 'm', '/m')")
    conn.execute("INSERT INTO behaviors (behavior_id, behavior_text, semantic_category) VALUES ('b', 't', 'cat')")
    conn.execute("INSERT INTO responses (comparison, model_name, behavior_id) VALUES ('c', 'm', 'b')")
    conn.commit()


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
            "uq_human_overrides_item",
        }
        assert expected.issubset(indexes), f"Missing indexes: {expected - indexes}"

    def test_human_overrides_unique_item(self, tmp_path):
        """One override per (comparison, model_name, behavior_id) — views LEFT
        JOIN human_overrides 1:1, a duplicate row would multiply join rows and
        inflate total/complied/asr_pct (lesson #76 arithmetic-corruption class)."""
        conn = get_db(str(tmp_path / "test.db"))
        _register_fixture(conn)
        conn.execute(
            "INSERT INTO human_overrides (comparison, model_name, behavior_id, verdict) "
            "VALUES ('c', 'm', 'b', 'refusal')"
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO human_overrides (comparison, model_name, behavior_id, verdict) "
                "VALUES ('c', 'm', 'b', 'no_refusal')"
            )
        conn.close()

    def test_human_overrides_dedupe_on_legacy_db(self, tmp_path):
        """A pre-index DB with duplicate overrides is deduped by get_db's schema
        script (newest row wins), then the unique index is created."""
        db_path = str(tmp_path / "legacy.db")
        conn = sqlite3.connect(db_path)
        # Build a legacy schema WITHOUT the dedupe/index block (the block sits
        # between the human_overrides table and the weight-results section).
        start = SCHEMA.index("-- One override per")
        end = SCHEMA.index("-- ============================================================", start)
        legacy = SCHEMA[:start] + SCHEMA[end:]
        conn.executescript(legacy)
        _register_fixture(conn)
        conn.execute(
            "INSERT INTO human_overrides (comparison, model_name, behavior_id, verdict, overridden_at) "
            "VALUES ('c', 'm', 'b', 'refusal', '2026-01-01T00:00:00Z')"
        )
        conn.execute(
            "INSERT INTO human_overrides (comparison, model_name, behavior_id, verdict, overridden_at) "
            "VALUES ('c', 'm', 'b', 'no_refusal', '2026-02-01T00:00:00Z')"
        )
        conn.commit()
        assert conn.execute("SELECT COUNT(*) FROM human_overrides").fetchone()[0] == 2
        conn.close()

        # Re-open through get_db: schema script dedupes, then enforces
        conn = get_db(db_path)
        rows = conn.execute("SELECT verdict FROM human_overrides").fetchall()
        assert rows == [("no_refusal",)]  # newest overridden_at wins
        conn.close()

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
