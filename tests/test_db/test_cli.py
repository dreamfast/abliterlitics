"""Tests for src/db/cli.py — export-review selection semantics.

Focus: the --reviewer re-review path (NOT EXISTS per-item predicate). The
pre-fix per-joined-row predicate leaked already-judged items via other
reviewers' rows and duplicated multi-reviewer items.
"""

from __future__ import annotations

import json

from src.db.cli import cmd_export_review
from src.db.core import get_db


def _setup(conn) -> None:
    """comparison with one model, three behaviors/responses, low-confidence
    classifications (the default export targets non-high-confidence items)."""
    conn.execute("INSERT INTO comparisons (name, base_model) VALUES ('tc', '/b')")
    conn.execute(
        "INSERT INTO models (comparison, model_name, display_name, model_path, variant_type) "
        "VALUES ('tc', 'v1', 'V1', '/m/v1', 'variant')"
    )
    for bid in ("b1", "b2", "b3"):
        conn.execute(
            "INSERT OR IGNORE INTO behaviors (behavior_id, behavior_text, semantic_category) VALUES (?, ?, 'chemical')",
            (bid, f"Behavior {bid}"),
        )
        conn.execute(
            "INSERT INTO responses (comparison, model_name, behavior_id, response_text, reasoning_text) "
            "VALUES ('tc', 'v1', ?, 'response text', 'thinking text')",
            (bid,),
        )
        conn.execute(
            "INSERT INTO classifications (comparison, model_name, behavior_id, classifier, label, "
            "is_refusal, layer, reason, confidence) "
            "VALUES ('tc', 'v1', ?, 'cls', 'not_refused', 0, 'L1', 'reason', 'medium')",
            (bid,),
        )
    conn.commit()


def _read(out) -> list[dict]:
    return json.loads(out.read_text())


class TestExportReviewSelection:
    def test_default_excludes_any_reviewed_item(self, tmp_path):
        conn = get_db(str(tmp_path / "test.db"))
        _setup(conn)
        conn.execute(
            "INSERT INTO llm_reviews (comparison, model_name, behavior_id, reviewer_model, verdict) "
            "VALUES ('tc', 'v1', 'b1', 'judge-a', 'no_refusal')"
        )
        conn.commit()
        out = tmp_path / "batch.json"
        cmd_export_review(conn, comparison="tc", output=str(out))
        items = _read(out)
        assert {i["behavior_id"] for i in items} == {"b2", "b3"}
        conn.close()

    def test_reviewer_excludes_only_that_reviewers_items(self, tmp_path):
        """--reviewer judge-b: items judged ONLY by judge-a still export."""
        conn = get_db(str(tmp_path / "test.db"))
        _setup(conn)
        conn.execute(
            "INSERT INTO llm_reviews (comparison, model_name, behavior_id, reviewer_model, verdict) "
            "VALUES ('tc', 'v1', 'b1', 'judge-a', 'no_refusal')"
        )
        conn.commit()
        out = tmp_path / "batch.json"
        cmd_export_review(conn, comparison="tc", output=str(out), reviewer="judge-b")
        items = _read(out)
        assert {i["behavior_id"] for i in items} == {"b1", "b2", "b3"}
        conn.close()

    def test_reviewer_no_leak_when_target_and_other_both_reviewed(self, tmp_path):
        """Regression: an item judged by BOTH the target reviewer and another
        reviewer must NOT export via the other reviewer's row."""
        conn = get_db(str(tmp_path / "test.db"))
        _setup(conn)
        for reviewer in ("judge-a", "judge-b"):
            conn.execute(
                "INSERT INTO llm_reviews (comparison, model_name, behavior_id, reviewer_model, verdict) "
                "VALUES ('tc', 'v1', 'b1', ?, 'no_refusal')",
                (reviewer,),
            )
        conn.commit()
        out = tmp_path / "batch.json"
        cmd_export_review(conn, comparison="tc", output=str(out), reviewer="judge-b")
        items = _read(out)
        assert all(i["behavior_id"] != "b1" for i in items)
        conn.close()

    def test_reviewer_multi_reviewer_item_exports_once(self, tmp_path):
        """Regression: an item with rows from two non-target reviewers exports
        once, not once per reviewer row."""
        conn = get_db(str(tmp_path / "test.db"))
        _setup(conn)
        for reviewer in ("judge-a", "judge-c"):
            conn.execute(
                "INSERT INTO llm_reviews (comparison, model_name, behavior_id, reviewer_model, verdict) "
                "VALUES ('tc', 'v1', 'b1', ?, 'refusal')",
                (reviewer,),
            )
        conn.commit()
        out = tmp_path / "batch.json"
        cmd_export_review(conn, comparison="tc", output=str(out), reviewer="judge-b")
        items = _read(out)
        b1_items = [i for i in items if i["behavior_id"] == "b1"]
        assert len(b1_items) == 1
        conn.close()

    def test_reviewed_items_excluded_once_reviewed_by_target(self, tmp_path):
        conn = get_db(str(tmp_path / "test.db"))
        _setup(conn)
        conn.execute(
            "INSERT INTO llm_reviews (comparison, model_name, behavior_id, reviewer_model, verdict) "
            "VALUES ('tc', 'v1', 'b2', 'judge-b', 'soft_refusal')"
        )
        conn.commit()
        out = tmp_path / "batch.json"
        cmd_export_review(conn, comparison="tc", output=str(out), reviewer="judge-b")
        items = _read(out)
        assert {i["behavior_id"] for i in items} == {"b1", "b3"}
        conn.close()

    def test_human_override_excludes_item(self, tmp_path):
        conn = get_db(str(tmp_path / "test.db"))
        _setup(conn)
        conn.execute(
            "INSERT INTO human_overrides (comparison, model_name, behavior_id, verdict) "
            "VALUES ('tc', 'v1', 'b3', 'degenerate')"
        )
        conn.commit()
        out = tmp_path / "batch.json"
        cmd_export_review(conn, comparison="tc", output=str(out))
        items = _read(out)
        assert {i["behavior_id"] for i in items} == {"b1", "b2"}
        conn.close()

    def test_full_reasoning_vs_truncation(self, tmp_path):
        """Lesson #38 contract: default truncates at 3000 chars; --full-reasoning
        exports the complete trace; reasoning_chars_total always exported."""
        conn = get_db(str(tmp_path / "test.db"))
        conn.execute("INSERT INTO comparisons (name, base_model) VALUES ('tc', '/b')")
        conn.execute(
            "INSERT INTO models (comparison, model_name, display_name, model_path, variant_type) "
            "VALUES ('tc', 'v1', 'V1', '/m/v1', 'variant')"
        )
        conn.execute("INSERT INTO behaviors (behavior_id, behavior_text, semantic_category) VALUES ('b1', 'B', 'cat')")
        long_reasoning = "x" * 5000
        conn.execute(
            "INSERT INTO responses (comparison, model_name, behavior_id, response_text, reasoning_text) "
            "VALUES ('tc', 'v1', 'b1', 'resp', ?)",
            (long_reasoning,),
        )
        conn.execute(
            "INSERT INTO classifications (comparison, model_name, behavior_id, classifier, label, "
            "is_refusal, layer, reason, confidence) "
            "VALUES ('tc', 'v1', 'b1', 'cls', 'not_refused', 0, 'L1', 'reason', 'medium')"
        )
        conn.commit()
        out = tmp_path / "batch.json"
        cmd_export_review(conn, comparison="tc", output=str(out))
        item = _read(out)[0]
        assert len(item["reasoning_text"]) == 3000
        assert item["reasoning_chars_total"] == 5000

        out_full = tmp_path / "batch_full.json"
        cmd_export_review(conn, comparison="tc", output=str(out_full), full_reasoning=True)
        item_full = _read(out_full)[0]
        assert len(item_full["reasoning_text"]) == 5000
        assert item_full["reasoning_chars_total"] == 5000
        conn.close()
