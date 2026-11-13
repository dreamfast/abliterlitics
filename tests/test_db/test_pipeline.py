"""Tests for src/db/pipeline.py — stage lifecycle helpers."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from src.db.core import get_db
from src.db.pipeline import (
    get_all_stages,
    get_stage,
    get_stale_stages,
    init_stages,
    stage_complete,
    stage_fail,
    stage_heartbeat,
    stage_is_completed,
    stage_skip,
    stage_start,
)


def _setup_comparison(conn, name="test-comp", models=("base", "variant_a")):
    """Helper: register a comparison with models."""
    conn.execute(
        "INSERT INTO comparisons (name, base_model) VALUES (?, '/models/base')",
        (name,),
    )
    for i, model_name in enumerate(models):
        conn.execute(
            "INSERT INTO models "
            "(comparison, model_name, display_name, model_path, variant_type) "
            "VALUES (?, ?, ?, ?, ?)",
            (name, model_name, model_name, f"/models/{model_name}", "base" if i == 0 else "variant"),
        )
    conn.commit()


class TestStageLifecycle:
    """Test stage init → start → complete transitions."""

    def test_init_stages(self, tmp_path):
        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)
        count = init_stages(conn, "test-comp", ["base", "variant_a"])
        assert count == 14  # 2 models x 7 stages (text_encoder added f75aed6)

        stages = get_all_stages(conn, "test-comp")
        assert len(stages) == 14  # 2 models x 7 stages
        assert all(s.status == "pending" for s in stages)
        conn.close()

    def test_start_then_complete(self, tmp_path):
        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)
        init_stages(conn, "test-comp", ["base"])

        stage_start(conn, "test-comp", "base", "weights")
        s = get_stage(conn, "test-comp", "base", "weights")
        assert s is not None
        assert s.status == "running"
        assert s.started_at is not None

        stage_complete(conn, "test-comp", "base", "weights", result_path="/results/weight.json")
        s = get_stage(conn, "test-comp", "base", "weights")
        assert s.status == "completed"
        assert s.finished_at is not None
        assert s.result_path == "/results/weight.json"
        assert s.duration_secs is not None
        assert s.duration_secs >= 0
        conn.close()

    def test_start_then_fail(self, tmp_path):
        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)
        init_stages(conn, "test-comp", ["base"])

        stage_start(conn, "test-comp", "base", "kl")
        stage_fail(conn, "test-comp", "base", "kl", "Docker OOM")

        s = get_stage(conn, "test-comp", "base", "kl")
        assert s.status == "failed"
        assert s.error_message == "Docker OOM"
        assert s.finished_at is not None
        conn.close()

    def test_heartbeat_updates_timestamp(self, tmp_path):
        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)
        init_stages(conn, "test-comp", ["base"])

        stage_start(conn, "test-comp", "base", "weights")
        s1 = get_stage(conn, "test-comp", "base", "weights")
        assert s1.heartbeat_at is not None

        # Heartbeat should advance the timestamp
        import time

        time.sleep(1)
        stage_heartbeat(conn, "test-comp", "base", "weights")
        s2 = get_stage(conn, "test-comp", "base", "weights")
        assert s2.heartbeat_at is not None
        assert s2.heartbeat_at >= s1.heartbeat_at
        conn.close()

    def test_stage_skip(self, tmp_path):
        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)
        init_stages(conn, "test-comp", ["base"])

        stage_skip(conn, "test-comp", "base", "report")
        s = get_stage(conn, "test-comp", "base", "report")
        assert s.status == "skipped"
        conn.close()

    def test_complete_with_metadata(self, tmp_path):
        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)
        init_stages(conn, "test-comp", ["base"])

        stage_start(conn, "test-comp", "base", "lm_eval")
        stage_complete(
            conn,
            "test-comp",
            "base",
            "lm_eval",
            result_path="/results/lm_eval_base.json",
            metadata={"phases": 2, "tasks": 8},
        )

        s = get_stage(conn, "test-comp", "base", "lm_eval")
        assert s.metadata == {"phases": 2, "tasks": 8}
        conn.close()

    def test_init_stages_idempotent(self, tmp_path):
        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)

        init_stages(conn, "test-comp", ["base"])
        init_stages(conn, "test-comp", ["base"])

        stages = get_all_stages(conn, "test-comp")
        assert len(stages) == 7  # Still 7, not 14
        conn.close()


class TestStageWriteFailExplicit:
    """Lesson #84: stage writes must be fail-explicit, never silent no-ops."""

    def test_stage_complete_unknown_row_returns_zero(self, tmp_path, caplog):
        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)
        init_stages(conn, "test-comp", ["base"])

        with caplog.at_level(logging.WARNING):
            rows = stage_complete(conn, "test-comp", "base", "no_such_stage")
        assert rows == 0
        assert "matched 0 rows" in caplog.text
        assert "no_such_stage" in caplog.text
        conn.close()

    def test_stage_fail_unknown_row_returns_zero(self, tmp_path, caplog):
        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)
        init_stages(conn, "test-comp", ["base"])

        with caplog.at_level(logging.WARNING):
            rows = stage_fail(conn, "test-comp", "ghost-model", "weights", "OOM")
        assert rows == 0
        assert "matched 0 rows" in caplog.text
        conn.close()

    def test_stage_skip_unknown_row_returns_zero(self, tmp_path, caplog):
        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)

        with caplog.at_level(logging.WARNING):
            rows = stage_skip(conn, "test-comp", "base", "report")
        assert rows == 0
        assert "matched 0 rows" in caplog.text
        conn.close()

    def test_stage_complete_success_returns_one(self, tmp_path):
        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)
        init_stages(conn, "test-comp", ["base"])
        assert stage_start(conn, "test-comp", "base", "weights") == 1
        assert stage_complete(conn, "test-comp", "base", "weights") == 1
        conn.close()

    def test_stage_skip_never_demotes_completed(self, tmp_path, caplog):
        """Lesson #83: a skip path must never overwrite a completed stage."""
        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)
        init_stages(conn, "test-comp", ["base"])
        stage_start(conn, "test-comp", "base", "weights")
        stage_complete(conn, "test-comp", "base", "weights")

        with caplog.at_level(logging.WARNING):
            rows = stage_skip(conn, "test-comp", "base", "weights")
        assert rows == 0
        assert "preserving status" in caplog.text
        assert get_stage(conn, "test-comp", "base", "weights").status == "completed"
        conn.close()

    def test_stage_start_never_demotes_completed(self, tmp_path, caplog):
        """Lesson #83: an unconditional re-run start must not demote completed."""
        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)
        init_stages(conn, "test-comp", ["base"])
        stage_start(conn, "test-comp", "base", "weights")
        stage_complete(conn, "test-comp", "base", "weights")

        with caplog.at_level(logging.WARNING):
            rows = stage_start(conn, "test-comp", "base", "weights")
        assert rows == 0
        assert "preserving status" in caplog.text
        assert get_stage(conn, "test-comp", "base", "weights").status == "completed"
        conn.close()

    def test_stage_is_completed_helper(self, tmp_path):
        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)
        init_stages(conn, "test-comp", ["base"])
        assert stage_is_completed(conn, "test-comp", "base", "weights") is False
        stage_start(conn, "test-comp", "base", "weights")
        assert stage_is_completed(conn, "test-comp", "base", "weights") is False
        stage_complete(conn, "test-comp", "base", "weights")
        assert stage_is_completed(conn, "test-comp", "base", "weights") is True
        # Unknown stage/model → False (no row)
        assert stage_is_completed(conn, "test-comp", "ghost", "weights") is False
        conn.close()


class TestStaleDetection:
    """Test stale running stage detection for crash recovery."""

    def test_stale_stages_detected(self, tmp_path):
        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)
        init_stages(conn, "test-comp", ["base"])

        # Manually insert a stale running stage using now_iso() format
        # (same format production code uses) — 3 hours ago
        from src.db.core import now_iso

        three_hours_ago = (datetime.fromisoformat(now_iso().replace("Z", "+00:00")) - timedelta(hours=3)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

        conn.execute(
            """UPDATE model_stages SET status = 'running', started_at = ?
            WHERE comparison = 'test-comp' AND model_name = 'base' AND stage = 'weights'""",
            (three_hours_ago,),
        )
        conn.commit()

        stale = get_stale_stages(conn, max_age_hours=2)
        assert len(stale) == 1
        assert stale[0].model_name == "base"
        assert stale[0].stage == "weights"
        conn.close()

    def test_recent_running_not_stale(self, tmp_path):
        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)
        init_stages(conn, "test-comp", ["base"])

        stage_start(conn, "test-comp", "base", "kl")

        stale = get_stale_stages(conn, max_age_hours=2)
        assert len(stale) == 0
        conn.close()


class TestPipelineViews:
    """Test v_pipeline_status view logic."""

    def test_pipeline_status_not_started(self, tmp_path):
        conn = get_db(str(tmp_path / "test.db"))
        conn.execute("INSERT INTO comparisons (name, base_model) VALUES ('test', '/m')")
        conn.commit()

        row = conn.execute("SELECT overall_status FROM v_pipeline_status WHERE comparison = 'test'").fetchone()
        assert row[0] == "not_started"
        conn.close()

    def test_pipeline_status_completed(self, tmp_path):
        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)
        init_stages(conn, "test-comp", ["base"])

        # Complete all stages
        for stage in ("weights", "kl", "lm_eval", "harmbench", "text_encoder", "graphs", "report"):
            stage_start(conn, "test-comp", "base", stage)
            stage_complete(conn, "test-comp", "base", stage)

        row = conn.execute(
            "SELECT overall_status, completed_stages, total_stages "
            "FROM v_pipeline_status WHERE comparison = 'test-comp'"
        ).fetchone()
        assert row[0] == "completed"
        assert row[1] == 7
        assert row[2] == 7
        conn.close()

    def test_pipeline_status_has_failures(self, tmp_path):
        conn = get_db(str(tmp_path / "test.db"))
        _setup_comparison(conn)
        init_stages(conn, "test-comp", ["base"])

        stage_start(conn, "test-comp", "base", "weights")
        stage_complete(conn, "test-comp", "base", "weights")
        stage_start(conn, "test-comp", "base", "kl")
        stage_fail(conn, "test-comp", "base", "kl", "OOM")

        row = conn.execute(
            "SELECT overall_status, failed_stages FROM v_pipeline_status WHERE comparison = 'test-comp'"
        ).fetchone()
        assert row[0] == "has_failures"
        assert row[1] == 1
        conn.close()
