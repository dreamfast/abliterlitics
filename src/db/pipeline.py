"""Abliterlitics SQLite database — pipeline stage lifecycle helpers.

Manages the model_stages table: pre-populating stages at init time,
tracking status transitions, and detecting stale running stages
(for crash recovery).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from typing import Any

from src.db.core import STAGES, ModelStage, now_iso

log = logging.getLogger(__name__)

# Columns selected for ModelStage rows — keep in sync with _row_to_stage()
_STAGE_COLS = (
    "comparison, model_name, stage, status, started_at, finished_at, "
    "heartbeat_at, result_path, error_message, duration_secs, metadata"
)


def _row_to_stage(row: tuple[Any, ...]) -> ModelStage:
    """Convert a result row to a ModelStage dataclass."""
    return ModelStage(
        comparison=row[0],
        model_name=row[1],
        stage=row[2],
        status=row[3],
        started_at=row[4],
        finished_at=row[5],
        heartbeat_at=row[6],
        result_path=row[7],
        error_message=row[8],
        duration_secs=row[9],
        metadata=json.loads(row[10]) if row[10] else {},
    )


def init_stages(
    conn: sqlite3.Connection,
    comparison: str,
    model_names: list[str],
    stages: tuple[str, ...] = STAGES,
) -> int:
    """Pre-populate model_stages rows as 'pending' for all models × stages.

    Returns the number of stage rows that were actually inserted
    (skips existing rows via INSERT OR IGNORE).
    """
    before = conn.total_changes
    for model_name in model_names:
        for stage in stages:
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO model_stages "
                    "(comparison, model_name, stage, status) VALUES (?, ?, ?, 'pending')",
                    (comparison, model_name, stage),
                )
            except sqlite3.Error:
                log.warning("Failed to init stage %s/%s/%s", comparison, model_name, stage)
    conn.commit()
    inserted = conn.total_changes - before
    log.info(
        "Initialized %d stage rows for %s (%d models × %d stages)",
        inserted,
        comparison,
        len(model_names),
        len(stages),
    )
    return inserted


def stage_is_completed(conn: sqlite3.Connection, comparison: str, model_name: str, stage: str) -> bool:
    """Read-only check: has this stage already reached 'completed'?

    Library helper so every caller (runners, watchdogs) shares one guard
    against demoting completed stages (lesson #83). Best-effort: any sqlite
    error returns False so callers fall through to their normal write path.
    """
    try:
        row = conn.execute(
            "SELECT 1 FROM model_stages WHERE comparison = ? AND model_name = ? AND stage = ? AND status = 'completed'",
            (comparison, model_name, stage),
        ).fetchone()
    except sqlite3.Error:
        return False
    return row is not None


def stage_start(conn: sqlite3.Connection, comparison: str, model_name: str, stage: str) -> int:
    """Set a stage to 'running' with started_at timestamp.

    Returns the number of stage rows affected (0 if the write was refused
    because the stage is already 'completed' — lesson #83: a re-run/skip-walk
    must never demote a completed stage; the stage stays 'completed' and the
    eventual stage_complete refreshes duration/result instead).
    """
    if stage_is_completed(conn, comparison, model_name, stage):
        log.warning(
            "stage-start: %s/%s/%s already completed — preserving status (lesson #83)",
            comparison,
            model_name,
            stage,
        )
        return 0
    ts = now_iso()
    cur = conn.execute(
        """INSERT INTO model_stages (comparison, model_name, stage, status, started_at, heartbeat_at)
        VALUES (?, ?, ?, 'running', ?, ?)
        ON CONFLICT(comparison, model_name, stage) DO UPDATE SET
            status = 'running', started_at = ?, heartbeat_at = ?, error_message = NULL""",
        (comparison, model_name, stage, ts, ts, ts, ts),
    )
    conn.commit()
    log.info("Stage started: %s/%s/%s", comparison, model_name, stage)
    return cur.rowcount


def stage_heartbeat(conn: sqlite3.Connection, comparison: str, model_name: str, stage: str) -> None:
    """Update heartbeat_at timestamp (for long-running stages)."""
    ts = now_iso()
    conn.execute(
        "UPDATE model_stages SET heartbeat_at = ? WHERE comparison = ? AND model_name = ? AND stage = ?",
        (ts, comparison, model_name, stage),
    )
    conn.commit()


def stage_complete(
    conn: sqlite3.Connection,
    comparison: str,
    model_name: str,
    stage: str,
    result_path: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> int:
    """Set a stage to 'completed' with finished_at and duration.

    Returns the number of rows updated. 0 means the stage name/model has no
    row in model_stages (wrong stage name, or the model was added to
    comparison.json after ``init`` — lesson #84): the caller is warned, never
    silently told "Stage completed".
    """
    ts = now_iso()
    meta_json = json.dumps(metadata) if metadata else None

    # Compute duration from started_at
    row = conn.execute(
        "SELECT started_at FROM model_stages WHERE comparison = ? AND model_name = ? AND stage = ?",
        (comparison, model_name, stage),
    ).fetchone()

    duration: int | None = None
    if row and row[0]:
        try:
            started = datetime.fromisoformat(row[0].replace("Z", "+00:00"))
            finished = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            duration = int((finished - started).total_seconds())
        except (ValueError, TypeError):
            pass

    cur = conn.execute(
        """UPDATE model_stages SET
            status = 'completed', finished_at = ?, heartbeat_at = ?,
            result_path = ?, metadata = COALESCE(?, metadata), duration_secs = ?
        WHERE comparison = ? AND model_name = ? AND stage = ?""",
        (ts, ts, result_path, meta_json, duration, comparison, model_name, stage),
    )
    conn.commit()
    if cur.rowcount == 0:
        log.warning(
            "stage-complete matched 0 rows: %s/%s/%s — check stage name and that "
            "the model is registered (lesson #84); re-run init_stages if the roster changed",
            comparison,
            model_name,
            stage,
        )
    else:
        log.info("Stage completed: %s/%s/%s (%ds)", comparison, model_name, stage, duration or 0)
    return cur.rowcount


def stage_fail(
    conn: sqlite3.Connection,
    comparison: str,
    model_name: str,
    stage: str,
    error_message: str,
) -> int:
    """Set a stage to 'failed' with error message.

    Returns the number of rows updated (0 = no such stage row — warned,
    lesson #84).
    """
    ts = now_iso()
    cur = conn.execute(
        """UPDATE model_stages SET
            status = 'failed', finished_at = ?, heartbeat_at = ?, error_message = ?
        WHERE comparison = ? AND model_name = ? AND stage = ?""",
        (ts, ts, error_message, comparison, model_name, stage),
    )
    conn.commit()
    if cur.rowcount == 0:
        log.warning(
            "stage-fail matched 0 rows: %s/%s/%s — check stage name and that the model is registered (lesson #84)",
            comparison,
            model_name,
            stage,
        )
    else:
        log.error("Stage failed: %s/%s/%s: %s", comparison, model_name, stage, error_message[:200])
    return cur.rowcount


def stage_skip(conn: sqlite3.Connection, comparison: str, model_name: str, stage: str) -> int:
    """Set a stage to 'skipped'.

    Returns the number of rows updated. Refuses (returns 0) when the stage is
    already 'completed' — a skip path must never overwrite a stronger terminal
    status (lesson #83: the re-run/skip-walk demotion loop).
    """
    if stage_is_completed(conn, comparison, model_name, stage):
        log.warning(
            "stage-skip: %s/%s/%s already completed — preserving status (lesson #83)",
            comparison,
            model_name,
            stage,
        )
        return 0
    cur = conn.execute(
        """UPDATE model_stages SET status = 'skipped'
        WHERE comparison = ? AND model_name = ? AND stage = ?""",
        (comparison, model_name, stage),
    )
    conn.commit()
    if cur.rowcount == 0:
        log.warning(
            "stage-skip matched 0 rows: %s/%s/%s — check stage name and that the model is registered (lesson #84)",
            comparison,
            model_name,
            stage,
        )
    else:
        log.info("Stage skipped: %s/%s/%s", comparison, model_name, stage)
    return cur.rowcount


def get_stale_stages(conn: sqlite3.Connection, max_age_hours: int = 2) -> list[ModelStage]:
    """Find 'running' stages whose last heartbeat (or start) is older than threshold.

    Uses COALESCE(heartbeat_at, started_at) so actively-heartbeating stages
    are not incorrectly flagged as stale.
    """
    rows = conn.execute(
        f"""SELECT {_STAGE_COLS}
           FROM model_stages
           WHERE status = 'running'
             AND COALESCE(heartbeat_at, started_at)
                 < strftime('%Y-%m-%dT%H:%M:%SZ', 'now', ? || ' hours')
           ORDER BY started_at""",
        (f"-{max_age_hours}",),
    ).fetchall()

    return [_row_to_stage(r) for r in rows]


def get_stage(conn: sqlite3.Connection, comparison: str, model_name: str, stage: str) -> ModelStage | None:
    """Get a single stage row."""
    row = conn.execute(
        f"SELECT {_STAGE_COLS} FROM model_stages WHERE comparison = ? AND model_name = ? AND stage = ?",
        (comparison, model_name, stage),
    ).fetchone()

    return _row_to_stage(row) if row else None


def get_all_stages(conn: sqlite3.Connection, comparison: str) -> list[ModelStage]:
    """Get all stage rows for a comparison."""
    rows = conn.execute(
        f"SELECT {_STAGE_COLS} FROM model_stages WHERE comparison = ? ORDER BY model_name, stage",
        (comparison,),
    ).fetchall()

    return [_row_to_stage(r) for r in rows]
