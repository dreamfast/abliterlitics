"""Abliterlitics SQLite database — core connection, dataclasses, helpers."""

from __future__ import annotations

import csv
import io
import logging
import sqlite3
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.db.schema import SCHEMA

log = logging.getLogger(__name__)

DB_PATH = "abliterlitics.db"

# Pinned HarmBench behaviors CSV (same as harmbench_generate.py)
HARMBENCH_BEHAVIORS_CSV = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench/"
    "main/data/behavior_datasets/harmbench_behaviors_text_all.csv"
)

# Canonical stage names for model_stages
STAGES = ("weights", "kl", "lm_eval", "harmbench", "graphs", "report")


# ---------------------------------------------------------------------------
# Dataclasses for DB row shapes
# ---------------------------------------------------------------------------


@dataclass
class ModelStage:
    """Row from model_stages table."""

    comparison: str
    model_name: str
    stage: str
    status: str  # pending, running, completed, failed, skipped
    started_at: str | None = None
    finished_at: str | None = None
    heartbeat_at: str | None = None
    result_path: str | None = None
    error_message: str | None = None
    duration_secs: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class KLResult:
    """Row from kl_results table."""

    comparison: str
    variant: str
    kl_batchmean: float | None = None
    kl_median: float | None = None
    kl_std: float | None = None
    kl_min: float | None = None
    kl_max: float | None = None
    num_prompts: int = 0
    rating: str | None = None
    result_path: str | None = None


@dataclass
class LMEvalMetric:
    """Row from lm_eval_results table."""

    comparison: str
    model_name: str
    task: str
    metric_name: str
    metric_value: float
    metric_stderr: float | None = None
    result_path: str | None = None


@dataclass
class WeightResult:
    """Row from weight_results table."""

    comparison: str
    variant: str | None = None
    variant_b: str | None = None
    script_name: str = ""
    result_path: str = ""
    summary: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------


def get_db(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Get a connection to the database, creating schema if needed."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")  # 10s — allow concurrent sub-agent writes
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    return conn


def now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_behaviors() -> list[dict[str, str]]:
    """Download HarmBench behaviors CSV and return as list of dicts."""
    log.info("Downloading HarmBench behaviors...")
    resp = urllib.request.urlopen(HARMBENCH_BEHAVIORS_CSV, timeout=30)
    content = resp.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(content))
    behaviors = list(reader)
    log.info("Loaded %d behaviors", len(behaviors))
    return behaviors


# ---------------------------------------------------------------------------
# Journal — inter-agent communication + project memory
# ---------------------------------------------------------------------------

# Valid values for the journal.category and journal.severity columns.
JOURNAL_CATEGORIES: tuple[str, ...] = (
    "discovery",
    "decision",
    "warning",
    "action",
    "lesson",
    "result",
    "error",
    "note",
)
JOURNAL_SEVERITIES: tuple[str, ...] = (
    "debug",
    "info",
    "warning",
    "error",
    "critical",
)


def journal_add(
    conn: sqlite3.Connection,
    *,
    agent: str,
    category: str,
    subject: str,
    title: str,
    body: str = "",
    comparison: str | None = None,
    severity: str = "info",
) -> int:
    """Append an entry to the journal table. Returns the new row id.

    Args:
        agent: name of the writing agent ('init','lm-eval','kl','harmbench',
            'weights','overseer','report','user','system').
        category: one of JOURNAL_CATEGORIES.
        subject: short tag for filtering ('chat_template','thinking_mode', ...).
        title: one-line summary.
        body: optional markdown body.
        comparison: comparison name, or None for project-wide entries.
        severity: one of JOURNAL_SEVERITIES.
    """
    if category not in JOURNAL_CATEGORIES:
        raise ValueError(f"Invalid category {category!r}. Must be one of {JOURNAL_CATEGORIES}")
    if severity not in JOURNAL_SEVERITIES:
        raise ValueError(f"Invalid severity {severity!r}. Must be one of {JOURNAL_SEVERITIES}")
    cur = conn.execute(
        """
        INSERT INTO journal (comparison, agent, category, severity, subject, title, body)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (comparison, agent, category, severity, subject, title, body),
    )
    conn.commit()
    lastrowid = cur.lastrowid
    # lastrowid is None only if the driver could not report it; -1 signals "no id".
    return lastrowid if lastrowid is not None else -1


def journal_list(
    conn: sqlite3.Connection,
    *,
    comparison: str | None = None,
    agent: str | None = None,
    category: str | None = None,
    subject: str | None = None,
    severity: str | None = None,
    limit: int = 50,
) -> list[sqlite3.Row]:
    """List journal entries, newest first. Any filter can be None (= no filter)."""
    where: list[str] = []
    params: list[Any] = []
    if comparison is not None:
        where.append("comparison = ?")
        params.append(comparison)
    if agent is not None:
        where.append("agent = ?")
        params.append(agent)
    if category is not None:
        where.append("category = ?")
        params.append(category)
    if subject is not None:
        where.append("subject = ?")
        params.append(subject)
    if severity is not None:
        where.append("severity = ?")
        params.append(severity)
    where_clause = ("WHERE " + " AND ".join(where)) if where else ""
    params.append(limit)
    rows = conn.execute(
        f"""
        SELECT id, timestamp, comparison, agent, category, severity, subject, title, body
        FROM journal
        {where_clause}
        ORDER BY timestamp DESC, id DESC
        LIMIT ?
        """,
        params,
    ).fetchall()
    return list(rows)


def journal_latest(
    conn: sqlite3.Connection,
    *,
    comparison: str | None = None,
) -> list[sqlite3.Row]:
    """Latest entry per subject (optionally filtered by comparison)."""
    where = "WHERE comparison = ?" if comparison is not None else ""
    params: list[Any] = [comparison] if comparison is not None else []
    rows = conn.execute(
        f"""
        SELECT comparison, subject, agent, category, severity, title, body, timestamp
        FROM v_journal_latest
        {where}
        ORDER BY subject
        """,
        params,
    ).fetchall()
    return list(rows)


def journal_search(
    conn: sqlite3.Connection,
    query: str,
    *,
    comparison: str | None = None,
    limit: int = 50,
) -> list[sqlite3.Row]:
    """Full-text search across title + body."""
    where_parts = ["(title LIKE ? OR body LIKE ?)"]
    params: list[Any] = [f"%{query}%", f"%{query}%"]
    if comparison is not None:
        where_parts.append("comparison = ?")
        params.append(comparison)
    where_clause = "WHERE " + " AND ".join(where_parts)
    params.append(limit)
    rows = conn.execute(
        f"""
        SELECT id, timestamp, comparison, agent, category, severity, subject, title, body
        FROM journal
        {where_clause}
        ORDER BY timestamp DESC, id DESC
        LIMIT ?
        """,
        params,
    ).fetchall()
    return list(rows)
