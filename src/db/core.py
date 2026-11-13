"""Abliterlitics SQLite database — core connection, dataclasses, helpers."""

from __future__ import annotations

import csv
import io
import logging
import os
import sqlite3
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
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
STAGES = ("weights", "kl", "lm_eval", "harmbench", "text_encoder", "graphs", "report")


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


def _behaviors_cache_path() -> Path:
    """Local cache path for the HarmBench behaviors CSV.

    Repo-root anchored (``<repo>/.cache/``, gitignored) so ``db init`` works
    from any CWD; ``ABL_BEHAVIORS_CACHE`` overrides (tests / sandboxed hosts).
    """
    env = os.environ.get("ABL_BEHAVIORS_CACHE")
    if env:
        return Path(env)
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / ".cache" / "harmbench_behaviors.csv"


def load_behaviors(force_download: bool = False) -> list[dict[str, str]]:
    """Load HarmBench behaviors CSV — local cache first, network fallback.

    Cached at ``<repo_root>/.cache/harmbench_behaviors.csv`` (override with
    ``ABL_BEHAVIORS_CACHE``) so ``db init``/``import_comparison`` work offline
    once the CSV has been fetched once. A network failure with no cache raises
    RuntimeError with a remediation hint instead of a bare URLError traceback.
    """
    cache = _behaviors_cache_path()
    if not force_download and cache.is_file():
        content = cache.read_text(encoding="utf-8")
        behaviors = list(csv.DictReader(io.StringIO(content)))
        log.info("Loaded %d behaviors from cache %s", len(behaviors), cache)
        return behaviors

    try:
        log.info("Downloading HarmBench behaviors...")
        resp = urllib.request.urlopen(HARMBENCH_BEHAVIORS_CSV, timeout=30)
        content = resp.read().decode("utf-8")
    except (urllib.error.URLError, OSError) as exc:
        msg = (
            f"Cannot load HarmBench behaviors: network fetch failed ({exc}) and no "
            f"local cache exists at {cache}. Run once with network access (or "
            f"pre-populate the cache file from the HarmBench repo) before running "
            f"'db init'."
        )
        raise RuntimeError(msg) from exc

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(content, encoding="utf-8")
    behaviors = list(csv.DictReader(io.StringIO(content)))
    log.info("Loaded %d behaviors (cached at %s)", len(behaviors), cache)
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
