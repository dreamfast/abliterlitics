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
