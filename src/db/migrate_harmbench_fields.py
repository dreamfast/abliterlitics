#!/usr/bin/env python3
"""One-off migration: HarmBench reasoning-first field semantics + is_degenerate flag.

Background (qwen38-27b, 2026-09): HarmBench items where the model NEVER emitted
``</think>`` (budget-truncated mid-think, or EOS inside the unclosed think
block) were stored with the ENTIRE raw output in ``response`` /
``response_text`` and an empty ``reasoning`` / ``reasoning_text`` — the old
``_split_reasoning`` fallback treated untagged output as content. That is
semantically wrong: with a Qwen3.x thinking template the ``<think>`` opener is
PREFILLED in the prompt, so untagged output is 100% deliberation with no
finalized answer.

This migration is a LOSSLESS field swap (reasoning-first convention):

    reasoning_text <- old response_text      (the full deliberation)
    response_text  <- ''                     (there is no final answer)
    reasoning_chars <- len(old response_text)

Idempotency guard: only rows/items with ``reasoning_chars = 0 AND
response_text <> '' AND reasoning_text = ''`` are migrated. Post-migration
rows have ``response_text = ''`` and can never match again.

Also adds and backfills ``responses.is_degenerate`` (generation-stability
flag, INDEPENDENT of refusal verdicts). Definition:

    is_degenerate = 1 iff
      (a) finish_reason = 'length' AND completion_tokens >= 0.99 * max_tokens
          -- degenerate loop: burned the whole generation budget
      OR (b) finish_reason = 'stop' AND response_text = '' AND no error
          -- EOS-inside-unclosed-think: model stopped without ever closing
             the think block (no final answer, no budget exhaustion)

    Bucket (b) is only meaningful AFTER the field swap (pre-migration those
    rows carry the deliberation in response_text). Error rows (client/HTTP
    failures) are excluded — they are transport failures, not model loops.

Disk files (``harmbench_<model>_responses.json`` and
``harmbench_<model>_classified.json``) get the same swap on their
``response``/``reasoning``/``reasoning_chars`` keys so disk and DB agree.
``.partial.json`` files are never touched.

llm_reviews / human_overrides / classifications verdicts are NOT touched.
After migrating, re-run classification (cmd_classify) so classified files and
the classifications table agree with the new fields: post-migration, a no-closer
item has response='' + reasoning=<deliberation> and classify_item's
empty-response rule scores it REFUSED (no finalized answer = no compliance).

Usage:
    python3 -m src.db.migrate_harmbench_fields --comparison qwen38-27b --dry-run
    python3 -m src.db.migrate_harmbench_fields --comparison qwen38-27b
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Any

from src.db.core import DB_PATH, get_db

log = logging.getLogger(__name__)

# Degenerate-loop budget threshold fraction: completion_tokens must reach
# 0.99 * max_tokens for finish_reason='length' to count as a degenerate loop.
DEGENERATE_LENGTH_FRACTION = 0.99


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    return any(row[1] == column for row in conn.execute(f"PRAGMA table_info({table})"))


def db_swap_counts(conn: sqlite3.Connection, comparison: str) -> list[tuple[Any, ...]]:
    """Per-arm counts of rows matching the migration guard (no writes)."""
    return conn.execute(
        """
        SELECT model_name, COUNT(*),
               SUM(CASE WHEN finish_reason = 'length' THEN 1 ELSE 0 END),
               SUM(CASE WHEN finish_reason = 'stop' THEN 1 ELSE 0 END)
        FROM responses
        WHERE comparison = ? AND reasoning_chars = 0
          AND response_text <> '' AND reasoning_text = ''
        GROUP BY model_name ORDER BY model_name
        """,
        (comparison,),
    ).fetchall()


def db_degenerate_counts(conn: sqlite3.Connection, comparison: str, max_tokens: int) -> list[tuple[Any, ...]]:
    """Per-arm counts of would-be is_degenerate rows as of NOW (no writes).

    Bucket (b) (stop + empty response) only becomes fully populated after the
    field swap — pre-swap those rows carry deliberation in response_text and
    appear in the swap counts instead. Dry-run output therefore shows bucket
    (a) exactly and bucket (b) as a lower bound.
    """
    thresh = int(DEGENERATE_LENGTH_FRACTION * max_tokens)
    return conn.execute(
        """
        SELECT model_name,
               SUM(CASE WHEN finish_reason = 'length'
                          AND completion_tokens >= ? THEN 1 ELSE 0 END),
               SUM(CASE WHEN finish_reason = 'stop'
                          AND response_text = ''
                          AND (error IS NULL OR error = '') THEN 1 ELSE 0 END)
        FROM responses
        WHERE comparison = ?
        GROUP BY model_name ORDER BY model_name
        """,
        (thresh, comparison),
    ).fetchall()


def migrate_db(conn: sqlite3.Connection, comparison: str, max_tokens: int) -> dict[str, Any]:
    """Run the DB field swap + is_degenerate backfill. Idempotent."""
    # 1. Field swap (guarded — post-migration rows have response_text = '')
    cur = conn.execute(
        """
        UPDATE responses
        SET reasoning_text = response_text,
            response_text = '',
            reasoning_chars = length(response_text)
        WHERE comparison = ? AND reasoning_chars = 0
          AND response_text <> '' AND reasoning_text = ''
        """,
        (comparison,),
    )
    swapped = cur.rowcount

    # 2. is_degenerate column (add if missing — SQLite has no column comments,
    #    the definition lives in this docstring and src/db/schema.py)
    if not _has_column(conn, "responses", "is_degenerate"):
        conn.execute("ALTER TABLE responses ADD COLUMN is_degenerate INTEGER NOT NULL DEFAULT 0")
        log.info("Added responses.is_degenerate column")

    # 3. Backfill: reset then apply, so re-runs are deterministic
    conn.execute("UPDATE responses SET is_degenerate = 0 WHERE comparison = ?", (comparison,))
    thresh = int(DEGENERATE_LENGTH_FRACTION * max_tokens)
    cur = conn.execute(
        """
        UPDATE responses SET is_degenerate = 1
        WHERE comparison = ? AND (
            (finish_reason = 'length' AND completion_tokens >= ?)
            OR (finish_reason = 'stop' AND response_text = ''
                AND (error IS NULL OR error = ''))
        )
        """,
        (comparison, thresh),
    )
    degenerate = cur.rowcount

    conn.commit()
    return {"swapped": swapped, "degenerate": degenerate}


def migrate_json_file(path: Path) -> tuple[int, int]:
    """Swap response/reasoning on guard-matching items of one results JSON.

    Returns (n_items, n_swapped). Writes atomically only if something changed.
    """
    data = json.loads(path.read_text())
    container_key = "harmbench" if "harmbench" in data else "harmbench_classified"
    items = data.get(container_key, [])
    swapped = 0
    for item in items:
        response = item.get("response", "") or ""
        reasoning = item.get("reasoning", "") or ""
        rc = item.get("reasoning_chars", 0) or 0
        if rc == 0 and response.strip() and not reasoning.strip():
            item["reasoning"] = response
            item["response"] = ""
            item["reasoning_chars"] = len(response)
            swapped += 1
    if swapped:
        tmp = path.with_suffix(path.suffix + ".migtmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(path)
    return len(items), swapped


def migrate_disk(results_dir: Path, dry_run: bool) -> list[tuple[str, int, int]]:
    """Migrate all harmbench_*_responses.json / _classified.json files."""
    out = []
    for pattern in ("harmbench_*_responses.json", "harmbench_*_classified.json"):
        for path in sorted(results_dir.glob(pattern)):
            if path.name.endswith(".partial.json"):
                continue  # transient, possibly in-flight — never touch
            if dry_run:
                data = json.loads(path.read_text())
                container_key = "harmbench" if "harmbench" in data else "harmbench_classified"
                items = data.get(container_key, [])
                n = sum(
                    1
                    for it in items
                    if (it.get("reasoning_chars", 0) or 0) == 0
                    and (it.get("response", "") or "").strip()
                    and not (it.get("reasoning", "") or "").strip()
                )
                out.append((path.name, len(items), n))
            else:
                total, swapped = migrate_json_file(path)
                out.append((path.name, total, swapped))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--comparison", default="qwen38-27b", help="Comparison name")
    parser.add_argument("--db", default=str(DB_PATH), help="Database path")
    parser.add_argument(
        "--results-dir",
        default=None,
        help="HarmBench results dir (default: comparisons/<comparison>/results/harmbench)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=15360,
        help="harmbench_max_tokens used for this comparison (degenerate-bucket-a threshold)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report counts only, write nothing")
    parser.add_argument("--skip-db", action="store_true", help="Skip the DB migration")
    parser.add_argument("--skip-disk", action="store_true", help="Skip the disk-file migration")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    results_dir = (
        Path(args.results_dir) if args.results_dir else Path("comparisons") / args.comparison / "results" / "harmbench"
    )

    print(f"Comparison: {args.comparison}  (dry_run={args.dry_run}, max_tokens={args.max_tokens})")

    if not args.skip_db:
        conn = get_db(args.db)
        print("\n== DB rows matching migration guard (rc=0, response_text non-empty):")
        for model, n, n_len, n_stop in db_swap_counts(conn, args.comparison):
            print(f"  {model:<20} swap={n:<5} (finish length={n_len}, stop={n_stop})")
        print("\n== is_degenerate (pre-swap snapshot; bucket b is a lower bound):")
        for model, n_len, n_stop in db_degenerate_counts(conn, args.comparison, args.max_tokens):
            print(f"  {model:<20} length-loop={n_len:<5} stop-empty={n_stop}")

        if args.dry_run:
            print("\n[dry-run] DB not modified.")
        else:
            res = migrate_db(conn, args.comparison, args.max_tokens)
            print(f"\nDB migration done: swapped={res['swapped']}, is_degenerate=1 rows={res['degenerate']}")
        conn.close()
    else:
        print("\n(--skip-db) DB untouched.")

    if not args.skip_disk:
        if not results_dir.exists():
            print(f"\nResults dir not found: {results_dir} — skipping disk migration")
        else:
            print(f"\n== Disk files in {results_dir}:")
            for name, total, swapped in migrate_disk(results_dir, args.dry_run):
                action = "WOULD swap" if args.dry_run else "swapped"
                if swapped or args.dry_run:
                    print(f"  {name:<50} items={total:<4} {action}={swapped}")
                else:
                    print(f"  {name:<50} items={total:<4} (no guard matches)")
            if args.dry_run:
                print("[dry-run] disk files not modified.")
    else:
        print("\n(--skip-disk) disk files untouched.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
