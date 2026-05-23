"""SQLite database for Abliterlitics review tracking.

Stores HarmBench responses, classifier results, LLM reviews, and human
overrides in a single queryable database. Designed so multiple classifier
versions and review rounds can coexist without losing history.

Usage:
    # Create/import from scratch
    python3 -m src.db init comparisons/gemma4-e2b/

    # Check status
    python3 -m src.db status

    # Export items needing LLM review
    python3 -m src.db export-review --model llmfan46 --output review_batch.json
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import sqlite3
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

DB_PATH = "abliterlitics.db"

# Pinned HarmBench behaviors CSV (same as harmbench_generate.py)
HARMBENCH_BEHAVIORS_CSV = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench/"
    "main/data/behavior_datasets/harmbench_behaviors_text_all.csv"
)

SCHEMA = """
-- HarmBench behavior definitions (shared across all models)
CREATE TABLE IF NOT EXISTS behaviors (
    behavior_id    TEXT PRIMARY KEY,
    behavior_text  TEXT NOT NULL,
    semantic_category TEXT NOT NULL,
    tags           TEXT NOT NULL DEFAULT ''
);

-- Models in the comparison
CREATE TABLE IF NOT EXISTS models (
    model_name     TEXT PRIMARY KEY,
    display_name   TEXT NOT NULL,
    model_path     TEXT NOT NULL,
    variant_type   TEXT NOT NULL DEFAULT 'variant',  -- 'base' or 'variant'
    comparison     TEXT NOT NULL DEFAULT ''           -- e.g. 'gemma4-e2b'
);

-- Raw model responses to HarmBench behaviors
CREATE TABLE IF NOT EXISTS responses (
    model_name       TEXT NOT NULL,
    behavior_id      TEXT NOT NULL,
    response_text    TEXT NOT NULL DEFAULT '',
    reasoning_text   TEXT NOT NULL DEFAULT '',
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    finish_reason    TEXT NOT NULL DEFAULT '',
    error            TEXT,
    is_refusal_kw    INTEGER,            -- original keyword matcher result
    PRIMARY KEY (model_name, behavior_id),
    FOREIGN KEY (model_name) REFERENCES models(model_name),
    FOREIGN KEY (behavior_id) REFERENCES behaviors(behavior_id)
);

-- Classifier results (one row per classifier version per response)
CREATE TABLE IF NOT EXISTS classifications (
    model_name       TEXT NOT NULL,
    behavior_id      TEXT NOT NULL,
    classifier       TEXT NOT NULL,      -- e.g. 'harmbench_classify.py v4.0'
    label            TEXT NOT NULL,      -- 'refused' or 'not_refused'
    is_refusal       INTEGER NOT NULL,   -- 1=refused, 0=complied
    layer            TEXT NOT NULL DEFAULT '',  -- L1_regex, L1_soft, etc
    reason           TEXT NOT NULL DEFAULT '',
    confidence       TEXT NOT NULL DEFAULT 'high', -- high, medium, low
    classified_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    PRIMARY KEY (model_name, behavior_id, classifier),
    FOREIGN KEY (model_name) REFERENCES models(model_name),
    FOREIGN KEY (behavior_id) REFERENCES behaviors(behavior_id)
);

-- LLM review results (one row per reviewer per response)
CREATE TABLE IF NOT EXISTS llm_reviews (
    model_name       TEXT NOT NULL,
    behavior_id      TEXT NOT NULL,
    reviewer_model   TEXT NOT NULL,      -- e.g. 'gpt-4o-mini', 'claude-haiku'
    verdict          TEXT NOT NULL,      -- 'refused' or 'complied'
    review_reasoning TEXT NOT NULL DEFAULT '',
    reviewed_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    PRIMARY KEY (model_name, behavior_id, reviewer_model),
    FOREIGN KEY (model_name) REFERENCES models(model_name),
    FOREIGN KEY (behavior_id) REFERENCES behaviors(behavior_id)
);

-- Human overrides (final authority)
CREATE TABLE IF NOT EXISTS human_overrides (
    model_name       TEXT NOT NULL,
    behavior_id      TEXT NOT NULL,
    verdict          TEXT NOT NULL,      -- 'refused' or 'complied'
    reason           TEXT NOT NULL DEFAULT '',
    overridden_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    overridden_by    TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (model_name, behavior_id),
    FOREIGN KEY (model_name) REFERENCES models(model_name),
    FOREIGN KEY (behavior_id) REFERENCES behaviors(behavior_id)
);

-- Useful views
CREATE VIEW IF NOT EXISTS v_latest_classification AS
    SELECT model_name, behavior_id, label, is_refusal, layer, reason, confidence, classifier
    FROM classifications c1
    WHERE classifier = (
        SELECT c2.classifier FROM classifications c2
        WHERE c2.model_name = c1.model_name AND c2.behavior_id = c1.behavior_id
        ORDER BY c2.classified_at DESC LIMIT 1
    );

CREATE VIEW IF NOT EXISTS v_review_status AS
    SELECT
        r.model_name,
        r.behavior_id,
        c.label AS classifier_label,
        c.confidence,
        c.classifier,
        lr.verdict AS llm_verdict,
        lr.reviewer_model AS llm_reviewer,
        ho.verdict AS human_verdict,
        CASE
            WHEN ho.verdict IS NOT NULL THEN 'resolved'
            WHEN lr.verdict IS NOT NULL THEN 'llm_reviewed'
            WHEN c.confidence != 'high' THEN 'needs_llm_review'
            ELSE 'classifier_only'
        END AS review_status
    FROM responses r
    LEFT JOIN v_latest_classification c ON r.model_name = c.model_name AND r.behavior_id = c.behavior_id
    LEFT JOIN llm_reviews lr ON r.model_name = lr.model_name AND r.behavior_id = lr.behavior_id
    LEFT JOIN human_overrides ho ON r.model_name = ho.model_name AND r.behavior_id = ho.behavior_id;
"""


def get_db(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Get a connection to the database, creating schema if needed."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    return conn


def load_behaviors() -> list[dict]:
    """Download HarmBench behaviors CSV and return as list of dicts."""
    log.info("Downloading HarmBench behaviors...")
    resp = urllib.request.urlopen(HARMBENCH_BEHAVIORS_CSV)
    content = resp.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(content))
    behaviors = list(reader)
    log.info("Loaded %d behaviors", len(behaviors))
    return behaviors


def import_comparison(conn: sqlite3.Connection, comparison_dir: str) -> None:
    """Import all HarmBench data for a comparison directory."""
    base = Path(comparison_dir)
    comp_file = base / "comparison.json"
    if not comp_file.exists():
        log.error("No comparison.json found in %s", comparison_dir)
        return

    comp = json.loads(comp_file.read_text())
    comparison_name = comp.get("name", base.name)

    # --- Import behaviors ---
    behaviors = load_behaviors()
    bcount = 0
    for b in behaviors:
        try:
            conn.execute(
                "INSERT OR IGNORE INTO behaviors (behavior_id, behavior_text, semantic_category, tags) VALUES (?, ?, ?, ?)",
                (b.get("BehaviorID", ""), b.get("Behavior", ""), b.get("SemanticCategory", ""), b.get("Tags", "")),
            )
            bcount += 1
        except sqlite3.Error:
            log.warning("Failed to import behavior: %s", b.get("BehaviorID", ""))
    conn.commit()
    log.info("Imported %d behaviors", bcount)

    # --- Import models ---
    base_name = comp.get("base", "").split("/")[-1] or "base"
    base_display = "Base (Google)"
    conn.execute(
        "INSERT OR IGNORE INTO models (model_name, display_name, model_path, variant_type, comparison) VALUES (?, ?, ?, 'base', ?)",
        (base_name, base_display, comp.get("base", ""), comparison_name),
    )

    variants = comp.get("variants", {})
    for vname, vinfo in variants.items():
        path = vinfo.get("path", "")
        display = vinfo.get("display_name", vname)
        conn.execute(
            "INSERT OR IGNORE INTO models (model_name, display_name, model_path, variant_type, comparison) VALUES (?, ?, ?, 'variant', ?)",
            (vname, display, path, comparison_name),
        )
    conn.commit()
    log.info("Imported %d models (1 base + %d variants)", 1 + len(variants), len(variants))

    # --- Import responses and classifications ---
    harmbench_dir = base / "results" / "harmbench"
    classified_dir = base / "results" / "harmbench_classified"

    if not harmbench_dir.exists():
        log.error("No harmbench results found in %s", harmbench_dir)
        return

    response_files = sorted(harmbench_dir.glob("harmbench_*_responses.json"))
    total_responses = 0
    total_classified = 0

    for rf in response_files:
        model_name = rf.stem.replace("harmbench_", "").replace("_responses", "")
        # Normalize model name to match comparison.json keys
        # harmbench uses underscores, comparison.json uses hyphens for huihui
        model_key = model_name.replace("_", "-")

        # Find the actual model_name in our models table
        # Try both forms
        actual_model = model_name
        if conn.execute("SELECT 1 FROM models WHERE model_name = ?", (model_name,)).fetchone():
            actual_model = model_name
        elif conn.execute("SELECT 1 FROM models WHERE model_name = ?", (model_key,)).fetchone():
            actual_model = model_key
        else:
            log.warning("Model %s not found in comparison.json, adding anyway", model_name)
            actual_model = model_name
            conn.execute(
                "INSERT OR IGNORE INTO models (model_name, display_name, model_path, variant_type, comparison) VALUES (?, ?, '', 'variant', ?)",
                (model_name, model_name, comparison_name),
            )

        raw = json.loads(rf.read_text())
        items = raw.get("harmbench", [])
        kw_count = 0

        for item in items:
            bid = item.get("behavior_id", "")
            is_ref_kw = 1 if item.get("is_refusal") else 0
            if is_ref_kw:
                kw_count += 1

            conn.execute(
                """INSERT OR REPLACE INTO responses
                (model_name, behavior_id, response_text, reasoning_text, completion_tokens, finish_reason, error, is_refusal_kw)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    actual_model,
                    bid,
                    item.get("response", ""),
                    item.get("reasoning", ""),
                    item.get("completion_tokens", 0),
                    item.get("finish_reason", ""),
                    item.get("error"),
                    is_ref_kw,
                ),
            )
            total_responses += 1

        # Import classified results if available
        cls_file = classified_dir / f"{model_name}_classified.json"
        if not cls_file.exists():
            # Try with hyphens
            cls_file = classified_dir / f"{model_key}_classified.json"

        if cls_file.exists():
            cls = json.loads(cls_file.read_text())
            classifier = cls.get("classifier", "unknown")

            for item in cls.get("harmbench_classified", []):
                bid = item.get("behavior_id", "")
                label = "refused" if item.get("is_refusal") else "not_refused"
                is_ref = 1 if item.get("is_refusal") else 0

                conn.execute(
                    """INSERT OR REPLACE INTO classifications
                    (model_name, behavior_id, classifier, label, is_refusal, layer, reason, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        actual_model,
                        bid,
                        classifier,
                        label,
                        is_ref,
                        item.get("classification_layer", ""),
                        item.get("classification_reason", ""),
                        item.get("classification_confidence", "high"),
                    ),
                )
                total_classified += 1

        conn.commit()
        log.info("  %s: %d responses, %d kw-refusals", actual_model, len(items), kw_count)

    log.info("Total: %d responses, %d classifications imported", total_responses, total_classified)


def cmd_status(conn: sqlite3.Connection) -> None:
    """Print database status summary."""
    models = conn.execute("SELECT model_name, display_name, variant_type, comparison FROM models ORDER BY variant_type, model_name").fetchall()
    print(f"Models: {len(models)}")
    for m in models:
        print(f"  {m[0]:<15} {m[1]:<30} {m[2]:<8} {m[3]}")

    print()

    behaviors = conn.execute("SELECT COUNT(*) FROM behaviors").fetchone()[0]
    print(f"Behaviors: {behaviors}")

    responses = conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
    print(f"Responses: {responses}")

    classifiers = conn.execute("SELECT DISTINCT classifier FROM classifications").fetchall()
    print(f"Classifiers: {[c[0] for c in classifiers]}")

    # Per-model summary from latest classifier
    print()
    print(f"{'Model':<15} {'Responses':>9} {'Refused':>7} {'ASR':>7} {'NeedsRev':>9} {'LLMRev':>6} {'Human':>5}")
    print("-" * 65)

    for m in models:
        name = m[0]
        row = conn.execute(
            """SELECT
                COUNT(*) as total,
                SUM(c.is_refusal) as refused,
                SUM(CASE WHEN c.confidence != 'high' THEN 1 ELSE 0 END) as needs_review,
                SUM(CASE WHEN lr.verdict IS NOT NULL THEN 1 ELSE 0 END) as llm_reviewed,
                SUM(CASE WHEN ho.verdict IS NOT NULL THEN 1 ELSE 0 END) as human_overridden
            FROM responses r
            LEFT JOIN v_latest_classification c ON r.model_name = c.model_name AND r.behavior_id = c.behavior_id
            LEFT JOIN llm_reviews lr ON r.model_name = lr.model_name AND r.behavior_id = lr.behavior_id
            LEFT JOIN human_overrides ho ON r.model_name = ho.model_name AND r.behavior_id = ho.behavior_id
            WHERE r.model_name = ?""",
            (name,),
        ).fetchone()

        if row and row[0]:
            total = row[0]
            refused = row[1] or 0
            asr = (total - refused) / total * 100 if total else 0
            needs = row[2] or 0
            llm = row[3] or 0
            human = row[4] or 0
            print(f"{name:<15} {total:>9} {refused:>7} {asr:>6.1f}% {needs:>9} {llm:>6} {human:>5}")


# Confidence sort order: low first, then medium, then high.
_CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}


def cmd_export_review(
    conn: sqlite3.Connection,
    model: str | None = None,
    output: str = "review_batch.json",
    confidence: str = "non-high",
    limit: int = 0,
) -> None:
    """Export items needing LLM review as a JSON batch.

    Results are ordered by confidence (low → medium → high) so the reviewer
    tackles the hardest items first. Use --limit to cap the batch size for
    incremental review loops.
    """
    where_parts = ["(lr.verdict IS NULL AND ho.verdict IS NULL)"]
    params: list[str] = []
    if model:
        where_parts.append("r.model_name = ?")
        params.append(model)
    if confidence == "non-high":
        where_parts.append("c.confidence != 'high'")
    # "all" = no confidence filter, export everything

    # ORDER BY: low confidence first (hardest to classify), then medium, then high.
    limit_clause = f"LIMIT {limit}" if limit > 0 else ""

    rows = conn.execute(
        f"""SELECT r.model_name, r.behavior_id, b.behavior_text, b.semantic_category,
               r.response_text, r.reasoning_text, r.completion_tokens,
               c.label, c.layer, c.reason, c.confidence, c.classifier
        FROM responses r
        JOIN behaviors b ON r.behavior_id = b.behavior_id
        LEFT JOIN v_latest_classification c ON r.model_name = c.model_name AND r.behavior_id = c.behavior_id
        LEFT JOIN llm_reviews lr ON r.model_name = lr.model_name AND r.behavior_id = lr.behavior_id
        LEFT JOIN human_overrides ho ON r.model_name = ho.model_name AND r.behavior_id = ho.behavior_id
        WHERE {' AND '.join(where_parts)}
        ORDER BY CASE c.confidence
            WHEN 'low' THEN 0
            WHEN 'medium' THEN 1
            WHEN 'high' THEN 2
            ELSE 3
        END, r.model_name, r.behavior_id
        {limit_clause}""",
        params,
    ).fetchall()

    items = []
    for r in rows:
        items.append(
            {
                "model_name": r[0],
                "behavior_id": r[1],
                "behavior_text": r[2][:200],
                "semantic_category": r[3],
                "response_text": r[4],
                "reasoning_text": r[5][:3000] if r[5] else "",
                "completion_tokens": r[6],
                "classifier_label": r[7],
                "classifier_layer": r[8],
                "classifier_reason": r[9],
                "classifier_confidence": r[10],
                "classifier": r[11],
            }
        )

    Path(output).write_text(json.dumps(items, indent=2))
    log.info("Exported %d items to %s", len(items), output)


def cmd_import_llm_reviews(conn: sqlite3.Connection, reviews_file: str, reviewer_model: str) -> None:
    """Import LLM review results from a JSON file."""
    reviews = json.loads(Path(reviews_file).read_text())
    count = 0
    for r in reviews:
        conn.execute(
            """INSERT OR REPLACE INTO llm_reviews
            (model_name, behavior_id, reviewer_model, verdict, review_reasoning)
            VALUES (?, ?, ?, ?, ?)""",
            (
                r["model_name"],
                r["behavior_id"],
                reviewer_model,
                r.get("verdict", ""),
                r.get("review_reasoning", ""),
            ),
        )
        count += 1
    conn.commit()
    log.info("Imported %d LLM reviews from %s (reviewer: %s)", count, reviews_file, reviewer_model)


def cmd_export_prompt(conn: sqlite3.Connection, model: str | None, output: str, batch_size: int = 10, confidence: str = "non-high", limit: int = 0) -> None:
    """Export items as a ready-to-use LLM prompt for review.

    Creates a JSON file with both the system prompt and the batched user prompts,
    so you can paste them directly into an LLM chat or API call.
    """
    from src.benchmark.reviewer_prompt import REVIEWER_SYSTEM_PROMPT, build_batch_prompt

    # Reuse export-review logic to get items
    where_parts = ["(lr.verdict IS NULL AND ho.verdict IS NULL)"]
    params: list[str] = []
    if model:
        where_parts.append("r.model_name = ?")
        params.append(model)
    if confidence == "non-high":
        where_parts.append("c.confidence != 'high'")

    limit_clause = f"LIMIT {limit}" if limit > 0 else ""

    rows = conn.execute(
        f"""SELECT r.model_name, r.behavior_id, b.behavior_text, b.semantic_category,
               r.response_text, r.reasoning_text, r.completion_tokens,
               c.label, c.layer, c.reason, c.confidence, c.classifier
        FROM responses r
        JOIN behaviors b ON r.behavior_id = b.behavior_id
        LEFT JOIN v_latest_classification c ON r.model_name = c.model_name AND r.behavior_id = c.behavior_id
        LEFT JOIN llm_reviews lr ON r.model_name = lr.model_name AND r.behavior_id = lr.behavior_id
        LEFT JOIN human_overrides ho ON r.model_name = ho.model_name AND r.behavior_id = ho.behavior_id
        WHERE {' AND '.join(where_parts)}
        ORDER BY CASE c.confidence
            WHEN 'low' THEN 0
            WHEN 'medium' THEN 1
            WHEN 'high' THEN 2
            ELSE 3
        END, r.model_name, r.behavior_id
        {limit_clause}""",
        params,
    ).fetchall()

    items = []
    for r in rows:
        items.append(
            {
                "model_name": r[0],
                "behavior_id": r[1],
                "behavior_text": r[2][:200],
                "semantic_category": r[3],
                "response_text": r[4],
                "reasoning_text": r[5][:3000] if r[5] else "",
                "completion_tokens": r[6],
                "classifier_label": r[7],
                "classifier_layer": r[8],
                "classifier_reason": r[9],
                "classifier_confidence": r[10],
                "classifier": r[11],
            }
        )

    # Build batches
    batches = []
    for i in range(0, len(items), batch_size):
        batch_items = items[i : i + batch_size]
        batches.append(
            {
                "batch_num": i // batch_size + 1,
                "item_count": len(batch_items),
                "items": batch_items,
                "user_prompt": build_batch_prompt(batch_items),
            }
        )

    export_data = {
        "system_prompt": REVIEWER_SYSTEM_PROMPT,
        "total_items": len(items),
        "batch_size": batch_size,
        "total_batches": len(batches),
        "batches": batches,
    }

    Path(output).write_text(json.dumps(export_data, indent=2))
    log.info("Exported %d items in %d batches to %s", len(items), len(batches), output)


def cmd_asr(conn: sqlite3.Connection, model: str | None = None) -> None:
    """Print ASR computed from the authoritative review chain.

    Priority: human_overrides > llm_reviews > classifier.
    """
    models = conn.execute("SELECT model_name, display_name FROM models ORDER BY variant_type, model_name").fetchall()

    print(f"{'Model':<15} {'Total':>5} {'Complied':>8} {'Refused':>8} {'ASR':>7} {'Source':>12}")
    print("-" * 60)

    for mname, dname in models:
        if model and mname != model:
            continue

        row = conn.execute(
            """SELECT
                COUNT(*) as total,
                SUM(CASE
                    WHEN ho.verdict IS NOT NULL THEN CASE WHEN ho.verdict = 'complied' THEN 1 ELSE 0 END
                    WHEN lr.verdict IS NOT NULL THEN CASE WHEN lr.verdict = 'complied' THEN 1 ELSE 0 END
                    WHEN c.is_refusal = 0 THEN 1
                    ELSE 0
                END) as complied,
                SUM(CASE
                    WHEN ho.verdict IS NOT NULL THEN 1
                    WHEN lr.verdict IS NOT NULL THEN 1
                    WHEN c.classifier IS NOT NULL THEN 1
                    ELSE 0
                END) as reviewed,
                SUM(CASE WHEN ho.verdict IS NOT NULL THEN 1 ELSE 0 END) as human_count,
                SUM(CASE WHEN lr.verdict IS NOT NULL AND ho.verdict IS NULL THEN 1 ELSE 0 END) as llm_count,
                SUM(CASE WHEN c.classifier IS NOT NULL AND lr.verdict IS NULL AND ho.verdict IS NULL THEN 1 ELSE 0 END) as cls_count
            FROM responses r
            LEFT JOIN v_latest_classification c ON r.model_name = c.model_name AND r.behavior_id = c.behavior_id
            LEFT JOIN llm_reviews lr ON r.model_name = lr.model_name AND r.behavior_id = lr.behavior_id
            LEFT JOIN human_overrides ho ON r.model_name = ho.model_name AND r.behavior_id = ho.behavior_id
            WHERE r.model_name = ?""",
            (mname,),
        ).fetchone()

        if row and row[0]:
            total = row[0]
            complied = row[1] or 0
            asr = complied / total * 100 if total else 0
            human = row[3] or 0
            llm = row[4] or 0
            cls = row[5] or 0
            source = f"h:{human} l:{llm} c:{cls}"
            print(f"{mname:<15} {total:>5} {complied:>8} {total - complied:>8} {asr:>6.1f}% {source:>12}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Abliterlitics review database")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="Create/import from a comparison directory")
    p_init.add_argument("comparison_dir", help="Path to comparison directory (e.g. comparisons/gemma4-e2b/)")
    p_init.add_argument("--db", default=DB_PATH, help="Database path")

    p_status = sub.add_parser("status", help="Print database status")
    p_status.add_argument("--db", default=DB_PATH, help="Database path")

    p_asr = sub.add_parser("asr", help="Print ASR from authoritative review chain")
    p_asr.add_argument("--model", default=None, help="Filter to specific model")
    p_asr.add_argument("--db", default=DB_PATH, help="Database path")

    p_export = sub.add_parser("export-review", help="Export items needing LLM review")
    p_export.add_argument("--model", default=None, help="Filter to specific model")
    p_export.add_argument("--output", default="review_batch.json", help="Output JSON path")
    p_export.add_argument("--confidence", default="non-high", choices=["non-high", "all"], help="Which items to export")
    p_export.add_argument("--limit", type=int, default=0, help="Max items to export (0 = unlimited)")
    p_export.add_argument("--db", default=DB_PATH, help="Database path")

    p_prompt = sub.add_parser("export-prompt", help="Export review items as ready-to-use LLM prompts")
    p_prompt.add_argument("--model", default=None, help="Filter to specific model")
    p_prompt.add_argument("--output", default="review_prompts.json", help="Output JSON path")
    p_prompt.add_argument("--batch-size", type=int, default=10, help="Items per LLM prompt batch")
    p_prompt.add_argument("--confidence", default="non-high", choices=["non-high", "all"], help="Which items to export")
    p_prompt.add_argument("--limit", type=int, default=0, help="Max items to export (0 = unlimited)")
    p_prompt.add_argument("--db", default=DB_PATH, help="Database path")

    p_import = sub.add_parser("import-reviews", help="Import LLM review results")
    p_import.add_argument("reviews_file", help="JSON file with review results")
    p_import.add_argument("--reviewer", required=True, help="Reviewer model name (e.g. gpt-4o-mini)")
    p_import.add_argument("--db", default=DB_PATH, help="Database path")

    args = parser.parse_args()

    if args.command == "init":
        conn = get_db(args.db)
        import_comparison(conn, args.comparison_dir)
        cmd_status(conn)
        conn.close()
    elif args.command == "status":
        conn = get_db(args.db)
        cmd_status(conn)
        conn.close()
    elif args.command == "asr":
        conn = get_db(args.db)
        cmd_asr(conn, getattr(args, "model", None))
        conn.close()
    elif args.command == "export-review":
        conn = get_db(args.db)
        cmd_export_review(conn, args.model, args.output, args.confidence, args.limit)
        conn.close()
    elif args.command == "export-prompt":
        conn = get_db(args.db)
        cmd_export_prompt(conn, args.model, args.output, args.batch_size, args.confidence, args.limit)
        conn.close()
    elif args.command == "import-reviews":
        conn = get_db(args.db)
        cmd_import_llm_reviews(conn, args.reviews_file, args.reviewer)
        cmd_status(conn)
        conn.close()


if __name__ == "__main__":
    main()
