"""Abliterlitics SQLite database — CLI entry point.

All subcommands for database management: init, status, asr, export-review,
export-prompt, import-reviews (existing), plus stage-start, stage-complete,
stage-fail, stage-skip, pipeline-status (new).
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from src.db.core import DB_PATH, get_db
from src.db.pipeline import (
    get_all_stages,
    get_stale_stages,
    stage_complete,
    stage_fail,
    stage_skip,
    stage_start,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Existing subcommands (adapted for composite PK)
# ---------------------------------------------------------------------------


def import_comparison(conn: sqlite3.Connection, comparison_dir: str) -> None:
    """Import all data for a comparison directory.

    Delegates to ``src.db.importers.import_comparison()`` which handles:
    comparison registration, model import, behavior import,
    HarmBench responses/classifications, pipeline stage init,
    and result file imports.
    """
    from src.db.importers import import_comparison as _import

    _import(conn, comparison_dir)


def cmd_status(conn: sqlite3.Connection, comparison: str | None = None) -> None:
    """Print database status summary."""
    where = "WHERE comparison = ?" if comparison else ""
    params: list[str] = [comparison] if comparison else []

    models = conn.execute(
        f"SELECT comparison, model_name, display_name, variant_type "
        f"FROM models {where} "
        f"ORDER BY comparison, variant_type, model_name",
        params,
    ).fetchall()

    comparisons = conn.execute("SELECT name, architecture, num_variants FROM comparisons ORDER BY name").fetchall()
    if comparisons:
        print("Comparisons:")
        for c in comparisons:
            print(f"  {c[0]:<20} arch={c[1] or '?':<10} variants={c[2]}")

    print(f"\nModels: {len(models)}")
    for m in models:
        print(f"  [{m[0]}] {m[1]:<15} {m[2]:<30} {m[3]}")

    behaviors = conn.execute("SELECT COUNT(*) FROM behaviors").fetchone()[0]
    print(f"\nBehaviors: {behaviors}")

    responses = conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
    print(f"Responses: {responses}")

    classifiers = conn.execute("SELECT DISTINCT classifier FROM classifications").fetchall()
    print(f"Classifiers: {[c[0] for c in classifiers]}")

    # Per-model HarmBench summary from view
    print(f"\n{'Comparison':<16} {'Model':<15} {'Total':>5} {'Complied':>8} {'Refused':>8} {'ASR':>7}")
    print("-" * 65)

    summary_where = "WHERE comparison = ?" if comparison else ""
    summary_params = [comparison] if comparison else []
    rows = conn.execute(
        f"SELECT comparison, model_name, total_behaviors, complied, refused, asr_pct "
        f"FROM v_harmbench_summary {summary_where} "
        f"ORDER BY comparison, model_name",
        summary_params,
    ).fetchall()
    for r in rows:
        print(f"{r[0]:<16} {r[1]:<15} {r[2]:>5} {r[3]:>8} {r[4]:>8} {r[5]:>6.1f}%")


def cmd_asr(conn: sqlite3.Connection, comparison: str | None = None, model: str | None = None) -> None:
    """Print ASR from authoritative review chain."""
    where_parts: list[str] = []
    params: list[str] = []

    if comparison:
        where_parts.append("m.comparison = ?")
        params.append(comparison)
    if model:
        where_parts.append("r.model_name = ?")
        params.append(model)

    where_clause = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""

    rows = conn.execute(
        f"""SELECT m.comparison, r.model_name, COUNT(*) as total,
               SUM(CASE
                   WHEN ho.verdict IS NOT NULL
                       THEN CASE WHEN ho.verdict = 'complied' THEN 1 ELSE 0 END
                   WHEN lr.verdict IS NOT NULL
                       THEN CASE WHEN lr.verdict = 'complied' THEN 1 ELSE 0 END
                   WHEN c.is_refusal = 0 THEN 1
                   ELSE 0
               END) as complied,
               SUM(CASE WHEN ho.verdict IS NOT NULL THEN 1 ELSE 0 END) AS human_count,
               SUM(CASE WHEN lr.verdict IS NOT NULL AND ho.verdict IS NULL
                   THEN 1 ELSE 0 END) AS llm_count,
               SUM(CASE WHEN c.classifier IS NOT NULL
                   AND lr.verdict IS NULL AND ho.verdict IS NULL
                   THEN 1 ELSE 0 END) AS cls_count
        FROM responses r
        JOIN models m ON r.comparison = m.comparison
            AND r.model_name = m.model_name
        LEFT JOIN v_latest_classification c
            ON r.comparison = c.comparison
            AND r.model_name = c.model_name
            AND r.behavior_id = c.behavior_id
        LEFT JOIN llm_reviews lr
            ON r.comparison = lr.comparison
            AND r.model_name = lr.model_name
            AND r.behavior_id = lr.behavior_id
        LEFT JOIN human_overrides ho
            ON r.comparison = ho.comparison
            AND r.model_name = ho.model_name
            AND r.behavior_id = ho.behavior_id
        {where_clause}
        GROUP BY m.comparison, r.model_name""",
        params,
    ).fetchall()

    print(f"{'Comparison':<16} {'Model':<15} {'Total':>5} {'Complied':>8} {'ASR':>7} {'Source':>12}")
    print("-" * 65)
    for r in rows:
        total = r[2]
        complied = r[3] or 0
        asr = complied / total * 100 if total else 0
        human = r[4] or 0
        llm = r[5] or 0
        cls = r[6] or 0
        source = f"h:{human} l:{llm} c:{cls}"
        print(f"{r[0]:<16} {r[1]:<15} {total:>5} {complied:>8} {asr:>6.1f}% {source:>12}")


def cmd_export_review(
    conn: sqlite3.Connection,
    comparison: str | None = None,
    model: str | None = None,
    output: str = "review_batch.json",
    confidence: str = "non-high",
    limit: int = 0,
) -> None:
    """Export items needing LLM review as a JSON batch."""
    where_parts = ["(lr.verdict IS NULL AND ho.verdict IS NULL)"]
    params: list[str] = []
    if comparison:
        where_parts.append("r.comparison = ?")
        params.append(comparison)
    if model:
        where_parts.append("r.model_name = ?")
        params.append(model)
    if confidence == "non-high":
        where_parts.append("c.confidence != 'high'")

    limit_clause = f"LIMIT {limit}" if limit > 0 else ""

    rows = conn.execute(
        f"""SELECT r.comparison, r.model_name, r.behavior_id,
               b.behavior_text, b.semantic_category,
               r.response_text, r.reasoning_text, r.completion_tokens,
               c.label, c.layer, c.reason, c.confidence, c.classifier
        FROM responses r
        JOIN behaviors b ON r.behavior_id = b.behavior_id
        LEFT JOIN v_latest_classification c
            ON r.comparison = c.comparison
            AND r.model_name = c.model_name
            AND r.behavior_id = c.behavior_id
        LEFT JOIN llm_reviews lr
            ON r.comparison = lr.comparison
            AND r.model_name = lr.model_name
            AND r.behavior_id = lr.behavior_id
        LEFT JOIN human_overrides ho
            ON r.comparison = ho.comparison
            AND r.model_name = ho.model_name
            AND r.behavior_id = ho.behavior_id
        WHERE {" AND ".join(where_parts)}
        ORDER BY CASE c.confidence
            WHEN 'low' THEN 0
            WHEN 'medium' THEN 1
            WHEN 'high' THEN 2
            ELSE 3
        END, r.comparison, r.model_name, r.behavior_id
        {limit_clause}""",
        params,
    ).fetchall()

    items = []
    for r in rows:
        items.append(
            {
                "comparison": r[0],
                "model_name": r[1],
                "behavior_id": r[2],
                "behavior_text": r[3][:200],
                "semantic_category": r[4],
                "response_text": r[5],
                "reasoning_text": r[6][:3000] if r[6] else "",
                "completion_tokens": r[7],
                "classifier_label": r[8],
                "classifier_layer": r[9],
                "classifier_reason": r[10],
                "classifier_confidence": r[11],
                "classifier": r[12],
            }
        )

    Path(output).write_text(json.dumps(items, indent=2))
    log.info("Exported %d items to %s", len(items), output)


def cmd_import_llm_reviews(
    conn: sqlite3.Connection,
    reviews_file: str,
    reviewer_model: str,
    comparison: str | None = None,
) -> None:
    """Import LLM review results from a JSON file."""
    reviews = json.loads(Path(reviews_file).read_text())
    count = 0
    for r in reviews:
        comp = comparison or r.get("comparison", "")
        conn.execute(
            """INSERT INTO llm_reviews
            (comparison, model_name, behavior_id, reviewer_model, verdict, review_reasoning)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(comparison, model_name, behavior_id, reviewer_model) DO UPDATE SET
                verdict = excluded.verdict,
                review_reasoning = excluded.review_reasoning""",
            (
                comp,
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


def cmd_pipeline_status(conn: sqlite3.Connection, comparison_dir: str) -> None:
    """Show rich pipeline status for a comparison directory."""
    base = Path(comparison_dir)
    comp_file = base / "comparison.json"
    if not comp_file.exists():
        log.error("No comparison.json found in %s", comparison_dir)
        return

    comp = json.loads(comp_file.read_text())
    comparison_name = comp.get("name", base.name)

    # Check if comparison is registered
    comp_row = conn.execute(
        "SELECT name, architecture, num_variants FROM comparisons WHERE name = ?",
        (comparison_name,),
    ).fetchone()

    if not comp_row:
        print(f"Comparison '{comparison_name}' not found in database. Run 'init' first.")
        return

    print(f"Comparison: {comp_row[0]}")
    print(f"Architecture: {comp_row[1] or 'unknown'}")
    print(f"Variants: {comp_row[2]}")

    # Pipeline stages
    stages = get_all_stages(conn, comparison_name)
    if not stages:
        print("\nNo stages tracked. Run 'init' first.")
        return

    print(f"\n{'Model':<15} {'Stage':<12} {'Status':<10} {'Duration':>8} {'Error'}")
    print("-" * 75)

    for s in stages:
        duration_str = f"{s.duration_secs}s" if s.duration_secs is not None else "-"
        error_str = (s.error_message or "")[:40] if s.status == "failed" else ""
        print(f"{s.model_name:<15} {s.stage:<12} {s.status:<10} {duration_str:>8} {error_str}")

    # Overall progress
    total = len(stages)
    completed = sum(1 for s in stages if s.status in ("completed", "skipped"))
    failed = sum(1 for s in stages if s.status == "failed")
    running = sum(1 for s in stages if s.status == "running")

    print(f"\nProgress: {completed}/{total} stages completed ({completed * 100 // total if total else 0}%)")
    if failed:
        print(f"  Failures: {failed}")
    if running:
        print(f"  Running: {running}")

    # Stale stage detection
    stale = get_stale_stages(conn, max_age_hours=2)
    stale_for_comp = [s for s in stale if s.comparison == comparison_name]
    if stale_for_comp:
        print("\nStale running stages (>2h, likely crashed):")
        for s in stale_for_comp:
            print(f"  {s.model_name}/{s.stage} (started: {s.started_at})")

    # Quick HarmBench summary
    hb_rows = conn.execute(
        "SELECT model_name, total_behaviors, asr_pct FROM v_harmbench_summary WHERE comparison = ? ORDER BY model_name",
        (comparison_name,),
    ).fetchall()
    if hb_rows:
        print("\nHarmBench ASR:")
        for r in hb_rows:
            print(f"  {r[0]:<15} {r[1]:>5} behaviors, ASR={r[2]:.1f}%")

    # Quick KL summary
    kl_rows = conn.execute(
        "SELECT variant, kl_batchmean, rating FROM kl_results WHERE comparison = ? ORDER BY variant",
        (comparison_name,),
    ).fetchall()
    if kl_rows:
        print("\nKL Divergence:")
        for r in kl_rows:
            print(f"  {r[0]:<15} {r[1]:.4f} ({r[2] or '?'})")

    # Quick benchmark summary (MMLU + GSM8K)
    bm_rows = conn.execute(
        """SELECT model_name, task, metric_value
        FROM lm_eval_results
        WHERE comparison = ? AND task IN ('mmlu', 'gsm8k')
          AND metric_name IN ('acc,none', 'exact_match,strict-match')
        ORDER BY model_name, task""",
        (comparison_name,),
    ).fetchall()
    if bm_rows:
        print("\nKey Benchmarks:")
        for r in bm_rows:
            print(f"  {r[0]:<15} {r[1]:<10} {r[2]:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Abliterlitics pipeline database")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- init ---
    p_init = sub.add_parser("init", help="Create/import from a comparison directory")
    p_init.add_argument("comparison_dir", help="Path to comparison directory")
    p_init.add_argument("--db", default=DB_PATH, help="Database path")

    # --- status ---
    p_status = sub.add_parser("status", help="Print database status")
    p_status.add_argument("--comparison", default=None, help="Filter to specific comparison")
    p_status.add_argument("--db", default=DB_PATH, help="Database path")

    # --- asr ---
    p_asr = sub.add_parser("asr", help="Print ASR from authoritative review chain")
    p_asr.add_argument("--comparison", default=None, help="Filter to specific comparison")
    p_asr.add_argument("--model", default=None, help="Filter to specific model")
    p_asr.add_argument("--db", default=DB_PATH, help="Database path")

    # --- export-review ---
    p_export = sub.add_parser("export-review", help="Export items needing LLM review")
    p_export.add_argument("--comparison", default=None, help="Filter to specific comparison")
    p_export.add_argument("--model", default=None, help="Filter to specific model")
    p_export.add_argument("--output", default="review_batch.json", help="Output JSON path")
    p_export.add_argument("--confidence", default="non-high", choices=["non-high", "all"])
    p_export.add_argument("--limit", type=int, default=0)
    p_export.add_argument("--db", default=DB_PATH, help="Database path")

    # --- export-prompt ---
    p_prompt = sub.add_parser("export-prompt", help="Export review items as LLM prompts")
    p_prompt.add_argument("--comparison", default=None, help="Filter to specific comparison")
    p_prompt.add_argument("--model", default=None, help="Filter to specific model")
    p_prompt.add_argument("--output", default="review_prompts.json")
    p_prompt.add_argument("--batch-size", type=int, default=10)
    p_prompt.add_argument("--confidence", default="non-high", choices=["non-high", "all"])
    p_prompt.add_argument("--limit", type=int, default=0)
    p_prompt.add_argument("--db", default=DB_PATH, help="Database path")

    # --- import-reviews ---
    p_import = sub.add_parser("import-reviews", help="Import LLM review results")
    p_import.add_argument("reviews_file", help="JSON file with review results")
    p_import.add_argument("--reviewer", required=True, help="Reviewer model name")
    p_import.add_argument("--comparison", default=None, help="Comparison name")
    p_import.add_argument("--db", default=DB_PATH, help="Database path")

    # --- import-results ---
    p_ires = sub.add_parser("import-results", help="Import result files (lm-eval, KL, weights)")
    p_ires.add_argument("comparison_dir", help="Path to comparison directory")
    p_ires.add_argument("--db", default=DB_PATH, help="Database path")

    # --- pipeline-status ---
    p_ps = sub.add_parser("pipeline-status", help="Show rich pipeline status for a comparison")
    p_ps.add_argument("comparison_dir", help="Path to comparison directory")
    p_ps.add_argument("--db", default=DB_PATH, help="Database path")

    # --- stage-start ---
    p_ss = sub.add_parser("stage-start", help="Mark a stage as running")
    p_ss.add_argument("--comparison", required=True)
    p_ss.add_argument("--model", required=True)
    p_ss.add_argument("--stage", required=True)
    p_ss.add_argument("--db", default=DB_PATH)

    # --- stage-complete ---
    p_sc = sub.add_parser("stage-complete", help="Mark a stage as completed")
    p_sc.add_argument("--comparison", required=True)
    p_sc.add_argument("--model", required=True)
    p_sc.add_argument("--stage", required=True)
    p_sc.add_argument("--result-path", default=None)
    p_sc.add_argument("--metadata", default=None, help="JSON metadata string")
    p_sc.add_argument("--db", default=DB_PATH)

    # --- stage-fail ---
    p_sf = sub.add_parser("stage-fail", help="Mark a stage as failed")
    p_sf.add_argument("--comparison", required=True)
    p_sf.add_argument("--model", required=True)
    p_sf.add_argument("--stage", required=True)
    p_sf.add_argument("--error", required=True)
    p_sf.add_argument("--db", default=DB_PATH)

    # --- stage-skip ---
    p_sk = sub.add_parser("stage-skip", help="Mark a stage as skipped")
    p_sk.add_argument("--comparison", required=True)
    p_sk.add_argument("--model", required=True)
    p_sk.add_argument("--stage", required=True)
    p_sk.add_argument("--db", default=DB_PATH)

    # --- journal (inter-agent communication + project memory) ---
    # Import journal constants here to avoid top-of-file coupling
    from src.db.core import JOURNAL_CATEGORIES, JOURNAL_SEVERITIES

    p_journal = sub.add_parser("journal", help="Read/write the project journal")
    journal_sub = p_journal.add_subparsers(dest="journal_command", required=True)

    # journal add
    p_ja = journal_sub.add_parser("add", help="Append a journal entry")
    p_ja.add_argument("--agent", required=True, help="Writing agent name (init/lm-eval/kl/...)")
    p_ja.add_argument("--category", required=True, choices=JOURNAL_CATEGORIES, help="Entry category")
    p_ja.add_argument("--severity", default="info", choices=JOURNAL_SEVERITIES, help="Entry severity (default: info)")
    p_ja.add_argument("--subject", required=True, help="Short tag (chat_template, thinking_mode, ...)")
    p_ja.add_argument("--title", required=True, help="One-line summary")
    p_ja.add_argument("--body", default="", help="Markdown body (optional)")
    p_ja.add_argument("--comparison", default=None, help="Comparison name (omit for project-wide)")
    p_ja.add_argument("--db", default=DB_PATH)

    # journal list
    p_jl = journal_sub.add_parser("list", help="List entries (newest first)")
    p_jl.add_argument("--comparison", default=None)
    p_jl.add_argument("--agent", default=None)
    p_jl.add_argument("--category", default=None, choices=JOURNAL_CATEGORIES)
    p_jl.add_argument("--subject", default=None)
    p_jl.add_argument("--severity", default=None, choices=JOURNAL_SEVERITIES)
    p_jl.add_argument("--limit", type=int, default=50)
    p_jl.add_argument("--db", default=DB_PATH)

    # journal latest
    p_jlat = journal_sub.add_parser("latest", help="Latest entry per subject")
    p_jlat.add_argument("--comparison", default=None)
    p_jlat.add_argument("--db", default=DB_PATH)

    # journal search
    p_js = journal_sub.add_parser("search", help="Full-text search title + body")
    p_js.add_argument("query", help="Search query")
    p_js.add_argument("--comparison", default=None)
    p_js.add_argument("--limit", type=int, default=50)
    p_js.add_argument("--db", default=DB_PATH)

    args = parser.parse_args()

    if args.command == "init":
        conn = get_db(args.db)
        import_comparison(conn, args.comparison_dir)
        cmd_status(conn)
        conn.close()

    elif args.command == "status":
        conn = get_db(args.db)
        cmd_status(conn, getattr(args, "comparison", None))
        conn.close()

    elif args.command == "asr":
        conn = get_db(args.db)
        cmd_asr(conn, getattr(args, "comparison", None), getattr(args, "model", None))
        conn.close()

    elif args.command == "export-review":
        conn = get_db(args.db)
        cmd_export_review(conn, args.comparison, args.model, args.output, args.confidence, args.limit)
        conn.close()

    elif args.command == "export-prompt":
        conn = get_db(args.db)
        # NOTE: Prompt building is not yet implemented in this package.
        # For now, this delegates to export-review (raw review items).
        # Use src/benchmark/reviewer_prompt.py for interactive prompt building.
        if args.batch_size != 10:
            log.warning("--batch-size is not yet supported by export-prompt; delegating to export-review")
        cmd_export_review(conn, args.comparison, args.model, args.output, args.confidence, args.limit)
        conn.close()

    elif args.command == "import-reviews":
        conn = get_db(args.db)
        cmd_import_llm_reviews(conn, args.reviews_file, args.reviewer, getattr(args, "comparison", None))
        cmd_status(conn)
        conn.close()

    elif args.command == "import-results":
        conn = get_db(args.db)
        from src.db.importers import import_all_results

        counts = import_all_results(conn, args.comparison_dir)
        if counts:
            print("Imported:")
            for k, v in counts.items():
                print(f"  {k}: {v} rows")
        else:
            print("No results found to import.")
        conn.close()

    elif args.command == "pipeline-status":
        conn = get_db(args.db)
        cmd_pipeline_status(conn, args.comparison_dir)
        conn.close()

    elif args.command == "stage-start":
        conn = get_db(args.db)
        stage_start(conn, args.comparison, args.model, args.stage)
        conn.close()

    elif args.command == "stage-complete":
        conn = get_db(args.db)
        metadata = json.loads(args.metadata) if args.metadata else None
        stage_complete(conn, args.comparison, args.model, args.stage, args.result_path, metadata)
        conn.close()

    elif args.command == "stage-fail":
        conn = get_db(args.db)
        stage_fail(conn, args.comparison, args.model, args.stage, args.error)
        conn.close()

    elif args.command == "stage-skip":
        conn = get_db(args.db)
        stage_skip(conn, args.comparison, args.model, args.stage)
        conn.close()

    elif args.command == "journal":
        handle_journal_command(args)


def handle_journal_command(args: argparse.Namespace) -> None:
    """Dispatch for the journal subcommand tree.

    Subcommands:
      add       Append a journal entry.
      list      List entries (newest first).
      latest    Latest entry per subject (optionally per comparison).
      search    Full-text search title + body.
    """
    from src.db.core import (
        journal_add,
        journal_latest,
        journal_list,
        journal_search,
    )

    sub = getattr(args, "journal_command", None)
    conn = get_db(args.db)

    try:
        if sub == "add":
            entry_id = journal_add(
                conn,
                agent=args.agent,
                category=args.category,
                severity=args.severity,
                subject=args.subject,
                title=args.title,
                body=args.body or "",
                comparison=args.comparison,
            )
            print(f"OK journal entry id={entry_id}")

        elif sub == "list":
            rows = journal_list(
                conn,
                comparison=args.comparison,
                agent=args.agent,
                category=args.category,
                subject=args.subject,
                severity=args.severity,
                limit=args.limit,
            )
            _print_journal_rows(rows)

        elif sub == "latest":
            rows = journal_latest(conn, comparison=args.comparison)
            if not rows:
                print("(no journal entries)")
                return
            print(f"{'Subject':<25} {'Category':<11} {'Severity':<9} {'Agent':<10} {'Timestamp':<21} Title")
            print("-" * 120)
            for r in rows:
                ts = r[7][:19].replace("T", " ") if r[7] else ""
                print(f"{(r[1] or '-'):<25} {r[3]:<11} {r[4]:<9} {r[2]:<10} {ts:<21} {r[5]}")

        elif sub == "search":
            rows = journal_search(
                conn,
                args.query,
                comparison=args.comparison,
                limit=args.limit,
            )
            _print_journal_rows(rows)

        else:
            print(f"Unknown journal subcommand: {sub!r}")
            print("Available: add, list, latest, search")
    finally:
        conn.close()


def _print_journal_rows(rows: list[Any]) -> None:
    """Pretty-print a list of journal rows."""
    if not rows:
        print("(no journal entries match)")
        return
    print(
        f"{'ID':<5} {'Timestamp':<21} {'Comparison':<14} {'Agent':<10} "
        f"{'Category':<10} {'Severity':<9} {'Subject':<20} Title"
    )
    print("-" * 140)
    for r in rows:
        # row layout: id, timestamp, comparison, agent, category, severity, subject, title, body
        ts = r[1][:19].replace("T", " ") if r[1] else ""
        comp = (r[2] or "-")[:13]
        print(f"{r[0]:<5} {ts:<21} {comp:<14} {r[3]:<10} {r[4]:<10} {r[5]:<9} {(r[6] or '-')[:19]:<20} {r[7]}")


if __name__ == "__main__":
    main()
