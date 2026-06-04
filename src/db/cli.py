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

from src.db.core import DB_PATH, get_db, load_behaviors, now_iso
from src.db.pipeline import (
    get_all_stages,
    get_stale_stages,
    init_stages,
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
    """Import all HarmBench data for a comparison directory.

    Also registers the comparison in the comparisons table and
    pre-populates model_stages for all models × all stages.
    """
    base = Path(comparison_dir)
    comp_file = base / "comparison.json"
    if not comp_file.exists():
        log.error("No comparison.json found in %s", comparison_dir)
        return

    comp = json.loads(comp_file.read_text())
    comparison_name = comp.get("name", base.name)

    # --- Register comparison ---
    settings = comp.get("settings", {})
    variants = comp.get("variants", {})
    # Detect architecture from base model config if available
    architecture = ""
    base_path = comp.get("base", "")
    base_config = base / base_path / "config.json"
    if not base_config.exists():
        # base_path might be absolute or relative differently
        base_config = Path(base_path) / "config.json"
    if base_config.exists():
        try:
            from src.config import _detect_architecture

            arch_path = Path(base_path).resolve() if Path(base_path).is_absolute() else base.resolve() / base_path
            architecture = _detect_architecture(arch_path)
        except Exception:
            pass

    conn.execute(
        """INSERT INTO comparisons (name, base_model, architecture, num_variants, settings_json)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET
            base_model = excluded.base_model,
            architecture = excluded.architecture,
            num_variants = excluded.num_variants,
            settings_json = excluded.settings_json,
            updated_at = ?""",
        (comparison_name, base_path, architecture, len(variants), json.dumps(settings), now_iso()),
    )
    conn.commit()
    log.info("Registered comparison: %s (%d variants, arch=%s)", comparison_name, len(variants), architecture)

    # --- Import behaviors ---
    behaviors = load_behaviors()
    bcount = 0
    for b in behaviors:
        try:
            conn.execute(
                "INSERT OR IGNORE INTO behaviors "
                "(behavior_id, behavior_text, semantic_category, tags) "
                "VALUES (?, ?, ?, ?)",
                (b.get("BehaviorID", ""), b.get("Behavior", ""), b.get("SemanticCategory", ""), b.get("Tags", "")),
            )
            bcount += 1
        except sqlite3.Error:
            log.warning("Failed to import behavior: %s", b.get("BehaviorID", ""))
    conn.commit()
    log.info("Imported %d behaviors", bcount)

    # --- Import models ---
    base_name = comp.get("base", "").split("/")[-1] or "base"
    base_display = "Base"
    conn.execute(
        """INSERT INTO models (comparison, model_name, display_name, model_path, variant_type)
        VALUES (?, ?, ?, ?, 'base')
        ON CONFLICT(comparison, model_name) DO UPDATE SET
            display_name = excluded.display_name,
            model_path = excluded.model_path""",
        (comparison_name, base_name, base_display, comp.get("base", "")),
    )

    for vname, vinfo in variants.items():
        path = vinfo.get("path", "")
        display = vinfo.get("display_name", vname)
        conn.execute(
            """INSERT INTO models (comparison, model_name, display_name, model_path, variant_type)
            VALUES (?, ?, ?, ?, 'variant')
            ON CONFLICT(comparison, model_name) DO UPDATE SET
                display_name = excluded.display_name,
                model_path = excluded.model_path""",
            (comparison_name, vname, display, path),
        )
    conn.commit()
    log.info("Imported %d models (1 base + %d variants)", 1 + len(variants), len(variants))

    # --- Pre-populate model_stages ---
    all_model_names = [base_name] + list(variants.keys())
    init_stages(conn, comparison_name, all_model_names)

    # --- Import responses and classifications ---
    harmbench_dir = base / "results" / "harmbench"
    classified_dir = base / "results" / "harmbench_classified"

    if not harmbench_dir.exists():
        log.info("No harmbench results found in %s", harmbench_dir)
        return

    response_files = sorted(harmbench_dir.glob("harmbench_*_responses.json"))
    total_responses = 0
    total_classified = 0

    for rf in response_files:
        model_name = rf.stem.replace("harmbench_", "").replace("_responses", "")
        model_key = model_name.replace("_", "-")

        # Find the actual model_name in our models table
        actual_model = model_name
        if conn.execute(
            "SELECT 1 FROM models WHERE comparison = ? AND model_name = ?",
            (comparison_name, model_name),
        ).fetchone():
            actual_model = model_name
        elif conn.execute(
            "SELECT 1 FROM models WHERE comparison = ? AND model_name = ?",
            (comparison_name, model_key),
        ).fetchone():
            actual_model = model_key
        else:
            log.warning("Model %s not found in comparison.json, adding anyway", model_name)
            actual_model = model_name
            conn.execute(
                """INSERT INTO models (comparison, model_name, display_name, model_path, variant_type)
                VALUES (?, ?, '', '', 'variant')
                ON CONFLICT(comparison, model_name) DO NOTHING""",
                (comparison_name, model_name),
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
                """INSERT INTO responses
                (comparison, model_name, behavior_id, response_text, reasoning_text,
                 completion_tokens, finish_reason, error, is_refusal_kw)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(comparison, model_name, behavior_id) DO UPDATE SET
                    response_text = excluded.response_text,
                    reasoning_text = excluded.reasoning_text,
                    completion_tokens = excluded.completion_tokens,
                    finish_reason = excluded.finish_reason,
                    error = excluded.error,
                    is_refusal_kw = excluded.is_refusal_kw""",
                (
                    comparison_name,
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
            cls_file = classified_dir / f"{model_key}_classified.json"

        if cls_file.exists():
            cls = json.loads(cls_file.read_text())
            classifier = cls.get("classifier", "unknown")

            for item in cls.get("harmbench_classified", []):
                bid = item.get("behavior_id", "")
                label = "refused" if item.get("is_refusal") else "not_refused"
                is_ref = 1 if item.get("is_refusal") else 0

                conn.execute(
                    """INSERT INTO classifications
                    (comparison, model_name, behavior_id, classifier, label, is_refusal,
                     layer, reason, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(comparison, model_name, behavior_id, classifier) DO UPDATE SET
                        label = excluded.label,
                        is_refusal = excluded.is_refusal,
                        layer = excluded.layer,
                        reason = excluded.reason,
                        confidence = excluded.confidence""",
                    (
                        comparison_name,
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


if __name__ == "__main__":
    main()
