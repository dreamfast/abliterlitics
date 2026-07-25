"""Abliterlitics — Markdown table generation from SQLite database.

Standalone module (not in ``src/db/`` — this is a report consumer, not DB
infrastructure).  Queries the database and returns formatted markdown tables
with best values bolded.

CLI: ``python3 -m src.db_tables <command> <comparison>``
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from typing import Any

from src.db.core import DB_PATH, get_db

log = logging.getLogger(__name__)

# Re-use the same primary metric mapping from importers.
# Key: lm-eval task name (as it appears in results dict).
# Value: list of metric names to import (e.g. "acc,none", "exact_match,strict-match").
# Ordered by category: v2 OLL suite first, then v1 supplement, then generative tasks.
TASK_PRIMARY_METRICS: dict[str, list[str]] = {
    # v2 Open LLM Leaderboard suite (aggregates only, subtasks are filtered)
    "leaderboard_mmlu_pro": ["acc,none"],
    "leaderboard_gpqa_diamond": ["acc_norm,none"],
    "leaderboard_bbh": ["acc_norm,none"],
    "leaderboard_musr": ["acc_norm,none"],
    "leaderboard_ifeval": ["prompt_level_strict_acc,none"],
    # v1 forensic supplement
    "hellaswag": ["acc_norm,none"],
    "arc_challenge": ["acc_norm,none"],
    "winogrande": ["acc,none"],
    "piqa": ["acc_norm,none"],
    "lambada_openai": ["perplexity,none"],
    # Legacy MMLU (kept for backwards compat with older comparisons)
    "mmlu": ["acc,none"],
    # Generative tasks
    "gsm8k": ["exact_match,strict-match", "exact_match,flexible-extract"],
    "truthfulqa_mc1": ["acc,none"],
    "truthfulqa_mc2": ["acc,none"],
    "truthfulqa_gen": ["bleu_acc,none"],
    # HumanEval (dead under chat-template scoring but tracked for completeness)
    "humaneval": ["pass@1,none"],
}

# Task display names for tables.
TASK_DISPLAY_NAMES: dict[str, str] = {
    "leaderboard_mmlu_pro": "MMLU-Pro",
    "leaderboard_gpqa_diamond": "GPQA Diamond",
    "leaderboard_bbh": "BBH",
    "leaderboard_musr": "MUSR",
    "leaderboard_ifeval": "IFEval",
    "hellaswag": "HellaSwag",
    "arc_challenge": "ARC-C",
    "winogrande": "WinoGrande",
    "piqa": "PIQA",
    "lambada_openai": "Lambada",
    "mmlu": "MMLU",
    "gsm8k": "GSM8K",
    "truthfulqa_mc1": "TQA-MC1",
    "truthfulqa_mc2": "TQA-MC2",
    "truthfulqa_gen": "TQA-Gen",
    "humaneval": "HumanEval",
}

# HarmBench category display names.
CATEGORY_DISPLAY_NAMES: dict[str, str] = {
    "chemical_biological": "Chemical/Bio",
    "copyright": "Copyright",
    "cybercrime_intrusion": "Cybercrime",
    "harassment_bullying": "Harassment",
    "harmful": "Harmful Content",
    "illegal": "Illegal Activity",
    "misinformation_disinformation": "Misinformation",
}

# Metrics where lower is better.
LOWER_IS_BETTER: set[str] = {"perplexity,none"}

# Tasks excluded from the headline benchmark table because chat-template scoring
# breaks them (AGENTS.md lesson #17): they are raw-completion tasks that an
# instruction-tuned model answers conversationally when wrapped as a chat turn.
# Their numbers are meaningless under our methodology and would mislead readers.
# They remain in the DB and can be queried directly if needed.
DEAD_TASKS: set[str] = {"lambada_openai", "humaneval"}


# ---------------------------------------------------------------------------
# Core helper
# ---------------------------------------------------------------------------


def _bold_best(values: list[str], direction: str) -> list[str]:
    """Bold the best value in a list of formatted strings.

    Args:
        values: Formatted string values (e.g. ["0.4076", "0.4080", "0.4060"]).
        direction: ``"higher"`` or ``"lower"`` — which direction is better.

    Returns:
        List with the best value wrapped in ``**...**``. Ties are NOT bolded.
    """
    if not values or len(values) < 2:
        return list(values)

    # Parse numeric values for comparison
    parsed: list[float | None] = []
    for v in values:
        try:
            parsed.append(float(v.replace("%", "").replace(",", "")))
        except (ValueError, AttributeError):
            parsed.append(None)

    # Find the best numeric value
    numeric = [(i, v) for i, v in enumerate(parsed) if v is not None]
    if len(numeric) < 2:
        return list(values)

    best_val = max(v for _, v in numeric) if direction == "higher" else min(v for _, v in numeric)

    # Check for ties
    best_indices = [i for i, v in numeric if v == best_val]
    if len(best_indices) != 1:
        return list(values)  # Tie — don't bold

    result = list(values)
    idx = best_indices[0]
    result[idx] = f"**{result[idx]}**"
    return result


def _fmt(value: float | None, fmt: str = ".4f") -> str:
    """Format a numeric value, returning '-' for None."""
    if value is None:
        return "-"
    return f"{value:{fmt}}"


def _fmt_pct(value: float | None) -> str:
    """Format as percentage string."""
    if value is None:
        return "-"
    return f"{value:.1f}%"


# ---------------------------------------------------------------------------
# Table generators
# ---------------------------------------------------------------------------


def _get_models(conn: sqlite3.Connection, comparison: str) -> list[tuple[str, str, str]]:
    """Get (model_name, display_name, variant_type) for a comparison."""
    return conn.execute(
        "SELECT model_name, display_name, variant_type FROM models "
        "WHERE comparison = ? ORDER BY variant_type, model_name",
        (comparison,),
    ).fetchall()


def _get_metric_direction(metric_name: str) -> str:
    """Return 'higher' or 'lower' for a metric."""
    return "lower" if metric_name in LOWER_IS_BETTER else "higher"


def generate_benchmark_table(conn: sqlite3.Connection, comparison: str, max_cols: int = 0) -> str:
    """Generate main benchmark comparison table.

    Columns: Task | Base | Variant1 | Variant2 | ...
    Bolds highest per row (except perplexity — bolds lowest).

    When ``max_cols > 0``, splits into multiple sub-tables of at most
    ``max_cols`` model columns each. Base always appears in every sub-table.
    """
    models = _get_models(conn, comparison)
    if not models:
        return "_No models found for this comparison._"

    # Split models into chunks if max_cols is set
    if max_cols > 0 and len(models) > max_cols:
        base = [m for m in models if m[2] == "base"]
        variants = [m for m in models if m[2] == "variant"]
        chunks: list[list[tuple[str, str, str]]] = []
        for i in range(0, len(variants), max_cols - 1 if base else max_cols):
            chunk = base + variants[i : i + (max_cols - 1 if base else max_cols)]
            chunks.append(chunk)
        tables = []
        for idx, chunk in enumerate(chunks):
            f" (part {idx + 1}/{len(chunks)})" if len(chunks) > 1 else ""
            t = _generate_benchmark_table_for_models(conn, comparison, chunk)
            tables.append(t)
        return "\n\n".join(tables)
    return _generate_benchmark_table_for_models(conn, comparison, models)


def _generate_benchmark_table_for_models(
    conn: sqlite3.Connection,
    comparison: str,
    models: list[tuple[str, str, str]],
) -> str:
    """Internal: generate benchmark table for a specific subset of models."""
    if not models:
        return "_No models found for this comparison._"

    model_names = [m[0] for m in models]
    display_names = [m[1] for m in models]

    # Collect all tasks that have data
    tasks_with_data: list[tuple[str, str]] = []  # (task, metric_name)
    for task, metrics in TASK_PRIMARY_METRICS.items():
        if task in DEAD_TASKS:
            continue
        for metric in metrics:
            row = conn.execute(
                "SELECT 1 FROM lm_eval_results WHERE comparison = ? AND task = ? AND metric_name = ?",
                (comparison, task, metric),
            ).fetchone()
            if row:
                tasks_with_data.append((task, metric))
                break  # One metric per task is enough for detection

    if not tasks_with_data:
        return "_No benchmark results found._"

    # Build header
    header = "| Task | " + " | ".join(display_names) + " |"
    sep = "|" + "|".join(["------"] + ["------" for _ in models]) + "|"

    lines = [header, sep]

    # Build rows
    for task, primary_metric in tasks_with_data:
        # Get the actual primary metric that has data
        metric_name: str = primary_metric
        for cand_metric in TASK_PRIMARY_METRICS.get(task, []):
            if conn.execute(
                "SELECT 1 FROM lm_eval_results WHERE comparison = ? AND task = ? AND metric_name = ?",
                (comparison, task, cand_metric),
            ).fetchone():
                metric_name = cand_metric
                break

        values: list[float | None] = []
        for mn in model_names:
            row = conn.execute(
                "SELECT metric_value FROM lm_eval_results "
                "WHERE comparison = ? AND model_name = ? AND task = ? AND metric_name = ?",
                (comparison, mn, task, metric_name),
            ).fetchone()
            values.append(row[0] if row else None)

        is_perplexity = "perplexity" in metric_name
        if is_perplexity:
            str_values = [_fmt(v, ".1f") for v in values]
        else:
            # Show as percentage (multiply by 100)
            str_values = [f"{v * 100:.2f}%" if v is not None else "-" for v in values]
        direction = _get_metric_direction(metric_name)
        bolded = _bold_best(str_values, direction)

        display_task = TASK_DISPLAY_NAMES.get(task, task)
        lines.append(f"| {display_task} | " + " | ".join(bolded) + " |")

    return "\n".join(lines)


def generate_benchmark_delta_table(conn: sqlite3.Connection, comparison: str, max_cols: int = 0) -> str:
    """Generate delta-vs-base benchmark table.

    Columns: Task | Variant1 | Variant2 | ...
    Shows difference from base model (e.g. +0.0004, -0.0016).

    When ``max_cols > 0``, splits into multiple sub-tables of at most
    ``max_cols`` variant columns each.
    """
    models = _get_models(conn, comparison)
    if not models:
        return "_No models found for this comparison._"

    # Split variants into chunks if max_cols is set
    variant_models = [m for m in models if m[2] == "variant"]
    if max_cols > 0 and len(variant_models) > max_cols:
        chunks = []
        for i in range(0, len(variant_models), max_cols):
            chunks.append(variant_models[i : i + max_cols])
        tables = []
        for _idx, chunk in enumerate(chunks):
            t = _generate_benchmark_delta_for_variants(conn, comparison, chunk)
            tables.append(t)
        return "\n\n".join(tables)
    return _generate_benchmark_delta_for_variants(conn, comparison, variant_models)


def _generate_benchmark_delta_for_variants(
    conn: sqlite3.Connection, comparison: str, variant_models: list[tuple[str, str, str]]
) -> str:
    """Internal: generate delta table for a specific subset of variants."""
    if not variant_models:
        return "_No variant models._"

    # Get base name
    base_name = conn.execute(
        "SELECT model_name FROM models WHERE comparison = ? AND variant_type = 'base'",
        (comparison,),
    ).fetchone()
    if not base_name:
        return "_No base model found._"
    base_name = base_name[0]

    variant_names = [m[0] for m in variant_models]
    variant_displays = [m[1] for m in variant_models]

    tasks_with_data: list[tuple[str, str]] = []
    for task, metrics in TASK_PRIMARY_METRICS.items():
        if task in DEAD_TASKS:
            continue
        for metric in metrics:
            row = conn.execute(
                "SELECT metric_value FROM lm_eval_results "
                "WHERE comparison = ? AND model_name = ? AND task = ? AND metric_name = ?",
                (comparison, base_name, task, metric),
            ).fetchone()
            if row:
                tasks_with_data.append((task, metric))
                break

    if not tasks_with_data:
        return "_No benchmark results found._"

    header = "| Task | " + " | ".join(variant_displays) + " |"
    sep = "|" + "|".join(["------"] + ["------" for _ in variant_models]) + "|"

    lines = [header, sep]

    for task, primary_metric in tasks_with_data:
        metric_name: str = primary_metric
        for cand_metric in TASK_PRIMARY_METRICS.get(task, []):
            if conn.execute(
                "SELECT 1 FROM lm_eval_results WHERE comparison = ? AND task = ? AND metric_name = ?",
                (comparison, task, cand_metric),
            ).fetchone():
                metric_name = cand_metric
                break

        base_row = conn.execute(
            "SELECT metric_value FROM lm_eval_results "
            "WHERE comparison = ? AND model_name = ? AND task = ? AND metric_name = ?",
            (comparison, base_name, task, metric_name),
        ).fetchone()
        if not base_row:
            continue
        base_val = base_row[0]

        deltas: list[str] = []
        for vn in variant_names:
            row = conn.execute(
                "SELECT metric_value FROM lm_eval_results "
                "WHERE comparison = ? AND model_name = ? AND task = ? AND metric_name = ?",
                (comparison, vn, task, metric_name),
            ).fetchone()
            if row:
                diff = row[0] - base_val
                pct = (diff / base_val * 100) if base_val != 0 else 0.0
                sign = "+" if diff >= 0 else ""
                deltas.append(f"{sign}{pct:.2f}%")
            else:
                deltas.append("-")

        direction = _get_metric_direction(metric_name)
        # For deltas: higher delta is better (for higher-is-better metrics)
        # and lower (more negative) delta is better for lower-is-better metrics
        bolded = _bold_best(deltas, direction)
        display_task = TASK_DISPLAY_NAMES.get(task, task)
        lines.append(f"| {display_task} | " + " | ".join(bolded) + " |")

    return "\n".join(lines)


def generate_harmbench_table(conn: sqlite3.Connection, comparison: str) -> str:
    """Generate HarmBench ASR summary table with reasoning-damage indicators.

    Adds three columns beyond the basic ASR table:
    - **Empty**: responses where the model produced no output (refusal loops,
      thinking-loop truncated responses, API errors). These are *not*
      refusals — they are failures to respond at all. High empty counts
      signal degraded generation reliability.
    - **Avg reasoning chars**: mean length of the reasoning/thinking trace
      (the ``reasoning_text`` captured before the final response). Sharp
      drops vs base suggest the abliteration damaged the model's ability
      to construct coherent reasoning before answering.
    - **Truncated**: responses where ``finish_reason='length'`` (hit the
      max_gen_toks ceiling before completing). High counts indicate
      runaway generation / thinking loops.

    Sorted by ASR descending (best abliteration at top, base near bottom).
    Bolds highest ASR among variants only (base is never bolded).
    """
    rows = conn.execute(
        """
        SELECT
            r.model_name,
            COUNT(*) AS total,
            SUM(CASE WHEN r.response_text = '' OR r.response_text IS NULL THEN 1 ELSE 0 END) AS empty,
            SUM(CASE WHEN r.finish_reason = 'length' THEN 1 ELSE 0 END) AS truncated,
            ROUND(AVG(r.reasoning_chars), 0) AS avg_reasoning_chars,
            ROUND(AVG(r.completion_tokens), 0) AS avg_completion_tokens
        FROM responses r
        WHERE r.comparison = ?
        GROUP BY r.model_name
        ORDER BY (
            SELECT asr_pct FROM v_harmbench_summary
            WHERE comparison = r.comparison AND model_name = r.model_name
        ) DESC
        """,
        (comparison,),
    ).fetchall()

    if not rows:
        return "_No HarmBench results found._"

    # Get ASR + variant_type via the v_harmbench_summary view
    asr_rows = {
        r[0]: (r[4], r[2], r[3])  # model -> (asr_pct, complied, refused)
        for r in conn.execute(
            "SELECT model_name, total_behaviors, complied, refused, asr_pct "
            "FROM v_harmbench_summary WHERE comparison = ?",
            (comparison,),
        ).fetchall()
    }

    model_types: dict[str, str] = {
        r[0]: r[1]
        for r in conn.execute(
            "SELECT model_name, variant_type FROM models WHERE comparison = ?",
            (comparison,),
        )
    }

    header = "| Model | ASR | Complied | Refused | Empty | Truncated | Avg Reasoning chars | Avg Tokens |"
    sep = "|-------|-----|----------|---------|-------|-----------|---------------------|------------|"
    lines = [header, sep]

    # Find max ASR among variants for bolding
    variant_asrs = [
        (i, asr_rows.get(r[0], (None,))[0])
        for i, r in enumerate(rows)
        if model_types.get(r[0]) != "base" and asr_rows.get(r[0], (None,))[0] is not None
    ]
    max_asr = max((a for _, a in variant_asrs), default=None)
    max_indices = [i for i, a in variant_asrs if a == max_asr] if max_asr is not None else []

    for i, r in enumerate(rows):
        model = r[0]
        r[1]
        empty = r[2]
        truncated = r[3]
        avg_reason = r[4] if r[4] is not None else 0
        avg_tokens = r[5] if r[5] is not None else 0

        asr_info = asr_rows.get(model)
        if asr_info:
            asr_val, complied, refused = asr_info
            asr_str = _fmt_pct(asr_val)
        else:
            asr_val, complied, refused = None, 0, 0
            asr_str = "-"

        is_base = model_types.get(model) == "base"
        if not is_base and len(max_indices) == 1 and i == max_indices[0]:
            asr_str = f"**{asr_str}**"

        lines.append(
            f"| {model} | {asr_str} | {complied} | {refused} | {empty} | {truncated} | "
            f"{int(avg_reason)} | {int(avg_tokens)} |"
        )

    return "\n".join(lines)


def generate_harmbench_category_table(conn: sqlite3.Connection, comparison: str, max_cols: int = 0) -> str:
    """Generate per-category HarmBench ASR table.

    When ``max_cols > 0``, splits models into multiple sub-tables.
    """
    rows = conn.execute(
        "SELECT category, model_name, total, complied, asr_pct "
        "FROM v_harmbench_by_category WHERE comparison = ? "
        "ORDER BY category, model_name",
        (comparison,),
    ).fetchall()

    if not rows:
        return "_No category-level HarmBench results found._"

    # Get unique categories and models
    categories = sorted(set(r[0] for r in rows))
    all_model_names = sorted(set(r[1] for r in rows))

    # Split models into chunks if needed
    if max_cols > 0 and len(all_model_names) > max_cols:
        tables = []
        for i in range(0, len(all_model_names), max_cols):
            chunk = all_model_names[i : i + max_cols]
            t = _generate_harmbench_category_for_models(rows, categories, chunk)
            tables.append(t)
        return "\n\n".join(tables)
    return _generate_harmbench_category_for_models(rows, categories, all_model_names)


def _generate_harmbench_category_for_models(
    rows: list[tuple[Any, ...]], categories: list[str], model_names: list[str]
) -> str:
    """Internal: generate category table for a specific subset of models."""
    header = "| Category | Items | " + " | ".join(model_names) + " |"
    sep = "|" + "|".join(["----------", "-------"] + ["------" for _ in model_names]) + "|"
    lines = [header, sep]

    for cat in categories:
        cat_rows = [r for r in rows if r[0] == cat]
        items = cat_rows[0][2] if cat_rows else 0
        asrs: list[str] = []
        for mn in model_names:
            match = [r for r in cat_rows if r[1] == mn]
            asrs.append(_fmt_pct(match[0][4]) if match else "-")
        bolded = _bold_best(asrs, "higher")
        display_cat = CATEGORY_DISPLAY_NAMES.get(cat, cat)
        lines.append(f"| {display_cat} | {items} | " + " | ".join(bolded) + " |")

    return "\n".join(lines)


def generate_gsm8k_reasoning_table(conn: sqlite3.Connection, comparison: str) -> str:
    """Generate GSM8K reasoning damage indicators table.

    Surfaces four signals derived from per-sample analysis of GSM8K
    responses (parsed from ``samples_gsm8k_*.jsonl`` files at import time).
    These complement the headline ``exact_match`` score by detecting
    reasoning damage that strict accuracy alone misses.

    Columns (per model):
    - **Strict / Flexible**: the two extraction-filter scores. Strict requires
      the canonical ``#### N`` answer marker; flexible accepts the last
      number in the response.
    - **Gap**: ``flexible − strict`` (percentage points). A widening gap
      means the model is producing correct numeric answers but in the wrong
      format — a direct signal of format/reasoning damage.
    - **Empty**: count of responses where strict-match extracted nothing.
      Catastrophic failure indicator (thinking loops, refusals, truncation).
    - **Adjusted**: ``correct / (total − empty)``. The "true" accuracy on
      responses the model actually attempted. Diverges from strict only
      when ``empty > 0``.
    - **Mean chars**: mean length of the model's raw response. Verbosity
      shift signal — sharp drops indicate degraded reasoning depth.

    Bolds best strict score among variants; flags large gaps with ⚠.
    """
    models = _get_models(conn, comparison)
    if not models:
        return "_No models found for this comparison._"

    # Check if GSM8K sample stats are available
    has_stats = conn.execute(
        "SELECT 1 FROM lm_eval_results WHERE comparison = ? AND task = 'gsm8k' "
        "AND metric_name = 'strict_flexible_gap,none' LIMIT 1",
        (comparison,),
    ).fetchone()
    if not has_stats:
        return (
            "_GSM8K sample stats not imported. Re-run `db import-results` after the runner has written samples files._"
        )

    # Sort models: base first, then variants by strict score descending
    scored_models: list[tuple[int, float, tuple[str, str, str]]] = []
    for m in models:
        row = conn.execute(
            "SELECT metric_value FROM lm_eval_results "
            "WHERE comparison = ? AND model_name = ? AND task = 'gsm8k' "
            "AND metric_name = 'exact_match,strict-match'",
            (comparison, m[0]),
        ).fetchone()
        score = row[0] if row else -1
        # Base always first, then variants by score desc
        sort_key = (0, -score) if m[2] == "base" else (1, -score)
        scored_models.append((sort_key[0], sort_key[1], m))
    scored_models.sort(key=lambda x: (x[0], x[1]))

    header = "| Model | Strict | Flexible | Gap | Empty | Adjusted | Mean chars |"
    sep = "|-------|--------|----------|------|-------|----------|------------|"
    lines = [header, sep]

    models = [sm[2] for sm in scored_models]

    # Find best strict among variants for bolding
    variant_stricts: list[tuple[int, float]] = []
    for i, m in enumerate(models):
        if m[2] != "base":
            row = conn.execute(
                "SELECT metric_value FROM lm_eval_results "
                "WHERE comparison = ? AND model_name = ? AND task = 'gsm8k' "
                "AND metric_name = 'exact_match,strict-match'",
                (comparison, m[0]),
            ).fetchone()
            if row:
                variant_stricts.append((i, row[0]))

    max_strict = max((v for _, v in variant_stricts), default=None)
    max_strict_indices = [i for i, v in variant_stricts if v == max_strict] if max_strict is not None else []

    for i, m in enumerate(models):
        model_name = m[0]
        is_base = m[2] == "base"

        strict_row = conn.execute(
            "SELECT metric_value FROM lm_eval_results "
            "WHERE comparison = ? AND model_name = ? AND task = 'gsm8k' "
            "AND metric_name = 'exact_match,strict-match'",
            (comparison, model_name),
        ).fetchone()
        flex_row = conn.execute(
            "SELECT metric_value FROM lm_eval_results "
            "WHERE comparison = ? AND model_name = ? AND task = 'gsm8k' "
            "AND metric_name = 'exact_match,flexible-extract'",
            (comparison, model_name),
        ).fetchone()
        gap_row = conn.execute(
            "SELECT metric_value FROM lm_eval_results "
            "WHERE comparison = ? AND model_name = ? AND task = 'gsm8k' "
            "AND metric_name = 'strict_flexible_gap,none'",
            (comparison, model_name),
        ).fetchone()
        empty_row = conn.execute(
            "SELECT metric_value FROM lm_eval_results "
            "WHERE comparison = ? AND model_name = ? AND task = 'gsm8k' "
            "AND metric_name = 'empty_count,none'",
            (comparison, model_name),
        ).fetchone()
        adj_row = conn.execute(
            "SELECT metric_value FROM lm_eval_results "
            "WHERE comparison = ? AND model_name = ? AND task = 'gsm8k' "
            "AND metric_name = 'exact_match_adjusted,strict-match'",
            (comparison, model_name),
        ).fetchone()
        chars_row = conn.execute(
            "SELECT metric_value FROM lm_eval_results "
            "WHERE comparison = ? AND model_name = ? AND task = 'gsm8k' "
            "AND metric_name = 'mean_response_chars,none'",
            (comparison, model_name),
        ).fetchone()

        strict_str = _fmt_pct(strict_row[0] * 100) if strict_row else "-"
        flex_str = _fmt_pct(flex_row[0] * 100) if flex_row else "-"
        gap_val = gap_row[0] if gap_row else None
        empty_val = int(empty_row[0]) if empty_row else 0
        adj_str = _fmt_pct(adj_row[0] * 100) if adj_row else "-"
        chars_str = f"{chars_row[0]:.0f}" if chars_row else "-"

        # Flag large gaps (>5pp = format damage, >10pp = severe damage)
        if gap_val is not None:
            gap_pp = gap_val * 100
            gap_str = f"{gap_pp:.2f}pp"
            if gap_pp > 10:
                gap_str = f"⚠⚠ {gap_str}"
            elif gap_pp > 5:
                gap_str = f"⚠ {gap_str}"
        else:
            gap_str = "-"

        # Bold best strict among variants only
        if not is_base and len(max_strict_indices) == 1 and i == max_strict_indices[0]:
            strict_str = f"**{strict_str}**"

        lines.append(
            f"| {model_name} | {strict_str} | {flex_str} | {gap_str} | {empty_val} | {adj_str} | {chars_str} |"
        )

    return "\n".join(lines)


# Published KL claims from model cards (from the model_card_audit).
# These are NOT measurements, they are what each card publishes about itself.
# Used by generate_kl_calibration_table() to compare claim vs our measurement.
# Source: model_card_audit.md cross-reference section, verified against card text.
CARD_KL_CLAIMS: dict[str, float] = {
    "abliterix": 0.0006,
    "apostate": 0.119,
    "mythos": 0.04,
    "heresy": 0.0140,
    "trevorjs": 0.068,
    "infinimind": 0.068,
    "nullpo": 0.0256,
    "heretic": 0.0076,
    "heretic-std": 0.0043,
    "coder3101": 0.0058,
    # Variants with no card KL claim: obliteratus, physshell, bendernina,
    # sdft, distill, claude-distill, deckard, deckard-expresso, huihui, wwt,
    # treadon, treadon-combo, treadon-disin
}


def generate_ifeval_table(conn: sqlite3.Connection, comparison: str) -> str:
    """Generate IFEval standalone table sorted by score with delta vs base.

    IFEval is the cleanest capability-damage signal in the suite because it
    is generative (directly comparable to Google's published methodology)
    and degrades before raw knowledge metrics do.
    """
    models = _get_models(conn, comparison)
    if not models:
        return "_No models found._"

    rows: list[tuple[Any, ...]] = []
    base_score: float | None = None
    for m in models:
        model_name, display_name, variant_type = m
        row = conn.execute(
            "SELECT metric_value FROM lm_eval_results "
            "WHERE comparison = ? AND model_name = ? AND task = 'leaderboard_ifeval' "
            "AND metric_name = 'prompt_level_strict_acc,none'",
            (comparison, model_name),
        ).fetchone()
        score = row[0] if row else None
        if variant_type == "base":
            base_score = score
        rows.append((model_name, display_name, variant_type, score))

    if not any(r[3] is not None for r in rows):
        return "_No IFEval results found._"

    # Sort: base first, then variants by score descending
    rows.sort(key=lambda r: (r[2] != "base", -(r[3] or -1)))

    header = "| Model | IFEval prompt-strict | Δ vs Base |"
    sep = "|-------|--------------------:|----------:|"
    lines = [header, sep]

    variant_deltas: list[tuple[int, float]] = []
    for i, r in enumerate(rows):
        model_name, _, variant_type, score = r
        if variant_type != "base" and score is not None and base_score is not None:
            variant_deltas.append((i, score - base_score))

    max_delta = max((d for _, d in variant_deltas), default=None)
    max_delta_indices = [i for i, d in variant_deltas if d == max_delta] if max_delta is not None else []

    for i, r in enumerate(rows):
        model_name, _, variant_type, score = r
        if score is None:
            score_str = "-"
            delta_str = "-"
        else:
            score_str = _fmt_pct(score * 100)
            if variant_type == "base" or base_score is None:
                delta_str = "-"
            else:
                delta = (score - base_score) * 100
                delta_str = f"{delta:+.1f}pp"
        # Bold best variant (least-negative delta = closest to base)
        if variant_type != "base" and len(max_delta_indices) == 1 and i == max_delta_indices[0]:
            score_str = f"**{score_str}**"
        prefix = "**Base**" if variant_type == "base" else model_name
        lines.append(f"| {prefix} | {score_str} | {delta_str} |")

    return "\n".join(lines)


def generate_non_abliteration_baselines_table(conn: sqlite3.Connection, comparison: str) -> str:
    """Generate filtered table of non-abliteration baseline variants.

    These are variants that modify the model without targeting the refusal
    direction: distillation, disinhibition, RP fine-tuning. Shows ASR, KL,
    GSM8K strict, and MMLU-Pro for direct comparison to base.
    """
    # The non-abliteration variants are identified by not achieving
    # meaningful ASR lift over base (within 10pp of base's 30.8%).
    base_row = conn.execute(
        "SELECT asr_pct FROM v_harmbench_summary WHERE comparison = ? AND variant_type = 'base'",
        (comparison,),
    ).fetchone()
    if not base_row:
        return "_No base ASR found._"
    base_asr = base_row[0]

    # Find variants within 10pp of base ASR
    rows = conn.execute(
        """SELECT model_name, asr_pct FROM v_harmbench_summary
           WHERE comparison = ? AND variant_type = 'variant'
           AND asr_pct < ? + 10 ORDER BY model_name""",
        (comparison, base_asr),
    ).fetchall()

    if not rows:
        return "_No non-abliteration baselines identified._"

    header = "| Model | ASR | What it is | KL | GSM8K strict | MMLU-Pro |"
    sep = "|-------|-----|------------|-----|-------|------|"
    lines = [header, sep]

    # Add base row first
    base_gsm8k = conn.execute(
        "SELECT metric_value FROM lm_eval_results WHERE comparison = ? AND model_name = "
        "(SELECT model_name FROM models WHERE comparison = ? AND variant_type = 'base') "
        "AND task = 'gsm8k' AND metric_name = 'exact_match,strict-match'",
        (comparison, comparison),
    ).fetchone()
    base_mmlu = conn.execute(
        "SELECT metric_value FROM lm_eval_results WHERE comparison = ? AND model_name = "
        "(SELECT model_name FROM models WHERE comparison = ? AND variant_type = 'base') "
        "AND task = 'leaderboard_mmlu_pro' AND metric_name = 'acc,none'",
        (comparison, comparison),
    ).fetchone()
    base_name = conn.execute(
        "SELECT model_name FROM models WHERE comparison = ? AND variant_type = 'base'",
        (comparison,),
    ).fetchone()[0]
    gsm8k_str = _fmt_pct(base_gsm8k[0] * 100) if base_gsm8k else "-"
    mmlu_str = _fmt_pct(base_mmlu[0] * 100) if base_mmlu else "-"
    lines.append(f"| **{base_name}** | **{_fmt_pct(base_asr)}** | Original model | - | {gsm8k_str} | {mmlu_str} |")

    for r in rows:
        model_name, asr = r
        kl_row = conn.execute(
            "SELECT kl_batchmean FROM kl_results WHERE comparison = ? AND variant = ?",
            (comparison, model_name),
        ).fetchone()
        gsm8k_row = conn.execute(
            "SELECT metric_value FROM lm_eval_results WHERE comparison = ? AND model_name = ? "
            "AND task = 'gsm8k' AND metric_name = 'exact_match,strict-match'",
            (comparison, model_name),
        ).fetchone()
        mmlu_row = conn.execute(
            "SELECT metric_value FROM lm_eval_results WHERE comparison = ? AND model_name = ? "
            "AND task = 'leaderboard_mmlu_pro' AND metric_name = 'acc,none'",
            (comparison, model_name),
        ).fetchone()
        kl_str = _fmt(kl_row[0]) if kl_row else "-"
        gsm8k_str = _fmt_pct(gsm8k_row[0] * 100) if gsm8k_row else "-"
        mmlu_str = _fmt_pct(mmlu_row[0] * 100) if mmlu_row else "-"
        # Description is editorial; leave blank for Python to fill, report overrides
        lines.append(f"| {model_name} | {_fmt_pct(asr)} |  | {kl_str} | {gsm8k_str} | {mmlu_str} |")

    return "\n".join(lines)


def generate_kl_calibration_table(conn: sqlite3.Connection, comparison: str) -> str:
    """Generate KL calibration table comparing card-published claims to our measurements.

    Card claims come from CARD_KL_CLAIMS (transcribed from model cards).
    Our measurements come from kl_results table. Only variants with a
    published claim appear.
    """
    rows = []
    for variant, claimed in CARD_KL_CLAIMS.items():
        measured_row = conn.execute(
            "SELECT kl_batchmean FROM kl_results WHERE comparison = ? AND variant = ?",
            (comparison, variant),
        ).fetchone()
        if not measured_row:
            continue
        measured = measured_row[0]
        # Always show ratio as >=1 (magnitude of discrepancy)
        # Direction conveys which way the card is off
        if claimed >= measured:
            ratio = claimed / measured if measured > 0 else float("inf")
            direction = "card over-reports"
        else:
            ratio = measured / claimed if claimed > 0 else float("inf")
            direction = "card under-reports"
        rows.append((variant, claimed, measured, ratio, direction))

    if not rows:
        return "_No KL calibration data available._"

    # Sort by ratio descending (biggest card over-reports first)
    rows.sort(key=lambda r: -r[3])

    header = "| Variant | Card claims | Measured | Claim ÷ Measured | Direction |"
    sep = "|---------|------------|----------|------------------|-----------|"
    lines = [header, sep]

    for variant, claimed, measured, ratio, direction in rows:
        # ratio is claimed/measured: >1 means card over-reports damage
        ratio_str = f"{ratio:.1f}×" if ratio < 100 else f"{ratio:.0f}×"
        lines.append(f"| {variant} | {_fmt(claimed)} | {_fmt(measured)} | {ratio_str} | {direction} |")

    return "\n".join(lines)


def generate_kl_impact_table(conn: sqlite3.Connection, comparison: str) -> str:
    """Generate KL vs benchmark impact cross-table.

    Shows how capability metrics degrade as KL increases. Curated subset
    spanning the full KL range from surgical (0.001) to heavy (1.10).
    """
    # Get all variants with KL data
    kl_rows = conn.execute(
        "SELECT variant, kl_batchmean FROM kl_results WHERE comparison = ? ORDER BY kl_batchmean",
        (comparison,),
    ).fetchall()

    if not kl_rows:
        return "_No KL data found._"

    base_name_row = conn.execute(
        "SELECT model_name FROM models WHERE comparison = ? AND variant_type = 'base'",
        (comparison,),
    ).fetchone()
    if not base_name_row:
        return "_No base model found._"
    base_name = base_name_row[0]

    # Get base scores for delta computation
    base_scores: dict[str, float | None] = {}
    for task, metric in [
        ("leaderboard_mmlu_pro", "acc,none"),
        ("gsm8k", "exact_match,strict-match"),
        ("leaderboard_ifeval", "prompt_level_strict_acc,none"),
    ]:
        row = conn.execute(
            "SELECT metric_value FROM lm_eval_results WHERE comparison = ? AND model_name = ? "
            "AND task = ? AND metric_name = ?",
            (comparison, base_name, task, metric),
        ).fetchone()
        base_scores[task] = row[0] if row else None

    header = "| Variant | KL | MMLU-Pro Δ | GSM8K strict Δ | IFEval Δ |"
    sep = "|---------|-----|------------|----------------|----------|"
    lines = [header, sep]

    # Select representative variants spanning the KL range
    # Take all variants but cap at ~12 rows for readability
    # Strategy: every Nth variant + always include key outliers
    selected = kl_rows
    if len(kl_rows) > 12:
        # Take roughly evenly spaced sample
        step = len(kl_rows) // 10
        selected = kl_rows[::step][:10]
        # Always include the worst
        if kl_rows[-1] not in selected:
            selected.append(kl_rows[-1])

    for variant, kl in selected:
        deltas: list[str] = []
        for task, metric in [
            ("leaderboard_mmlu_pro", "acc,none"),
            ("gsm8k", "exact_match,strict-match"),
            ("leaderboard_ifeval", "prompt_level_strict_acc,none"),
        ]:
            row = conn.execute(
                "SELECT metric_value FROM lm_eval_results WHERE comparison = ? AND model_name = ? "
                "AND task = ? AND metric_name = ?",
                (comparison, variant, task, metric),
            ).fetchone()
            if row and base_scores.get(task) is not None:
                delta = (row[0] - base_scores[task]) * 100
                deltas.append(f"{delta:+.1f}pp")
            else:
                deltas.append("-")
        lines.append(f"| {variant} | {_fmt(kl)} | {deltas[0]} | {deltas[1]} | {deltas[2]} |")

    return "\n".join(lines)


def generate_kl_table(conn: sqlite3.Connection, comparison: str) -> str:
    """Generate KL divergence summary table.

    Bolds lowest KL divergence.
    """
    rows = conn.execute(
        "SELECT variant, kl_batchmean, rating FROM kl_results WHERE comparison = ? ORDER BY kl_batchmean",
        (comparison,),
    ).fetchall()

    if not rows:
        return "_No KL divergence results found._"

    header = "| Variant | KL Divergence | Rating |"
    sep = "|---------|--------------|--------|"
    lines = [header, sep]

    # Bold lowest KL
    kl_values = [_fmt(r[1]) for r in rows]
    bolded = _bold_best(kl_values, "lower")

    for i, r in enumerate(rows):
        lines.append(f"| {r[0]} | {bolded[i]} | {r[2] or '-'} |")

    return "\n".join(lines)


def generate_weight_summary_table(conn: sqlite3.Connection, comparison: str) -> str:
    """Generate weight modification summary table, one row per variant.

    Columns: Variant | Tensors | Total | % | Types | Layers | E% | M% | L%

    Data comes from the fingerprint JSON's ``scope``, ``targeting``, and
    ``layer_profile`` sections, all stored in the ``weight_results.summary``
    column as JSON.
    """
    rows = conn.execute(
        "SELECT variant, summary FROM weight_results "
        "WHERE comparison = ? AND variant_b = '' AND script_name = 'fingerprint' "
        "ORDER BY variant",
        (comparison,),
    ).fetchall()

    if not rows:
        return "_No weight analysis results found._"

    # Parse each variant's fingerprint data
    variants: list[dict[str, Any]] = []
    for variant_name, summary_json in rows:
        try:
            d = json.loads(summary_json) if isinstance(summary_json, str) else summary_json
        except (json.JSONDecodeError, TypeError):
            continue

        changed = d.get("changed_tensors")
        total = d.get("total_tensors")
        pct = d.get("changed_pct")
        types = d.get("num_types", len(d.get("tensor_types", {})) if d.get("tensor_types") else 0)
        layers = d.get("layers_modified")
        early = d.get("early_layer_pct") or 0
        mid = d.get("mid_layer_pct") or 0
        late = d.get("late_layer_pct") or 0

        variants.append(
            {
                "name": variant_name,
                "changed": changed,
                "total": total,
                "pct": pct,
                "types": types,
                "layers": layers,
                "early": early,
                "mid": mid,
                "late": late,
            }
        )

    if not variants:
        return "_No fingerprint data available._"

    # Sort by changed tensor count ascending (least modified first = best)
    variants.sort(key=lambda v: v["changed"] or 9999)

    header = "| Variant | Changed | Total | % | Types | Layers | E% | M% | L% |"
    sep = "|---------|---------|-------|------|-------|--------|----|----|----|"
    lines = [header, sep]

    # Find best (fewest changes) for bolding
    min_changed = min(v["changed"] for v in variants if v["changed"] is not None)

    for v in variants:
        changed_str = str(v["changed"]) if v["changed"] is not None else "-"
        if v["changed"] == min_changed:
            changed_str = f"**{changed_str}**"
        pct_str = f"{v['pct']:.1f}%" if v["pct"] is not None else "-"
        total_str = str(v["total"]) if v["total"] is not None else "-"
        types_str = str(v["types"]) if v["types"] else "-"
        layers_str = str(v["layers"]) if v["layers"] is not None else "-"
        early_str = f"{v['early']:.0f}" if v["early"] else "0"
        mid_str = f"{v['mid']:.0f}" if v["mid"] else "0"
        late_str = f"{v['late']:.0f}" if v["late"] else "0"
        lines.append(
            f"| {v['name']} | {changed_str} | {total_str} | {pct_str} | "
            f"{types_str} | {layers_str} | {early_str} | {mid_str} | {late_str} |"
        )

    lines.append("")
    lines.append("*E% / M% / L% = early (0-13) / mid (14-27) / late (28-41) layer distribution.*")

    return "\n".join(lines)


def generate_combined_summary_table(conn: sqlite3.Connection, comparison: str) -> str:
    """Generate the master summary table: one row per model, key metrics as columns.

    This is the "at a glance" table. Models as rows (base first, then variants
    sorted by ASR descending). Columns: ASR, GSM8K strict, MMLU-Pro, IFEval,
    KL, Tensors. Bolds best value per column among variants.
    """
    models = _get_models(conn, comparison)
    if not models:
        return "_No models found._"

    base_model = next((m for m in models if m[2] == "base"), None)
    [m for m in models if m[2] == "variant"]
    if not base_model:
        return "_No base model found._"

    base_model[0]

    # Collect data for each model
    model_data: list[dict[str, Any]] = []
    for m in models:
        model_name, display_name, variant_type = m
        row_data: dict[str, Any] = {"name": model_name, "display": display_name, "type": variant_type}

        # ASR
        asr = conn.execute(
            "SELECT asr_pct FROM v_harmbench_summary WHERE comparison = ? AND model_name = ?",
            (comparison, model_name),
        ).fetchone()
        row_data["asr"] = asr[0] if asr else None

        # GSM8K strict
        gsm8k = conn.execute(
            "SELECT metric_value FROM lm_eval_results WHERE comparison = ? AND model_name = ? "
            "AND task = 'gsm8k' AND metric_name = 'exact_match,strict-match'",
            (comparison, model_name),
        ).fetchone()
        row_data["gsm8k"] = gsm8k[0] if gsm8k else None

        # MMLU-Pro
        mmlu = conn.execute(
            "SELECT metric_value FROM lm_eval_results WHERE comparison = ? AND model_name = ? "
            "AND task = 'leaderboard_mmlu_pro' AND metric_name = 'acc,none'",
            (comparison, model_name),
        ).fetchone()
        row_data["mmlu_pro"] = mmlu[0] if mmlu else None

        # IFEval
        ifeval = conn.execute(
            "SELECT metric_value FROM lm_eval_results WHERE comparison = ? AND model_name = ? "
            "AND task = 'leaderboard_ifeval' AND metric_name = 'prompt_level_strict_acc,none'",
            (comparison, model_name),
        ).fetchone()
        row_data["ifeval"] = ifeval[0] if ifeval else None

        # KL
        kl = conn.execute(
            "SELECT kl_batchmean FROM kl_results WHERE comparison = ? AND variant = ?",
            (comparison, model_name),
        ).fetchone()
        row_data["kl"] = kl[0] if kl else None

        # Tensors changed
        wr = conn.execute(
            "SELECT summary FROM weight_results WHERE comparison = ? AND variant = ? "
            "AND script_name = 'fingerprint' AND variant_b = ''",
            (comparison, model_name),
        ).fetchone()
        if wr:
            try:
                s = json.loads(wr[0])
                row_data["tensors"] = s.get("changed_tensors")
                row_data["tensor_pct"] = s.get("changed_pct")
            except (json.JSONDecodeError, TypeError):
                row_data["tensors"] = None
                row_data["tensor_pct"] = None
        else:
            row_data["tensors"] = None
            row_data["tensor_pct"] = None

        model_data.append(row_data)

    # Sort: base first, then variants by ASR desc
    model_data.sort(key=lambda d: (0 if d["type"] == "base" else 1, -(d["asr"] or 0)))

    # Find best per metric among variants for bolding
    # For ASR/GSM8K/MMLU-Pro/IFEval: higher is better
    # For KL/tensors: lower is better
    def _best(field: str, higher: bool) -> float | None:
        vals = [d[field] for d in model_data if d["type"] != "base" and d[field] is not None]
        if not vals:
            return None
        return max(vals) if higher else min(vals)

    best_asr = _best("asr", True)
    best_gsm8k = _best("gsm8k", True)
    best_mmlu = _best("mmlu_pro", True)
    best_ifeval = _best("ifeval", True)
    best_kl = _best("kl", False)
    best_tensors = _best("tensors", False)

    header = "| Model | ASR | GSM8K Δ | MMLU-Pro Δ | IFEval Δ | KL | Tensors |"
    sep = "|-------|-----|---------|------------|----------|----|---------|"
    lines = [header, sep]

    # Get base values for delta computation
    base_data = next((d for d in model_data if d["type"] == "base"), None)

    for d in model_data:
        name = f"**{d['name']}**" if d["type"] == "base" else d["name"]
        is_base = d["type"] == "base"

        # ASR: show absolute for base, delta for variants
        if is_base:
            asr_str = f"**{_fmt_pct(d['asr'])}**" if d["asr"] is not None else "-"
        else:
            if d["asr"] is not None and base_data and base_data["asr"] is not None:
                delta = d["asr"] - base_data["asr"]
                asr_str = f"{delta:+.1f}pp"
                if best_asr is not None and d["asr"] == best_asr:
                    asr_str = f"**{asr_str}**"
            else:
                asr_str = "-"

        # Benchmarks: base shows reference value, variants show delta.
        # `is_base` bound as a default arg to close over the current loop value
        # (avoids late-binding closure bug flagged by B023).
        def delta_str(
            val: float | None,
            base_val: float | None,
            best: float | None,
            is_base: bool = is_base,
        ) -> str:
            if is_base:
                if val is None:
                    return "-"
                return f"{val * 100:.1f}%"
            if val is None or base_val is None:
                return "-"
            delta = (val - base_val) * 100
            s = f"{delta:+.1f}pp"
            if best is not None and val == best:
                s = f"**{s}**"
            return s

        gsm8k_str = delta_str(d["gsm8k"], base_data["gsm8k"] if base_data else None, best_gsm8k)
        mmlu_str = delta_str(d["mmlu_pro"], base_data["mmlu_pro"] if base_data else None, best_mmlu)
        ifeval_str = delta_str(d["ifeval"], base_data["ifeval"] if base_data else None, best_ifeval)

        # KL: absolute value (no base reference)
        if d["kl"] is not None:
            kl_str = _fmt(d["kl"])
            if not is_base and best_kl is not None and d["kl"] == best_kl:
                kl_str = f"**{kl_str}**"
        else:
            kl_str = "-"

        # Tensors: absolute count
        if d["tensors"] is not None:
            tensor_str = str(d["tensors"])
            if not is_base and best_tensors is not None and d["tensors"] == best_tensors:
                tensor_str = f"**{tensor_str}**"
        else:
            tensor_str = "-"

        lines.append(f"| {name} | {asr_str} | {gsm8k_str} | {mmlu_str} | {ifeval_str} | {kl_str} | {tensor_str} |")

    return "\n".join(lines)


def generate_all(conn: sqlite3.Connection, comparison: str, max_cols: int = 8) -> str:
    """Generate all tables as a single markdown string.

    ``max_cols`` controls splitting of wide tables (benchmark, delta,
    harmbench-categories) into multiple narrower sub-tables.
    Set to 0 to disable splitting.
    """
    sections = [
        ("## Benchmarks", generate_benchmark_table(conn, comparison, max_cols=max_cols)),
        ("## Benchmark Deltas (vs Base)", generate_benchmark_delta_table(conn, comparison, max_cols=max_cols)),
        ("## IFEval (Instruction-Following)", generate_ifeval_table(conn, comparison)),
        ("## GSM8K Reasoning Damage", generate_gsm8k_reasoning_table(conn, comparison)),
        ("## HarmBench ASR", generate_harmbench_table(conn, comparison)),
        ("## HarmBench by Category", generate_harmbench_category_table(conn, comparison, max_cols=max_cols)),
        ("## Non-Abliteration Baselines", generate_non_abliteration_baselines_table(conn, comparison)),
        ("## KL Divergence", generate_kl_table(conn, comparison)),
        ("## KL Calibration (Card Claims vs Measured)", generate_kl_calibration_table(conn, comparison)),
        ("## KL vs Benchmark Impact", generate_kl_impact_table(conn, comparison)),
        ("## Weight Analysis", generate_weight_summary_table(conn, comparison)),
        ("## Combined Summary", generate_combined_summary_table(conn, comparison)),
    ]
    parts: list[str] = []
    for title, table in sections:
        parts.append(title)
        parts.append(table)
        parts.append("")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for markdown table generation."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Generate markdown tables from Abliterlitics DB")
    sub = parser.add_subparsers(dest="command", required=True)

    # All subcommands take a comparison name
    for cmd in [
        "benchmark",
        "benchmark-delta",
        "ifeval",
        "gsm8k-reasoning",
        "harmbench",
        "harmbench-categories",
        "non-abliteration-baselines",
        "kl",
        "kl-calibration",
        "kl-impact",
        "weight-summary",
        "summary",
        "all",
    ]:
        p = sub.add_parser(cmd)
        p.add_argument("comparison", help="Comparison name")
        p.add_argument("--db", default=DB_PATH, help="Database path")

    args = parser.parse_args()
    conn = get_db(args.db)

    generators = {
        "benchmark": generate_benchmark_table,
        "benchmark-delta": generate_benchmark_delta_table,
        "ifeval": generate_ifeval_table,
        "gsm8k-reasoning": generate_gsm8k_reasoning_table,
        "harmbench": generate_harmbench_table,
        "harmbench-categories": generate_harmbench_category_table,
        "non-abliteration-baselines": generate_non_abliteration_baselines_table,
        "kl": generate_kl_table,
        "kl-calibration": generate_kl_calibration_table,
        "kl-impact": generate_kl_impact_table,
        "weight-summary": generate_weight_summary_table,
        "summary": generate_combined_summary_table,
        "all": generate_all,
    }

    func = generators.get(args.command)
    if func:
        print(func(conn, args.comparison))

    conn.close()


if __name__ == "__main__":
    main()
