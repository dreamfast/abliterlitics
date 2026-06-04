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

from src.db.core import DB_PATH, get_db

log = logging.getLogger(__name__)

# Re-use the same primary metric mapping from importers.
TASK_PRIMARY_METRICS: dict[str, list[str]] = {
    "mmlu": ["acc,none"],
    "gsm8k": ["exact_match,strict-match", "exact_match,flexible-extract"],
    "hellaswag": ["acc_norm,none"],
    "arc_challenge": ["acc_norm,none"],
    "winogrande": ["acc,none"],
    "truthfulqa_mc1": ["acc,none"],
    "truthfulqa_mc2": ["acc,none"],
    "truthfulqa_gen": ["bleu_acc,none"],
    "piqa": ["acc_norm,none"],
    "lambada_openai": ["perplexity,none"],
}

# Task display names for tables.
TASK_DISPLAY_NAMES: dict[str, str] = {
    "mmlu": "MMLU",
    "gsm8k": "GSM8K",
    "hellaswag": "HellaSwag",
    "arc_challenge": "ARC-C",
    "winogrande": "WinoGrande",
    "truthfulqa_mc1": "TQA-MC1",
    "truthfulqa_mc2": "TQA-MC2",
    "truthfulqa_gen": "TQA-Gen",
    "piqa": "PIQA",
    "lambada_openai": "Lambada",
}

# Metrics where lower is better.
LOWER_IS_BETTER: set[str] = {"perplexity,none"}


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


def generate_benchmark_table(conn: sqlite3.Connection, comparison: str) -> str:
    """Generate main benchmark comparison table.

    Columns: Task | Base | Variant1 | Variant2 | ...
    Bolds highest per row (except perplexity — bolds lowest).
    """
    models = _get_models(conn, comparison)
    if not models:
        return "_No models found for this comparison._"

    model_names = [m[0] for m in models]
    display_names = [m[1] for m in models]

    # Collect all tasks that have data
    tasks_with_data: list[tuple[str, str]] = []  # (task, metric_name)
    for task, metrics in TASK_PRIMARY_METRICS.items():
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

        fmt_str = ".1f" if "perplexity" in metric_name else ".4f"
        str_values = [_fmt(v, fmt_str) for v in values]
        direction = _get_metric_direction(metric_name)
        bolded = _bold_best(str_values, direction)

        display_task = TASK_DISPLAY_NAMES.get(task, task)
        lines.append(f"| {display_task} | " + " | ".join(bolded) + " |")

    return "\n".join(lines)


def generate_benchmark_delta_table(conn: sqlite3.Connection, comparison: str) -> str:
    """Generate delta-vs-base benchmark table.

    Columns: Task | Variant1 | Variant2 | ...
    Shows difference from base model (e.g. +0.0004, -0.0016).
    """
    models = _get_models(conn, comparison)
    if not models:
        return "_No models found for this comparison._"

    base_name = None
    variant_models = []
    for m in models:
        if m[2] == "base":
            base_name = m[0]
        else:
            variant_models.append(m)

    if not base_name or not variant_models:
        return "_Need at least one base and one variant model._"

    variant_names = [m[0] for m in variant_models]
    variant_displays = [m[1] for m in variant_models]

    tasks_with_data: list[tuple[str, str]] = []
    for task, metrics in TASK_PRIMARY_METRICS.items():
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
                sign = "+" if diff >= 0 else ""
                deltas.append(f"{sign}{diff:.4f}")
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
    """Generate HarmBench ASR summary table.

    Bolds highest ASR among variants only (base is never bolded).
    """
    rows = conn.execute(
        "SELECT model_name, total_behaviors, complied, refused, asr_pct FROM v_harmbench_summary WHERE comparison = ?",
        (comparison,),
    ).fetchall()

    if not rows:
        return "_No HarmBench results found._"

    # Get model types to distinguish base from variants
    model_types: dict[str, str] = {}
    for r in conn.execute(
        "SELECT model_name, variant_type FROM models WHERE comparison = ?",
        (comparison,),
    ):
        model_types[r[0]] = r[1]

    header = "| Model | ASR | Complied | Refused | Total |"
    sep = "|-------|-----|----------|---------|-------|"
    lines = [header, sep]

    # Collect ASR values for bolding (variants only)
    variant_asrs: list[tuple[int, float]] = []
    for i, r in enumerate(rows):
        if model_types.get(r[0]) != "base":
            variant_asrs.append((i, r[4]))

    # Find max ASR among variants
    max_asr = None
    max_indices: list[int] = []
    if variant_asrs:
        max_asr = max(a for _, a in variant_asrs)
        max_indices = [i for i, a in variant_asrs if a == max_asr]

    for i, r in enumerate(rows):
        asr_str = _fmt_pct(r[4])
        # Bold only if variant and has unique max
        is_base = model_types.get(r[0]) == "base"
        if not is_base and len(max_indices) == 1 and i == max_indices[0]:
            asr_str = f"**{asr_str}**"

        lines.append(f"| {r[0]} | {asr_str} | {r[2]} | {r[3]} | {r[1]} |")

    return "\n".join(lines)


def generate_harmbench_category_table(conn: sqlite3.Connection, comparison: str) -> str:
    """Generate per-category HarmBench ASR table."""
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
    model_names = sorted(set(r[1] for r in rows))

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
        lines.append(f"| {cat} | {items} | " + " | ".join(bolded) + " |")

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
    """Generate weight modification summary table."""
    rows = conn.execute(
        "SELECT variant, variant_b, script_name, summary FROM weight_results "
        "WHERE comparison = ? AND variant_b = '' ORDER BY variant, script_name",
        (comparison,),
    ).fetchall()

    if not rows:
        return "_No weight analysis results found._"

    # Group by variant
    variants = sorted(set(r[0] for r in rows if r[0]))

    header = "| Metric | " + " | ".join(variants) + " |"
    sep = "|" + "|".join(["--------"] + ["------" for _ in variants]) + "|"
    lines = [header, sep]

    # Collect metrics across variants
    metrics: dict[str, dict[str, str]] = {}  # metric_name -> {variant: value}
    for r in rows:
        try:
            summary = json.loads(r[3]) if isinstance(r[3], str) else r[3]
        except (json.JSONDecodeError, TypeError):
            summary = {}

        script = r[2]
        if script == "fingerprint":
            key = "Tensors changed"
            val = summary.get("changed_pct")
            if val is not None:
                metrics.setdefault(key, {})[r[0]] = f"{val:.2f}%"
        elif script == "layer_analysis":
            key = "Layers modified"
            val = summary.get("layers_modified")
            if val is not None:
                metrics.setdefault(key, {})[r[0]] = str(val)

    if not metrics:
        return "_No summary metrics available in weight results._"

    for metric_name in sorted(metrics.keys()):
        vals = [metrics[metric_name].get(v, "-") for v in variants]
        # Try to bold lowest (fewer changes = less modified)
        bolded = _bold_best(vals, "lower")
        lines.append(f"| {metric_name} | " + " | ".join(bolded) + " |")

    return "\n".join(lines)


def generate_combined_summary_table(conn: sqlite3.Connection, comparison: str) -> str:
    """Generate combined summary table with key metrics across all dimensions.

    Rows: HarmBench ASR, MMLU, GSM8K, KL divergence.
    Columns: Variant1 | Variant2 | ...
    """
    models = _get_models(conn, comparison)
    variant_models = [m for m in models if m[2] == "variant"]
    if not variant_models:
        return "_No variant models found._"

    variant_names = [m[0] for m in variant_models]
    variant_displays = [m[1] for m in variant_models]

    header = "| Metric | " + " | ".join(variant_displays) + " |"
    sep = "|" + "|".join(["--------"] + ["------" for _ in variant_models]) + "|"
    lines = [header, sep]

    # HarmBench ASR
    asr_vals: list[str] = []
    for vn in variant_names:
        row = conn.execute(
            "SELECT asr_pct FROM v_harmbench_summary WHERE comparison = ? AND model_name = ?",
            (comparison, vn),
        ).fetchone()
        asr_vals.append(_fmt_pct(row[0]) if row else "-")
    bolded = _bold_best(asr_vals, "higher")
    lines.append("| **HarmBench ASR** | " + " | ".join(bolded) + " |")

    # Key benchmarks
    key_tasks = [
        ("MMLU", "mmlu", "acc,none"),
        ("GSM8K", "gsm8k", "exact_match,strict-match"),
        ("HellaSwag", "hellaswag", "acc_norm,none"),
        ("ARC-C", "arc_challenge", "acc_norm,none"),
    ]
    for display, task, metric in key_tasks:
        vals: list[str] = []
        for vn in variant_names:
            row = conn.execute(
                "SELECT metric_value FROM lm_eval_results "
                "WHERE comparison = ? AND model_name = ? AND task = ? AND metric_name = ?",
                (comparison, vn, task, metric),
            ).fetchone()
            vals.append(_fmt(row[0]) if row else "-")
        bolded = _bold_best(vals, "higher")
        lines.append(f"| **{display}** | " + " | ".join(bolded) + " |")

    # KL divergence
    kl_vals: list[str] = []
    for vn in variant_names:
        row = conn.execute(
            "SELECT kl_batchmean FROM kl_results WHERE comparison = ? AND variant = ?",
            (comparison, vn),
        ).fetchone()
        kl_vals.append(_fmt(row[0]) if row else "-")
    bolded = _bold_best(kl_vals, "lower")
    lines.append("| **KL Divergence** | " + " | ".join(bolded) + " |")

    return "\n".join(lines)


def generate_all(conn: sqlite3.Connection, comparison: str) -> str:
    """Generate all tables as a single markdown string."""
    sections = [
        ("## Benchmarks", generate_benchmark_table(conn, comparison)),
        ("## Benchmark Deltas (vs Base)", generate_benchmark_delta_table(conn, comparison)),
        ("## HarmBench ASR", generate_harmbench_table(conn, comparison)),
        ("## HarmBench by Category", generate_harmbench_category_table(conn, comparison)),
        ("## KL Divergence", generate_kl_table(conn, comparison)),
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
        "harmbench",
        "harmbench-categories",
        "kl",
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
        "harmbench": generate_harmbench_table,
        "harmbench-categories": generate_harmbench_category_table,
        "kl": generate_kl_table,
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
