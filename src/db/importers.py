"""Abliterlitics SQLite database — result file parsing and import.

Parses lm-eval, KL divergence, weight analysis, and HarmBench result JSON
files and imports them into the database. All imports are idempotent
(ON CONFLICT DO UPDATE) and handle malformed files gracefully (log warning,
skip, continue).
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import Any

from src.db.pipeline import init_stages

log = logging.getLogger(__name__)

# Primary metrics to import per lm-eval task (skip aliases and subtasks).
# Key: lm-eval task name (as it appears in results dict).
# Value: list of metric names to import (e.g. "acc,none", "exact_match,strict-match").
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


def import_lm_eval_results(
    conn: sqlite3.Connection,
    comparison_name: str,
    results_dir: Path,
) -> int:
    """Import lm-eval result JSON files into lm_eval_results table.

    Scans for ``lm_eval_<model>.json`` files. Only imports primary metrics
    (per TASK_PRIMARY_METRICS). Skips ``alias`` keys and subtask rows
    (e.g. ``mmlu_humanities``).

    Also imports per-sample stats for GSM8K (empty response count, mean
    response length, adjusted score, strict-flex gap) by parsing the
    ``samples_gsm8k_*.jsonl`` files written by lm-eval ``--log_samples``.
    See ``_import_gsm8k_sample_stats()`` for details.

    Returns the number of metric rows imported.
    """
    pattern = "lm_eval_*.json"
    files = sorted(results_dir.glob(pattern))
    if not files:
        log.warning("No %s files found in %s", pattern, results_dir)
        return 0

    count = 0
    for f in files:
        # Model name from filename: lm_eval_<model>.json -> <model>
        model_name = f.stem.replace("lm_eval_", "", 1)
        try:
            raw = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Skipping %s: %s", f.name, exc)
            continue

        results = raw.get("results", {})
        if not results:
            log.warning("No 'results' key in %s", f.name)
            continue

        for task, metrics in results.items():
            # Skip alias entries (lm-eval puts alias inside task dicts, not as a key)
            if task == "alias":
                continue
            # Skip subtask entries (mmlu_* but not mmlu itself, etc.)
            if "_" in task and task.split("_")[0] in TASK_PRIMARY_METRICS:
                continue

            # Determine which metrics to import
            primary = TASK_PRIMARY_METRICS.get(task)
            if primary is None:
                # Unknown task — import all numeric metrics
                import_keys = [k for k in metrics if k != "alias" and isinstance(metrics[k], (int, float))]
            else:
                import_keys = [k for k in primary if k in metrics]

            for metric_name in import_keys:
                value = metrics.get(metric_name)
                if value is None or not isinstance(value, (int, float)):
                    continue

                # stderr: look for <metric_name>_stderr,<higher_is_better>
                stderr = None
                stderr_key = metric_name + "_stderr"
                # lm-eval uses "acc,none" and "acc_stderr,none"
                base, agg = metric_name.rsplit(",", 1) if "," in metric_name else (metric_name, "")
                stderr_alt = f"{base}_stderr,{agg}" if agg else f"{base}_stderr"
                if stderr_key in metrics:
                    stderr = metrics[stderr_key]
                elif stderr_alt in metrics:
                    stderr = metrics[stderr_alt]

                stderr = stderr if isinstance(stderr, (int, float)) else None

                try:
                    conn.execute(
                        """INSERT INTO lm_eval_results
                        (comparison, model_name, task, metric_name, metric_value, metric_stderr, result_path)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(comparison, model_name, task, metric_name) DO UPDATE SET
                            metric_value = excluded.metric_value,
                            metric_stderr = excluded.metric_stderr,
                            result_path = excluded.result_path""",
                        (comparison_name, model_name, task, metric_name, float(value), stderr, str(f)),
                    )
                    count += 1
                except sqlite3.Error as exc:
                    log.warning("Failed to insert lm_eval metric %s/%s/%s: %s", model_name, task, metric_name, exc)

        # Per-sample stats for GSM8K (empty count, mean response length, etc.)
        # Parsed from samples_gsm8k_*.jsonl files. New runner writes them as
        # <model>__samples_gsm8k_*.jsonl; legacy runs have samples_gsm8k_<ts>.jsonl
        # which we match by mtime to the result file.
        count += _import_gsm8k_sample_stats(conn, comparison_name, model_name, f, results_dir)

    conn.commit()
    log.info("Imported %d lm-eval metrics from %d files", count, len(files))
    return count


def _import_gsm8k_sample_stats(
    conn: sqlite3.Connection,
    comparison_name: str,
    model_name: str,
    result_file: Path,
    results_dir: Path,
) -> int:
    """Parse GSM8K samples JSONL and import per-sample reasoning stats.

    Computes four signals of reasoning damage from the raw sample responses:

    1. ``empty_count,none`` — number of responses where the strict-match filter
       extracted an empty string. Captures catastrophic failures (thinking
       loops that never produced an answer, refusals, OOM-truncated outputs).
    2. ``mean_response_chars,none`` — mean character length of the model's
       raw response (excluding any thinking tags). Verbosity shift signal;
       sharp drops indicate degraded reasoning depth.
    3. ``exact_match_adjusted,strict-match`` — strict accuracy recomputed as
       correct / (total - empty). When empty_count > 0, this is the "true"
       accuracy on responses the model actually attempted. Equal to the
       raw strict score when empty_count == 0.
    4. ``strict_flexible_gap,none`` — (flexible_extract_score − strict_score).
       A widening gap means the model's answers are increasingly
       ill-formatted (e.g., missing ``#### N`` marker, extra prose around
       the answer). Direct signal of format/reasoning damage.

    Sample file lookup (in priority order):
      1. New runner: ``<results_dir>/__model/<model>__samples_gsm8k_*.jsonl``
      2. Legacy: ``<results_dir>/__model/samples_gsm8k_*.jsonl`` matched by
         file mtime closest to ``result_file.stat().st_mtime``.

    Returns the number of metric rows imported (0 if no samples file found).
    """
    samples_dir = results_dir / "__model"
    if not samples_dir.is_dir():
        return 0

    # Try new naming convention first: <model>__samples_gsm8k_*.jsonl
    candidates = sorted(samples_dir.glob(f"{model_name}__samples_gsm8k_*.jsonl"))

    # Fall back to legacy: match by mtime
    if not candidates:
        result_mtime = result_file.stat().st_mtime
        all_samples = list(samples_dir.glob("samples_gsm8k_*.jsonl"))
        # Exclude any already renamed files
        all_samples = [sf for sf in all_samples if "__" not in sf.stem]
        if all_samples:
            closest = min(all_samples, key=lambda sf: abs(sf.stat().st_mtime - result_mtime))
            # Only use if within 10 minutes — otherwise it's a false match
            if abs(closest.stat().st_mtime - result_mtime) < 600:
                candidates = [closest]

    if not candidates:
        log.debug("No GSM8K samples file found for %s", model_name)
        return 0

    samples_file = candidates[-1]  # Latest if multiple

    # Parse the JSONL. Each line is one (doc_id, filter) combination.
    # GSM8K runs both strict-match and flexible-extract filters, so each
    # doc appears twice. We split by filter for per-filter stats.
    stats: dict[str, dict[str, Any]] = {
        "strict-match": {"total": 0, "correct": 0, "empty": 0, "resp_lens": []},
        "flexible-extract": {"total": 0, "correct": 0, "empty": 0, "resp_lens": []},
    }

    try:
        with open(samples_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                filt = d.get("filter", "strict-match")
                if filt not in stats:
                    continue
                resps = d.get("resps", [[]])
                filtered = d.get("filtered_resps", [])
                em = d.get("exact_match", 0)
                resp_text = resps[0][0] if resps and resps[0] else ""
                filt_text = filtered[0] if filtered else ""
                stats[filt]["total"] += 1
                if em == 1.0:
                    stats[filt]["correct"] += 1
                if not filt_text.strip():
                    stats[filt]["empty"] += 1
                stats[filt]["resp_lens"].append(len(resp_text))
    except OSError as exc:
        log.warning("Failed to read GSM8K samples %s: %s", samples_file.name, exc)
        return 0

    s = stats["strict-match"]
    f_stat = stats["flexible-extract"]
    if s["total"] == 0:
        return 0

    strict_pct = s["correct"] / s["total"]
    flex_pct = f_stat["correct"] / f_stat["total"] if f_stat["total"] else 0.0
    mean_resp = sum(s["resp_lens"]) / len(s["resp_lens"]) if s["resp_lens"] else 0.0
    adjusted = s["correct"] / (s["total"] - s["empty"]) if (s["total"] - s["empty"]) > 0 else 0.0
    gap = flex_pct - strict_pct

    rows = [
        ("empty_count,none", float(s["empty"])),
        ("mean_response_chars,none", float(mean_resp)),
        ("exact_match_adjusted,strict-match", float(adjusted)),
        ("strict_flexible_gap,none", float(gap)),
    ]

    count = 0
    for metric_name, value in rows:
        try:
            conn.execute(
                """INSERT INTO lm_eval_results
                (comparison, model_name, task, metric_name, metric_value, result_path)
                VALUES (?, ?, 'gsm8k', ?, ?, ?)
                ON CONFLICT(comparison, model_name, task, metric_name) DO UPDATE SET
                    metric_value = excluded.metric_value,
                    result_path = excluded.result_path""",
                (comparison_name, model_name, metric_name, value, str(samples_file)),
            )
            count += 1
        except sqlite3.Error as exc:
            log.warning("Failed to insert GSM8K stat %s/%s: %s", model_name, metric_name, exc)

    log.info(
        "  %s: GSM8K sample stats — empty=%d, mean_chars=%.0f, strict=%.4f, flex=%.4f, gap=%+.4f, adjusted=%.4f",
        model_name,
        s["empty"],
        mean_resp,
        strict_pct,
        flex_pct,
        gap,
        adjusted,
    )
    return count


def import_kl_results(
    conn: sqlite3.Connection,
    comparison_name: str,
    results_dir: Path,
) -> int:
    """Import KL divergence result JSON files into kl_results table.

    Scans for ``kl_*.json`` files. Extracts summary statistics,
    skips the ``per_prompt_results`` array.

    Returns the number of variant rows imported.
    """
    files = sorted(results_dir.glob("kl_*.json"))
    if not files:
        log.warning("No kl_*.json files found in %s", results_dir)
        return 0

    count = 0
    for f in files:
        # Variant name from filename: kl_<variant>.json -> <variant>
        variant = f.stem.replace("kl_", "", 1)
        try:
            raw = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Skipping %s: %s", f.name, exc)
            continue

        kl_batchmean = raw.get("kl_divergence_batchmean")
        kl_median = raw.get("kl_divergence_per_prompt_median")
        kl_std = raw.get("kl_divergence_per_prompt_std")
        kl_min = raw.get("kl_divergence_per_prompt_min")
        kl_max = raw.get("kl_divergence_per_prompt_max")
        num_prompts = raw.get("num_prompts", 0)
        rating = raw.get("interpretation")

        try:
            conn.execute(
                """INSERT INTO kl_results
                (comparison, variant, kl_batchmean, kl_median, kl_std, kl_min, kl_max,
                 num_prompts, rating, result_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(comparison, variant) DO UPDATE SET
                    kl_batchmean = excluded.kl_batchmean,
                    kl_median = excluded.kl_median,
                    kl_std = excluded.kl_std,
                    kl_min = excluded.kl_min,
                    kl_max = excluded.kl_max,
                    num_prompts = excluded.num_prompts,
                    rating = excluded.rating,
                    result_path = excluded.result_path""",
                (
                    comparison_name,
                    variant,
                    _to_float(kl_batchmean),
                    _to_float(kl_median),
                    _to_float(kl_std),
                    _to_float(kl_min),
                    _to_float(kl_max),
                    num_prompts or 0,
                    rating,
                    str(f),
                ),
            )
            count += 1
        except sqlite3.Error as exc:
            log.warning("Failed to insert KL result %s: %s", variant, exc)

    conn.commit()
    log.info("Imported %d KL divergence results", count)
    return count


def import_weight_results(
    conn: sqlite3.Connection,
    comparison_name: str,
    results_dir: Path,
) -> int:
    """Import weight analysis JSON files into weight_results table.

    Recursively scans for ``*.json`` files under the weight results directory.
    Determines variant/variant_b/script_name from filename patterns:
      - ``<variant>/edit_vector_<variant>.json`` → single-variant
      - ``<variant>/fingerprint_<variant>.json`` → single-variant
      - ``<variant>/svd_<variant>.json`` → single-variant
      - ``<variant>/layer_analysis_<variant>.json`` → single-variant
      - ``correlation_<v1>_vs_<v2>.json`` → pairwise
      - ``multi_model_panel.json`` → global

    Returns the number of rows imported.
    """
    files = sorted(results_dir.rglob("*.json"))
    if not files:
        log.warning("No JSON files found in %s", results_dir)
        return 0

    count = 0
    for f in files:
        variant = ""
        variant_b = ""
        script_name = f.stem

        fname = f.name
        recognized = False
        # Pairwise: correlation_<v1>_vs_<v2>.json
        m = re.match(r"correlation_(.+)_vs_(.+)\.json$", fname)
        if m:
            variant, variant_b = m.group(1), m.group(2)
            script_name = "correlation"
            recognized = True
        # Global: multi_model_panel.json
        elif fname == "multi_model_panel.json":
            variant, variant_b = "", ""
            script_name = "panel_comparison"
            recognized = True
        else:
            # Single-variant scripts: try to extract variant name
            m = re.match(
                r"(edit_vector|fingerprint|svd|layer_analysis|expert_analysis)_(.+)\.json$",
                fname,
            )
            if m:
                variant = m.group(2)
                variant_b = ""
                script_name = m.group(1)
                recognized = True
            else:
                # Cross-variant low-rank / subspace analyses
                m = re.match(r"(lowrank|subspace)_(.+)_vs_(.+)\.json$", fname)
                if m:
                    variant, variant_b = m.group(2), m.group(3)
                    script_name = m.group(1)
                    recognized = True

        if not recognized:
            # Skip files that are not weight-analysis outputs (e.g. lm_eval/kl/harmbench JSONs)
            continue

        # Extract key metrics for summary
        summary: dict[str, Any] = {}
        try:
            raw = json.loads(f.read_text())
            summary = _extract_weight_summary(script_name, raw)
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Skipping %s: %s", f.name, exc)
            summary = {"imported": True, "error": str(exc)}

        try:
            conn.execute(
                """INSERT INTO weight_results
                (comparison, variant, variant_b, script_name, result_path, summary)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(comparison, variant, variant_b, script_name) DO UPDATE SET
                    result_path = excluded.result_path,
                    summary = excluded.summary""",
                (comparison_name, variant, variant_b, script_name, str(f), json.dumps(summary)),
            )
            count += 1
        except sqlite3.Error as exc:
            log.warning("Failed to insert weight result %s: %s", f.name, exc)

    conn.commit()
    log.info("Imported %d weight analysis results from %s", count, results_dir)
    return count


def import_all_results(
    conn: sqlite3.Connection,
    comparison_dir: str,
) -> dict[str, int]:
    """Import all result files for a comparison into the database.

    Looks for result files in the standard directory layout:
      - ``<comparison_dir>/results/lm_eval/`` — lm-eval results
      - ``<comparison_dir>/results/kl/`` — KL divergence results
      - ``<comparison_dir>/results/<comparison_name>/weight/`` — weight analysis

    Returns a dict with counts per import type.
    """
    base = Path(comparison_dir)
    comp_file = base / "comparison.json"
    if not comp_file.exists():
        log.error("No comparison.json found in %s", comparison_dir)
        return {}

    comp = json.loads(comp_file.read_text())
    comparison_name = comp.get("name", base.name)

    counts: dict[str, int] = {}

    # lm-eval results
    lm_eval_dir = base / "results" / "lm_eval"
    if lm_eval_dir.exists():
        counts["lm_eval"] = import_lm_eval_results(conn, comparison_name, lm_eval_dir)
    else:
        log.info("No lm_eval results directory found")

    # KL divergence results
    kl_dir = base / "results" / "kl"
    if kl_dir.exists():
        counts["kl"] = import_kl_results(conn, comparison_name, kl_dir)
    else:
        log.info("No KL results directory found")

    # Weight analysis results
    weight_dir = base / "results" / comparison_name / "weight"
    if not weight_dir.exists():
        # Try flat layout
        weight_dir = base / "results" / "weight"
    if weight_dir.exists():
        counts["weight"] = import_weight_results(conn, comparison_name, weight_dir)
    else:
        # Per-variant layout: results/<variant>/<script>_<variant>.json plus root-level
        # cross-variant files (multi_model_panel.json, correlation_*.json, ...).
        # The importer skips non-weight files, so pointing it at results/ is safe.
        per_variant = any((base / "results" / v / f"fingerprint_{v}.json").exists() for v in comp.get("variants", {}))
        if per_variant:
            counts["weight"] = import_weight_results(conn, comparison_name, base / "results")
        else:
            log.info("No weight results directory found")

    # HarmBench results (responses, classifications, scores)
    harmbench_dir = base / "results" / "harmbench"
    if harmbench_dir.exists():
        _import_harmbench(conn, comparison_name, base)
        counts["harmbench"] = len(list(harmbench_dir.glob("harmbench_*_responses.json")))
    else:
        log.info("No harmbench results directory found")

    log.info("Import complete for %s: %s", comparison_name, counts)
    return counts


def import_comparison(
    conn: sqlite3.Connection,
    comparison_dir: str,
) -> None:
    """Import comparison configuration, models, behaviors, and results.

    This is the main entry point called by ``init``. It:
    1. Registers the comparison in the ``comparisons`` table
    2. Imports models from ``comparison.json``
    3. Imports HarmBench behaviors, responses, and classifications
    4. Pre-populates ``model_stages`` for all models × stages
    5. Calls ``import_all_results()`` for any existing result files
    """
    base = Path(comparison_dir)
    comp_file = base / "comparison.json"
    if not comp_file.exists():
        log.error("No comparison.json found in %s", comparison_dir)
        return

    comp = json.loads(comp_file.read_text())
    comparison_name = comp.get("name", base.name)

    # --- Register comparison ---
    base_path = comp.get("base", "")
    base_name = base_path.split("/")[-1] or "base"
    variants = comp.get("variants", {})
    settings_json = json.dumps(comp.get("settings", {}))

    # Detect architecture
    architecture = ""
    try:
        from src.config import _detect_architecture

        architecture = _detect_architecture(base_path) or ""
    except Exception as exc:
        log.debug("Architecture detection skipped: %s", exc)

    conn.execute(
        """INSERT INTO comparisons (name, base_model, architecture, num_variants, settings_json)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET
            base_model = excluded.base_model,
            architecture = excluded.architecture,
            num_variants = excluded.num_variants,
            settings_json = excluded.settings_json""",
        (comparison_name, base_name, architecture, len(variants), settings_json),
    )
    conn.commit()
    log.info("Registered comparison: %s (%d variants)", comparison_name, len(variants))

    # --- Import models ---
    conn.execute(
        """INSERT INTO models (comparison, model_name, display_name, model_path, variant_type)
        VALUES (?, ?, ?, ?, 'base')
        ON CONFLICT(comparison, model_name) DO UPDATE SET
            display_name = excluded.display_name,
            model_path = excluded.model_path""",
        (comparison_name, base_name, "Base", base_path),
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

    # --- Import behaviors ---
    from src.db.core import load_behaviors

    behaviors = load_behaviors()
    bcount = 0
    for b in behaviors:
        try:
            conn.execute(
                "INSERT OR IGNORE INTO behaviors (behavior_id, behavior_text, semantic_category, tags) "
                "VALUES (?, ?, ?, ?)",
                (b.get("BehaviorID", ""), b.get("Behavior", ""), b.get("SemanticCategory", ""), b.get("Tags", "")),
            )
            bcount += 1
        except sqlite3.Error:
            log.warning("Failed to import behavior: %s", b.get("BehaviorID", ""))
    conn.commit()
    log.info("Imported %d behaviors", bcount)

    # --- Import HarmBench responses + classifications ---
    _import_harmbench(conn, comparison_name, base)

    # --- Init pipeline stages ---
    model_names = [base_name] + list(variants.keys())
    init_stages(conn, comparison_name, model_names)

    # --- Import any existing results ---
    import_all_results(conn, comparison_dir)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _import_harmbench(
    conn: sqlite3.Connection,
    comparison_name: str,
    base: Path,
) -> None:
    """Import HarmBench responses and classifications from result files."""
    harmbench_dir = base / "results" / "harmbench"
    classified_dir = base / "results" / "harmbench_classified"

    if not harmbench_dir.exists():
        log.info("No harmbench results found — skipping HarmBench import")
        return

    response_files = sorted(harmbench_dir.glob("harmbench_*_responses.json"))
    total_responses = 0
    total_classified = 0

    for rf in response_files:
        model_name = rf.stem.replace("harmbench_", "").replace("_responses", "")
        # Normalize: harmbench uses underscores, comparison.json may use hyphens
        model_key = model_name.replace("_", "-")

        # Find the actual model name in our models table
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
                (comparison_name, actual_model),
            )

        try:
            raw = json.loads(rf.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Skipping %s: %s", rf.name, exc)
            continue

        items = raw.get("harmbench", [])
        for item in items:
            bid = item.get("behavior_id", "")
            is_ref_kw = 1 if item.get("is_refusal") else 0
            reasoning_text = item.get("reasoning", "")
            reasoning_chars = item.get("reasoning_chars", 0) or len(reasoning_text)
            try:
                conn.execute(
                    """INSERT INTO responses
                    (comparison, model_name, behavior_id, response_text, reasoning_text,
                     reasoning_chars, completion_tokens, finish_reason, error, is_refusal_kw)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(comparison, model_name, behavior_id) DO UPDATE SET
                        response_text = excluded.response_text,
                        reasoning_text = excluded.reasoning_text,
                        reasoning_chars = excluded.reasoning_chars,
                        completion_tokens = excluded.completion_tokens,
                        finish_reason = excluded.finish_reason,
                        error = excluded.error,
                        is_refusal_kw = excluded.is_refusal_kw""",
                    (
                        comparison_name,
                        actual_model,
                        bid,
                        item.get("response", ""),
                        reasoning_text,
                        reasoning_chars,
                        item.get("completion_tokens", 0),
                        item.get("finish_reason", ""),
                        item.get("error"),
                        is_ref_kw,
                    ),
                )
                total_responses += 1
            except sqlite3.Error as exc:
                log.warning("Failed to insert response %s/%s: %s", actual_model, bid, exc)

        # Import classified results if available
        # Check both harmbench/ dir (where runner saves) and harmbench_classified/ dir
        cls_candidates = [
            harmbench_dir / f"harmbench_{model_name}_classified.json",
            harmbench_dir / f"harmbench_{model_key}_classified.json",
            classified_dir / f"harmbench_{model_name}_classified.json",
            classified_dir / f"harmbench_{model_key}_classified.json",
            classified_dir / f"{model_name}_classified.json",
            classified_dir / f"{model_key}_classified.json",
        ]
        cls_file = None
        for candidate in cls_candidates:
            if candidate.exists():
                cls_file = candidate
                break

        if cls_file is not None:
            try:
                cls = json.loads(cls_file.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("Skipping classified %s: %s", cls_file.name, exc)
                continue

            classifier = cls.get("classifier", "unknown")

            for item in cls.get("harmbench_classified", []):
                bid = item.get("behavior_id", "")
                label = "refused" if item.get("is_refusal") else "not_refused"
                is_ref = 1 if item.get("is_refusal") else 0
                try:
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
                except sqlite3.Error as exc:
                    log.warning("Failed to insert classification %s/%s: %s", actual_model, bid, exc)

        # Import score files (ASR summary + CoT analysis)
        score_candidates = [
            harmbench_dir / f"harmbench_{model_name}_scores.json",
            harmbench_dir / f"harmbench_{model_key}_scores.json",
        ]
        for score_path in score_candidates:
            if score_path.exists():
                try:
                    _import_harmbench_scores(conn, comparison_name, actual_model, score_path)
                except Exception as exc:
                    log.warning("Failed to import scores for %s: %s", actual_model, exc)
                break

        conn.commit()
        log.info("  %s: %d responses imported", actual_model, len(items))

    log.info("Total HarmBench: %d responses, %d classifications", total_responses, total_classified)


def _import_harmbench_scores(
    conn: sqlite3.Connection,
    comparison_name: str,
    model_name: str,
    score_path: Path,
) -> None:
    """Import HarmBench score summary (ASR + CoT analysis) into harmbench_scores table."""
    raw = json.loads(score_path.read_text())
    classifier = raw.get("classifier", "unknown")
    total = raw.get("total", 0)
    harmful = raw.get("harmful_count", 0)
    safe = raw.get("safe_count", 0)
    ambiguous = raw.get("ambiguous_count", 0)
    asr = raw.get("asr", 0)
    asr_excl = raw.get("asr_excluding_ambiguous", 0)
    cot_analysis = json.dumps(raw.get("cot_analysis", {}))
    by_category = json.dumps(raw.get("by_category", {}))

    conn.execute(
        """INSERT INTO harmbench_scores
        (comparison, model_name, classifier, total, harmful_count, safe_count,
         ambiguous_count, asr, asr_excl_ambig, cot_analysis, by_category, result_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(comparison, model_name, classifier) DO UPDATE SET
            total = excluded.total,
            harmful_count = excluded.harmful_count,
            safe_count = excluded.safe_count,
            ambiguous_count = excluded.ambiguous_count,
            asr = excluded.asr,
            asr_excl_ambig = excluded.asr_excl_ambig,
            cot_analysis = excluded.cot_analysis,
            by_category = excluded.by_category,
            result_path = excluded.result_path""",
        (
            comparison_name,
            model_name,
            classifier,
            total,
            harmful,
            safe,
            ambiguous,
            asr,
            asr_excl,
            cot_analysis,
            by_category,
            str(score_path),
        ),
    )
    log.info("  %s: imported scores (ASR=%.1f%%, %d harmful/%d total)", model_name, asr * 100, harmful, total)


def _extract_weight_summary(script_name: str, raw: dict[str, Any]) -> dict[str, Any]:
    """Extract key metrics from a weight analysis JSON for the summary column."""
    summary: dict[str, Any] = {"script": script_name}

    if script_name == "panel_comparison":
        # multi_model_panel.json
        if "pairwise_changed_counts" in raw:
            summary["pairwise_changed_counts"] = raw["pairwise_changed_counts"]
        if "total_tensors" in raw:
            summary["total_tensors"] = raw["total_tensors"]

    elif script_name == "fingerprint":
        scope = raw.get("scope", {})
        if scope:
            summary["changed_tensors"] = scope.get("changed_tensors")
            summary["changed_pct"] = scope.get("changed_pct")
            summary["total_tensors"] = scope.get("total_tensors")
        magnitude = raw.get("magnitude", {})
        if magnitude:
            summary["magnitude"] = magnitude
        targeting = raw.get("targeting", {})
        if targeting:
            summary["tensor_types"] = targeting.get("tensor_types", {})
            summary["num_types"] = len(targeting.get("tensor_types", {}))
            summary["layers_modified"] = targeting.get("layers_modified")
        layer_profile = raw.get("layer_profile", {})
        if layer_profile:
            summary["early_layer_pct"] = layer_profile.get("early_layer_pct")
            summary["mid_layer_pct"] = layer_profile.get("mid_layer_pct")
            summary["late_layer_pct"] = layer_profile.get("late_layer_pct")

    elif script_name == "svd":
        ranks = raw.get("effective_ranks", {})
        if ranks:
            summary["effective_ranks"] = ranks

    elif script_name == "layer_analysis":
        if "layers_modified" in raw:
            summary["layers_modified"] = raw["layers_modified"]
        if "total_layers" in raw:
            summary["total_layers"] = raw["total_layers"]

    elif script_name == "correlation":
        if "correlation_coefficient" in raw:
            summary["correlation_coefficient"] = raw["correlation_coefficient"]

    elif script_name == "edit_vector":
        if "total_changes" in raw:
            summary["total_changes"] = raw["total_changes"]

    return summary


def _to_float(value: Any) -> float | None:
    """Safely convert a value to float, returning None on failure."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None
