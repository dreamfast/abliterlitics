"""SQLite database for Abliterlitics — schema definitions.

All CREATE TABLE/VIEW/INDEX statements. The database tracks:
- Comparisons and their configurations
- Models per comparison (base + variants)
- Pipeline stage status per model
- HarmBench behaviors, responses, classifications, reviews, overrides
- Weight analysis results (per-variant, pairwise, global)
- KL divergence results per variant
- LM-eval benchmark results per model per task
"""

from __future__ import annotations

SCHEMA = """
-- ============================================================
-- HarmBench behavior definitions (shared across all models)
-- ============================================================
CREATE TABLE IF NOT EXISTS behaviors (
    behavior_id    TEXT PRIMARY KEY,
    behavior_text  TEXT NOT NULL,
    semantic_category TEXT NOT NULL,
    tags           TEXT NOT NULL DEFAULT ''
);

-- ============================================================
-- Comparisons (top-level tracking)
-- ============================================================
CREATE TABLE IF NOT EXISTS comparisons (
    name           TEXT PRIMARY KEY,              -- e.g. 'gemma4-e4b'
    base_model     TEXT NOT NULL,                 -- path to base model
    architecture   TEXT NOT NULL DEFAULT '',
    num_variants   INTEGER NOT NULL DEFAULT 0,
    settings_json  TEXT NOT NULL DEFAULT '{}',
    created_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

-- ============================================================
-- Models per comparison (base + variants)
-- Composite PK (comparison, model_name) for multi-comparison support
-- ============================================================
CREATE TABLE IF NOT EXISTS models (
    comparison     TEXT NOT NULL DEFAULT '',
    model_name     TEXT NOT NULL,
    display_name   TEXT NOT NULL,
    model_path     TEXT NOT NULL,
    variant_type   TEXT NOT NULL DEFAULT 'variant',  -- 'base' or 'variant'
    PRIMARY KEY (comparison, model_name),
    FOREIGN KEY (comparison) REFERENCES comparisons(name) ON DELETE CASCADE
);

-- ============================================================
-- Pipeline stage tracking (per model per stage)
-- ============================================================
CREATE TABLE IF NOT EXISTS model_stages (
    comparison     TEXT NOT NULL,
    model_name     TEXT NOT NULL,
    stage          TEXT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'pending'
                    CHECK(status IN ('pending','running','completed','failed','skipped')),
    started_at     TEXT,
    finished_at    TEXT,
    heartbeat_at   TEXT,
    result_path    TEXT,
    error_message  TEXT,
    duration_secs  INTEGER,
    metadata       TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (comparison, model_name, stage),
    FOREIGN KEY (comparison) REFERENCES comparisons(name) ON DELETE CASCADE
);

-- ============================================================
-- HarmBench raw model responses
-- ============================================================
CREATE TABLE IF NOT EXISTS responses (
    comparison      TEXT NOT NULL DEFAULT '',
    model_name      TEXT NOT NULL,
    behavior_id     TEXT NOT NULL,
    response_text   TEXT NOT NULL DEFAULT '',
    reasoning_text  TEXT NOT NULL DEFAULT '',
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    finish_reason   TEXT NOT NULL DEFAULT '',
    error           TEXT,
    is_refusal_kw   INTEGER,
    PRIMARY KEY (comparison, model_name, behavior_id),
    FOREIGN KEY (comparison, model_name) REFERENCES models(comparison, model_name),
    FOREIGN KEY (behavior_id) REFERENCES behaviors(behavior_id)
);

-- ============================================================
-- Classifier results (one row per classifier version per response)
-- ============================================================
CREATE TABLE IF NOT EXISTS classifications (
    comparison      TEXT NOT NULL DEFAULT '',
    model_name      TEXT NOT NULL,
    behavior_id     TEXT NOT NULL,
    classifier      TEXT NOT NULL,
    label           TEXT NOT NULL,               -- 'refused' or 'not_refused'
    is_refusal      INTEGER NOT NULL,            -- 1=refused, 0=complied
    layer           TEXT NOT NULL DEFAULT '',
    reason          TEXT NOT NULL DEFAULT '',
    confidence      TEXT NOT NULL DEFAULT 'high',
    classified_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    PRIMARY KEY (comparison, model_name, behavior_id, classifier),
    FOREIGN KEY (comparison, model_name) REFERENCES models(comparison, model_name),
    FOREIGN KEY (behavior_id) REFERENCES behaviors(behavior_id)
);

-- ============================================================
-- LLM review results (one row per reviewer per response)
-- ============================================================
CREATE TABLE IF NOT EXISTS llm_reviews (
    comparison      TEXT NOT NULL DEFAULT '',
    model_name      TEXT NOT NULL,
    behavior_id     TEXT NOT NULL,
    reviewer_model  TEXT NOT NULL,
    verdict         TEXT NOT NULL,               -- 'refused' or 'complied'
    review_reasoning TEXT NOT NULL DEFAULT '',
    reviewed_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    PRIMARY KEY (comparison, model_name, behavior_id, reviewer_model),
    FOREIGN KEY (comparison, model_name) REFERENCES models(comparison, model_name),
    FOREIGN KEY (behavior_id) REFERENCES behaviors(behavior_id)
);

-- ============================================================
-- Human overrides (final authority)
-- ============================================================
CREATE TABLE IF NOT EXISTS human_overrides (
    comparison      TEXT NOT NULL DEFAULT '',
    model_name      TEXT NOT NULL,
    behavior_id     TEXT NOT NULL,
    verdict         TEXT NOT NULL,               -- 'refused' or 'complied'
    reason          TEXT NOT NULL DEFAULT '',
    overridden_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    overridden_by   TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (comparison, model_name, behavior_id),
    FOREIGN KEY (comparison, model_name) REFERENCES models(comparison, model_name),
    FOREIGN KEY (behavior_id) REFERENCES behaviors(behavior_id)
);

-- ============================================================
-- Weight analysis results (per-variant, pairwise, or global)
-- ============================================================
CREATE TABLE IF NOT EXISTS weight_results (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    comparison     TEXT NOT NULL,
    variant        TEXT NOT NULL DEFAULT '',     -- empty string for global scripts
    variant_b      TEXT NOT NULL DEFAULT '',     -- empty string unless pairwise
    script_name    TEXT NOT NULL,
    result_path    TEXT NOT NULL,
    summary        TEXT NOT NULL DEFAULT '{}',
    created_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    UNIQUE(comparison, variant, variant_b, script_name),
    FOREIGN KEY (comparison) REFERENCES comparisons(name) ON DELETE CASCADE
);

-- ============================================================
-- KL divergence results per variant
-- ============================================================
CREATE TABLE IF NOT EXISTS kl_results (
    comparison     TEXT NOT NULL,
    variant        TEXT NOT NULL,
    kl_batchmean   REAL,
    kl_median      REAL,
    kl_std         REAL,
    kl_min         REAL,
    kl_max         REAL,
    num_prompts    INTEGER NOT NULL DEFAULT 0,
    rating         TEXT,
    result_path    TEXT,
    created_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    PRIMARY KEY (comparison, variant),
    FOREIGN KEY (comparison) REFERENCES comparisons(name) ON DELETE CASCADE
);

-- ============================================================
-- LM-eval benchmark results per model per task
-- ============================================================
CREATE TABLE IF NOT EXISTS lm_eval_results (
    comparison     TEXT NOT NULL,
    model_name     TEXT NOT NULL,
    task           TEXT NOT NULL,
    metric_name    TEXT NOT NULL,
    metric_value   REAL NOT NULL,
    metric_stderr  REAL,
    result_path    TEXT,
    created_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    PRIMARY KEY (comparison, model_name, task, metric_name),
    FOREIGN KEY (comparison) REFERENCES comparisons(name) ON DELETE CASCADE
);

-- ============================================================
-- Indexes for query-heavy columns
-- ============================================================
CREATE INDEX IF NOT EXISTS idx_model_stages_comp
    ON model_stages(comparison, model_name);
CREATE INDEX IF NOT EXISTS idx_model_stages_status
    ON model_stages(comparison, status);
CREATE INDEX IF NOT EXISTS idx_lm_eval_comp_model
    ON lm_eval_results(comparison, model_name);
CREATE INDEX IF NOT EXISTS idx_lm_eval_task
    ON lm_eval_results(comparison, task);
CREATE INDEX IF NOT EXISTS idx_weight_comp
    ON weight_results(comparison);
CREATE INDEX IF NOT EXISTS idx_responses_comp
    ON responses(comparison, model_name);
CREATE INDEX IF NOT EXISTS idx_classifications_comp
    ON classifications(comparison, model_name);

-- ============================================================
-- Views: Latest classification per response
-- ============================================================
CREATE VIEW IF NOT EXISTS v_latest_classification AS
    SELECT comparison, model_name, behavior_id, label, is_refusal,
           layer, reason, confidence, classifier
    FROM classifications c1
    WHERE classifier = (
        SELECT c2.classifier FROM classifications c2
        WHERE c2.comparison = c1.comparison
          AND c2.model_name = c1.model_name
          AND c2.behavior_id = c1.behavior_id
        ORDER BY c2.classified_at DESC LIMIT 1
    );

-- ============================================================
-- Views: Review status per response
-- ============================================================
CREATE VIEW IF NOT EXISTS v_review_status AS
    SELECT
        r.comparison,
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
        AND r.behavior_id = ho.behavior_id;

-- ============================================================
-- Views: HarmBench ASR summary (computed from existing tables)
-- Single source of truth — no duplicate summary tables needed.
-- ============================================================
CREATE VIEW IF NOT EXISTS v_harmbench_summary AS
    SELECT
        m.comparison,
        r.model_name,
        m.display_name,
        m.variant_type,
        COUNT(*) AS total_behaviors,
        SUM(CASE
            WHEN ho.verdict IS NOT NULL THEN CASE WHEN ho.verdict = 'complied' THEN 1 ELSE 0 END
            WHEN lr.verdict IS NOT NULL THEN CASE WHEN lr.verdict = 'complied' THEN 1 ELSE 0 END
            WHEN c.is_refusal = 0 THEN 1
            ELSE 0
        END) AS complied,
        SUM(CASE
            WHEN ho.verdict IS NOT NULL THEN CASE WHEN ho.verdict = 'refused' THEN 1 ELSE 0 END
            WHEN lr.verdict IS NOT NULL THEN CASE WHEN lr.verdict = 'refused' THEN 1 ELSE 0 END
            WHEN c.is_refusal = 1 THEN 1
            ELSE 0
        END) AS refused,
        ROUND(CAST(SUM(CASE
            WHEN ho.verdict IS NOT NULL THEN CASE WHEN ho.verdict = 'complied' THEN 1 ELSE 0 END
            WHEN lr.verdict IS NOT NULL THEN CASE WHEN lr.verdict = 'complied' THEN 1 ELSE 0 END
            WHEN c.is_refusal = 0 THEN 1
            ELSE 0
        END) AS REAL) / COUNT(*) * 100, 1) AS asr_pct
    FROM responses r
    JOIN models m
        ON r.comparison = m.comparison
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
    GROUP BY m.comparison, r.model_name, m.display_name, m.variant_type;

-- ============================================================
-- Views: HarmBench per-category ASR
-- ============================================================
CREATE VIEW IF NOT EXISTS v_harmbench_by_category AS
    SELECT
        m.comparison,
        r.model_name,
        b.semantic_category AS category,
        COUNT(*) AS total,
        SUM(CASE
            WHEN ho.verdict IS NOT NULL THEN CASE WHEN ho.verdict = 'complied' THEN 1 ELSE 0 END
            WHEN lr.verdict IS NOT NULL THEN CASE WHEN lr.verdict = 'complied' THEN 1 ELSE 0 END
            WHEN c.is_refusal = 0 THEN 1
            ELSE 0
        END) AS complied,
        ROUND(CAST(SUM(CASE
            WHEN ho.verdict IS NOT NULL THEN CASE WHEN ho.verdict = 'complied' THEN 1 ELSE 0 END
            WHEN lr.verdict IS NOT NULL THEN CASE WHEN lr.verdict = 'complied' THEN 1 ELSE 0 END
            WHEN c.is_refusal = 0 THEN 1
            ELSE 0
        END) AS REAL) / COUNT(*) * 100, 1) AS asr_pct
    FROM responses r
    JOIN models m
        ON r.comparison = m.comparison
        AND r.model_name = m.model_name
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
    GROUP BY m.comparison, r.model_name, b.semantic_category;

-- ============================================================
-- Views: Pipeline overview (one row per comparison)
-- ============================================================
CREATE VIEW IF NOT EXISTS v_pipeline_status AS
    SELECT
        c.name AS comparison,
        c.architecture,
        c.num_variants,
        COUNT(DISTINCT ms.model_name) AS tracked_models,
        SUM(CASE WHEN ms.status = 'completed' THEN 1 ELSE 0 END) AS completed_stages,
        SUM(CASE WHEN ms.status = 'failed' THEN 1 ELSE 0 END) AS failed_stages,
        SUM(CASE WHEN ms.status = 'running' THEN 1 ELSE 0 END) AS running_stages,
        SUM(CASE WHEN ms.status = 'skipped' THEN 1 ELSE 0 END) AS skipped_stages,
        COUNT(ms.stage) AS total_stages,
        CASE
            WHEN COUNT(ms.stage) = 0 THEN 'not_started'
            WHEN SUM(CASE WHEN ms.status = 'failed' THEN 1 ELSE 0 END) > 0 THEN 'has_failures'
            WHEN SUM(CASE WHEN ms.status = 'running' THEN 1 ELSE 0 END) > 0 THEN 'running'
            WHEN SUM(CASE WHEN ms.status IN ('completed', 'skipped') THEN 1 ELSE 0 END)
                 = COUNT(ms.stage) THEN 'completed'
            ELSE 'in_progress'
        END AS overall_status,
        c.updated_at
    FROM comparisons c
    LEFT JOIN model_stages ms ON c.name = ms.comparison
    GROUP BY c.name;
"""
