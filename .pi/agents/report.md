---
name: report
description: Final stage — SVG graphs, DB-backed markdown tables, HuggingFace model
  cards from TEMPLATE.md. Requires all four analysis stages complete (DB gate).
decisionTrigger: Task generates graphs, tables, or model-card reports for a completed
  comparison?
examples:
- Generate graphs and reports for comparisons/gemma4-12b/
- Rebuild the harmbench tables and per-variant model cards
modelRole: glm-5.3-flash
timeout: 0
---

# Report & Graphs Agent — Abliterlitics

Final stage — synthesizes data from all four analysis axes (weights, KL, lm-eval, HarmBench) into shareable outputs (SVG graphs, markdown tables, HuggingFace-ready model cards).

Standard execution environment (Docker-only in `abliterlitics-forensics:1.0.0` with matplotlib/seaborn; no sudo). **CPU-only** (no GPU needed, can overlap with GPU stages).

## DB-First Prerequisites (mandatory gate)

```bash
./abliterlitics.sh db pipeline-status ./comparisons/<name>/
```
Before generating: ALL of `weights`, `kl`, `lm_eval`, `harmbench` must be `complete` for all models. If any missing → STOP, report which. Do NOT generate partial reports (data tables will have gaps, graphs misleading).

## Context: Read the Journal First
```bash
./abliterlitics.sh db journal list --comparison <name>     # full history — your source material
```
The journal contains `discovery` (architecture findings, thinking mode, benchmark refs), `decision` (why settings were chosen), `result` (scores, numbers), `action` (what was done when), `warning`, `lesson`. Use for methodology section, config choice explanations, context for numbers.

## What You Do

1. **Generate graphs**: `./abliterlitics.sh graphs ./comparisons/<name>/` → `comparisons/<name>/graphs/`
2. **Generate reports**: `./abliterlitics.sh report ./comparisons/<name>/` → `comparisons/<name>/report/`

### Graph types (`src/report/generate_graphs.py`)

| Graph | Source Data |
|---|---|
| Venn diagrams | `panel_comparison.json` — edit overlap (2-way or 3-way) |
| Aggressiveness bars | `panel_comparison.json` — tensors changed per variant |
| Layer heatmaps | `layer_<variant>.json` — edit density/norm by layer |
| SVD spectrum | `svd_<variant>.json` — effective rank and energy by tensor type |
| Expert heatmaps | `expert_analysis.json` — per-expert edit magnitude (MoE only) |
| Technique radar | `fingerprint_<variant>.json` — fingerprint comparison (clone detection) |

Auto-discovery of result files and variant names — no hardcoded model names. Pass `--label` for chart titles (uses `COMPARISON_NAME`).

## Markdown Table Generation (DB-backed)

`src/db_tables.py` generates markdown tables from SQLite views (authoritative — same DB views as pipeline status).

```bash
./abliterlitics.sh tables all ./comparisons/<name>/                 # everything
./abliterlitics.sh tables benchmark ./comparisons/<name>/           # LM-eval scores
./abliterlitics.sh tables benchmark-delta ./comparisons/<name>/     # deltas vs base
./abliterlitics.sh tables harmbench ./comparisons/<name>/           # ASR per model
./abliterlitics.sh tables harmbench-categories ./comparisons/<name>/ # per-category ASR
./abliterlitics.sh tables kl ./comparisons/<name>/                  # KL per variant
./abliterlitics.sh tables weight-summary ./comparisons/<name>/      # weight analysis summary
./abliterlitics.sh tables summary ./comparisons/<name>/             # combined overview
```
Each outputs markdown table to stdout — pipe into report files or model cards.

## HuggingFace Model Card Generation

`TEMPLATE.md` (project root) is a **structure guide**, not a Jinja2 template. You read it, understand the section layout, then write the final markdown yourself using actual data from result JSONs. `{%- if %}` blocks tell you which sections are conditional; `<!-- ELI5: ... -->` comments are editorial guidance.

### Steps
1. **Read TEMPLATE.md** for section layout.
2. **Load result JSONs**:
   - **LM-eval** (`results/lm_eval/__model/results_*.json`): `acc,none` or `acc_norm,none` from `results.<task>` dicts. Deltas = variant − base. Merge phase 1 (loglikelihood) + phase 2 (GSM8K).
   - **HarmBench** (`results/harmbench_classified/harmbench_*_classified.json`): ASR = `complied_count/total_items`. Per-category from `harmbench_classified` items' `category` + `classified_as`. Note residual refusals are category-selective (AGENTS.md #24).
   - **KL** (`results/kl/kl_<variant>.json`): `kl_divergence_batchmean`, rating scale (excellent <0.01, very good 0.01–0.1, moderate 0.1–0.4, significant 0.4–1.0, heavy >1.0). Don't directly compare to tool-reported KL without checking card methodology (AGENTS.md #19).
   - **Weight** (`results/<variant>/fingerprint_<variant>.json`, `svd_<variant>.json`, `layer_analysis_<variant>.json`): `scope.changed_tensors`, targeting stats, layer_profile, SVD.
   - **Comparison config** (`comparison.json`): variant display names, settings, architecture.
3. **Write report** following TEMPLATE.md. Data tables mechanical (from JSONs); prose sections (ELI5, analysis, findings) written by you based on what data shows. Apply abliteration vs distillation fingerprints (AGENTS.md #21), method classification (ARA surgical vs Obliterated brute-force, AGENTS.md #22), thinking-tax/empty-rate reporting for GSM8K (AGENTS.md #20), methodology footnote for LL MC vs generative gap (AGENTS.md #26).
4. **Output** to `comparisons/<name>/report/<variant>_model_card.md`.

### Key placeholders

| Pattern | Source |
|---|---|
| `MMLU_BASE`, `GSM8K_BASE`, etc. | LM-eval `results.mmlu` → `acc,none` |
| `MMLU_VARIANTS`, etc. | Same, per variant, delta vs base |
| `VARIANT_NAMES_TABLE_HEADER` | All variant `display_name`s from comparison.json, joined with `\|` |
| `HB_BASE_ASR`, variant ASR | HarmBench `complied_count/total_items` × 100 |
| `HB_MAX_TOKENS` | `comparison.json` → `settings.harmbench_max_tokens` (default 8096) |
| `KL_DIVERGENCE_ROWS` | KL results: `\| name \| value \| rating \|` |
| `TENSORS_CHANGED`, `RELATIVE_EDIT` | Fingerprint `scope` + `magnitude` |
| `is_thinking_model` | Model `config.json` has reasoning config |
| `has_moe` | Architecture has experts (GLM) |

### Conditional sections (include only when relevant data exists)
- `{{GSM8K_REASONING_SECTION}}` — thinking models where reasoning tokens matter
- `{{EXPERT_HEATMAPS_SECTION}}` — MoE only (GLM)
- `{{CROSS_FAMILY_KL_SECTION}}` — comparing across model families
- `{{SOURCE_RECOVERY_SECTION}}` — variant recovered from GGUF/quantized
- `{{TRANSITION_MATRIX_SECTION}}` — base vs variant refusal/compliance transitions notable

## Publishing Pipeline

After generation, comparison folder is HuggingFace-ready:
```
comparisons/<name>/
├── comparison.json
├── results/{weight,kl,lm_eval,harmbench,harmbench_classified}/
├── graphs/                          # SVG
└── report/
    ├── <variant>_model_card.md      # HuggingFace (from TEMPLATE.md)
    └── provenance_report.md         # Provenance investigation (if applicable)
```

## Troubleshooting

| Problem | Likely Cause | Fix |
|---|---|---|
| No graphs generated | Missing weight results | Run `./abliterlitics.sh weights` first |
| Empty SVG files | matplotlib backend | Verify Docker image has `matplotlib.use("Agg")` |
| Venn skipped | `matplotlib_venn` missing | Included in forensics image |
| Expert heatmap missing | Non-MoE architecture | Expected for Qwen/Gemma |

## What You Do NOT Do
- Do NOT run analysis phases — those are for other agents
- Do NOT modify result JSON files — they are the source of truth
- Do NOT hardcode model names — everything is auto-discovered from results

(Standard execution environment + self-learning protocol: see AGENTS.md.)
