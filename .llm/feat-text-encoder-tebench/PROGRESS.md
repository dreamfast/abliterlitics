# Progress

> Update after every task. **Never modify `PLAN.md`** — only this file tracks status.
> Convention: `Status` = pending | in_progress | done | blocked. `Review` = deepseek:Y/N hy3:Y/N | fast-path.

> Updated: 2026-08-14
> Status: IN PROGRESS — Phase 1 complete, Phase 2 (backend) blocked on model checkpoints, Phase 3 complete

## Task status

| ID | Task | Status | Review | Commit | Notes |
|----|------|--------|--------|--------|-------|
| **Phase 1 — Core metrics (no GPU)** | | | | | |
| T1.1 | Core data types (`artifact.py`, `__init__.py`) | done | deepseek:✓ hy3:✓ | 5d5c2a6 | 30 tests; mask/dim/layers validation per reviewer feedback |
| T1.2 | CKA linear + RBF (`metrics.py`) | done | deepseek:✓ hy3:✓ | 35091ca | 27 known-answer tests; fixed RBF centering formula |
| T1.3 | Companion metrics (cosine/L2/rank/per-dim) | done | deepseek:✓ hy3:✓ | 56195d1 | 17 new tests; per_dim_drift n>=2 guard |
| T1.4 | Noise-floor + MSE damage (`compare.py`) | done | deepseek:✓ hy3:✓ | 7bb1111 | StreamReport, damage formula; masked MSE per W1 |
| T1.5 | Bootstrap CIs (`bootstrap.py`) | done | deepseek:✓ hy3:✓ | 7347fd9 | Joint ratio bootstrap (C4); assert→ValueError |
| T1.6 | Prompt suite (`suite.py`, `te_suite.yaml`) | done | deepseek:✓ hy3:✓ | 7347fd9 | 60 prompts, 10/category; fixed German typos |
| T1.7 | Report schema (`report.py`) | done | deepseek:✓ hy3:✓ | 7347fd9 | JSON + markdown; TE_RESULTS_VERSION stamped |
| T1.8 | Activation cache (`cache.py`) | done | deepseek:✓ hy3:✓ | ce75cc3 | transform field in key (C1); get_or_compute sync |
| **Phase 2 — LTX-2 backend** | | | | | |
| T2.1 | BaseBackend ABC + HookManager + FakeQuantizer | done | deepseek:✓ hy3:✓ | 1be578c | 10 tests; review found C1 path bug, fixed |
| T2.2 | ltx-core integration | blocked | — | | Needs ltx-core package + real model |
| T2.3 | LTX2Backend | blocked | — | | Needs Gemma3 12B checkpoint |
| T2.4 | ACCEPTANCE GATE — run_analysis.py | blocked | — | | Needs T2.2+T2.3 |
| T2.5 | "2% revisited" result | blocked | — | | Needs T2.4 |
| **Phase 3 — Integration** | | | | | |
| T3.1 | Dedicated Docker image (`Dockerfile.te`) | done | deepseek:✓ hy3:✓ | f75aed6 | abliterlitics-te:1.0.0, scipy+pyyaml |
| T3.2 | Schema + config | done | deepseek:✓ hy3:✓ | f75aed6 | comparison.schema.json + config.py tebench fields |
| T3.3 | _load_comparison.py exports | done | deepseek:✓ hy3:✓ | f75aed6 | TEBENCH_* env vars + VARIANT_SKIP_TE_* |
| T3.4 | DB schema + importer + stages | done | deepseek:✓ hy3:✓ | 112aa2f | te_results table; slug-based; defensive import |
| T3.5 | CLI + runner | done | deepseek:✓ hy3:✓ | 112aa2f | docker_run_cpu (C3) + container path translation (C1 fix) |
| T3.6 | Deps + lint config | done | deepseek:✓ hy3:✓ | f75aed6 | Dockerfile.te deps; pyproject deferred |
| **Phase 4 — H3 backend** | | | | | |
| T4.1 | Resolve H3 architecture (research) | pending | — | | layer-50-of-64 vs truncation |
| T4.2 | H3Backend + headline run | pending | — | | answers PSA |
| **Phase 5 — Reports** | | | | | |
| T5.1 | SVG graphs | pending | — | | |
| T5.2 | Markdown report | pending | — | | HF + Reddit |

## Milestones
- **M1**: Core validated (Phase 1 done, no GPU)
- **M2**: Oracle passed (T2.4 gate green)
- **M3**: First real result (T2.5 "2% revisited")
- **M4**: Headline (T4.2 H3 conditioning damage)
- **M5**: Publishable (T5.1-T5.2 graphs + report)

## Notes
- 2026-08-13: Plan approved v2 after 2 review rounds (deepseek + hy3). CPU-first, MSE damage,
  dedicated Docker, separate TE_RESULTS_VERSION, pooled CKA. Starting T1.1.
- 2026-08-14: Phase 1 (core metrics) complete — 8 tasks, 120 tests, all quality gates green.
  CKA linear+RBF, cosine/L2/MSE/rank_corr/per_dim_drift, MSE-based damage with noise floor,
  joint-ratio bootstrap, prompt suite (60 prompts/6 categories), JSON+markdown report, disk
  cache with transform-field key (C1). Every task reviewed by both deepseek+hy3 code reviewers.
- 2026-08-14: Phase 2 T2.1 (BaseBackend ABC + HookManager + FakeQuantizer) complete.
- 2026-08-14: Phase 3 (integration) complete — schema/config/DB/CLI/runner/Docker all wired.
  Phase 3 review found: C1 host-path-to-container bug (fixed), W1 importer defensiveness (fixed),
  W3 variant slug consistency (fixed), W5 tebench-not-configured guard (added), W7 build wiring (added).
- Remaining: T2.2-T2.5 (LTX-2 backend, needs model checkpoint + ltx-core package),
  T4.1-T4.2 (H3 backend, needs Qwen3-VL-32B), T5.1-T5.2 (reports, needs results).
  These are GPU/model-dependent and require the actual text encoder checkpoints to proceed.
