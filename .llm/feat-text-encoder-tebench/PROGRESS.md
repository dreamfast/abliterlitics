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
| T2.2 | ComfyUI-native loader (comfyui_loader.py) | done | deepseek:✓ hy3:✓ | bfea6bd | Import guard + selective DIT load + fail-loud checks |
| T2.3 | ZImageBackend (z_image.py) | done | deepseek:✓ hy3:✓ | b22aa3d | VERIFIED on real models, GPU; full conditioning chain |
| T2.4 | LTX2Backend (ltx2.py) | done | deepseek:✓ hy3:✓ | bfea6bd | Same pattern; CONNECTOR_OUT DiT-side for dual_linear |
| T2.5 | First real result: base-vs-heretic + quant variants | done | — | b22aa3d | Z-Image+Qwen3-4B: 3 heretic quants, damage table below |
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
| T5.1 | SVG graphs | done | deepseek:✓ hy3:✓ | 42cfa22 | 4 graphs: layer profile w/ consumption marker, damage ranking, category gradient, heatmap |
| T5.2 | HTML report | done | deepseek:✓ hy3:✓ | 42cfa22 | Self-contained te_report.html; auto key findings; methodology footer |

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
- 2026-08-14: PIVOT to ComfyUI-native backends. The models are ComfyUI-format safetensors
  (INT8 ConvRot/NVFP4) — HF transformers path was wrong. New approach: import ComfyUI's own
  model classes (comfy.text_encoders.*) via a safe import guard (argv neutralization + cpu/cuda
  mode). Verified empirically: base+heretic Qwen3-4B load with 0 missing keys; full conditioning
  chain runs; GPU works in Docker (RTX 5090).
  - ZImageBackend: Qwen3-4B (36 layers + consumed_hidden @ -2) + cap_embedder + pad-to-32 +
    context_refiner from z_image_turbo_bf16.safetensors (faithful embed_cap replication).
  - LTX2Backend: Gemma3-12B + dual_linear projection (video 4096/audio 2048); CONNECTOR_OUT
    is DiT-side for dual_linear models — declared absent.
  - FIRST REAL RESULTS (Z-Image + Qwen3-4B heretic, 60 prompts, GPU):
    * layer_35 (heretic edit site): damage 121x/97x/63x (int8/fp8/nvfp4 quant floors)
    * layer_34 = consumed_hidden (what Z-Image reads): damage 0.34x/0.89x/6.46x
    * refined_context (enters DiT): damage 0.85x/1.03x/2.76x; taboo-ratio 4.2x/3.3x/1.3x
    * KEY FINDING: heretic's abliteration edits layer 35 — one layer AFTER what Z-Image
      consumes (layer -2 of 36 = 34). The massive 121x damage barely transmits: 0.85x
      quant-noise at the DiT input for int8, but NVFP4 quant amplifies it to 2.76x.
  - Backend reviews (deepseek APPROVE, hy3 NEEDS CHANGES) — all findings fixed in b22aa3d:
    stage-order cache bug, _te_sd 24GB memory leak, --force no-op, assert->RuntimeError,
    cache-key context gap, variant-file validation.
- Remaining: H3 backend (T4.x, models available), LTX-2 base TE acquisition, GGUF loader
  (needs ComfyUI-GGUF custom node or gguf-lib dequant path — RESEARCH.md R8).
- 2026-08-14: External base-quant preparation complete (05780a1). Quant/abliteration
  decomposition: te_reference matched-quant comparisons + dual-run (__vs_base) + real
  quant_floor_file. Two reviewer rounds: dual-run architecture (combined column was a
  duplicate of pure ablation), floor_base inf bug, skip-gate ordering (two-phase flow).
  All verified via stand-in simulation (pure_ab=0.00x for identical files — math proof),
  fully reverted. Files expected from external tool:
  qwen3-4b_{int8,fp8_e4m3fn,nvfp4}.safetensors in comfyui-models/text_encoders/.

## GGUF section (2026-08-13, uncommitted)
- G1-G3: vendored city96 dequant (bit-exact vs gguf numpy reference on real tensors),
  gguf_loader (fail-loud qwen3 key-set validation, GGUFMeta with measured bpw),
  z_image seam + artifact labeling gguf_<quant_label>, GGUFMeta->JSON threading.
  deepseek: APPROVE (hy3 aborted -> degraded mode). Findings fixed: assert->ValueError,
  BF16 container refusal, filename/content consistency warning, orig_shape metadata tests.
- G4-G5: 16 GGUF variants (8 quants x base/heretic, te_reference dual-runs),
  abliterlitics-te:1.1.0 (gguf==0.19.0) built + smoke-tested.
- G6: _gguf_section ladder (bpw from persisted GGUFMeta, not constants),
  graph_gguf_ladder SVG, decomposition suffix tuple (mxfp8 BEFORE fp8 — endswith trap),
  findings aggregation (16 GGUF variants -> 2 summary lines), GGUF methodology para.
- G8 gates: Gate A base_gguf_f16 = 0.0008x floor (loader reconstruction ~perfect).
  Gate B cross-loader ratio 1.000 (33.26x vs 33.27x safetensors). Ladder: Q8_0 0.3x,
  Q6_K 2.8x, Q5_K 7-8x, Q4_K 24-30x, Q3_K_M 83x; heretic pure-ablation survives all
  quants (33-47x); divergence 1.0-1.6% of noise-control for all 16.
- Hotfixes found during real-data verification (regression tested): graph crashed on
  gguf=None mixin variants; heretic_mxfp8 decomposition row lost pure_ab (endswith fp8);
  ladder 'vs' annotation double-escaped into visible text (user-reported).
- 190 tests, ruff+mypy --strict clean. Lesson #28b bit again (backgrounded runner died
  at 7/16 — foreground rerun; skip-gates + cache made it cheap).

## R2/R3 session (2026-08-13, commits 10f5d96 57d1e69 68131ed)
- R1 grouping committed (10f5d96): 5-group family×container×quant ordering across
  damage/divergence/ladder tables; damage_ranking + category_gradient graphs
  redesigned (DiT-input-only log bars, two-panel base/heretic); heatmap grouped;
  layer profile filtered to safetensors. User manually reviewed (sub-agents skipped).
- R2a directional (57d1e69): directional.py + te-directional CLI. RESULT: heretic edit
  touches taboo axis (cos ~ -0.15, z 50-134) but 97-99% orthogonal energy, no
  toward-innocent movement — NOT a taboo→harmless mapping. 14 matched-quant pairs.
- R2b/c (57d1e69): per-step divergence curves SVG; semantic-unit definition documented
  (1770 distinct-prompt pairs); CKA/MSE-disconnect auto-finding.
- R3 A/B (57d1e69 + 68131ed): run_ab_test.py vs live ComfyUI (host.docker.internal:8191,
  Z-Image Turbo). User-supplied workflows moved to comparisons/te-zimage/workflows/
  (safetensors node 35, GGUF node 38 — auto-dispatched). Stress prompts researched:
  'ABLITERLITICS' chrome graffiti scene w/ countable elements + spray-can label.
  Live bugs fixed: prompt_id polling hang (ComfyUI assigns its own), /view subfolder
  404, const redeclaration. Perf: server_gen_ms, queue_number, cached_nodes, te_cached
  (true cold/warm). 112 images (28 TEs × 2 prompts × 2 passes, ~9 min, fixed seed 42,
  cold==warm byte-identical). GGUF cold +4-6s vs safetensors; warm 4s flat.
- JS comparator (68131ed): dropdown prompt/A/B + swap, hover clip-path reveal, metric
  captions per side, timing table. Images re-encoded 768px/q80 (report 6.9MB).
  DB: te_ab_images 332 rows (paths+timing only, no blobs). node --check clean.
- Remaining: full-matrix re-runs not needed (all cached/cheap); next comparison
  could extend A/B to more seeds/prompts; publish writeup (HF full + Reddit condensed).
