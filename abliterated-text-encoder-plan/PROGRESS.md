# Progress

> Update after every task. **Never modify `PLAN.md`** — only this file tracks status.
> Convention: `Status` = pending | in_progress | done | blocked. `Review` = hy3:✓/✗ deepseek:✓/✗ | fast-path.

> Updated: 2026-08-13
> Status: NOT STARTED — planning complete, awaiting relocation to standalone project folder

## Task status

| ID | Task | Status | Review | Commit | Notes |
|----|------|--------|--------|--------|-------|
| **Phase 0 — Setup** | | | | | |
| T0.1 | Repo scaffold + tooling | pending | — | | `pyproject`, `ruff`, `pytest`, module layout |
| T0.2 | Determinism + dtype policy | pending | — | | |
| **Phase 1 — Core metrics** | | | | | **No GPU / no model deps** |
| T1.1 | Core data types (`artifact.py`) | pending | — | | |
| T1.2 | CKA linear + RBF | pending | — | | known-answer tests = correctness gate |
| T1.3 | Companion metrics (cosine/L2/rank/per-dim) | pending | — | | |
| T1.4 | Noise-floor + relative damage | pending | — | | |
| T1.5 | Bootstrap CIs | pending | — | | |
| T1.6 | Per-category breakdown | pending | — | | |
| T1.7 | Report schema + renderers | pending | — | | |
| **Phase 2 — Suite + harness** | | | | | |
| T2.1 | Prompt suite (`suite.yaml`) | pending | — | | ≥80/category; multilingual stratum |
| T2.2 | Comparison runner (`compare()`) | pending | — | | |
| **Phase 3 — LTX-2 backend** | | | | | **Reference + acceptance oracle** |
| T3.1 | LTX-2 loader (Gemma3 + ltx-core) | pending | — | | |
| T3.2 | RAW_HIDDEN → reproduce blog | pending | — | | **GATE** — innocent 0.869 … extreme 0.825 |
| T3.3 | PROJECTED stage | pending | — | | re-derive "layer48 = 2.04%, rank 26/49" |
| T3.4 | CONNECTOR_OUT stage | pending | — | | true DiT input |
| T3.5 | "2% revisited" result | pending | — | | blog-ready |
| **Phase 4 — H3 backend** | | | | | **Headline** |
| T4.1 | Resolve H3 architecture (research) | pending | — | | ⚠️ layer-50-of-64 vs truncation |
| T4.2 | H3 loader + extraction | pending | — | | ComfyUI validation gate |
| T4.3 | H3 headline run | pending | — | | answers PSA's "if anything" |
| **Phase 5 — LTX-2.5 backend** | | | | | **Novelty** |
| T5.1 | Resolve LTX-2.5 extractor (research) | pending | — | | ⚠️ V2 + video/audio |
| T5.2 | LTX-2.5 loader + dual projection | pending | — | | two streams |
| T5.3 | Video/audio split result | pending | — | | unreported axis |
| **Phase 6 — Quant sweep** | | | | | **Second headline** |
| T6.1 | Variant loaders (fp8/nxfp4/GGUF) | pending | — | | gguf dequant fallback |
| T6.2 | Damage ranking table | pending | — | | recommend H3 |
| T6.3 | Heretic × quant interaction | pending | — | | |
| **Phase 7 — Polish** | | | | | |
| T7.1 | CLI | pending | — | | |
| T7.2 | Multilingual extension | pending | — | | cross-lingual drift |
| T7.3 | Reproducibility + docs | pending | — | | optional: DiT-step divergence |

## Milestones

- **M1 — Core validated (no GPU):** Phase 1 + 2 done. `compare()` works on synthetic artifacts.
- **M2 — Oracle passed:** T3.2 green. The tool reproduces the author's published numbers → trustworthy for novel claims.
- **M3 — First real result:** T3.5 (LTX-2 "2% revisited").
- **M4 — Headline:** T4.3 (H3 conditioning damage, in fp8-floor units, per category).
- **M5 — Novelty:** T5.3 (LTX-2.5 video/audio split).
- **M6 — Practical guide:** T6.2 (quant damage ranking).

## Notes

- 2026-08-13: Planning package authored. Scope locked to LLM-TE models (H3, LTX-2, LTX-2.5). Loader = standalone (ComfyUI only for one-time validation gate). First backend = LTX-2 (author's published cosine table is the acceptance oracle), then H3 (headline). KL rejected as primary metric in favor of CKA + noise-floor-relative reporting. Folder to be relocated to a standalone project outside ComfyUI before T0.1.
- Critical checkpoint: **T3.2** — do not proceed past it until the blog's cosine numbers reproduce within tolerance.
