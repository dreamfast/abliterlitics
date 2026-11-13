# Implementation Plan

> **Read first when building.** This is the operational plan. `PROGRESS.md` tracks live status; the reference docs (`ARCHITECTURE.md`, `METRICS.md`, `RESEARCH.md`, `MODELS.md`) hold the "why."

**Scope:** LLM text-encoder models only — H3 (Qwen3-VL-32B), LTX-2 (Gemma 3 12B), LTX-2.5 (Gemma 4 12B). Flux/Krea (T5+CLIP) deliberately out. → `README.md`.

**Hard rules:**
1. **CKA primary, never KL** for conditioning (`METRICS.md` §1–2).
2. **Every number reported relative to a noise floor** (`METRICS.md` §4).
3. **Core never imports a model** (`ARCHITECTURE.md` §1).
4. **One forward pass per (ckpt, batch)**; stages via hooks.
5. **Standalone loader**, ComfyUI used only for the one-time per-backend validation gate.
6. Measurement, not advocacy — never claim "uncensoring works" (`RESEARCH.md` R5–R6).

---

## Phase 0 — Setup

### T0.1 Repo scaffold + tooling
**Goal:** runnable empty package, deps, lint, test runner.
- Create `tebench/`, `core/`, `backends/`, `prompts/`, `tests/` per `ARCHITECTURE.md` §7.
- `pyproject.toml`: python ≥3.11, deps `torch transformers safetensors numpy scipy pyyaml jinja2`, dev `pytest ruff pytest-cov`.
- `ruff` config, `.gitignore`, `README.md` stub.
- **Acceptance:** `pip install -e .` works; `pytest` runs (0 tests, exit 0); `ruff check .` clean.

### T0.2 Determinism + dtype policy
**Goal:** no hidden nondeterminism.
- Central config: global seed, default `dtype="bf16"`, `device="cuda"`, fp32-for-math flag.
- **Acceptance:** running any metric twice on identical tensors gives bit-identical results.

---

## Phase 1 — Core metrics (no model)

### T1.1 Core data types
**Files:** `core/artifact.py`.
- `Stage`, `Pooling`, `StreamSpec`, `Stream`, `ConditioningArtifact` per `ARCHITECTURE.md` §2.
- Construction invariants (fp32 cast, mask rules, length alignment).
- **Tests (`tests/test_artifact.py`):** shape/type invariants; reject mismatched prompt/category lengths; fp32 cast on load.

### T1.2 CKA (linear + RBF)
**Files:** `core/metrics.py`.
- `cka_linear` per `METRICS.md` §2 (vendor the reference impl).
- `cka_rbf` with optional sigma (median-heuristic default).
- **Tests (`tests/test_metrics.py`) — known-answer, the critical correctness gate:**
  - `cka(X, X) == 1.0` (identity).
  - `cka(X, X @ R) ≈ 1.0` for random orthogonal `R` (orthogonal invariance).
  - `cka(X, s·X) == 1.0` for scalar `s` (isotropic-scale invariance).
  - `cka(X, Y) ≈ 0` for independent random `X, Y` (independence).
  - symmetric: `cka(X,Y) == cka(Y,X)`.
  - bounded `[0,1]`.

### T1.3 Companion metrics
**Files:** `core/metrics.py`.
- `cosine_all_tokens` (masked), `cosine_last_token`, `l2_distance`, `mse`, `rank_correlation` (Spearman of pairwise distances), `per_dim_drift`.
- **Tests:** cosine of identical = 1; cosine of antipodal = −1; rank_corr of identical geometry = 1; per_dim_drift zero on identical.

### T1.4 Noise-floor + relative damage
**Files:** `core/compare.py`.
- `damage_fp8_units = (m(base, variant) − m(base, base_fp32)) / (m(base, base_fp8) − m(base, base_fp32))` (`METRICS.md` §4).
- Handle zero-denominator edge (quant floor ≈ numerical floor) gracefully.
- **Tests:** damage = 1.0 when variant == base_fp8 (by construction); damage ≈ 0 when variant == base.

### T1.5 Bootstrap CIs
**Files:** `core/bootstrap.py`.
- Nonparametric over prompt axis, `B=1000`, percentile CIs, vectorized.
- **Tests:** CI contains the point estimate; narrower with more prompts; deterministic with seed.

### T1.6 Per-category breakdown
**Files:** `core/compare.py`.
- Group by `categories`, compute per-category CKA + CIs; category-discrimination ratio (between/within-category divergence).
- **Tests:** synthetic categories recover known grouping.

### T1.7 Report schema + renderers
**Files:** `core/report.py`.
- `Report`/`VariantReport`/`StreamReport`; markdown + json renderers (html optional later).
- Fidelity index = pooled damage in fp8 units across streams.
- **Tests:** round-trip `to_json` → reconstruct; markdown contains all stream rows.

**Phase 1 exit:** the entire core is correct and unit-tested with **zero GPU and zero model dependency**. This is the foundation everything else trusts.

---

## Phase 2 — Prompt suite + comparison harness

### T2.1 Prompt suite
**Files:** `prompts/suite.yaml`, `core/suite.py`.
- Stratified categories: `innocent, mild, violent, sexual, extreme, multilingual`. ≥80 prompts/category.
- Loader → `(prompts, categories)`; tokenization-alignment assertion helper.
- **Acceptance:** loads; counts per category ≥ 80.

### T2.2 Comparison runner
**Files:** `core/compare.py` (the `compare()` orchestrator).
- Assert base/variant share prompts, categories, tokenization hash.
- For each stream × stage present in both artifacts → `StreamReport`; assemble `Report`.
- **Tests (fake tensors):** end-to-end on synthetic artifacts produces a well-formed `Report`.

**Phase 2 exit:** `compare(base_art, variant_art, ...)` → `Report`, fully working on hand-built artifacts. Ready to plug a real backend.

---

## Phase 3 — LTX-2 backend  (reference + acceptance oracle)

> **Why first:** the author's published cosine table is a free correctness oracle. If the tool reproduces it, the core is proven before we trust it for novel claims.

### T3.1 LTX-2 loader
**Files:** `backends/ltx2.py`, `backends/base.py`.
- Load Gemma 3 12B via `transformers` (`Gemma3ForConditionalGeneration`, use `.model` to skip lm_head), pad to 1024.
- Vendor/call `ltx-core` feature extractor V1 + `Embeddings1DConnector` (or replicate faithfully from `feature_extractor.py` / `embeddings_connector.py`).
- Weight mapping per `MODELS.md`.
- **Acceptance:** loads without OOM on target GPU; hidden_states shape correct.

### T3.2 RAW_HIDDEN extraction → reproduce the blog  ← ACCEPTANCE TEST
**Goal:** validate the entire core against a known answer.
- Encode base + `DreamFast/gemma-3-12b-it-heretic` over the suite; capture `RAW_HIDDEN`, last-token cosine.
- **Acceptance (gate):** reproduce within tolerance:
  - innocent 0.869, mild 0.850, violent 0.836, sexual 0.827, extreme 0.825.
  - per-layer: layers 0–47 ≈ 1.0, layer 48 ≈ 0.859 / 0.820.
- **If this fails, stop and debug the core — nothing downstream is trustworthy until it passes.**

### T3.3 PROJECTED stage
- Hook `feature_extractor` output (post `aggregate_embed`); capture `[n, L, 3840]`.
- Re-derive the author's "layer 48 = 2.04%, rank 26/49" from the projection weight as a cross-check.

### T3.4 CONNECTOR_OUT stage
- Hook `Embeddings1DConnector` output; capture the true DiT-input conditioning.
- Exclude learnable-register tokens (pad replacements) from comparison via mask.

### T3.5 "2% revisited" result
- Full `compare()` across all three stages with CKA + noise floor + CIs.
- **Deliverable:** the corrected measurement of how much abliteration actually reaches the LTX-2 DiT (vs the 2% projection-weight proxy). Blog-ready.

**Phase 3 exit:** core validated against a published oracle; first real result produced.

---

## Phase 4 — H3 backend  (headline)

### T4.1 Resolve H3 architecture (research, ⚠️ load-bearing) — BEFORE T4.2
- Load the published MiniMax-H3 text-encoder ckpt; count transformer blocks; find the extraction index and the conditioning projection module from the real loader (FastVideo `MiniMaxH3Qwen3VLConditioner` or ComfyUI).
- Decide: **layer-50-of-64** vs **50-layer truncation**.
- Document findings in `MODELS.md` (resolve the ⚠️).
- **Acceptance:** written determination + the exact module path to hook.

### T4.2 H3 loader + extraction
**Files:** `backends/h3.py`.
- Load Qwen3-VL-32B LM part (`language_model.model`), skip lm_head, `output_hidden_states=True`.
- Hooks at the resolved layer (RAW_HIDDEN) + conditioning projection (PROJECTED/DIT_INPUT).
- Single stream per stage.
- **Acceptance:** one-prompt ComfyUI validation gate passes (standalone ≈ ComfyUI capture, bf16 tol).

### T4.3 H3 headline run
- Base vs `ethanfel/Qwen3-VL-32B-Ultra-Heretic-MiniMax-H3-ComfyUI-INT8-ConvRot` (and/or NVFP4 variant).
- Full `compare()` + report.
- **Deliverable:** "how much does an abliterated TE perturb H3's conditioning, in fp8-floor units, per category." The headline blog result + the answer to the PSA's "if anything" hedge.

**Phase 4 exit:** the contested question has defensible numbers.

---

## Phase 5 — LTX-2.5 backend  (novelty: video/audio split)

### T5.1 Resolve LTX-2.5 extractor (research, ⚠️)
- Confirm Gemma 4 layer count + aggregated layers; confirm V2 extractor + exact norm/rescale for the shipped ckpt. Resolve `MODELS.md` ⚠️s.

### T5.2 LTX-2.5 loader + dual projection
**Files:** `backends/ltx25.py`.
- Gemma 4 (unified) backbone; V2 feature extractor; capture `video_proj` + `audio_proj` + their connector outputs as **two streams**.
- ComfyUI validation gate.

### T5.3 Video/audio split result
- `compare()` per stream independently.
- **Deliverable:** "does abliteration damage LTX-2.5 speech/phonetic conditioning differently from visual conditioning?" Unreported axis.

**Phase 5 exit:** novel finding.

---

## Phase 6 — Quantization sweep  (second headline)

### T6.1 Variant loaders
**Files:** `backends/base.py` helpers.
- fp8 (`torch.float8_e4m3fn`), nxfp4, GGUF Q2–Q8 (via `gguf` lib + dequant-to-safetensors; note ComfyUI-GGUF Gemma support is incomplete — `RESEARCH.md` R8).

### T6.2 Damage ranking
- Run all variants on one model (recommend H3 — most contested) on the same metric axis.
- **Deliverable:** a single table — heretic vs fp8 vs nxfp4 vs GGUF Q2/Q4/Q6/Q8 — ranked by fidelity index. Directly answers "is heretic damage ≪ or ≫ one fp8 step?" and the GGUF-quality debate.

### T6.3 Interaction (heretic × quant)
- Combinations (heretic+fp8, heretic+nxfp4) — does damage compound or saturate?

**Phase 6 exit:** the practical buyer's guide ("what does each compression actually cost?").

---

## Phase 7 — Polish

### T7.1 CLI
**Files:** `cli.py`.
- `tebench compare --backend h3 --base <ckpt> --variant <ckpt[,..]> --prompts suite.yaml --out report.md`
- `tebench sweep ...`, `tebench validate <ckpt> <comfyui-ref.npz>`.

### T7.2 Multilingual extension
- Expand the multilingual stratum; produce the cross-lingual drift profile (`RESEARCH.md` R9).

### T7.3 Reproducibility + docs
- Top-level `README.md` with run instructions; record env (torch/transformers/ltx-core versions) in every `Report`.
- Optional: Phase-7 stretch downstream DiT-step divergence (`METRICS.md` §3.5) on the headline model only.

---

## Dependency graph

```
T0.1 ─┬─► T0.2
      ├─► T1.1 ─► T1.2 ─► T1.3 ─► T1.4 ─► T1.5 ─► T1.6 ─► T1.7
      │                                              │
      └──────────────────────────────────► T2.1 ─► T2.2 ─┐
                                                          │
                        T3.1 ─► T3.2(gate) ─► T3.3 ─► T3.4 ─► T3.5   (LTX-2)
                                                                      │
              T4.1(research) ─► T4.2 ─► T4.3                          │ (core validated here)
                                                                      │
              T5.1(research) ─► T5.2 ─► T5.3   (LTX-2.5)               │
                                                                      │
                                  T6.1 ─► T6.2 ─► T6.3   (quant sweep) │
                                                                      │
                                              T7.1  T7.2  T7.3 ◄──────┘
```

**Critical path:** T0.1 → T1.1 → T1.2 → T1.3 → T1.7 → T2.2 → T3.1 → **T3.2 (gate)** → T4.* .

**The single most important checkpoint is T3.2:** if the tool does not reproduce the author's published cosine numbers, everything downstream is suspect. Do not proceed past T3.2 until it passes.

---

## Risk register

| Risk | Mitigation |
|---|---|
| T3.2 fails to reproduce blog numbers | Debug core (most likely: tokenization/padding/last-token extraction). The blog numbers are the oracle. |
| H3 extraction layer ambiguous (T4.1) | Resolve from the real loader; document both hypotheses; the headline changes framing, not validity. |
| VRAM: 32B Qwen on target GPU | Batch encode (8–16 prompts), CPU-offload activations, del backbone between base/variant. |
| GGUF TE loading broken (T6.1) | Dequant via `gguf` lib → safetensors, bypass ComfyUI-GGUF. |
| `ltx-core` API churn | Pin a version; vendor the feature-extractor + connector code if unstable. |
| Overclaiming | Hard rule 6; every report says "measures perturbation, not uncensoring efficacy." |
