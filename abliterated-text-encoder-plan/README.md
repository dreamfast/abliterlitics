# Text-Encoder Fidelity Benchmarker

**Working title:** `tebench` — *quantifying what abliteration, quantization, and GGUF actually do to the conditioning tensor that feeds a diffusion transformer.*

> Status: **PLANNING** — this folder is the complete build brief. It will be relocated to a standalone project (`tebench/`) outside the ComfyUI tree before implementation begins.

---

## The thesis (one paragraph)

The "use a heretic text encoder to uncensor image/video models" debate is unresolved because **nobody has measured the right thing.** The creator of Heretic (p-e-w) is correct that abliteration cannot add knowledge the diffusion model lacks, and he *guesses* it "if anything makes outputs worse" — but he explicitly hedges ("if anything") because he never tested it. Meanwhile practitioners *feel* it helps, also untested. Both camps are, in one Redditor's words, "purely vibes based." The missing artifact is a **rigorous, reproducible measurement of how much a modified text encoder perturbs the conditioning tensor** that actually enters cross-attention, expressed relative to a meaningful noise floor, across the LLM-encoder models people actually use (H3/Qwen3-VL, LTX-2/Gemma3, LTX-2.5/Gemma4). This project builds that tool and produces the first defensible numbers.

## Scope

**In:** diffusion/video models whose text encoder is a **causal decoder LLM** (the only kind abliteration even applies to):

| Model | LLM text encoder | Consumption pattern |
|---|---|---|
| **MiniMax H3** | Qwen3-VL-32B | single layer (50th) → project |
| **LTX-2 (2.3)** | Gemma 3 12B (stock) | multi-layer → norm → project → 1D connector |
| **LTX-2.5** | Gemma 4 12B (unified, FT) | same + **dual video/audio projections** |

**Out (deliberate):**
- **Flux / Krea-flux** and any T5+CLIP dual-encoder model — CLIP/T5 are not LLMs you abliterate; the question doesn't apply.
- **Krea2-native** — parked until its text encoder is confirmed LLM-based. If T5-based, stays out.
- **Ideogram-style active refusal** — different problem (DiT-side abliteration), explicitly out of scope per p-e-w's own caveat.

## The two methodological commitments

1. **CKA is the primary metric, not KL divergence.** Hidden states / conditioning tensors are *not probability distributions*; KL is undefined on them without a lossy softmax projection that discards the geometry cross-attention actually uses. Heretic's own "KL divergence" number is valid but measures *next-token-logit* (chat) damage, not *conditioning* damage. Centered Kernel Alignment (Kornblith et al. 2019) is the established tool for "did two networks produce similar representations" and operates on the full activation matrix, capturing geometry. → see `METRICS.md`.
2. **Every number is reported relative to a noise floor.** A raw `CKA = 0.94` means nothing. Reported *relative* to `div(base_bf16, base_fp32)` (numerical floor) and `div(base, fp8)` (quantization floor), it becomes an argument: "abliteration perturbs taboo conditioning **3× beyond the fp8 noise floor**." This also defuses the knowledge-level confound (low divergence between two weak models ≠ safe). → see `METRICS.md`.

## Documents in this folder

| File | Purpose |
|---|---|
| **`PLAN.md`** | The build plan — phases, tasks, dependencies, tests, acceptance criteria. **Read this first when building.** |
| **`PROGRESS.md`** | Live task tracker. Update after every task. |
| **`ARCHITECTURE.md`** | Detailed design: `Backend` / `StreamSpec` / `ConditioningArtifact` interfaces, metrics core, module layout, report schema. Code-level. |
| **`METRICS.md`** | Metrics methodology: CKA derivation + why primary, cosine-all-tokens, L2/MSE, rank-correlation, noise-floor, bootstrap CIs. Includes the full "why KL is wrong" argument. |
| **`RESEARCH.md`** | Empirical findings + lessons learned that drive the design: H3 layer-50/truncated encoder, the LTX multi-stage conditioning pipeline (norm→project→connector), V1/V2 extractor split, the abliteration-refusal-layer fact, the knowledge confound, and what the Reddit PSA establishes. |
| **`MODELS.md`** | Per-backend implementation specs with exact extraction points, weight mappings, and **known unknowns** to resolve against the real loader. |

## Build order (summary)

1. Core metrics + noise-floor + report generator (no model)
2. **LTX-2 backend @ raw-hidden-state → reproduce the author's published cosine table** ← built-in acceptance test
3. Extend LTX-2 to connector-stage + CKA → "the 2% revisited" result
4. **H3 backend** → the headline (contested, single-layer)
5. **LTX-2.5 backend** → Gemma 4, video/audio damage split (the novelty)
6. GGUF / fp8 / nxfp4 quant sweep → damage ranking table

Full rationale and task-by-task detail: `PLAN.md`.

## Key non-obvious findings (so you don't re-discover them)

- **LTX does NOT just "aggregate 49 layers equally."** There is a full **1D transformer connector** (self-attention + RoPE + RMSNorm + learnable registers) *after* the linear projection, which is non-linear and can amplify or null directions. The author's "layer 48 = 2% of the projection weight" is a *lower-bound proxy*, not the truth. The real measurement point is after the connector. → `RESEARCH.md`
- **H3 may use a truncated encoder.** The published MiniMax-H3 Qwen3-VL checkpoint is "truncated" (a companion repo *reconnects the generation tail* to make it generate text again), and the README says it feeds "the 50th layer" while Qwen3-VL-32B has 64 layers. Confirm: is it layer-50-of-64, or a 50-layer truncation? This is load-bearing for the H3 headline. → `MODELS.md`
- **LTX-2.5 has separate video and audio conditioning projections.** Abliteration damage to *speech/phonetic* conditioning can be measured independently of visual conditioning. Likely an unreported result. → `RESEARCH.md`

## Tech stack

- Python 3.11+, PyTorch (CUDA), `transformers`, `safetensors`
- `ltx-core` (vendored or pip) for the LTX feature extractor + connector replication
- `numpy`, `scipy` (stats), `pyyaml`, `jinja2` (report)
- `pytest`, `ruff`
- Optional: `gguf` / ComfyUI-GGUF loader for the quant sweep
