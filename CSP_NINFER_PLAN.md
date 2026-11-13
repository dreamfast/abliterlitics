# CSP on NInfer — Implementation Plan

Status: PLANNED. Execute when GPU 0 (5090) frees from the qwen38-27b harmbench queue.
Target: `github.com/Neroued/ninfer` @ b88c0f6, serving Qwen3.8-27B (panel base) on the 5090.
Lab evidence base: `.cache/opencode/lab08/` (M1-M1g), journals #300-303.

## What we're building

A serving-layer CSP (counterfactual self-patching) mode for NInfer: a request may carry a
benign twin prompt; the engine runs a hidden twin prefill capturing last-k residual states
at the readout layers, splices them into the real prompt's prefill at the same layers, and
(optionally, phase 2) steers decode with the twin-real delta. Weights never touched.

## Ground truth from the 0.8B lab (all on Qwen3.5-0.8B, 24 layers)

| finding | value |
|---|---|
| best splice config | layers {21,22,23} (last 3), k=8 positions, prefill only |
| single layer 23 alone | 40/46 (87%) flips |
| + decode delta steering alpha 0.25 | 44/46 (96%) clean |
| alpha 0.4 / constant force | degeneration (coherent-regime edge) |
| prefix patches (0..L) | useless — refusal re-derived from recurrent state |
| hard holds | 2/46: need better twins (twin-bounded law), not stronger patches |

## Phase plan

### Phase 0 — bring-up harness (no HTTP, no engine changes)
- CLI one-shot path (`apps/cli/main.cpp`, `Engine::prepare_tokens`) gives exact-token
  control for twin + target prompts. Use causal-scoring Engine mode precedent if needed.
- Purpose: confirm we can drive two prefills and read engine internals on the 5090 box.

### Phase 1 — MVP: capture tap + prefill splice (the 87% feature)
1. **CspTap** (clone of `DFlashFeatureSink`, `text_context_impl.h:158-189`):
   - `capture_layer` copies last-k COLUMNS of residual `x` (BF16 [hidden, T]) for a
     configured layer subset into a persistent tensor `[n_layers, hidden, k]` added to
     `PersistentLayout` (`layouts.h:36-48`, sized in `layouts_impl.h:95-167`). Memcpy2D
     pattern already proven. Key on ABSOLUTE positions (`capture_positions` precedent)
     to survive chunk boundaries (prefill chunks are 1024).
   - Tap call sites already exist: `text_context_impl.h:1034,1055,1198,1200`.
2. **Twin prefill driver** (port of `causal_score`, `program_impl.h:1007-1160`):
   reserve_reset a state slot -> temp KVAddressSpaceHandle -> chunked
   `prefill_text_chunk(..., finalize_at_end=false)` NO sampling -> CspTap harvest ->
   release. Runs inside the same worker mutation boundary, immediately before the real
   request's `advance_prefill`, gated on the request flag.
   - Budget: one spare state slot + transient KV pages. Restrict CSP requests to
     concurrency < max until admission math (request_plan_impl.h) learns the cost.
   - Disable MTP + DFlash for CSP requests v0 (interactions uncharted: mtp_prefill_chunk
     consumes xf right after run_layers; DFlash owns the same Tap seam).
3. **Splice** in `run_layers` at the same post-mlp_tail seam: overwrite x's last-k
   columns with twin values, position-keyed. Mirror of capture (Memcpy2D or small gather).
   - Default layers {61,62,63}, k=8 (27B analog of lab {21,22,23}/k8; layer 63 is
     full-attention, 60-62 GDN — splicing above GDN is INTENDED: lab showed the
     readout, not the memory, is the effective lever).
4. **Request surface**: `"csp_twin": "<text>"` (or null). Thread:
   `openai_chat_request.cpp:887` parse -> `request.h:174` -> `translate.cpp` ->
   `types.h:247 ExecutionOptions` -> `runtime/contract/types.h:158` ->
   `program.h:470 RequestControl::Prefill`. Keep out of advertised schema v0; docs +
   schema tests (`tests/test_openai_schema.cpp`) updated when it goes public.
5. **Tests**: per-op unit test for tap capture + splice kernels (oracle pattern in
   `tests/ops/op_tester.h`); Program behavior test in `tests/targets/qwen3_6_27b/`
   (prefix/causal-scoring isolation tests are the template).

### Phase 2 — decode delta steering (the 96% feature)
- Per-decode-step add of `alpha * (twin_state - real_state)` at splice layers, captured
  at prefill. MUST be CUDA-graph-capture-safe: fixed kernel inside run_layers between
  chosen layers, per-row enable flags + per-row [hidden, B] delta buffers refilled via
  the round ingress pattern (`round_state.h:30-38`, `decode_impl.h:20-22` — the
  state_source_slots per-row selector trick).
- alpha exposed per-request, default 0.25, hard cap 0.4 (degeneration edge measured).

### Phase 3 — GDN state transplant (the deep intervention, research)
- `StateImageDevicePool::copy_slot` + chunked `gated_delta_net` give slot-to-slot GDN
  transplant without replay-fold's geometry locks (replay_fold width<=16, 2 geometries).
- Purpose: test whether transplanting the twin's recurrent memory (not just its readout
  state) beats readout splicing on the hard holds. This is the lab's open lane.

### Phase 4 — auto-twin (removes the manual twin)
- Template transformations first (proven in lab M1a: fiction frame, safety-research
  frame), model-generated twins second (the lab's twin-bounded law says twin quality
  is the compliance ceiling — a small dedicated reframer pays for itself).

## Evaluation plan (when live on 5090)

1. Pair corpus: apostate harmful.txt sample + panel harmbench behaviors (base refuses
   ~96% of them — rich source). Twin templates from lab M1a. ~50 pairs.
2. Sweep: layers {63}, {62,63}, {61,62,63} x k {1,4,8} -> flip rates vs 0.8B ladder.
3. Judge-grade scoring (glm-5.3-flash-v2 4-way rubric) on all flips — marker
   classifiers already proven insufficient in the lab tail.
4. Capability gates: unpatched requests must be BIT-IDENTICAL behavior (structural);
   patched-mode spot checks on TQA items (the trigger story: v0 is user-opt-in per
   request, so benign traffic never sees CSP).
5. Compare against panel numbers: CSP-on-base vs the 12 abliterated arms' judge ASR
   and TQA deltas. The pitch writes itself: runtime arm with 0 capability cost by
   construction vs weight arms paying 1.2-11.2pp TQA.

## Risks (from the codebase map)

- fused residual writes: no intra-block interception; post-layer seam only (fine for us)
- CUDA-graph decode: phase 2 kernel constraints (stable addresses, no host reads)
- chunk boundaries: position-keyed everything
- KV pool pressure: twin pass costs real pages; cap concurrency for CSP v0
- sm_120a build gate: dev happens on the 5090 itself; no 4090 port needed
- 0.8B not a registered checkpoint: no small-model engine tests; all engine validation
  is on the 27B directly (CLI harness first, cheap)

## Execution order when GPU frees

harmbench queue drains -> KL stage for 4 arms -> review batch -> THEN ninfer build on
the 5090 (tens of minutes one-time, incremental cheap) -> Phase 0 harness -> Phase 1.
Lab 4090 work continues in parallel: judge-grade scoring of the 0.8B 46-pair corpus.
