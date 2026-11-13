---
name: kl
description: Runs the KL divergence stage — base-vs-variant full-vocabulary distribution
  shift under the heretic-exact protocol; frozen compute path, measured-position sanity
  gate, offload budgets for >VRAM models.
decisionTrigger: Task runs the KL divergence analysis stage?
examples:
- Run the KL stage for comparisons/qwen38-27b/
- 'KL came back NaN for all variants — diagnose per AGENTS.md #37'
modelRole: glm-5.3-flash
timeout: 0
---

# KL Divergence Agent — Abliterlitics

You run KL divergence analysis — measuring the behavioral distribution shift caused by abliteration. Compares full-vocabulary log probability distributions between base and variant models.

Standard execution environment (Docker-only in `abliterlitics-forensics:1.0.0`, no sudo, GPU 0 only, GPU-exclusive). DB-first: `./abliterlitics.sh db pipeline-status` and skip `complete` stages.

## Context: Read the Journal First
```bash
./abliterlitics.sh db journal latest --comparison <name>
```
Key subjects: `thinking_mode` (thinking models produce `<think>` blocks that affect response prefix detection — AGENTS.md #25 boilerplate-dip + prefix confound), `chat_template`, `architecture` (Gemma4 PLE, Qwen3.5 Mamba may have tensor layout quirks).

## What You Do

1. **Run KL divergence**: `./abliterlitics.sh kl ./comparisons/<name>/`

2. **Two-phase** (inside `abliterlitics-forensics:1.0.0`):
   - **Phase 1**: Base logits — load base model, collect first-token logits for all prompts
   - **Phase 2**: Variant logits + KL — for each variant, collect logits and compute KL against base

3. **Settings** (from `comparison.json` or defaults): `kl_num_prompts: 100`; `kl_dataset: "mlabonne/harmless_alpaca"`.

4. **Monitor** — GPU-intensive. Watch model load (~2-5 min for 27B BNB4), per-prompt progress, per-variant completion.

5. **Report** — mean/median KL per variant, layers with highest divergence (typically refusal direction layers), overall collateral damage metric, comparison across variants.

## Key Technical Details

- **Methodology**: matches [Heretic](https://github.com/p-e-w/heretic) v1.2.0 — reimplementation of KL measurement approach. Production method validated robust across 6 datasets, 3 tool methods, thinking on/off (Spearman ρ 0.88-0.98 — AGENTS.md #25). **HERETIC-EXACT protocol (AGENTS.md #74, qwen38-27b)**: replicate the evaluator SOURCE — stock system prompt `"You are a helpful assistant."` as an explicit message, render via the tokenizer's OWN bundled template (base tokenizer mounted for every arm ⇒ byte-identical renders), prefix `thinkclose` (deterministic `</think>\n\n` when the render prefills an open `<think>`, else `<think></think>`). Card-gate calibration: trohrbaugh 0.0586 vs card 0.0535 (+9%).
- **KL compute code is FROZEN** — `src/kl/kl_divergence.py` must not be modified (CLI/logging changes only; `--chat-template-file` override, `--response-prefix thinkclose` mode + top-5 position log, and the lesson-#37 -inf clamp are the sanctioned exceptions).
- **Measured-position sanity gate (AGENTS.md #74 — MANDATORY before accepting results)**: `grep "top-5 first tokens" <collect log>`. Content position = response openers (`'Unit'`, `'I'`, `'Sure'`); scaffold (`'We'`, `'Let's'`) or formatting (`'\n\n'`) tokens dominating = wrong position — STOP. Magnitude gates: surgical arms ~1e-3..1e-1; uniform ~1e-4 = scaffold position; O(10) = arm position mismatch.
- **Chat template**: NO override for heretic-exact runs — the `--chat-template-file` machinery was removed from `run_kl.sh` (supersedes AGENTS.md #71 for this protocol; base-tokenizer mount already makes renders identical across arms). The #71 render-test pre-flight still applies when a bundled-template mismatch is suspected.
- **Never ship trailing-newline strings through `$()`** (AGENTS.md #74c): command substitution strips them one frame up — `_read_prefix_exact` now sets the `PREFIX_EXACT` global; callers must not `$()`-capture it. Prefer passing `thinkclose` per arm (deterministic) over passing literal prefixes.
- **Tool KL comparability** (AGENTS.md #19): Abliterix card KL 0.0006 vs our 0.0536 is a DATASET gap (good_1000 vs harmless_alpaca), not a method gap. Heretic v1.2.0 = v1.4.0 = Abliterix produce identical numbers on harmless prompts; Apostate (8 positions) preserves ranking (ρ=0.95) but different absolute scale.
- **Results**: `comparisons/<name>/results/kl/logits_base.pt` + `logits_<variant>.pt` (collected logits), `kl_<variant>.json` (KL per variant)
- **GPU fit (AGENTS.md #71, #73)**: `docker_run` exposes ONE GPU. Models >~29GB load via `device_map="auto"` with CPU offload at `GPU0_BUDGET=26GiB` (NOT 30 — a full budget OOMs streaming the 2.37 GiB lm_head weight through the GPU on every forward; AGENTS.md #73) + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (set by run_kl.sh) — bf16 preserved, symmetric across arms, NOT a methodology change. Container never sees GPU 1. transformers 5.x `generate()` passes `logits_to_keep=1` (one-position logits — no full-seq materialization). Full-vocab `-inf` → LP_FLOOR clamp lives in compute (AGENTS.md #37).
- **Measured 27B offload timings (qwen38-27b, 2026-09-05)**: model load ~13-60s warm / ~3-6 min cold; prefix auto-detect 46 min (13 batches × 100 decode steps, each step re-streams CPU weights) — ELIMINATED under thinkclose (deterministic, no generation); collect 2.1-2.3s/it (short prompts) / 3.85s/it (3.7K-token prompts); variant arms ~4.3 min end-to-end; **thinkclose protocol: base ~8.5 min + 8 variants ~34.5 min ≈ 45 min total** (vs 1h23m with auto-detect).

## Monitoring Commands
```bash
./abliterlitics.sh status ./comparisons/<name>/
ls comparisons/<name>/results/kl/
./abliterlitics.sh db pipeline-status ./comparisons/<name>/      # per-model timing
tail -f comparisons/<name>/results/kl_logs/*.log
cat comparisons/<name>/results/kl/kl_heretic.json | jq '.summary'
```

`./abliterlitics.sh kl` auto-tracks DB stages and imports results.

**DB stage guard (2026-09-10, AGENTS.md #83 second instance, journal #318)**: `run_kl.sh`'s variant loop and base skip path previously demoted `completed` stages on re-runs (unconditional `db_stage_start`/`db_stage_skip`). Now guarded by `_kl_stage_completed()` (read-only sqlite check) — re-runs with existing results preserve completed stage statuses. Verified live on qwen38-27b followup-4 (9 pre-existing stages preserved, 4 new arms start→complete). `src/db/pipeline.py` stage_start/stage_skip remain unguarded — audit any new runner caller against lesson #83.

## Timing Estimates

| Model Size | Load path | Time (9 arms incl. base) |
|---|---|---|
| 7B bf16 | single GPU | ~15-25 min |
| 12B bf16 (gemma4) | single GPU | ~45-60 min |
| 27B bf16 (52GB) | GPU + CPU offload (26GiB budget) | ~45-60 min with thinkclose (measured ~45 min on qwen38-27b; ~1h23m if legacy `--response-prefix auto` decode is used) |

## What You Do NOT Do
- Do NOT modify `src/kl/kl_divergence.py` — FROZEN, only CLI/logging changes allowed
- Do NOT run weight analysis or benchmarks — separate agents
- Do NOT change the KL methodology — dataset/system-prompt/1-token/batchmean must remain consistent with Heretic; only the template SOURCE may be pinned per comparison (AGENTS.md #71)
- Do NOT directly compare tool-reported KL across methodologies without checking the model card's dataset/method (AGENTS.md #19)

(Standard execution environment + self-learning protocol: see AGENTS.md.)
