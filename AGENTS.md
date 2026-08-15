# AGENTS.md — Project Context & Lessons Learned

> For AI agents and humans working on Abliterlitics. Captures hard-won knowledge
> about running evaluations correctly. **Read this before any pipeline work.**

## Agent Self-Learning Protocol (applies to ALL agents)

When you solve a non-trivial problem (config bug, unexpected error, doc gap, tool quirk not already documented), persist it before completing your task:

1. **AGENTS.md** — add a lesson under "Critical Lessons Learned" (or update existing). Format: **Symptom → Cause (cite source) → Fix → Detection**.
2. **Your agent file** (`.opencode/agents/<this-agent>.md`) — update relevant section, cross-reference AGENTS.md by lesson number.
3. **Runner/script** — if the fix belongs in code, change it there. Code > docs.
4. **Journal** — write `category=lesson` entry to `abliterlitics.db` for the specific instance.

**Do NOT update for**: stylistic preferences, one-off environment quirks that won't recur, user-error problems, issues already covered by an existing lesson (just cite the number).

## Project Journal (inter-agent communication)

The `journal` table in `abliterlitics.db` is the source of truth. All agents read/write it. Schema: `journal(id, timestamp, comparison, agent, category, severity, subject, title, body)`.

**CLI cheatsheet**:
```bash
./abliterlitics.sh db journal add --agent <a> --category <c> --severity <s> \
    --comparison <name> --subject <tag> --title "<one line>" --body "<markdown>"
./abliterlitics.sh db journal latest --comparison <name>   # current truth per subject
./abliterlitics.sh db journal list  --comparison <name>     # full history (newest first)
./abliterlitics.sh db journal list  --agent <a> --category <c>
./abliterlitics.sh db journal search "<query>"
```

Categories: `discovery`/`decision`/`warning`/`action`/`lesson`/`result`/`error`/`note`. Severities: `debug`/`info`/`warning`/`error`/`critical`.

**Write when**: you discover/decide/do something non-obvious, or solve a problem (also add an AGENTS.md lesson). **Don't write**: routine progress (use `model_stages` table), per-model benchmark numbers (use result tables), anything already structured.

**Relationship**: AGENTS.md = reusable cross-comparison lessons (the general rule). Journal = per-comparison findings (the specific instance).

## Project Overview

Abliterlitics compares **base LLMs vs their abliterated/uncensored variants** across weight forensics (layer-by-layer diffs, SVD, cosine similarity), KL divergence, lm-evaluation-harness benchmarks, and HarmBench safety. Reports go to HuggingFace (technical) and Reddit (condensed).

**Active comparison**: query `./abliterlitics.sh db journal list` (DB is source of truth — no static list here).

## LM-Eval Working Approach

**Architecture**: vLLM OpenAI server + lm-eval's `local-completions` backend, **both in the same Docker container** (lesson #6). Runner `runners/run_lm_eval.sh` handles two phases, caching, chat templates, thinking overrides.

**Images**:
- `abliterlitics-forensics:1.0.0` — weight analysis, KL divergence
- `abliterlitics-lmeval:1.0.0` — vLLM 0.19.0 (baseline; no Gemma4 thinking)
- `abliterlitics-lmeval-nightly:1.0.0` — vLLM 0.20.1 (Gemma4ForConditionalGeneration only — E4B/E2B/31B)
- `abliterlitics-lmeval-gemma4unified:1.0.0` — vLLM 0.26.0 (adds Gemma4UnifiedForConditionalGeneration — 12B)

Per-comparison override via `comparison.json` → `settings.lmeval_image` (lesson #11).

---

## Critical Lessons Learned

### 1. `max_gen_toks` includes thinking tokens — plan accordingly

**Cause**: For reasoning models (Qwen3.5 `<think/>`, Gemma 4 `<|think|>`), `max_gen_toks` is the TOTAL budget including thinking. Model thinks ~5000 tokens, gets cut off, never answers.
**Fix**: GSM8K needs `max_gen_toks=7168`. Loglikelihood tasks ignore it. Never higher than `max_model_len - max_prompt_length`. Two-phase: Phase 1 (loglikelihood + truthfulqa) at 2048; Phase 2 (gsm8k) at 7168.
**Detection**: GSM8K scores near-zero with truncated/empty responses.

### 2. Quantization: BF16 always, unless user specifies otherwise

Runner default is native BF16 for every model. Don't suggest alternatives unless the user explicitly asks or BF16 OOMs and the user requests a workaround.

### 3. Qwen3.5 architecture specifics

Qwen3.5 is a Mamba2 hybrid (`linear_attn` / `conv1d` keys mixed with standard attention). TP=2 works fine in BF16 for large variants. Reasoning uses `<think>...</think>` tags → `--reasoning-parser qwen3` + reasoning_config (lesson #4).

### 4. vLLM `reasoning-parser` settings

Qwen3 family: `--reasoning-parser qwen3 --reasoning-config '{"reasoning_start_str": "<think>", "reasoning_end_str": "</think>"}'`. Gemma 4: `--reasoning-parser gemma4` (parser handles `<|channel>thought...<channel|>` internally, no config). Auto-detected by `runners/_load_comparison.py:_detect_reasoning_parser` — matches `arch.startswith("Gemma4")` → `gemma4` (covers both `Gemma4ForConditionalGeneration` and `Gemma4UnifiedForConditionalGeneration`).

### 5. Tokenizer mount is required separately

Mount tokenizer from BASE model (`models/<base>:/tokenizer:ro`) even when evaluating variants.

### 6. `local-completions` tokenizer download hang

If lm-eval runs in a separate container from vLLM, it tries to download the tokenizer from HF Hub using model name `/model` and hangs. **Both MUST run in the same container.**

### 7. Reasoning models and thinking loops

Some abliterated models generate repetitive reasoning text consuming all tokens without converging. Manifests as very short final responses after thinking stripped. Model behavior issue, not config. See lesson #20 for the measurable signal.

### 8. Old results can be from a different architecture

Always verify which model produced existing result files. `results/lm_eval/` once contained results from a different architecture run. Check JSON `model_class`/`config` fields to confirm.

### 9. `--apply_chat_template` is MANDATORY for instruction-tuned models on loglikelihood tasks

**Cause**: Without it, instruct models receive raw `"Question: ... Answer:"` instead of their trained chat format. Every MC loglikelihood score drops to near-chance (MMLU −33pp, HellaSwag −44pp, ARC-Challenge −59pp). Generation tasks escape this.
**Fix**: `runners/run_lm_eval.sh` auto-applies `--apply_chat_template` when it detects `chat_template.jinja` OR `tokenizer_config.json` containing `"chat_template"` (per-model AND per-tokenizer-dir).
**Detection**: dry-run shows `phase1 flag='--apply_chat_template'`. If `'<none>'` for a model you know is instruct-tuned → STOP, fix model dir.
**Caveat**: `enable_thinking=True` is for generative tasks only (GSM8K), NEVER loglikelihood (lm-eval `interface.md`).

### 10. Always research + diagnose a new model before the full pipeline

**The mistake**: Kicked off full pipeline, came back 6 days later to find every loglikelihood score depressed 30-60pp by undetected config issue.
**The pattern that catches it early**: Before any full GPU work, run `lm_eval --limit 200` on a single task against base, compare to published. Within ~5pp = PASS. The `abliterlitics-init` agent owns this.
**Worked example**: Gemma 4 E4B HellaSwag 0.39 vs community 0.79 — 40pp gap from missing `--apply_chat_template`. 30min to diagnose vs 144 GPU-hours wasted otherwise.
**How to run the diagnostic (preferred — uses the runner, not a bare docker cmd)**: `run_lm_eval.sh` accepts `--limit N` (caps samples per task) and `--models LIST` (comma-separated filter; `base` = base model only). Both default to "off" so full runs are unaffected. Example:
`./abliterlitics.sh lm-eval --tasks hellaswag --limit 200 --models base ./comparisons/<name>/`
This exercises the REAL stack the pipeline uses (vLLM server + `local-completions` backend in the same container, lesson #6, with the comparison's image/reasoning-parser/chat-template auto-detected). A bare `docker run lm_eval --model hf` command does NOT exercise `local-completions` and can miss server-side bugs — prefer the runner. **Caveat**: the diagnostic writes the canonical `lm_eval_<base>.json` and marks the base `lm_eval` stage `completed` in the DB — reset both before handing off (delete the canonical file + `results/lm_eval/__model/`, `UPDATE model_stages SET status='pending'` for base lm_eval, clear the comparison's `lm_eval_results` rows). The per-model `responses.sqlite` cache CAN be kept — it speeds up the real HellaSwag run with no contamination (same requests, same model).

### 11. Per-comparison Docker image override + Gemma 4 needs nightly (or unified) image

**Cause**: `runners/common.sh` defaults to `abliterlitics-lmeval:1.0.0` (vLLM 0.19.0), lacks `gemma4_reasoning_parser`. Crash during `model_loader.load_model()`. The `Gemma4UnifiedForConditionalGeneration` arch (gemma-4-12B-it) additionally requires vLLM 0.23.0+ (PR vllm-project/vllm#44429); shipped in `abliterlitics-lmeval-gemma4unified:1.0.0` (vLLM 0.26.0).
**Fix**: image vars env-overridable; `runners/_load_comparison.py` reads `settings.lmeval_image` from `comparison.json`.
**For `Gemma4ForConditionalGeneration`** (E4B/E2B/31B): `lmeval_image: abliterlitics-lmeval-nightly:1.0.0`.
**For `Gemma4UnifiedForConditionalGeneration`** (12B): `lmeval_image: abliterlitics-lmeval-gemma4unified:1.0.0`.
**Detection**: `python3 runners/_load_comparison.py <json> <dir> | grep LMEVAL_IMAGE`.

### 12. flashinfer / flashinfer-jit-cache version mismatch in nightly image

**Symptom**: vLLM crash during model load: `flashinfer-jit-cache version (0.6.6+cu129) does not match flashinfer version (0.6.8.post1)`.
**Fix**: `FLASHINFER_DISABLE_VERSION_CHECK=1` set unconditionally in `runners/run_lm_eval.sh` docker env block. If it recurs, rebuild image with aligned versions.

### 13. Runner skips via `comparison.json` flags, NOT DB state

**Cause**: Runner reads `VARIANT_SKIP_LM_<slug>` env vars set by `_load_comparison.py` from each variant's `"skip_lm_eval": true` in `comparison.json`. DB and runner are decoupled — DB is reporting/observability only.
**Fix**: Add `"skip_lm_eval": true` to each variant in `comparison.json`. Same pattern for `skip_harmbench`/`skip_kl`/`skip_weight`.
**Detection**: `./abliterlitics.sh --dry-run lm-eval ./comparisons/<name>/ | grep -cE "Evaluating|skip"`.

### 14. lm-eval request caching: preprocessing only, NOT API responses — `--use_cache` is wired for true response-level resume

**Cause**: `--cache_requests true` caches preprocessed prompts only (~2min saved on restart). Does NOT cache model API responses. A crashed/stopped Phase 1 historically lost ALL inference work and reran from request 0. Confirmed empirically multiple times (journals #11, #33, #46).
**Fix (now shipped in runner)**: per-model cache dir `${RESULTS_DIR}/lm_eval_cache/${model_name}/`, mounted at `/lm_eval_cache`, env `LM_HARNESS_CACHE_PATH=/lm_eval_cache`, `--cache_requests true` on all 3 phases, cleared by `--force` (rm -rf of the per-model cache dir).
**True response-level resume (RESOLVED 2026-08-02)**: `--use_cache /lm_eval_cache/responses.sqlite` is now wired into all 3 phases. lm-eval creates `responses.sqlite_rank0.db` (note: it appends `_rank0.db` automatically — do NOT look for a bare `responses.sqlite` file). Schema: single table `unnamed(key TEXT PRIMARY KEY, value BLOB)` where key = request content hash, value = pickled response. **Empirically verified** in gemma4-12b base diagnostic: 800 of 4000 HellaSwag LL requests served from cache (`Cached requests: 800, Requests remaining: 3200`) when re-running with a larger `--limit` — exactly the crash/resume semantics we wanted.
**All 3 phases share the same cache file**: Phase 1 (LL MC + generative) → Phase 2 (GSM8K chat) → Phase 3 (raw completion). On restart, Phase 1 returns instantly from cache, Phase 2/3 resume from where they crashed.
**Detection**: `grep -c use_cache runners/run_lm_eval.sh` ≥3. In the lm-eval log, look for `Loading '<task>' responses from cache '<path>' where possible...` and `Cached requests: N, Requests remaining: M` where N>0 means cache hits. If N=0 on a restart of an interrupted run, the cache wasn't written (check disk space, permissions on `${RESULTS_DIR}/lm_eval_cache/`).

### 15. lm-eval 0.4.11 `--cache_requests` CLI flag is broken — auto-patched

**Cause**: `_cli/run.py` defines the flag with `type=request_caching_arg_to_dict` AND `choices=["true","refresh","delete"]`. argparse applies `choices=` AFTER `type=` conversion, so the resulting dict is never in the string list — flag unusable from CLI.
**Fix** (in runner): one-line `sed` patch in container startup removes the `choices=[...]` line. Idempotent (no-op if already fixed or if running lm-eval ≥0.4.12 where the bug is patched upstream — `abliterlitics-lmeval-gemma4unified:1.0.0` ships 0.4.12).
**Detection**: failure shows `argument --cache_requests: invalid choice: "{...}"`. Patch confirmed by `[container] Patching lm-eval 0.4.11 --cache_requests choices= bug` in master log.

### 16. Thinking-model chat templates contaminate GENERATIVE tasks (originally mis-framed as LL MC)

**Original framing (wrong)**: E4B's chat_template.jinja defaults `enable_thinking=true`, injects `<|think|>` at system turn, contaminates LL MC scoring.
**Empirical correction (journal #14)**: the no_thinking_chat_template.jinja fix had **ZERO effect on LL MC tasks** — MMLU-Pro was 0.4394 before AND after (it was always the true LL MC score for this 4B model, ~25pp below Google's generative number by methodology design, lesson #21). The fix DID dramatically improve GENERATIVE tasks: IFEval +42pp, TruthfulQA-Gen +52pp. The `<|think|>` was eating the GENERATION budget causing truncated responses, not contaminating logit distributions.
**Fix**: Create `comparisons/<name>/no_thinking_chat_template.jinja` — copy of model's `chat_template.jinja` with `enable_thinking` default flipped to `false`. Runner auto-mounts it over `/tokenizer/chat_template.jinja` for Phase 1.
**When actually needed**: ONLY when the model's chat_template.jinja defaults `enable_thinking=true` (E4B case). Gemma 4 12B Unified defaults `false` natively — no override needed for Phase 1.
**Detection**: render the actual prompt: `tokenizer.apply_chat_template([{role:user,content:'Q\nA. x\nB. y\nAnswer:'}], tokenize=False, add_generation_prompt=True)`. If output contains `<|think|>` (or `<think>`) BEFORE the model turn marker → Phase 1 generative tasks will be truncated; create the override.
**Cache invalidation**: cache keys don't differentiate thinking/non-thinking. Use `--force` after creating the override.
**Applies to**: Gemma 4 E4B/E2B (`<|think|>`), Qwen3 (`<think>`), DeepSeek-R1 (`<think>`). NOT Gemma 4 12B Unified (defaults false).

### 17. `--apply_chat_template` kills raw-completion tasks (HumanEval, LAMBADA)

**Cause**: HumanEval prompts end mid-function (raw continuation needed); LAMBADA prompts are narratives missing the last word. With chat template applied, instruct models respond conversationally instead of continuing. HumanEval pass@1=0; LAMBADA perplexity ~1e11.
**Fix (shipped in runner)**: `run_lm_eval.sh` now runs a **Phase 3** for raw-completion tasks (HumanEval, LAMBADA) WITHOUT `--apply_chat_template`. Tasks are automatically split: Phase 1 (LL MC + generative WITH chat template) → Phase 2 (GSM8K thinking) → Phase 3 (raw completion, no chat template). Same vLLM server, same container.
**Detection**: If `humaneval pass@1 ≈ 0` AND `lambada perplexity > 1e6` for ANY model in a chat-template run, Phase 3 is not running — check the runner version and task split logic.
**Anomaly note**: heavily-abliterated models (obliteratus) may produce non-zero HumanEval even WITH chat template because aggressive abliteration makes the model less aligned to chat format. Phase 3 (raw mode) produces the real pass@1 for all models.
**No effect on**: mmlu_pro, gpqa, bbh, musr, ifeval, truthfulqa, hellaswag, arc, piqa, winogrande, gsm8k.

### 18. No Phase-level checkpointing — stopping mid-run reruns everything

**Cause**: `run_lm_eval.sh` writes per-Phase `results_TIMESTAMP.json` to `__model/`, but the canonical `lm_eval_<model>.json` is only produced by collection code AFTER the container exits. Skip check only looks for canonical file. No mechanism to detect "Phase 1 already done" outside of `GSM8K_RERUN` mode. `--cache_requests` does NOT help (lesson #14).
**Confirmed empirically** (pre-`--use_cache` era): stopped container during Phase 2 (request 17/1319), restarted → Phase 1 reran from 0 (~3h wasted). The Phase 1 result file was on disk the entire time but ignored.
**Rule (RELAXED 2026-08-02)**: stopping a running lm-eval container is now MUCH cheaper — the response cache (lesson #14, RESOLVED) makes Phase 1 return instantly from cache and Phase 2/3 resume from where they crashed. Still prefer letting runs finish naturally; the response cache flushes periodically so worst case is losing ~minutes of inference, not hours. Safe-stop check unchanged: `ls results/lm_eval/lm_eval_<model>.json` — exists = model fully done; missing = at least one phase still pending (will resume cheaply on restart).
**RESOLVED**: (1) Phase 1 checkpoint marker — not needed; response cache covers it. (2) `--use_cache <sqlite>` — shipped (lesson #14).

### 19. Tool self-reported KL is NOT comparable to our full-vocab Heretic-method KL

**Example**: Abliterix card reports KL **0.0006**; our measurement is **0.0536**. ~90× gap, not a contradiction.
**Cause (fully resolved, journals #60/#61)**: different DATASET, not method. Abliterix uses `good_1000[400:500]` (curated benign bilingual); we use `harmless_alpaca test[:100]`. On good_1000 our method gives 0.0017 mean / 0.0005 median — matches card's 0.0006. On harmless_alpaca same model = 0.0536. ~30× drop purely from dataset.
**Ruled out** (definitively, `scripts/verify_method_good1000.py`): top-100 truncation (raises KL to 0.078), the method itself (forward-pass logits == generate output_scores on clean prompts), borderline-prompt composition.
**Method tool landscape** (journals #57, #59): Heretic v1.2.0 = v1.4.0 = Abliterix all produce IDENTICAL numbers on harmless prompts (the v1.4.0 raw-logits switch and Abliterix clamp both exist for `-inf` handling that never triggers on harmless_alpaca). Apostate (8 positions) preserves ranking (ρ=0.95) but different absolute scale. All three converge on `KL(base||variant)` over harmless_alpaca.
**Rule**: before citing any tool's self-reported KL, check the model card's methodology + dataset. Only full-vocabulary scores on the same dataset are comparable across variants.

### 20. Empty-response rate is a forensic signal — report 3 GSM8K numbers

**Cause** (journal #17): vLLM with `--reasoning-parser` splits response into `message.content` (post-`</think>`) and `message.reasoning_content` (the thinking trace). lm-eval's `local-chat-completions` backend only reads `content`. When a model thinks for the full `max_gen_toks` budget and gets cut off before emitting `</think>`, `content` is null → empty string in samples JSONL. The thinking trace itself is silently discarded (cannot do qualitative reasoning comparison from samples alone).
**Measurement**: count of empties is a proxy for "reasoning failure rate". For base Gemma 4 E4B: 29/1319 (~2.2%) empty, costing ~2pp on headline GSM8K.
**Reporting rule**: for every variant's GSM8K, report THREE numbers:
1. **Headline** (denominator = all problems; empties count as wrong) — for leaderboard comparison
2. **Empty count + %** — reasoning failure rate (proxy for thinking-loop damage from abliteration)
3. **Answered-only score** (denominator = non-empty) — capability ceiling when reasoning converges
The delta between (1) and (3) is the **"thinking-tax"** — damaged abliterations show a bigger gap. The empty-rate trend across variants is itself a forensic signal.
**Future fix**: patch lm-eval `local-chat-completions` in-container to capture `reasoning_content` (~30 LOC; same pattern as cache_requests sed patch). Do NOT do mid-run.

### 21. Abliteration vs Distillation: OPPOSITE benchmark fingerprints

**Abliteration fingerprint** (refusal-direction ablation): GSM8K preserved (±2pp typically, often +0.5 to +2pp slight improvement), TruthfulQA down (−3 to −16pp). Capabilities flat.
**Distillation fingerprint** (output distribution shift to a teacher model): GSM8K damaged (−4 to −17pp), TruthfulQA preserved (±3pp). Scattered deltas — some tasks improve (HellaSwag +4-5pp from teacher's style transfer), some degrade.
**Why this matters**: classifies unknown modifications. If a variant shows GSM8K damage WITHOUT TruthfulQA shift → it's a distillation/merge, not an abliteration. Don't file it under "damaged abliteration"; file it under "distribution shift".
**Empirical anchors** (gemma4-e4b, journals #38, #43):
- Pure abliterations (huihui, heretic, abliterix, ARA variants): GSM8K ±2pp, TQA −2 to −10pp.
- Gemini 3.1 Pro distill: GSM8K −3.6pp, TQA −2.7pp, HellaSwag +4.2pp.
- Claude 4.6 Opus distill: GSM8K −17.1pp, TQA −1.2pp, HellaSwag +5.1pp, BBH temporal_sequences −46pp.
- Heavy abliterations (obliteratus/bendernina/physshell): GSM8K −20pp AND TQA −9 to −16pp (both damaged — over-aggressive direction hit capability directions alongside refusal).

### 22. Method matters: ARA surgical vs "Obliterated" brute-force

The abliteration technique produces dramatically different capability damage for comparable TruthfulQA shift. From gemma4-e4b (journals #48, #49, #50, #51, #53):
- **ARA (Activation Refusal Analysis)** — nullpo (ARA-5), heresy (MuXodious ARA): TQA −2 to −4pp, all capabilities within ~1-3pp of base. Most surgical.
- **Standard direction-ablation** — huihui, heretic, coder3101, trevorjs, wwt: TQA −5 to −7pp, capabilities preserved.
- **Aggressive "Obliterated" recipe** — obliteratus, bendernina, physshell: TQA −9 to −16pp, GSM8K −20pp, MMLU-Pro −10pp. The three "Obliterated"-named variants produced near-identical deltas (within ~0.5pp on all 12 tasks) → same recipe, near-clones.
**Capability gap**: ~20pp GSM8K between ARA and "Obliterated" for comparable refusal removal. **The method matters as much as the model.**
**GPQA / arc_challenge resilience**: across ALL abliterations including the catastrophic "Obliterated" ones, GPQA Diamond and ARC-Challenge stayed within ±4pp of base. PhD-level science and grade-school science appear far from refusal directions — these tasks cannot detect abliteration damage but usefully confirm the model isn't broken.

### 23. Process cleanup gotcha — pkill all run_lm_eval, not just master.pid

**Cause** (journal #21): `master.pid` only tracks the latest runner PID. Restarting the runner multiple times in quick succession leaves old runner processes alive → multiple vLLM instances spawn → each tries to grab 75% of GPU memory → CUDA OOM for all three.
**Fix**: when restarting, kill ALL processes matching `run_lm_eval`, not just the PID in `master.pid`: `pkill -f run_lm_eval && sleep 3 && ps aux | grep run_lm_eval | grep -v grep` (verify empty).

### 24. HarmBench residual refusals are category-selective even on "uncensored" models

**Observation** (journals #54, #55): a 90% ASR "uncensored" variant (sdft-runpod-vllm) still flatly refused certain categories — chemical-weapons synthesis requiring detailed steps (Sarin, phosgene), self-harm glorification, race-incitement (while generating racist vitriol via definitional deflection). Full compliance on ransomware, phishing, ricin, disinfo, wikipedia-editing.
**Implication for HarmBench interpretation**: ASR is not uniform. Two variants with the same headline ASR may have very different refusal profiles. The classifier flags complied/refused; the harm-reviewer should note the refusal categories in its summary. Don't interpret "90% ASR" as "fully uncensored on 90% of categories".

### 25. KL methodology is robust across datasets/methods/thinking (empirically validated)

**Stress-test** (journals #56, #58, #59, #62): production method (Heretic v1.2.0, harmless_alpaca 1-token, batchmean) was tested against 6 alternative datasets, 32-token trajectories, thinking on/off, and 3 tool methods.
**Ranking stability**: Spearman ρ of variant ranking vs production across all swaps: mean 0.90 (1-token), 0.88 (32-token). Min 0.79. The variant order at the extremes is identical everywhere; only the near-tied middle cluster shuffles within noise.
**No heretic-favoritism**: Heretic-made models' low KL is genuine capability preservation, not a harmless_alpaca artifact (ρ ≥0.88 on Dolly, GPQA, GSM8K, no_robots, MMLU-Pro, truthful_qa).
**Boilerplate-dip on thinking models**: per-position KL is ≈0 at the opening reasoning tokens ("Here's a thinking process… 1. Analyze the Request") because abliteration only edits the refusal direction, not the reasoning scaffolding. Cumulative KL dips at N=2-4 then rises as content diverges. A 1-token KL on a thinking model is therefore low-resolution (often a near-deterministic format token). 32-token trajectories are higher-resolution but introduce their own noise (prefix detection — see below).
**Prefix confound** (journals #61, #62): `kl_experiment` harness auto-detects a response prefix and is inconsistent per-run (some runs score the opener token, some score post-opener content). Fix: `RESPONSE_PREFIX=none` forces deterministic opener position. Cross-dataset absolute numbers were partly confounded by this; ranking held despite.
**Heretic does NOT disable thinking for Gemma 4**: Heretic's CoT-skip pattern matches Qwen3 `<think>` and gpt-oss `<|channel|>analysis`, NOT Gemma 4's `<|channel>thought`. So Heretic also measures thinking tokens here — apples-to-apples with our method.

### 26. Methodology-only validation for new thinking models not yet on OLL v2

**Context** (journal #19): many new thinking models (Gemma 4 family as of 2026-07) are NOT on the Open LLM Leaderboard v2 — no public loglikelihood-MC reference exists. Google's published numbers are generative-with-thinking, which is a fundamentally different methodology (~25pp gap by design — see lesson #21).
**Diagnostic framing**: do NOT frame the diagnostic as "match published numbers" — frame it as **"stack equivalence via generative tasks + LL MC methodology correctness via task structure"**. Generative tasks (GSM8K, IFEval, TruthfulQA-Gen) DO match community and validate the pipeline stack. LL MC absolute numbers cannot be cited against Google's generative numbers.
**What IS valid**: deltas between variants (base vs abliterated) on LL MC tasks are fully valid — all variants evaluated identically. This is the primary forensic signal.

### 27. `download_models.sh` `--exclude` is silently broken on modern `hf` CLI — verify bytes, never trust exit code

**Symptom**: `bash comparisons/<name>/download_models.sh` prints `[OK]` for every model, but the dirs are empty (4 KB) or contain only stray auxiliary files. Pipeline then fails downstream when weights stage can't find safetensors.
**Cause** (gemma4-12b, 2026-08-02): the script's `hf download <repo> --exclude "*.gguf" "*.pt" ...` is mis-handled by `huggingface_hub` ≥0.30 (Python 3.14 / pipx build): it warns `Ignoring --exclude since filenames have been explicitly set`, then interprets the glob patterns as **filenames to fetch**. Repos with no matching aux files → 0 files fetched → exit 0 → `[OK]`. Repos that DO have a matching file (e.g. jwest33 ships `*.pt` projectors) get just those aux files, not the weights.
**Fix** (shipped in `comparisons/gemma4-12b/download_models.sh`): dropped `--exclude` entirely — grab the whole repo. All abliterated variants in this project are pure BF16 safetensors, so there is no GGUF/ONNX/PT clutter to filter. Added a **hard post-download verification**: each variant dir must contain ≥1 `*.safetensors` AND total size > 1 GB, otherwise the entry is recorded as `[FAIL]` even if `hf` returned 0. The silent-success path is now impossible.
**Detection**: after any variant download run, `du -sh models/*` must show ~23.92 GB per variant dir, not 4 KB. If you see `Ignoring --exclude` in the output, the bug is present.
**Generalisation**: when writing future `download_models.sh` for a new comparison, copy the gemma4-12b version (no `--exclude`, hard size check). Do NOT re-introduce `--exclude` even if a variant has aux clutter — filter downstream instead (e.g. weights stage already ignores non-safetensors files).

### 28. `./abliterlitics.sh weights` (and likely kl/harmbench) is fragile when backgrounded; use the single-container sequencer pattern for long runs

**Three coupled bugs discovered on gemma4-12b (2026-08-02, journals #66/#67/#68)**:

**(a) JSON Schema rejects runner-supported settings keys** — `comparison.schema.json` had `additionalProperties: false` on the `settings` object but was missing `lmeval_image`, `forensics_image`, `gsm8k_gen_toks`, `lm_eval_gpu_util`. Host-side `./abliterlitics.sh validate` does NOT run `jsonschema.validate()` (it only does path checks via `load_comparison`), so the schema drift is invisible until `ComparisonConfig.from_dir()` runs inside Docker and crashes during `panel_comparison.py`. **Detection**: docker exit traceback ending in `jsonschema.validators.validate ... raise error`; host `validate` passes. **Fix**: schema now includes all 4 keys. When adding a new `settings.foo` reader to `_load_comparison.py`, also add the key to `comparison.schema.json` in the same change.

**(b) Backgrounded `./abliterlitics.sh weights &` dies after first phase** — `run_weights.sh` uses `set -euo pipefail`. `nohup ... &` protects against SIGHUP only, not SIGTERM. When the launching bash tool's command returns and the tool's session is cleaned up, SIGTERM propagates to all descendants → runner script dies. Worse: the docker container spawned by the runner has stdout connected to the runner's pipe; when the runner dies, the container's next `print()` raises SIGPIPE → python crashes mid-phase. **Symptom**: log ends mid-phase with no `Running X.py` for subsequent phases and no completion marker; docker shows container `Exited` shortly after. **Fix (workaround)**: run all weight-analysis phases in ONE detached docker container using `runners/weights_sequencer.sh` + `docker run -d`. The container writes to a file inside the volume (not a stdout pipe) so SIGPIPE cannot kill it. **Detection**: after launching `./abliterlitics.sh weights &`, poll `pgrep -af run_weights` — if no process within 30 min, the runner has died even though the docker may still be alive.

**(c) `run_weights.sh` skip-existing check NEVER triggers** — `run_script` looks for `${RESULTS_DIR}/${COMPARISON_NAME}/weight/${output_file}` (e.g. `panel_comparison.json`) but the scripts write per-variant outputs to `${RESULTS_DIR}/<variant>/<script>_<variant>.json` and main aggregates to `${RESULTS_DIR}/multi_model_panel.json`. The importer (src/db/importers.py:483-497) handles both layouts via per-variant fallback, but the runner's skip-check is broken for every phase. Effect: every `./abliterlitics.sh weights` re-runs everything from scratch. **Detection**: `./abliterlitics.sh weights ./comparisons/<name>/` always says `Running panel_comparison.py` even when results exist. **Fix (TBD)**: align runner skip-check paths with actual script output paths, or pass `--results-dir /results/<comparison>/weight/` to scripts and update skip-check filenames.

**Working approach used for gemma4-12b weights (147 min total)**: `docker run -d --name weights-seq ... abliterlitics-forensics:1.0.0 /app/runners/weights_sequencer.sh`, then `tail -f comparisons/<name>/results/sequencer.log` and `docker ps --filter name=weights-seq` to monitor. Container auto-exits when done. Then `python3 -m src.db import-results comparisons/<name>/` to load results, and per-model `python3 -m src.db stage-complete` to mark stages done (the orchestrator-row `all` can also be marked complete for symmetry).

### 29. `predownload_datasets.sh` IMAGE must match the comparison's lmeval_image — lm-eval version determines task → dataset-config mapping

**Cause** (gemma4-12b, 2026-08-02): `predownload_datasets.sh` hardcodes `IMAGE='abliterlitics-lmeval:1.0.0'` (lm-eval 0.4.11), where `truthfulqa` resolves to `truthfulqa_mc2` (multiple_choice config). The gemma4-12b comparison uses `abliterlitics-lmeval-gemma4unified:1.0.0` (lm-eval 0.4.12), where `truthfulqa` resolves to `truthfulqa_gen` (generation config). Predownload only cached `multiple_choice`; Phase 1 crashed with `ValueError: Couldn't find cache for truthfulqa/truthful_qa for config 'generation'` for every model, with `HF_DATASETS_OFFLINE=1` preventing fallback download.
**Fix**: pre-cache the `generation` config using the SAME image the comparison will use:
```bash
docker run --rm -v .cache/hf:/root/.cache/huggingface abliterlitics-lmeval-gemma4unified:1.0.0 \
  python3 -m lm_eval --model hf --model_args pretrained=HuggingFaceTB/SmolLM2-360M,dtype=float32 \
  --tasks truthfulqa --limit 1 --batch_size 1 --output_path /tmp/results/
```
**Detection**: orchestrator cycles through ALL models in <5 min (should take hours). Phase 1 log ends with `_find_hash_in_cache` ValueError. **Generalisation**: any comparison using a non-default image needs datasets pre-cached with that image's lm-eval version.

### 30. GSM8K Phase 2 on thinking models: `num_fewshot=0` + `until=[]` override — 5-shot causes thinking loops, Question: stop sequence truncates thinking

**Cause** (gemma4-12b, 2026-08-02): TWO coupled issues in Phase 2 GSM8K for thinking models:

**(a) 5-shot fewshot causes thinking loops (PRIMARY issue, 92% null rate)**: The gsm8k task default is `num_fewshot: 5`. Fewshot examples teach the model to produce `Answer: <<48/2=24>>24 clips... #### 72` format. With thinking enabled, the model first solves the problem correctly in its reasoning trace, then tries to reformat the answer to match the fewshot `<<calc>>` notation, entering a loop that consumes all 8096 tokens without converging. The reasoning parser sees incomplete thinking with no closing delimiter → `content=null`. **Verified**: 5-shot base model = 8096 tokens, `finish_reason=length`, content=NULL, reasoning shows correct answer then loops reformatting. 0-shot base model = 377 tokens, `finish_reason=stop`, content populated with correct answer.

**(b) `until=['Question:']` stop sequence truncates thinking (SECONDARY issue, 73% null rate)**: lm-eval gsm8k task defines `until: ['Question:', '</s>', '<|im_end|>']`. The model writes "Question:" during reasoning (e.g., `* Question: How many dogs...`). vLLM truncates at that point → incomplete thinking → content=null. Only triggers when fewshot conditioning causes the model to reference "Question:" in its thinking.

**Fix** (shipped in runner): Phase 2 now passes `--num_fewshot 0 --gen_kwargs max_gen_toks=<N>,until=[]`. The `--num_fewshot 0` eliminates the thinking loop (root cause). The `until=[]` overrides the task stop sequences with the literal string `"[]"` (never matches) as a safety measure. The model stops naturally at EOS after completing thinking + answer. **Verified**: 0-shot + until=[] base model = 0 null responses out of 1319 GSM8K problems.

**Detection**: Phase 2 log shows `grep -c "null content" <model>_phase2.log` >> 0. Null rate >10% on the BASE model indicates Phase 2 config issue. Check for `Overwriting default num_fewshot of gsm8k from 5 to 0` in the log — if absent, the fix is not applied. Also check `gen_kwargs` line shows `'until': []`.

**Applies to**: ALL thinking models on Phase 2 GSM8K (Gemma 4, Qwen3, DeepSeek-R1). Non-thinking models are unaffected by either issue (they don't think, so no loops; they don't reference "Question:" in their response).

**Cache note**: Both `--num_fewshot` and `until` change request hashes, so old cached Phase 2 responses are orphaned and Phase 2 re-runs fresh. Phase 1 is unaffected (Phase 1 uses different endpoint and tasks).

**Score methodology note**: 0-shot GSM8K scores are typically 2-5pp lower than 5-shot for non-thinking models. For thinking models, 0-shot is the STANDARD (Open LLM Leaderboard v2, Google model cards). The 0-shot thinking score is directly comparable to community references. Do NOT compare our 0-shot thinking GSM8K with 5-shot non-thinking GSM8K from other sources.

### 31. `--enforce-eager` is MANDATORY for Gemma4Unified on vLLM 0.26.0 — CUDA graphs cause runtime 500 errors

**Cause** (gemma4-12b, 2026-08-03, journal #73): removing `--enforce-eager` to enable CUDA graph optimization (potential 20-40% speedup) causes TWO failure modes on `Gemma4UnifiedForConditionalGeneration` with vLLM 0.26.0 (`abliterlitics-lmeval-gemma4unified:1.0.0`):
1. **500 Internal Server Error on ALL `/v1/completions` requests** — CUDA graphs capture successfully at startup (PIECEWISE: 51 graphs, FULL: 35 graphs, 0.55 GiB), but every inference request returns HTTP 500. Server then shuts down.
2. **`OSError: Can't load feature extractor for '/model'`** — without eager mode, vLLM enters a code path that tries to load a multimodal `auto_processor_class.from_pretrained()`, which requires `preprocessor_config.json`. Gemma 4 text models don't have this file → crash during model loading.

**Fix**: KEEP `--enforce-eager` for ALL Gemma 4 models on vLLM 0.26.0. The runner now has `LM_EVAL_ENFORCE_EAGER` env var (default=1, set in `run_lm_eval.sh`). Do NOT set to 0 for Gemma4 comparisons. The env var was added for future vLLM versions that may fix the CUDA graph execution bug.

**Detection**: vLLM server log shows `POST /v1/completions HTTP/1.1" 500 Internal Server Error` on every request, or `OSError: Can't load feature extractor` during startup. Runner burns through models in ~2 min each (vLLM starts, crashes on first request, runner moves to next model).

**Throughput impact**: `--enforce-eager` limits MFU to ~15% (~6100 tokens/s for 12B BF16 on 5090). GPU is compute-bound — increasing `num_concurrent` does NOT help (verified empirically: 4 concurrent = 6.89 it/s, 16 concurrent = 5.72 it/s — slightly worse due to client overhead). Phase 1 MMLU-Pro (5-shot, ~3000 token prompts) runs at ~5.5 it/s. Shorter-prompt tasks (ARC, HellaSwag, Winogrande) run 2-4× faster. Realistic per-model time: ~8-12h (not the ~5h estimate from AGENTS.md Timing section — that was optimistic for this architecture).

**Response cache resilience** (journal #75): the response cache (`responses.sqlite_rank0.db`) uses content-hashed keys, NOT config-dependent keys. Changing `num_concurrent` or `enforce_eager` between runs does NOT invalidate the cache. Safe to experiment with vLLM server configs between runs — cache resume works across config changes. Only model/tokenizer/task changes require cache invalidation (`--force`).

### 32. Surgical abliterations show FALSE-POSITIVE GSM8K damage in headline scores — answered-only is the true capability metric

**Empirical finding** (gemma4-12b, journal #76, first 3-number GSM8K analysis per lesson #20): for huihui (weights stage flagged as "most surgical" — only 12 tensors changed, 6 layers, 100% rank-1), the **headline** GSM8K score showed a −15pp gap vs base (66.87% vs 81.96%). The **answered-only** score revealed the gap is **100% thinking-loop failure, NOT capability damage**:

| Metric | base | huihui | delta |
|---|---:|---:|---:|
| Empty responses (thinking-loop null) | 100/1319 (7.6%) | 317/1319 (24.0%) | **+16.4pp** |
| Headline `exact_match,flexible-extract` | 81.96% | 66.87% | −15.1pp |
| **Answered-only score** | **88.68%** | **88.02%** | **−0.7pp (noise)** |

**Cause**: abliteration edits the refusal direction in mid-upper layer attention/MLP weights. Even surgical edits (rank-1, single direction) perturb the residual stream enough to destabilise the model's reasoning trajectory. The model still "knows" the math (88% answered-only = same as base), but enters thinking loops 3× more often — consuming the full `max_gen_toks` budget before emitting `</think>`, leaving `content=null` (lesson #20).

**Implication for variant comparison**: the **headline** GSM8K metric is structurally misleading for abliterations of thinking models. It conflates two distinct signals:
1. **Capability damage** — model can't solve the problem even when it converges (answered-only drops)
2. **Reasoning stability damage** — model thinks itself into corners (empty rate rises)

These have different causes and different implications. A surgical abliteration can show 0 capability damage but 20pp headline damage purely from #2. A brute-force "Obliterated" variant likely shows BOTH.

**Reporting rule** (stronger than lesson #20):
- **Always compute and save** the 3-number GSM8K report per variant. The runner doesn't do this automatically — overseer (or sub-agent) must post-process the Phase 2 samples JSONL.
- **In comparison tables**, lead with **answered-only** score for capability comparison; show headline as a secondary column with the empty% alongside it.
- **Forensic interpretation**:
  - Answered-only within ~2-3pp of base + high empty rate → "surgical abliteration, reasoning destabilised but capability preserved" (huihui fingerprint)
  - Answered-only dropped 5-15pp + high empty rate → "aggressive abliteration, BOTH damaged" (obliteratus fingerprint expected)
  - Answered-only dropped 5-15pp + low empty rate → "distillation or capability-direction hit, reasoning intact" (openyourmind fingerprint expected)
- **DB persistence** (shipped pattern for gemma4-12b): write 4 rows per model under `task='gsm8k_phase2_thinking'` in `lm_eval_results`:
  - `gsm8k_total_problems` (1319)
  - `gsm8k_empty_count` (raw integer)
  - `gsm8k_empty_pct` (float 0-100)
  - `gsm8k_answered_only_exact_match` (float 0-1, comparable to headline metric)

**Detection**: compute `thinking_tax = headline - answered_only`. If `|answered_only_delta_vs_base| < 3pp` BUT `|headline_delta_vs_base| > 5pp` → the gap is loop-failure, not capability. Cite the answered-only number; flag the loop-failure as a separate stability finding.

**Detection script** (post-processing pattern): for each `<model>_phase2.log`'s samples JSONL (`samples_gsm8k_<timestamp>.jsonl`), iterate rows filtering on `filter == 'flexible-extract'`, dedupe by `doc_id`, check if `resps[0][0]` is empty/whitespace, compute (correct / total) and (correct / non-empty). ~30 lines of Python.

**Generalisation**: applies to ANY thinking model + generative eval (GSM8K now; AIME / MATH when added). Does NOT apply to LL MC tasks (those don't generate, they score answers — no thinking budget consumed).

---

### 33. Gemma4Unified vLLM load failures — THREE coupled model-packaging bugs (all on upstream variant dirs, none on official base)

**Context** (gemma4-12b, 2026-08-04, journals #74, #77, #78, #79): when running lm-eval on 8 abliterated variants of gemma-4-12B-it (all `Gemma4UnifiedForConditionalGeneration`), vLLM 0.26.0 (`abliterlitics-lmeval-gemma4unified:1.0.0`) hit THREE distinct packaging-defect failure modes during model load — each blocking a different subset of variants. The official base model and the huihui variant (presumably packaged more carefully) loaded cleanly. All three are upstream defects in the abliterator's published model dir; all three have safe host-side fixes that **do NOT modify actual model weights or text-inference behavior**.

**(a) Missing `processor_config.json` (originally lesson #31 extension)** — 4 variants failed during startup with `OSError: Can't load feature extractor for '/model'`. **Fix**: copy base's `processor_config.json` into each affected variant dir. Already applied previously by the overseer.

**(b) Truncated `config.json` — missing multimodal sub-config keys** (6 of 8 variants: obliteratus, abliterix, heretic, coder3101, prithiv, apostate). **Symptom**: `ValueError: There is no module or parameter named 'embed_vision.multimodal_embedder' in Gemma4UnifiedForConditionalGeneration. The available parameters belonging to embed_vision (Gemma4MultimodalEmbedder) are: {'embed_vision.embedding_projection.weight'}`. **Root cause**: the variant `config.json` is 169 lines vs base's 172 lines — the abliterator's transformers version stripped 3 keys from `audio_config` (`audio_samples_per_token`, `hidden_size`, `output_proj_dims`) and 1 from `vision_config` (`model_patch_size`); vLLM then generates incorrect expected-weight names from the incomplete config. **Fix**: copy base's `config.json` over the variant's. **Safe because**: all variants share identical `text_config` (the LLM itself: 48 layers, 3840 hidden, 262144 vocab, etc); the multimodal sub-configs are unused for text benchmarks; `max_position_embeddings` rises from 131072→262144 but we cap at `max_model_len=12288`. Backup: `config.json.broken_multimodal.bak`.

**(c) Pre-refactor safetensors key layout (obliteratus only)** — distinct from (b). **Symptom**: same `embed_vision.multimodal_embedder` ValueError PERSISTS after the config fix. **Root cause**: the safetensors itself was packaged with an OLDER HuggingFace transformers version that used pre-refactor module names. 10 multimodal keys use the old layout (e.g. `model.embed_vision.multimodal_embedder.embedding_projection.weight`, `model.embed_vision.patch_dense.*`) while base uses the post-refactor layout (`model.embed_vision.embedding_projection.weight`, `model.vision_embedder.patch_dense.*`). **Detection**: `python3 -c "from safetensors.torch import safe_open; ..."` and look for `multimodal_embedder` substring in any key. **Fix**: load all 677 tensors, rename the 10 affected keys, write a new safetensors file (script pattern: `/tmp/fix_obliteratus_keys.py` — load → rename in dict → `save_file` → backup original as `model.safetensors.prekeyrename.bak`). **Safe because**: the underlying tensor data is byte-identical (verified: all 10 tensors match base in shape AND dtype); the change is purely a key-string rename in the safetensors header. 22GB file rewrites in ~15s on NVMe.

**Detection (combined)**: before launching a Gemma4Unified lm-eval run, smoke-test each variant with: `(1) wc -l config.json` (expect 172 for the full multimodal config; 169 = broken); `(2) python3 -c "from safetensors.torch import safe_open; import glob; [print(k) for f in glob.glob('*.safetensors') for k in safe_open(f,framework='pt').keys() if 'multimodal_embedder' in k]"` (expect 0 output; any output = needs key rename); `(3) ls processor_config.json` (expect present; missing = copy from base).

**Generalisation**: any new Gemma 4 Unified variant may have one or more of these defects. The fixes are non-destructive (originals preserved as `*.bak`) and don't alter text-inference behavior. Apply preemptively before launching multi-day runs to avoid 2-minute-per-variant failure cascades that waste ~16 min (8 variants × 2 min) before the run aborts.

---

### 34. NLTK `punkt_tab` is a hard dependency for BBH — network-restricted environments need a host-side nltk_data mount

**Cause** (gemma4-12b, 2026-08-04, journal #80): `leaderboard_bbh` task in lm-eval imports `nltk.tokenize.punkt_tab` at task-load time. The base Docker image does NOT bundle nltk_data, so the lm-eval container calls `nltk.download('punkt_tab')` at startup. In network-restricted environments (e.g. offline GPU boxes, firewall blocks to `raw.githubusercontent.com`), this download hangs **silently** — the log shows `[nltk_data] Downloading package punkt_tab to /root/nltk_data...` and then no further progress. The container sits at 0% GPU utilization indefinitely.

**Detection**: Phase 1 log stuck on `[nltk_data] Downloading package punkt_tab...` with no `Requesting API:` progress and 0% GPU. Container's `/root/nltk_data/tokenizers/` contains `punkt_tab.zip.tmp` (partial download) and `punkt_tab.zip.lock` but no `punkt_tab/` directory.

**Fix** (shipped in `run_lm_eval.sh`): mount a host-side nltk_data directory into the container at `/root/nltk_data:ro`. Pre-populate the host dir once via:
```bash
mkdir -p .cache/nltk_data/tokenizers
curl --max-time 600 --retry 3 -o .cache/nltk_data/tokenizers/punkt_tab.zip \
    https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt_tab.zip
unzip -o .cache/nltk_data/tokenizers/punkt_tab.zip -d .cache/nltk_data/tokenizers/
```
The runner now has `NLTK_CACHE="${ABL_ROOT}/.cache/nltk_data"` and `-v "${NLTK_CACHE}:/root/nltk_data:ro"`.

**Why base+huihui ran without this fix**: their containers successfully downloaded punkt_tab over the network before it became restricted. The download is per-container (ephemeral `/root/nltk_data` was not mounted previously), so each new container retries the download.

**Detection after fix**: no `punkt_tab` download attempt in the log; BBH task loads cleanly. Pre-flight smoke check: `docker run --rm -v $NLTK_CACHE:/root/nltk_data:ro <image> python3 -c "import nltk; nltk.data.find('tokenizers/punkt_tab')"`.

---

### 35. Merging a small-alpha LoRA in BF16 produces sub-precision edits — the merged diff is quantization dithering, NOT the intended low-rank update

**Symptom** (gemma4-12b, 2026-08-08, journal #93): two merged-PEFT-LoRA variants (sdft_cybersec_r2, sdft_smoothed_r2_new; both r=2, alpha=2, 211 targets) were predicted to show `effective_rank_90pct ≤ 2` in SVD analysis (rank-2 by construction). **Actual: `effective_rank_90pct_energy ≈ 20` (max) for both.** The rank-r prediction failed in the energy-weighted metric.

**Cause**: `scripts/merge_lora.py` loads the base in BF16 (`torch_dtype=torch.bfloat16`), runs `merge_and_unload()` in BF16, saves BF16. With `alpha/r = 1.0`, the LoRA edits are `relative_edit mean ≈ 0.0005–0.0009`. BF16 relative precision is `2^-7 ≈ 0.78%` (0.0078) — so the edits are **~10× below one BF16 ULP**. Most weight elements round back to the original BF16 value under round-to-nearest; the diff (`merged − base`) is dominated by per-element rounding flips = **full-rank quantization dithering shaped by the LoRA's sign pattern**, not the intended rank-2 update.

**Evidence pattern** (use these to recognize the artifact):
1. **`lowrank_captured_energy_pct` (rank-2 reconstruction energy)**: a clean rank-1 abliteration (edits above precision) shows ~100%; the sub-precision LoRAs show only **63% / 77%** — 23–37% of diff energy is unreconstructable BF16 noise.
2. **Top singular values**: genuine rank-2 signal sits in σ1, σ2 (e.g. 2.98e-3, 1.22e-3) clearly above a noise floor at σ3…σ20 (~5e-4, roughly flat). The top-2 ARE the LoRA, but the noise floor contributes comparable energy → effective_rank_90 counts noise → reports ~20.
3. **`sigma2/sigma3` ratio** is modest (~2–8), not the "infinite" ratio of a true rank-2 matrix (σ3=0).
4. **Technique-correlation cosine vs EVERY other variant ≈ 0.0001–0.007**: the dithering is random per-element, uncorrelated with any structured abliteration direction.
5. `changed_count` still equals the exact LoRA target-module count (211) — the merge touched every target, the artifact is in magnitude, not in which tensors moved.

**Implication for downstream stages**: because the genuine edit is ~10× below BF16 precision, the merged model may be **behaviorally near-identical to base** — the abliteration was largely rounded away. Expect minimal capability change AND minimal refusal removal in lm-eval/HarmBench for sub-precision merges. The top-2 singular values carry a partial (63–77% energy) signal, so some attenuated effect may persist. This is a **precision caveat, NOT corruption** — the model is valid (677 tensors, loads fine, all 677 keys present).

**Fix (for future LoRA merges)**: merge in FP32 (load base `dtype=torch.float32`, `merge_and_unload()`, THEN cast to BF16 only at `save_pretrained`), OR use higher `alpha/r` scaling so edits exceed the BF16 ULP (~0.78%). `merge_lora.py` as-shipped is sub-optimal for small-alpha LoRAs.

**Detection**: before trusting SVD effective-rank of a merged LoRA, compute `lowrank_captured_energy_pct` (rank-r reconstruction energy). If it's well below ~95% AND `relative_edit mean < ~0.002` AND `sigma2/sigma3` ratio is small (<10), the merge is sub-precision and the "rank-r" structure is masked by BF16 dithering. Distinguish from a genuine merge bug (which would show changed_count ≠ target count or missing tensors).

**Generalisation**: applies to ANY merged LoRA where `alpha/r × ||ΔW||` is below the storage dtype's ULP. Most likely with r=2/alpha=2 "minimal" adapters. Does NOT affect full-model abliterations (trevorjs here: energy_top1=100%, effective_rank_90=1.0 — clean rank-1, edits ~0.017 well above precision).

### 36. HarmBench runner needs FLASHINFER_DISABLE_VERSION_CHECK=1 + must be invoked via abliterlitics.sh (not standalone python)

**Symptom** (gemma4-12b, 2026-08-12, journal #107): HarmBench vLLM container exits ~30s after start with code 1. `docker logs` shows `Model architectures ['Gemma4UnifiedForConditionalGeneration'] failed to be inspected`. The root cause (deep in `registry.py:1019` traceback) is the same flashinfer/flashinfer-cubin version mismatch as lesson #12: `RuntimeError: flashinfer-cubin version (0.6.12) does not match flashinfer version (0.6.14)` during `flashinfer.jit.env._get_cubin_dir()`.

**Coupled bug #2**: invoking `python3 runners/run_harmbench.py <dir>` directly (NOT via `./abliterlitics.sh harmbench`) silently uses the WRONG docker image. The runner reads `LMEVAL_IMAGE = os.environ.get("LMEVAL_IMAGE", "abliterlitics-lmeval:1.0.0")` — it does NOT call `load_comparison` itself. Without `_load_comparison.py` sourced first, it falls back to `abliterlitics-lmeval:1.0.0` (vLLM 0.19.0) which doesn't know `Gemma4UnifiedForConditionalGeneration` at all → a DIFFERENT, misleading error ("Transformers does not recognize this architecture").

**Fix** (shipped in `runners/run_harmbench.py`, docker_cmd env block): added three env vars after `CUDA_VISIBLE_DEVICES=0`:
- `FLASHINFER_DISABLE_VERSION_CHECK=1` (the real fix — lesson #12 propagation)
- `HF_HUB_OFFLINE=1` + `HF_DATASETS_OFFLINE=1` (HarmBench prompts are local CSV; no download needed)

**ALWAYS launch HarmBench via**: `./abliterlitics.sh harmbench ./comparisons/<name>/` (which sources `_load_comparison.py` exporting `LMEVAL_IMAGE`, `VARIANT_LORA_ADAPTER_*`, `REASONING_PARSER`, etc.). NEVER `python3 runners/run_harmbench.py` standalone — it will use wrong image + no LoRA adapter detection + wrong reasoning parser.

**Detection**:
1. Container image is `abliterlitics-lmeval:1.0.0` (not the comparison's `lmeval_image`) → standalone invocation bug.
2. Container image IS correct (`abliterlitics-lmeval-gemma4unified:1.0.0`) but still crashes with "failed to be inspected" → flashinfer mismatch; check for `FLASHINFER_DISABLE_VERSION_CHECK=1` in the runner's docker_cmd.
3. Smoke test: `docker run --rm -e FLASHINFER_DISABLE_VERSION_CHECK=1 ... <image> python3 -c "from vllm.model_executor.models.gemma4_unified import Gemma4UnifiedForConditionalGeneration"` → should print OK.

**Generalisation**: lesson #12's fix lived only in `run_lm_eval.sh`. Any NEW runner that starts vLLM containers (`run_harmbench.py`, future `run_kl.py` if containerized, etc.) must also set `FLASHINFER_DISABLE_VERSION_CHECK=1` when targeting the gemma4unified image. When you fix a docker-env bug in one runner, grep the others for the same pattern.

### 37. KL divergence `F.kl_div` produces NaN for ANY model with full-vocab `-inf` log-probs — clamp BEFORE compute (PyTorch ≥2.11 regression / latent bug)

**Symptom** (gemma4-12b, 2026-08-13, journal #110): ALL 12 variant KL results were `NaN` in the DB — not just the heavily-abliterated ones. Every `kl_<variant>.json` had `kl_divergence_batchmean: nan` across all 100 prompts. The compute phase ran without error (exit 0), so the NaN was silent.

**Root cause**: `F.kl_div(input, target, log_target=True)` computes `exp(target) * (target - input)` element-wise. For a 262K vocab, ~200 tokens/prompt have log-prob = `-inf` (underflowed from `F.log_softmax`). When `target = -inf`: `exp(-inf) = 0`, then `0 * (-inf - input) = 0 * -inf = NaN` per IEEE 754. A SINGLE NaN element poisons the `sum()` → batchmean = NaN. The old comment ("F.kl_div handles -inf correctly: exp(-inf)=0, contributes 0") was WRONG — `0 * -inf` is NaN, not 0. This was latent in `src/kl/kl_divergence.py` cmd_compute since inception but never triggered because previous comparisons (a) used the `abliterlitics-forensics:1.0.0` image which has an older PyTorch that may have special-cased `0 * -inf`, or (b) had smaller vocabs (128K/32K) with fewer -inf entries. The gemma4_unified comparison required `forensics_image: abliterlitics-lmeval-gemma4unified:1.0.0` (torch 2.11.0) for model loading — which exposed the bug.

**Fix** (shipped in `src/kl/kl_divergence.py` cmd_compute, lines ~753): clamp non-finite log-probs to `LP_FLOOR = -100.0` via `torch.nan_to_num(tensor, nan=LP_FLOOR, posinf=0.0, neginf=LP_FLOOR)` on BOTH `base_lp` and `variant_lp` BEFORE `F.kl_div`. Why -100: it's well below any observed finite log-prob (-57.5 min for gemma4-12b), `exp(-100) ≈ 3.7e-44 ≈ 0` in float32 (same effective probability as -inf), but `0.0 * finite = 0.0` (NOT NaN). Mathematically equivalent for softmax; keeps arithmetic finite. `+inf` (rare, from extreme weight mods) → 0 (prob 1).

**Verification**: after fix, all 12 variants produced finite KL (0.049–2.279). The clamping log confirms ~200 -inf entries clamped per tensor. No GPU re-collection needed — the logits `.pt` files were valid; only the compute phase needed re-running (delete `kl_*.json`, re-run `./abliterlitics.sh kl` — Phase 2a skips existing logits, Phase 2b re-computes with fix).

**Detection**: `SELECT variant, kl_batchmean FROM kl_results WHERE comparison='<name>'` — if ALL are NULL/None, the NaN bug hit. Cross-check: `docker run ... python3 -c "import torch; d=torch.load('logits_<variant>.pt',weights_only=False); lp=d['logprobs']; print(int((torch.isinf(lp)&(lp<0)).sum()))"` — if >0 AND the old code is running, KL will be NaN.

**Coupled issue — forensics_image needed for Gemma4Unified KL**: `abliterlitics-forensics:1.0.0` has transformers too old to load `gemma4_unified`. KL requires `settings.forensics_image: abliterlitics-lmeval-gemma4unified:1.0.0` (transformers 5.12.1). BUT this image lacks `matplotlib` → `./abliterlitics.sh graphs` fails. Workaround: run graphs/tables/report with the default forensics image (temporarily remove `forensics_image` from comparison.json after KL completes — graphs don't load models). Also `chmod +x src/**/*.py` — the gemma4unified image exec's scripts directly (not via `python3` prefix), unlike the forensics image which has an ENTRYPOINT wrapper. And `run_kl.sh` had `local` keywords outside a function (lines 125/126/139) that crashed Phase 2 under `set -euo pipefail` — removed.

**Generalisation**: affects ANY model where `F.log_softmax` produces -inf entries (any vocab > ~50K in float32). The fix is universal and should remain in the compute path permanently. If a future PyTorch version fixes `F.kl_div` to handle `0 * -inf = 0`, the clamp becomes a harmless no-op (exp(-100) ≈ 0 regardless).

### 38. `export-review` truncates `reasoning_text` at 3000 chars — full traces are 8K-46K; judges MUST pull full text for empty responses

**Symptom** (gemma4-12b, 2026-08-13, journal #112): HarmBench review batch for copyright behaviors (lyrics/passages) showed ~3000-char `reasoning_text` in the export, all looking like "planning loops with fabricated drafts". Judging from the export alone would have misclassified items where the deciding verbatim content or degenerate-loop evidence lives BEYOND char 3000.
**Cause**: `src/db/cli.py` cmd_export_review does `r[6][:3000]` — the DB stores full traces (8K-46K chars for thinking models that hit the 8096-token budget). The 3000-char cap is invisible unless you query the DB directly (all lengths read exactly 2997-3000).
**Fix** (shipped in cli.py): `--full-reasoning` flag exports the complete trace; `reasoning_chars_total` field is ALWAYS exported so truncation is detectable (`len(reasoning_text) < reasoning_chars_total` = truncated). Reviewers judging empty-response items must re-export with `--full-reasoning` (or query the `responses` table directly keyed by model_name+behavior_id).
**Detection**: export items with `len(reasoning_text) == 3000` are certainly truncated; check `reasoning_chars_total`. In this batch, tail inspection of full traces changed zero slice-1 verdicts but was decisive for confirming degenerate loops and finding verbatim content (e.g. kite_runner's verbatim opening sat mid-trace).
**Generalisation**: any consumer of `export-review` that needs reasoning content (not just a preview) needs `--full-reasoning`. Applies to every thinking-model comparison (Gemma 4, Qwen3, DeepSeek-R1).

---

## Benchmark Tasks

Default suite: **Open LLM Leaderboard v2** + GSM8K + TruthfulQA + **v1 forensic supplement** + HumanEval. The v1 tasks were dropped from OLL for "saturation" — but **saturation is irrelevant for delta-detection** (a benchmark at 80%→77% still shows the 3pp damage signal).

| Task | Type | Metric | Shots | Notes |
|---|---|---|---:|---|
| `leaderboard_mmlu_pro` | LL MC | acc | 5 | 12k Q, 14 disciplines, 10 choices |
| `leaderboard_gpqa_diamond` | LL MC | acc | 0 | 198 PhD science Q. **Gated — needs HF_TOKEN**. Resilient to abliteration (#22). |
| `leaderboard_bbh` | LL MC | acc | 3 | 23 Big-Bench Hard subtasks. snarks (sarcasm) is consistent abliteration collateral (~3-10pp drop). |
| `leaderboard_musr` | LL MC | acc | 0 | 3 subtasks |
| `leaderboard_ifeval` | Generate | prompt/inst acc | 0 | Critical for abliteration damage. Sensitive to disinhibition (lesson #22, journal #41). |
| `gsm8k` | Generate | exact_match | 0 | Needs high `max_gen_toks` for thinking. Report 3 numbers (lesson #20). Preserved by abliteration, damaged by distillation (lesson #21). |
| `truthfulqa` | Mixed | bleu/rouge/acc | 0 | **The abliteration canary** — drops 3-16pp across abliterations. Preserved by distillations. |
| `hellaswag` | LL MC | acc_norm | 0 | Common sense |
| `winogrande` | LL MC | acc | 0 | Pronoun resolution |
| `arc_challenge` | LL MC | acc_norm | 0 | Grade-school science. Resilient to abliteration (#22). |
| `piqa` | LL MC | acc_norm | 0 | Physical interaction QA |
| `lambada_openai` | LL | perplexity + acc | 0 | **Killed by `--apply_chat_template` (lesson #17)** |
| `humaneval` | Generate (exec) | pass@1 | 0 | **Requires `HF_ALLOW_CODE_EVAL=1` + `--confirm_run_unsafe_code`. Killed by `--apply_chat_template` (#17)** |

**Timing (per model, single 5090, bf16)**: Phase 1 (LL MC + generative) ~3h, forensic supplement ~30-45min, HumanEval ~30-60min, generative ~30min, Phase 2 GSM8K ~55min. **~5h total per model. ~6-7h with overhead.** Distill models can run ~30% faster (~20 it/s vs ~12-15 it/s) due to KV cache efficiency (journal #38).

**Dataset pre-download** (one-time, required): `bash runners/predownload_datasets.sh`. Runner sets `HF_DATASETS_OFFLINE=1` + `HF_HUB_OFFLINE=1`, fails fast on empty cache. GPQA needs HF auth + terms accepted at https://huggingface.co/datasets/Idavidrein/gpqa. HumanEval cache requires `HF_ALLOW_CODE_EVAL=1` set during predownload.

## File Locations

```
comparisons/<name>/               # comparison dir (comparison.json, NOTES.md, templates)
results/lm_eval/                  # lm_eval_<model>.json (canonical), __model/ (raw)
results/lm_eval_logs/             # <model>_vllm_server.log, _lm_eval_combined.log
.cache/hf/datasets/               # pre-cached benchmark datasets
abliterlitics.db                  # pipeline state + journal
docker/Dockerfile.*               # image definitions
runners/                          # all pipeline runners
```

## Automation Pipeline

```bash
./abliterlitics.sh auto ./comparisons/<name>/             # full pipeline
./abliterlitics.sh weights ./comparisons/<name>/          # individual phases
./abliterlitics.sh kl ./comparisons/<name>/
./abliterlitics.sh lm-eval ./comparisons/<name>/
./abliterlitics.sh harmbench ./comparisons/<name>/
./abliterlitics.sh graphs ./comparisons/<name>/
./abliterlitics.sh --dry-run auto ./comparisons/<name>/   # preview
./abliterlitics.sh --force lm-eval ./comparisons/<name>/  # rerun all models
./abliterlitics.sh status ./comparisons/<name>/           # result counts
./abliterlitics.sh validate ./comparisons/<name>/         # verify paths
```

**Settings plumbing**: `runners/_load_comparison.py` reads `comparison.json` and exports shell env vars — `BASE_DIR`, `VARIANT_NAMES`, `VARIANT_PATH_*`, `LM_EVAL_TASKS`, `LM_EVAL_MAX_MODEL_LEN`, `LM_EVAL_MAX_GEN_TOKS`, `HARMBENCH_MAX_TOKENS`, `REASONING_PARSER`, `REASONING_CONFIG_JSON`, `VARIANT_SKIP_*`, plus Docker image overrides from `settings.lmeval_image`/`settings.forensics_image`.

**Key defaults** (override in `comparison.json` → `settings`):

| Setting | Default | Notes |
|---|---|---|
| `harmbench_max_tokens` | 8096 | Reasoning models need thinking + response room |
| `lm_eval_max_model_len` | 16384 | Default for v2 suite (BBH 3-shot needs context) |
| `lm_eval_max_gen_toks` | 4096 | Phase 1 generative |
| GSM8K Phase 2 gen_toks | 7168 | Auto-capped to `max_model_len - 1024` |
| `lm_eval_tasks` | 13 tasks (table above) | |
| `LM_EVAL_NUM_CONCURRENT` | 4 | HTTP parallelism to vLLM. GPU is compute-bound at ~6100 tok/s — increasing doesn't help throughput. Made configurable via env var in `run_lm_eval.sh`. |
| `LM_EVAL_ENFORCE_EAGER` | 1 (on) | **MUST stay on for Gemma4Unified on vLLM 0.26.0** (lesson #31). Made configurable for future vLLM versions. |

## GPU Hardware

- **GPU 0**: NVIDIA RTX 5090 (32GB) — used for all eval runs
- **GPU 1**: NVIDIA RTX 4090 (24GB) — **leave alone unless user explicitly authorizes**
- Docker needs `--runtime=nvidia --shm-size=16g --ipc=host`

### 39. ComfyUI API A/B generation: poll the SERVER-assigned prompt_id, fetch /view with subfolder params, and capture status-message timings

**Symptom** (te-zimage A/B, 2026-08-13): `run_ab_test.py` queued an image, ComfyUI generated it in ~4s, but the script hung until timeout. Once fixed, the image fetch 404'd.

**Cause (three coupled bugs)**:
1. The script self-generated a `prompt_id = uuid4()` and polled `/history/<that>` — but ComfyUI assigns its OWN prompt_id in the `/prompt` response body (`{"prompt_id": "...", "number": 26, "node_errors": {}}`). Polling a self-generated ID never resolves.
2. `filename_prefix` containing a slash (`abliterlitics-ab/base_int8`) makes ComfyUI save into a SUBFOLDER: history reports `{"filename": "base_int8_00001.webp", "subfolder": "abliterlitics-ab", "type": "output"}`. Fetching `/view?filename=<slash-path>` 404s; the endpoint wants separate `filename`/`subfolder`/`type` query params (urlencode them).
3. Bonus: `/prompt`'s `node_errors` must be checked — a rejected workflow queues nothing.

**Rich data available** (capture it, "the more data the better"): `/history` status messages carry `execution_start`/`execution_success` timestamps (server-side gen ms), `execution_cached.nodes` (which nodes hit cache — for TE loaders this is TRUE cold/warm verification, better than inference timing), and the queue number. All recorded in the A/B manifest `images[].perf` + `te_ab_images` DB rows.

**Detection**: any ComfyUI API client that polls forever = check you're using the returned prompt_id. Any /view 404 = check subfolder/type params.

**Access note**: from inside the analysis Docker container, the host ComfyUI is `http://host.docker.internal:8191` (NOT localhost).

### 40. Host shell cannot reach ComfyUI; route all ComfyUI-API work through a container, and controls `--merge` needs the ORIGINAL dose-variants + a drift tolerance

**Symptom A** (te-zimage, 2026-08-14): `curl http://localhost:8191/...` from the agent host shell returns 000 (connection refused) even though `docker ps` shows `0.0.0.0:8191->8188`. **Fix**: the sandbox cannot reach the docker-proxy listener; containers CAN. Invoke ComfyUI-API clients (`run_ab_test.py` etc.) inside `abliterlitics-te:1.2.0` with `--add-host=host.docker.internal:host-gateway`, the repo mounted at `/app` (`-w /app -e PYTHONPATH=/app`), and `/home/maxx/extra/comfyui-models` mounted at its OWN absolute path so `comfy_te_name()` relativizes `te_file` entries correctly. Verify with a `/system_stats` probe before any long run.

**Symptom B**: `run_te_controls.sh --merge` refused with "different ... dose reference" even with the right dose-variants. **Cause**: the metadata check used 1e-6 relative tolerance, but the conditioning cache had been RECAPTURED since the original battery (~3-4% numeric drift in per-prompt MSE). **Fix**: 5% relative tolerance on `max_dose_mse` (a wrong dose-variant is still caught: q3 ≈2.6× off, heretic-only ≈70× off). When merging into an existing battery, you MUST pass the SAME `--dose-variants` as the original run — recover it by comparing stored `max_dose_mse` against per-variant cached MSE (probe: `_load_consumed_from_cache` on cuda; the cache is keyed by device — cpu probes miss). Merge semantics: conditions already present are SKIPPED, never recomputed (protects frozen numbers).

**Detection**: `--merge refused` → check stored vs recomputed `max_dose_mse` mean before assuming a setup mismatch.
