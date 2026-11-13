---
name: lm-eval
description: Runs the lm-eval capability stage — 13-task suite, two-phase per model,
  --apply_chat_template enforcement, thinking-model GSM8K handling, watchdog restart
  + response-cache resume.
decisionTrigger: Task runs the lm-evaluation-harness capability stage?
examples:
- Run the lm-eval stage for comparisons/qwen38-27b/
- Phase 1 died with EngineDeadError — verify watchdog state and resume
modelRole: glm-5.3-flash
timeout: 0
---

# LM-Eval Agent — Abliterlitics

You run capability benchmarks via lm-evaluation-harness — measuring whether abliteration damages model reasoning, knowledge, and language abilities. Longest-running phase (~5 hrs per model on the extended 13-task suite).

Standard execution environment (Docker-only — vLLM server AND lm-eval in the SAME container, no sudo, GPU 0 only, GPU-exclusive — cannot overlap harmbench/KL). DB-first: check `./abliterlitics.sh db pipeline-status` and skip `complete` models.

## Context: Read the Journal First
```bash
./abliterlitics.sh db journal latest --comparison <name>
```
Key subjects: `chat_template`, `thinking_mode`, `benchmark_references`, `diagnostic_*`. If any missing, init agent didn't finish — escalate.

## What You Do

1. **Run lm-eval**: `./abliterlitics.sh lm-eval ./comparisons/<name>/`

2. **Architecture**: vLLM OpenAI server + `local-completions` backend, both in the SAME container. Single-container design avoids the tokenizer download hang (AGENTS.md #6).

3. **Two-phase per model**:
   - **Phase 1**: Loglikelihood MC (`leaderboard_mmlu_pro`, `leaderboard_gpqa_diamond`, `leaderboard_bbh`, `leaderboard_musr`) + v1 forensic supplement (`hellaswag`, `winogrande`, `arc_challenge`, `piqa`, `lambada_openai`) + `humaneval` + TruthfulQA gen — `max_gen_toks=4096`. **`--apply_chat_template` mandatory for instruct models** (auto-detected — AGENTS.md #9). **`HF_ALLOW_CODE_EVAL=1`** for HumanEval pass@k.
   - **Phase 2**: GSM8K — `max_gen_toks=7168` (reasoning models need thinking room), auto-capped to `max_model_len - 1024` (AGENTS.md #1). Uses `local-chat-completions` endpoint.

4. **Monitor progress** — takes hours. Logs: `<model>_vllm_server.log`, `<model>_phase1.log`, `<model>_phase2.log`.

5. **Report results** — per-model scores across 13 tasks. Validation: generative tasks (GSM8K, IFEval, TruthfulQA) should match community. LL MC tasks NOT comparable to Google's published generative numbers (~25pp gap by design — AGENTS.md #21, #26).

## Critical Technical Details

### `--apply_chat_template` MANDATORY for instruct models (AGENTS.md #9)
**The #1 rule.** Without it, LL MC scores depressed 30-60pp. Runner auto-detects via `chat_template.jinja` OR `tokenizer_config.json` containing `"chat_template"` (per-model AND per-tokenizer-dir).
```bash
./abliterlitics.sh --dry-run --force lm-eval ./comparisons/<name>/
# Look for: phase1 flag='--apply_chat_template'. If '<none>' for an instruct model → STOP.
```

### Thinking models (AGENTS.md #16, originally mis-framed)
- `enable_thinking=True` is **only for generative tasks** (GSM8K). Never loglikelihood.
- For models whose chat_template defaults `enable_thinking=true` (Gemma 4 E4B), the `<|think|>` injection eats the GENERATION budget → truncated generative responses (IFEval, TruthfulQA-Gen depressed). The fix `comparisons/<name>/no_thinking_chat_template.jinja` (copy with default flipped false) restores them. **No effect on LL MC tasks** — those were always the true LL MC scores.
- For models that default `enable_thinking=false` (Gemma 4 12B Unified) — no override needed.
- Phase 2 GSM8K uses `thinking_chat_template.jinja` override via vLLM `--chat-template` — correct for generative.

### Other details (see AGENTS.md)
- `max_gen_toks` includes thinking tokens (AGENTS.md #1)
- Quantization default is **BF16 always** — only deviate if user explicitly asks (AGENTS.md #2)
- Reasoning parser auto-detect: Gemma4* → `gemma4`; qwen3 → needs reasoning_config (AGENTS.md #4)
- Phase 1 + Phase 2 results merged into canonical `lm_eval_<model>.json`
- lm-eval 0.4.11 `--cache_requests` CLI bug auto-patched in runner (AGENTS.md #15; no-op on 0.4.12+)

## Monitoring Commands
```bash
ls comparisons/<name>/results/lm_eval/lm_eval_*.json                  # completed models
./abliterlitics.sh db pipeline-status ./comparisons/<name>/           # per-model progress + timing
tail -f comparisons/<name>/results/lm_eval_logs/<model>_phase1.log    # live
cat comparisons/<name>/results/lm_eval/lm_eval_base.json | jq '.results | to_entries[] | {task: .key, acc: .value.acc, acc_norm: .value.acc_norm}'
```

`./abliterlitics.sh lm-eval` auto-tracks DB stages and imports results. No manual DB steps.

## Timing Estimates (per model, single 5090, native bf16)

Phase 1 (LL MC + v1 supplement + humaneval + truthfulqa): ~4h. Phase 2 (GSM8K): ~55 min. **Total: ~5 hrs/model.** Distill models can run ~30% faster (~20 it/s vs ~12-15 it/s) due to KV cache efficiency (AGENTS.md → Benchmark Tasks).

## Troubleshooting

| Problem | Likely Cause | Fix |
|---|---|---|
| vLLM OOM | Model too large for single GPU at BF16 | Use `--gpus dual` for TP=2 (Qwen3.5 supports TP in BF16) |
| GSM8K all zeros | `max_gen_toks` too low | Auto-caps to `max_model_len - 1024`. If model_len=4096 → 3072 may be too low for reasoning |
| Thinking loops | Abliterated model behavior | Known (AGENTS.md #7, #20). Not a config issue. |
| Phase 1 results missing | Multi-phase merge failed | Verify `lm_eval_<model>.json` has keys from all tasks |
| **MMLU/HellaSwag/ARC near chance** | **Missing `--apply_chat_template`** | Verify dry-run shows `phase1 flag='--apply_chat_template'` (AGENTS.md #9) |
| **IFEval/TruthfulQA-Gen depressed on thinking model** | **`<\|think\|>` eating generation budget** | Create `comparisons/<name>/no_thinking_chat_template.jinja` (AGENTS.md #16). Re-run with `--force` to clear cache. |
| **Gemma4Unified vLLM crash: `embed_vision.multimodal_embedder` not found** | **Truncated `config.json` (169 vs 172 lines)** | Copy base's `config.json` to variant. Backup as `config.json.broken_multimodal.bak` (AGENTS.md #33b). |
| **Same crash persists after config fix (obliteratus only)** | **Pre-refactor safetensors key layout** | Rename 10 multimodal keys in safetensors (`multimodal_embedder.*` → flatten, `embed_vision.patch_*` → `vision_embedder.*`). Backup as `model.safetensors.prekeyrename.bak` (AGENTS.md #33c). |
| **Phase 1 stuck at 0% GPU on `[nltk_data] Downloading punkt_tab`** | **Network blocked; punkt_tab not bundled in image** | Pre-download to `.cache/nltk_data/` on host; runner now mounts it at `/root/nltk_data:ro` (AGENTS.md #34). |
| **Dry-run dies: `VARIANT_LORA_ADAPTER_X.Y_Z: invalid variable name`** | **Dots in base/variant dir name break bash `${!indirect}`** | Fixed in exporter+consumers (AGENTS.md #64). Regression = someone re-introduced `tr '-' '_'`-only slug derivation. |
| **Dry-run shows "native bf16"/`GPUs=dual` on an FP8 comparison** | **Runner captured settings env before `load_comparison` exported them** | Fixed (AGENTS.md #65). Verify dry-run prints the comparison recipe per arm. |
| **vLLM startup crash: pydantic "Invalid JSON" on `--reasoning-config`** | **`printf %q` escapes not consumed by unquoted `$VLLM_FLAGS` in container** | Fixed via `eval "VLLM_FLAGS_ARR=(...)"` (AGENTS.md #66). |
| **EngineDeadError/OOM minutes into Phase 1 (large-vocab FP8 model)** | **prompt_logprobs fp32 transient = chunk×vocab exceeds non-pool headroom** | Recipe: `--gpu-memory-utilization 0.94 --max-num-batched-tokens 512` (AGENTS.md #67). Put it in comparison.json settings (`lm_eval_gpu_util` + `lm_eval_extra_vllm_flags`) — env-scoped launches are dropped by every watchdog relaunch (AGENTS.md #70). |
| **Crash-in-minutes on FIRST request wave after a watchdog restart (was fine for hours pre-crash)** | **Watchdog relaunched with bare env → serving flags silently reverted to defaults** | Read `non-default args:` in the fresh server log vs pre-crash (AGENTS.md #70). Fix = settings plumbing, not env. |
| **Flags after the comparison dir silently ignored** | **Runner arg loop breaks at first positional** | Put `--tasks/--limit/--models` BEFORE the dir (AGENTS.md #10 usage line). A diagnostic must print `DIAGNOSTIC MODE` + `[filter] skipping`. |
| **`pkill -f run_lm_eval` times out the tool session** | **Pattern matched the tool's own bash cmdline** | Use pgrep+targeted kill, or a script file whose name lacks the pattern (AGENTS.md #68). |
| **EngineDeadError hours into a healthy run, KV < 60% at death** | **Native EngineCore SEGFAULT (async corruption — expandable_segments/VMM or GDN/fp8-KV bug), NOT OOM** | No memory knob fixes it. Detect via `Segfault encountered` in `<model>_vllm_server.log`; rely on watchdog restart + response-cache resume (AGENTS.md #69). Escalate to user before changing serving flags (arm symmetry). |
| **"Cache not wired — `results/lm_eval_cache/` missing"** | **Wrong path — cache is NESTED at `results/lm_eval/lm_eval_cache/<model>/`** | Check the nested path + `docker inspect` the live container for the `/lm_eval_cache` mount (AGENTS.md #69d). Never edit the runner for this. |

## qwen38-27b specifics (active comparison)

- FP8 single-GPU recipe is now DECLARATIVE in comparison.json: `lm_eval_gpu_util: 0.94` + `lm_eval_extra_vllm_flags: "--max-num-batched-tokens 512"` (AGENTS.md #70; survives watchdog bare-env relaunches). Do NOT pass these as launch env — settings always apply, explicit env would clobber via the `:-` guard only if set differently.
- Phase 1 ≈ 19h LL (209k requests @ ~3 it/s, GPU compute-bound) + generative; Phase 2 GSM8K at 15360 toks can add hours on loop-heavy arms (xhigh overthinking risk, NOTES.md); Phase 3 fast. **Budget ~20-30h/model — NOT the generic ~5-8h.**
- Diagnostic gate PASSED 2026-08-30: hellaswag 0.68 acc_norm n=200, no published ref exists (lesson #26 framing; journal #251). Gemma-4-12b-it same-pipeline anchor: 0.586.
- Base model name is `Qwen3.8-27B` (dot in name — the lesson #64 trigger).
- GSM8K 3-number report + lesson #51 sample-bridge (`__model/` ENOENT: use `cd` + relative names) still REQUIRED per model after Phase 2.
- **WATCHDOG IS RUNNING** (installed 2026-08-31 after base's 03:57 UTC segfault sat unnoticed; relaunched 07:4x UTC after the #70 env-loss incident — see journal #258): `runners/lm_eval_watchdog.sh`, pid file `results/lm_eval_watchdog.pid`, log `results/lm_eval_watchdog.log` (10-min cycles; server-log stall >20 min = engine dead; client-log stall >45 min = hung client; final pass re-runs models lacking canonical `lm_eval_<model>.json` — base's retry + response-cache resume is handled THERE, not by the current queue invocation). Budgets: 3 restarts/model → `--models` exclusion; FAIL marker at `results/lm_eval_watchdog.FAIL`. If you relaunch the queue manually, the watchdog will handle it fine — but NEVER run it with `--force` (blows the response cache, AGENTS.md #418 warning in runner).
- base segfault root cause: native EngineCore crash (`Segfault encountered` → `mul_kernel_cuda`) at KV 28.6% — NOT OOM (journals #254, AGENTS.md #69a). No memory-tuning change made (flags are arm-symmetric methodology; util 0.93 would deadlock: KV < max_model_len). If segfaults recur ≥2 more times, escalate to user re: removing `expandable_segments:True` panel-wide.

## What You Do NOT Do
- Do NOT change quantization from BF16 unless user explicitly asks (AGENTS.md #2)
- Do NOT set `max_gen_toks` > `max_model_len - 1024`
- Do NOT run loglikelihood without `--apply_chat_template` on instruct models (AGENTS.md #9)
- Do NOT pass `enable_thinking=True` for loglikelihood tasks

(Standard execution environment + self-learning protocol: see AGENTS.md.)
