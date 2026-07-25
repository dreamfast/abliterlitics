# AGENTS.md — Project Context & Lessons Learned

> This file is for AI agents (and humans) working on the Abliterlitics project.
> It captures hard-won knowledge about how to run evaluations correctly.

## Agent Self-Learning Protocol (applies to ALL agents)

When you encounter and solve a non-trivial problem during your workflow — a configuration bug, an unexpected error, a documentation gap, a tool quirk, a gotcha that wasn't already documented — you MUST persist the learning before completing your task. **This is not optional.**

**What to do:**

1. **Add a lesson to AGENTS.md** under "Critical Lessons Learned" (or update an existing lesson if it overlaps). Format:
   - **Symptom:** What you observed (error message, wrong behavior, surprising number)
   - **Root cause:** Why it happened (cite source: docs URL, GitHub issue, file:line, or commit)
   - **Fix:** The specific change that resolved it (command/flag/config)
   - **Detection:** How to prevent or detect it next time (one-line check)

2. **Update your own agent file** (`.opencode/agents/<this-agent>.md`) in the relevant section (Critical Technical Details, Troubleshooting, What You Do NOT Do, etc.). Cross-reference AGENTS.md by lesson number.

3. **Update the runner/script** if the fix belongs in code (e.g., adding a missing flag, validation check, or warning). Code-level fixes are stronger than documentation alone.

**Why:** Future instances of you — and other agents working in adjacent areas — will repeat the same mistake otherwise. The cost of writing 5-15 lines of documentation is much less than the cost of re-debugging the same issue in a future session.

**Do NOT update for:**
- Stylistic preferences or subjective taste
- One-off environment quirks that won't recur in a fresh setup
- Problems caused solely by user-provided bad input
- Issues already covered by an existing lesson (just cite the existing lesson number)

**Worked example (DO NOT include this literal text — for shape reference only):**

> ### N. `[flag]` is required for `[scenario]`
>
> **The mistake:** Tried to do X without flag Y. Got result Z (wrong by ~N points).
>
> **The fix:** Add `--flag-y` to the command. Now matches expected behavior.
>
> **Detection:** Always check `[file]:[line]` for `[condition]` before running.

---

## Project Journal (inter-agent communication)

The `journal` table in `abliterlitics.db` is the inter-agent communication channel. It replaces `NOTES.md` as the source of truth for what's been discovered, decided, and done. All agents read from and write to it.

### Schema

```
journal(id, timestamp, comparison, agent, category, severity, subject, title, body)
```

- **comparison**: NULL for project-wide entries, otherwise the comparison name
- **agent**: who wrote it (`init` / `lm-eval` / `kl` / `harmbench` / `weights` / `overseer` / `report` / `user` / `system`)
- **category**: `discovery` / `decision` / `warning` / `action` / `lesson` / `result` / `error` / `note`
- **severity**: `debug` / `info` / `warning` / `error` / `critical`
- **subject**: short tag for filtering (`chat_template`, `thinking_mode`, `kl`, `benchmark`, etc.)
- **title**: one-line summary
- **body**: optional markdown body

### CLI

```bash
# Write an entry
./abliterlitics.sh db journal add \
  --agent init \
  --category discovery \
  --subject chat_template \
  --title "Gemma 4 chat template defaults enable_thinking=True" \
  --body "Details..." \
  --comparison gemma4-e4b

# Read latest state per subject (the "current truth" view)
./abliterlitics.sh db journal latest --comparison gemma4-e4b

# Full history (newest first)
./abliterlitics.sh db journal list --comparison gemma4-e4b

# Filter by agent / category / subject / severity
./abliterlitics.sh db journal list --agent init --category discovery

# Full-text search
./abliterlitics.sh db journal search "chat_template"
```

### When to write

- **Init agent**: every architecture detection, thinking-mode detection, model research finding, diagnostic result, and decision goes into the journal. This becomes the baseline knowledge for all downstream agents.
- **Any agent**: when you discover something, decide something, or do something non-obvious. When you encounter and solve a problem (also add a lesson to AGENTS.md).
- **Any agent**: at the start of your task, READ the journal to see what prior agents have learned.

### When NOT to write

- Routine progress updates (use `model_stages` table instead)
- Per-model benchmark results (use `lm_eval_results` / `kl_results` / etc. tables instead)
- Anything that's already in a structured table

### Relationship to AGENTS.md lessons

- **AGENTS.md** = project-wide, reusable lessons (codified for all future agents)
- **Journal** = per-comparison (or per-session) findings, decisions, and context

A journal entry with `category=lesson` should typically also be added as a numbered lesson in AGENTS.md if it's reusable across comparisons. The journal captures the specific instance; AGENTS.md captures the general rule.

---

## Project Overview

Abliterlitics compares **base LLMs vs their abliterated/uncensored variants** across multiple axes:
- **Weight forensics**: Layer-by-layer weight diffs, SVD, cosine similarity
- **KL divergence**: Distribution shifts between base and modified models
- **Benchmark evaluation**: lm-evaluation-harness on 8 standard tasks
- **HarmBench**: Safety benchmark via vLLM + classifier
- **Reports**: Technical reports published to HuggingFace, condensed versions to Reddit

### Current Comparison: Qwen3.6-27B Family

6 models, all `Qwen3_5ForConditionalGeneration` architecture (~51-52GB bf16):

| Short Name | Model Path | Type |
|---|---|---|
| `base` | `models/Qwen3.6-27B/` | Base (vanilla) |
| `aeon` | `models/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16/` | Uncensored |
| `huihui` | `models/Huihui-Qwen3.6-27B-abliterated/` | Abliterated (Chinese community) |
| `heretic` | `models/Qwen3.6-27B-uncensored-heretic-v2/` | Uncensored v2 |
| `hauhau` | `models/Qwen3.6-27B-HauhauCS-Q8KP-recovered/` | Recovered variant |
| `abliterix` | `models/Qwen3.6-27B-abliterated-v2/` | Abliterated v2 |

---

## LM-Eval: The Proven Working Approach

### Architecture: vLLM OpenAI Server + `local-completions` Backend

**Why this approach?** The `vllm_causallms` direct backend OOMs with 27B BNB4 on a single 32GB GPU. Running vLLM as an OpenAI-compatible server and using lm-eval's `local-completions` backend avoids this because the server manages memory more efficiently.

**Both vLLM server AND lm-eval must run inside the SAME Docker container** — if lm-eval runs outside, the `local-completions` model init tries to download the tokenizer from HF Hub using the model name `/model` and hangs.

### Docker Images

- `abliterlitics-lmeval:1.0.0` — vLLM 0.19.0, lm-eval, proven working
- `abliterlitics-forensics:1.0.0` — Weight analysis, KL divergence

### Working Docker Command Template

```bash
docker run --rm --runtime=nvidia --shm-size=16g --ipc=host \
  --name <MODEL>-lmeval \
  -e "NVIDIA_VISIBLE_DEVICES=0" -e "CUDA_VISIBLE_DEVICES=0" \
  -v "<model_path>:/model:ro" \
  -v "models/Qwen3.6-27B:/tokenizer:ro" \
  -v "results/lm_eval:/results" \
  -v "results/lm_eval_logs:/logs" \
  -v ".cache/hf:/root/.cache/huggingface" \
  -v ".cache/vllm:/root/.cache/vllm" \
  -e HF_HOME=/root/.cache/huggingface \
  abliterlitics-lmeval:1.0.0 \
  bash -c '
    # Start vLLM server
    python3 -m vllm.entrypoints.openai.api_server \
      --model /model --dtype bfloat16 --quantization bitsandbytes \
      --load-format bitsandbytes --trust-remote-code --max-model-len 8192 \
      --gpu-memory-utilization 0.90 --enforce-eager \
      --reasoning-parser qwen3 \
      --reasoning-config '\''{"reasoning_start_str": "<think}", "reasoning_end_str": "</think"}'\'' \
      --port 8080 --host 127.0.0.1 > /logs/<model>_vllm_server.log 2>&1 &
    SERVER_PID=$!

    # Wait for ready
    for i in $(seq 1 120); do
      if curl -s http://127.0.0.1:8080/v1/models | grep -q model; then break; fi
      sleep 5
    done

    # Phase 1: Loglikelihood + truthfulqa (fast, 2048 gen tokens)
    python3 -m lm_eval --model local-completions \
      --model_args "base_url=http://127.0.0.1:8080/v1/completions,model=/model,tokenizer=/tokenizer,tokenizer_backend=huggingface,num_concurrent=4,max_length=8192" \
      --tasks "mmlu,hellaswag,arc_challenge,winogrande,truthfulqa,piqa,lambada_openai" \
      --batch_size 4 --gen_kwargs max_gen_toks=2048 \
      --output_path /results/ --log_samples 2>&1 | tee /logs/<model>_phase1.log

    # Phase 2: GSM8K only (needs high gen tokens for reasoning models)
    python3 -m lm_eval --model local-completions \
      --model_args "base_url=http://127.0.0.1:8080/v1/completions,model=/model,tokenizer=/tokenizer,tokenizer_backend=huggingface,num_concurrent=4,max_length=8192" \
      --tasks "gsm8k" \
      --batch_size 4 --gen_kwargs max_gen_toks=7168 \
      --output_path /results/ --log_samples 2>&1 | tee /logs/<model>_gsm8k.log

    kill $SERVER_PID 2>/dev/null
  '
```

---

## Critical Lessons Learned

### 1. `max_gen_toks` includes thinking tokens — plan accordingly

**The mistake:** Set `max_gen_toks=2048` thinking that was the "response budget". But for reasoning models (Qwen3.5 with `<think/>` tags), `max_gen_toks` is the TOTAL generation budget including thinking tokens. The model would think for 1900 tokens, get cut off, and never produce an answer.

**The fix:** GSM8K needs `max_gen_toks=7168` to give the model room for extended reasoning (~5000 thinking tokens) + answer (~2048 response tokens). But this is ONLY needed for generative tasks (gsm8k). Loglikelihood tasks (mmlu, hellaswag, arc, winogrande, piqa, lambada) ignore `max_gen_toks` entirely.

**Never set `max_gen_toks` higher than `max_model_len - max_prompt_length`.** With `max_model_len=8192` and prompts up to ~1024 tokens, `max_gen_toks=7168` is the safe maximum. Setting 8092 caused context overflow crashes.

**Two-phase approach:** Run loglikelihood + truthfulqa with `max_gen_toks=2048` (keeps truthfulqa fast), then run gsm8k separately with `max_gen_toks=7168`.

### 2. BNB4 quantization damages math reasoning significantly

GSM8K scores under BNB4 quantization are MUCH lower than published bf16 scores. This is expected and consistent across all models — the comparison is still valid since all models use the same quantization. Do NOT compare these scores with published leaderboard scores run in fp16/bf16.

### 3. Qwen3.5 architecture cannot do BNB4 + tensor parallelism

`Qwen3_5ForConditionalGeneration` raises `NotImplementedError` for BNB4 with TP>1. This works for GLM architecture but not Qwen3.5. Must use single GPU.

### 4. vLLM `reasoning-parser` settings

For Qwen3.5 models that use extended `<think/>` reasoning:
```
--reasoning-parser qwen3
--reasoning-config '{"reasoning_start_str": "<think}", "reasoning_end_str": "</think"}'
```
This is needed for the vLLM server to properly handle the thinking tags. Without it, the server may not parse responses correctly.

### 5. Tokenizer mount is required separately

The tokenizer must be mounted from the BASE model (`models/Qwen3.6-27B:/tokenizer:ro`) even when evaluating variants. The variant models' tokenizers are identical but the mount ensures lm-eval can find it without downloading.

### 6. `local-completions` tokenizer download hang

If lm-eval runs in a separate container from the vLLM server, the `local-completions` model initialization tries to download the tokenizer from HF Hub using the model name `/model` and hangs indefinitely. Both MUST run in the same container.

### 7. Reasoning models and thinking loops

Some Qwen3.5 abliterated models exhibit "thinking loops" — they generate repetitive reasoning text that consumes generation tokens without converging. This manifests as very short final responses after the thinking is stripped. It's a model behavior issue, not an eval config issue.

### 8. Old results can be from a different architecture

Always verify which model produced existing result files. The `results/lm_eval/` directory contained results from a GLM-4.7-Flash run (April), not Qwen3.6-27B. Check the JSON for `model_class` or `config` fields to confirm.

### 9. `--apply_chat_template` is MANDATORY for instruction-tuned models on loglikelihood tasks

**The mistake:** Ran lm-eval on Gemma 4 E4B (an instruction-tuned chat model) without the `--apply_chat_template` flag. The model received raw `"Question: ... Answer:"` text instead of its trained chat format. Every multiple-choice loglikelihood score was depressed to near the chance floor:

| Task | Our score (no chat_template) | Community reference | Gap |
|---|---|---|---|
| MMLU | 40.8% | ~74% | **−33 pp** |
| HellaSwag | 34.9% | ~79% | **−44 pp** |
| ARC-Challenge | 25.1% (chance floor) | ~84% | **−59 pp** |
| WinoGrande | 48.7% (below chance) | — | — |
| TruthfulQA-MC1 | 27.8% | — | — |
| GSM8K (generate task) | 86.6% | ~85% | ✅ matches |

**Why generation tasks escaped this:** GSM8K and TruthfulQA-Gen only require the model to *generate* text, where it can still produce reasonable output from out-of-distribution prompts. Loglikelihood scoring compares per-token probabilities — much more sensitive to format mismatch.

**The fix:** Add `--apply_chat_template` to the lm-eval command for any model with a chat template (instruction-tuned, chat, instruct variants). This is documented in the [lm-eval chat templates guide](https://lm-evaluation-harness.readthedocs.io/running_evals/chat_templates/) and is now auto-applied by `runners/run_lm_eval.sh` when a chat template is detected.

**Diagnostic confirmation (Gemma 4 E4B HellaSwag, 200 samples, base model):**
- Without `--apply_chat_template`: 0.39 acc_norm (matches our broken 34.88%)
- With `--apply_chat_template`: 0.555 acc_norm (**+16.5 pp** at 0-shot)
- With `--apply_chat_template` + 10-shot: 0.695 acc_norm (**+30.5 pp**)

**Detection rule:** A model is instruction-tuned (needs `--apply_chat_template`) if ANY of:
- `<model_dir>/chat_template.jinja` exists
- `<model_dir>/tokenizer_config.json` contains `"chat_template"`
- `<tokenizer_dir>/chat_template.jinja` exists (tokenizer mounted separately)
- `<tokenizer_dir>/tokenizer_config.json` contains `"chat_template"`

**Caveat for thinking/reasoning models:** Per [lm-eval interface docs](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md), `enable_thinking=True` is **only for generative tasks** (GSM8K). Do NOT pass `enable_thinking=True` for loglikelihood tasks. The model's chat template default thinking behavior is fine for loglikelihood scoring.

**Never skip this check.** The relative comparison between variants remains valid even with the bug (all models evaluated identically), but absolute scores are nonsense and cannot be cited. Always check `runners/run_lm_eval.sh` dry-run output for `phase1 flag='--apply_chat_template'` before trusting results.

### 10. Always research + diagnose a new model before the full pipeline

**The mistake:** Set up a new comparison, kicked off the full pipeline, came back 6 days later to discover every loglikelihood score was depressed by 30-60 pp because of an undetected configuration issue (chat template, BOS handling, lm-eval version, prompt format, etc.).

**The pattern that catches it early:** Before any full GPU work, run a tiny diagnostic — `lm_eval --limit 200` on a single task against the base model — and compare to published numbers.

**The init agent (`abliterlitics-init.md`) is responsible for:**
1. Detecting the model's architecture from `config.json` + safetensors keys
2. Detecting thinking mode (chat template scanning, special tokens)
3. Fetching the model card + technical report from HuggingFace
4. Finding community-published benchmark numbers (MMLU-Pro, GPQA, GSM8K, IFEval)
5. Running a `--limit 200` diagnostic on the base model
6. Comparing to published numbers — within ~5 pp = PASS, way off = diagnose
7. Documenting findings in `comparisons/<name>/NOTES.md`

**Detection rule:** The init agent must NOT mark a comparison as ready until the diagnostic passes. The overseer's Phase 0 pre-flight checks include "No NOTES.md" as a warning gate — if init didn't complete, don't proceed.

**Worked example (the bug that motivated this lesson):** Gemma 4 E4B diagnostic showed HellaSwag acc_norm=0.39 vs community 0.79 — a 40 pp gap that traced to missing `--apply_chat_template`. Cost: 30 minutes to diagnose. Had we not caught it, the full rerun would have wasted ~144 GPU-hours producing nonsense numbers.

---

### 11. Per-comparison Docker image override + Gemma 4 requires nightly image

**The mistake:** Assumed the runner would use the right Docker image automatically. In fact `runners/common.sh` hardcoded `LMEVAL_IMAGE="abliterlitics-lmeval:1.0.0"` (vLLM 0.19.0), which lacks the `gemma4_reasoning_parser`. This causes vLLM to crash on `--reasoning-parser gemma4` for any `Gemma4ForConditionalGeneration` model.

**Symptoms:** vLLM server fails to start during `model_loader.load_model()`. The import chain is `gemma4.py → compilation.backends → passes.pass_manager → fusion.allreduce_rms_fusion → flashinfer.comm`, so the failure surfaces as either "unknown reasoning parser" or a flashinfer import error (see lesson #12).

**The fix:** Two-part plumbing (both shipped 2026-07-19):

1. `runners/common.sh` — made all image vars env-overridable:
   ```bash
   LMEVAL_IMAGE="${LMEVAL_IMAGE:-abliterlitics-lmeval:1.0.0}"
   FORENSICS_IMAGE="${FORENSICS_IMAGE:-abliterlitics-forensics:1.0.0}"
   # (also LLAMACPP_IMAGE, IK_LLAMACPP_IMAGE)
   ```
2. `runners/_load_comparison.py` — reads `settings.lmeval_image` and `settings.forensics_image` from `comparison.json` and exports the override.
3. `runners/_load_comparison.py` — extended `_detect_reasoning_parser()` to also auto-detect `Gemma4*` architectures (returns parser `"gemma4"` with empty `reasoning_config_json`, since the parser handles `<|channel>thought...<channel|>` internally). Previously only `Qwen3*` was detected.

**Usage in comparison.json:**
```json
"settings": {
  "lmeval_image": "abliterlitics-lmeval-nightly:1.0.0",
  ...
}
```

**Detection rule:** When setting up a Gemma 4 comparison, ALWAYS set `lmeval_image` in comparison.json. The standard `abliterlitics-lmeval:1.0.0` lacks the gemma4 parser — it will fail. Verify via dry-run:
```bash
python3 runners/_load_comparison.py comparisons/<name>/comparison.json comparisons/<name> | grep -E "(LMEVAL_IMAGE|REASONING_PARSER)"
# Expect: export LMEVAL_IMAGE=abliterlitics-lmeval-nightly:1.0.0
# Expect: export REASONING_PARSER=gemma4
```

### 12. flashinfer / flashinfer-jit-cache version mismatch in nightly image

**The mistake:** Launched Gemma 4 eval with `abliterlitics-lmeval-nightly:1.0.0`. vLLM crashed during model load with:
```
RuntimeError: flashinfer-jit-cache version (0.6.6+cu129) does not match flashinfer version (0.6.8.post1). 
Please install the same version of both packages. Set FLASHINFER_DISABLE_VERSION_CHECK=1 to bypass this check.
```

**Root cause:** The Docker image ships `flashinfer` 0.6.8.post1 but `flashinfer-jit-cache` 0.6.6+cu129. The check fails on import even though vLLM isn't actually using flashinfer for attention (Gemma 4 forces `TRITON_ATTN` backend due to heterogeneous head dimensions). The flashinfer import is incidental — pulled in via the compilation pass manager (`vllm.compilation.passes.fusion.allreduce_rms_fusion`).

**The fix:** Set `FLASHINFER_DISABLE_VERSION_CHECK=1` in the runner's `docker run` env block (`runners/run_lm_eval.sh`). The error message itself recommends this. Harmless for non-flashinfer code paths.

**Detection rule:** If you see `flashinfer-jit-cache version ... does not match flashinfer version` in `<model>_vllm_server.log`, this is the cause. The env var is now set unconditionally in the runner so this shouldn't recur — but if it does, the Docker image itself needs rebuilding with aligned package versions.

### 13. Runner skips via comparison.json `skip_lm_eval`, NOT DB state

**The mistake:** Marked variants as `lm_eval: skipped` in the `model_stages` DB table and assumed the runner would honor that. It does not — the runner only checks `VARIANT_SKIP_LM_<slug>` env vars, which `_load_comparison.py` sets from each variant's `"skip_lm_eval": true` field in `comparison.json`. DB state and runner state are decoupled by design.

**Symptom:** Dry-run shows variants you expected to skip being queued for evaluation. Example output without the fix:
```
[2026-07-19 10:08:19] Evaluating huihui (14GB, native bf16, ...)  ← unexpected!
[2026-07-19 10:08:19] Evaluating abliterix (14GB, ...)             ← unexpected!
```

**The fix:** Add `"skip_lm_eval": true` to each variant in `comparison.json` (the canonical source of truth for runner behavior). The DB `model_stages` table tracks state for reporting/observability only.

**Detection rule:** Before any scoped run, ALWAYS dry-run first and verify the expected skip behavior:
```bash
./abliterlitics.sh --dry-run lm-eval ./comparisons/<name>/ | grep -cE "Evaluating|skip"
# Should show 1 Evaluating (base) + N skip lines if you intended base-only
```

The same pattern applies to `skip_harmbench`, `skip_kl`, `skip_weight` for their respective runners.

### 14. lm-eval request caching is available and OFF by default — enable it

**The mistake:** Ran a ~7h lm-eval Phase 1 job (158,686 requests across MMLU-Pro + GPQA + BBH + MUSR + IFEval + TruthfulQA bundled into ONE `python3 -m lm_eval` call). At ~12% through (~2h in), the user asked "if I stop this, will it pick up where it left off?" Answer: no. The runner had no request-level caching, and Phase 1 results are only written to disk when the entire call completes — so stopping meant losing all 2h of work and restarting from request 0.

**Root cause:** `runners/run_lm_eval.sh` did not pass `--cache_requests true` to lm-eval. lm-eval has had this flag since v0.4.x. The cache is content-addressed (pickle files keyed by hash of request + model args), so it's safe across runs and only skips requests that would produce identical outputs.

**The fix** (shipped 2026-07-19, `runners/run_lm_eval.sh`):

1. Per-model cache dir: `${RESULTS_DIR}/lm_eval_cache/${model_name}/` (created with `mkdir -p` before each run)
2. Mounted into container at `/lm_eval_cache`
3. `LM_HARNESS_CACHE_PATH=/lm_eval_cache` env var tells lm-eval where to write. **Critical:** the default cache path is inside the lm_eval Python package directory (`{MODULE_DIR}/.cache`) — gone when the container exits. The env var override is mandatory for persistence.
4. `--cache_requests true` added to BOTH Phase 1 and Phase 2 lm_eval invocations
5. `--force` (i.e., `SKIP_EXISTING=0`) clears the per-model cache dir so `--force` re-runs are truly fresh

**How lm-eval's cache works** (verified by reading `lm_eval/caching/cache.py` + observed behavior):
- `--cache_requests {true,refresh,delete}` — `true` reads+writes, `refresh` forces rewrite, `delete` clears
- Cache location: `LM_HARNESS_CACHE_PATH` env var, falls back to `{lm_eval install dir}/.cache` (useless inside Docker)
- Files: `{LM_HARNESS_CACHE_PATH}/requests-{task}-{config}-...{hash}.pickle` — pickle files keyed by `(model_args, request_content)` hash
- **CRITICAL CORRECTION (2026-07-24):** `--cache_requests` caches **preprocessed prompts only** (the built request objects), NOT model API responses. On restart, the preprocessing is skipped (~2 min saved), but ALL API calls are re-made from scratch. This was confirmed empirically: nullpo Phase 1 was stopped after completion, restarted, and ran at full API speed (~3h) despite 40 cache files being present and accessible.
- **`--cache_requests` does NOT enable recovery from container restart.** A stopped/crashed Phase 1 loses all API work and must rerun from request 0.
- Response-level caching requires `--use_cache <path>` — a separate SQLite loglikelihood cache. The runner does NOT currently set this. See lesson #18.

**~~Effect: A container crash / OOM / `docker stop` / Ctrl-C at any point during Phase 1 or Phase 2 now costs ~5 min~~** **WRONG — see lesson #18.** Stopping mid-Phase-1 costs the full Phase 1 rerun (~3h). Do NOT tell users it's safe to stop and resume.

**Detection rule:** Verify caching is wired before any long lm-eval run:
```bash
grep -c "cache_requests" runners/run_lm_eval.sh    # expect ≥2 (Phase 1 + Phase 2)
ls results/lm_eval/lm_eval_cache/ 2>/dev/null      # per-model dirs appear once a run starts
```
If the runner is ever reverted or forked, this is the first thing to re-add.

**Tradeoff:** Cache files are ~1-3 MB per 1k requests → ~200-500 MB per full Phase 1 run per model. For a 24-model comparison that's ~10 GB. Acceptable; stored on the same volume as `results/lm_eval/`.

### 15. lm-eval 0.4.11 `--cache_requests` CLI flag is broken — patch the container at startup

**The mistake:** Lesson #14 added `--cache_requests true` to both Phase 1 and Phase 2 invocations. The first run on `abliterlitics-lmeval-nightly:1.0.0` (lm-eval 0.4.11) failed instantly with:
```
usage: lm-eval run --model <model> --tasks <task> <task> --model_args <arg=value> <arg=value> [options]
lm-eval run: error: argument --cache_requests: invalid choice: "{'cache_requests': True, 'rewrite_requests_cache': False, 'delete_requests_cache': False}" (choose from true, refresh, delete)
```

**Root cause:** lm-eval 0.4.11 `_cli/run.py` defines the flag with **both** `type=request_caching_arg_to_dict` AND `choices=["true", "refresh", "delete"]`:
```python
cache_group.add_argument(
    "--cache_requests",
    type=request_caching_arg_to_dict,        # converts "true" → dict
    default=None,
    choices=["true", "refresh", "delete"],   # checks the POST-CONVERSION dict against STRING choices
    help="Cache preprocessed prompts (true|refresh|delete)",
)
```
In Python's argparse, `choices=` is applied **after** `type=` conversion. So `type` converts the string `"true"` to the dict `{"cache_requests": True, "rewrite_requests_cache": False, "delete_requests_cache": False}`, and then `choices=` checks whether that dict is in `["true", "refresh", "delete"]` — it never is. **Any** value supplied to `--cache_requests` is rejected. The flag is unusable from the CLI in this version. (Reproduced on both `abliterlitics-lmeval:1.0.0` and `abliterlitics-lmeval-nightly:1.0.0`, both shipping 0.4.11.)

**The fix** (shipped 2026-07-19, `runners/run_lm_eval.sh`): add a one-line `sed` patch in the container's startup `bash -c` block, right before starting the vLLM server:
```bash
LMEVAL_RUN_PY=/usr/local/lib/python3.12/dist-packages/lm_eval/_cli/run.py
if grep -q 'choices=\["true", "refresh", "delete"\],' "$LMEVAL_RUN_PY" 2>/dev/null; then
    echo "[container] Patching lm-eval 0.4.11 --cache_requests choices= bug"
    sed -i '/choices=\["true", "refresh", "delete"\],/d' "$LMEVAL_RUN_PY"
fi
```
Removing the `choices=[...]` line lets the `type=` conversion (which still returns a dict) pass through to the evaluator unchanged. Idempotent — once patched, the grep finds nothing and skips. The patch lives only inside the ephemeral container; it does not modify the Docker image.

**Detection rule:** When the runner log shows:
```
lm-eval run: error: argument --cache_requests: invalid choice: "{'cache_requests': True, ...}" (choose from true, refresh, delete)
```
the patch is missing (e.g., on an older runner fork). Look for `[container] Patching lm-eval 0.4.11 --cache_requests choices= bug` in the master log — that line confirms the patch fired.

When upgrading to a future lm-eval version that fixes this upstream, the grep simply finds nothing and the patch becomes a no-op — safe to leave in place.

### 16. Thinking-model chat templates contaminate loglikelihood scoring — disable for Phase 1

**The mistake:** Ran the v2 leaderboard suite on Gemma 4 E4B with `--apply_chat_template` correctly applied (lesson #9). The init agent's HellaSwag diagnostic showed acc_norm=0.555 (vs 0.39 without chat_template) — interpreted as "chat template works, +16pp improvement". The full Phase 1 run completed and produced:

| Task | Score | Google-published | Gap |
|---|---:|---:|---:|
| MMLU-Pro (5-shot) | 0.4392 | 0.694 | **−25 pp** |
| GPQA Diamond (0-shot) | 0.3434 | 0.586 | **−24 pp** (barely above 25% chance floor) |
| BBH (3-shot, agg) | 0.6327 | — | (tracking_shuffled subtasks at chance: 11-34%) |
| TruthfulQA-MC1 | 0.3843 | — | (better than broken v1's 0.278, still low) |
| IFEval prompt_strict | 0.4399 | — | (thinking eats generation budget) |

**Root cause:** The model's `chat_template.jinja` (line 1) defaults `enable_thinking=true`, which (lines 180-184) injects `<|think|>\n` at the start of the system turn — an **unclosed** thinking tag. The resulting prompt format for MMLU-Pro loglikelihood scoring was:
```
<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\n[5-shot examples]\n<|turn>user\n[Q]Answer:<turn|>\n<|turn>model\n[SCORE POSITION]
```
At the scoring position (`<|turn>model\n`), the model believes it is **inside a thinking block** — the `<|think|>` was never closed. The model's logit distribution here is over **thinking prose tokens**, not over A/B/C/D answer letters. The per-choice loglikelihoods are near-random, producing chance-level accuracy on sensitive tasks (MMLU-Pro, GPQA, BBH-tracking_shuffled).

The init agent's HellaSwag diagnostic at 200 samples showed the same pattern (0.555 vs community 0.79 = −24 pp gap) but was misread as "chat template works" because it was compared to the even-worse no-chat-template baseline (0.39) instead of the community target.

**The fix:** Create a `no_thinking_chat_template.jinja` in the comparison directory — a copy of the model's `chat_template.jinja` with **line 1 changed** from `enable_thinking = true` to `enable_thinking = false`. The runner auto-detects this file and mounts it over `/tokenizer/chat_template.jinja` inside the container, so Phase 1 loglikelihood uses the clean non-thinking format:
```
<bos><|turn>user\n[Q]Answer:<turn|>\n<|turn>model\n[SCORE POSITION]
```
No system turn, no `<|think|>` injection. The model scores answer choices directly. Phase 2 (GSM8K via `local-chat-completions`) is **unaffected** — it goes through the vLLM chat endpoint which uses the `thinking_chat_template.jinja` override (thinking enabled, as intended for generative tasks).

**Confirmed working** (verified inside the container with `tokenizer.apply_chat_template`):
- Default template: `'<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\n[Q]Answer:<turn|>\n<|turn>model\n'`
- Non-thinking template: `'<bos><|turn>user\n[Q]Answer:<turn|>\n<|turn>model\n'`

**Detection rule:** For ANY thinking model (Gemma 4, Qwen3, DeepSeek-R1, etc.), the init agent's diagnostic must compare the chat-template loglikelihood score against the **published community/Google number**, not just against the no-template baseline. If the gap is >10 pp, suspect thinking contamination and check the chat template format by running:
```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("<model_dir>")
print(repr(tok.apply_chat_template(
    [{"role": "user", "content": "Q\nA. x\nB. y\nAnswer:"}],
    tokenize=False, add_generation_prompt=True
)))
```
If the output contains `<|think|>` (or equivalent) BEFORE the model turn marker, loglikelihood scoring is contaminated. Create `no_thinking_chat_template.jinja` and re-run.

**Cache invalidation:** The Phase 1 cache (`lm_eval_cache/<model>/`) stores preprocessed requests containing the thinking-format prompts. The cache key includes `-chat_template` but does NOT differentiate between thinking/non-thinking variants. **Must use `--force`** (which clears the per-model cache dir) when re-running after creating `no_thinking_chat_template.jinja`, otherwise the broken thinking-format requests are reused from cache.

**Applies to:** All thinking models with chat templates that inject thinking tags by default. Known affected: Gemma 4 (`<|think|>`), Qwen3 (`<think>`), DeepSeek-R1 (`<think>`). The `no_thinking_chat_template.jinja` approach is model-agnostic — just change the template's thinking-default variable.

### 17. `--apply_chat_template` kills raw-completion tasks (HumanEval, LAMBADA) — exclude or run raw

**Symptom:** Gemma 4 E4B base finished Phase 1 of the extended 13-task suite. Two tasks produced garbage:
- `humaneval` pass@1 = **0.0000** (0/161 passed; the `create_test` filter *did* execute — the code just never compiled)
- `lambada_openai` perplexity = **4.59e11** (acc = 0.0)

**Root cause:** Both tasks are **raw-completion** tasks. HumanEval's prompt is a partial Python function ending mid-docstring (`"""`); the model must continue the function *body*. LAMBADA's prompt is a narrative passage missing its last word. With `--apply_chat_template` (even with the `no_thinking_chat_template.jinja` override from lesson #16), the prompt is wrapped as `<bos><|turn>user\n<prompt><turn|>\n<|turn>model\n`. An instruction-tuned/chat model then **responds conversationally** rather than raw-continuing. Concrete evidence from `samples_humaneval_*.jsonl[0]`:
- Prompt ended at: `...    """\n` (mid-function, awaiting body)
- Model generated: `` ```python\nfrom typing import List\n\n `` — a markdown code fence that restates the signature and **never writes the body**.
- `create_test` extracts `` ```python\nfrom typing import List\n `` → no function body → compile/exec fails → pass@1=0.

For LAMBDA, the chat-turn wrapper tokens (`<|turn>`, `<turn|>`) have near-zero probability under narrative-continuation scoring, so perplexity explodes to ~1e11.

**This is the generative counterpart of lesson #16.** Lesson #16's `no_thinking_chat_template.jinja` fix solved the *loglikelihood* MC tasks (mmlu_pro, gpqa, bbh, musr, hellaswag, arc, piqa, winogrande) but does NOT help HumanEval/LAMBADA, because those tasks need **no chat template at all** — not a non-thinking chat template.

**The fix (do NOT apply mid-run):** For future comparisons involving instruction/chat/thinking models, either:
1. Run HumanEval + LAMBADA in a **separate Phase without `--apply_chat_template`** (raw completion mode), with its own result file; or
2. Drop them from the suite for instruction-tuned models (they add ~1h of dead compute each otherwise).

HumanEval's value for abliteration *capability-damage* detection is real (coding ability is a known abliteration casualty), so option (1) — a raw-completion HumanEval phase — is worth wiring into the runner next time. The runner's task-list split (Phase 1 generative vs Phase 2 gsm8k) could gain a Phase 0 / Phase 3 for raw-completion tasks.

**Detection rule:** If `humaneval pass@1` ≈ 0 **AND** `lambada perplexity` > 1e6 for ANY model in a `--apply_chat_template` run, both tasks are dead for the **entire comparison** (all models share the same flag) — exclude them from delta analysis. Do NOT interpret pass@1=0 as "the model cannot code"; it is a format artifact. Verified on the `gemma4-e4b` comparison (base + all variants expected to show the same pattern).

**No effect on the 11 valid tasks:** mmlu_pro, gpqa_diamond, bbh (×23 subtasks), musr (×3), ifeval, truthfulqa (mc1/mc2/gen), hellaswag, arc_challenge, piqa, winogrande, gsm8k all produce sane scores and remain the delta-detection workhorses.

### 18. No Phase-level checkpointing — stopping mid-run reruns everything

**The mistake:** An agent advised the user it was safe to `docker stop` a container after Phase 1 finished (Phase 1 result JSON was written to `__model/results_*.json` at 09:12). The user stopped the container during Phase 2 (GSM8K). On restart, the runner re-ran Phase 1 from scratch (~3.5h wasted) instead of skipping to Phase 2. The Phase 1 results were sitting on disk the entire time — the runner just didn't look for them.

**Root cause (runner design):** `runners/run_lm_eval.sh` has two phases:
- Phase 1: loglikelihood + generative tasks (mmlu_pro, gpqa, bbh, musr, ifeval, truthfulqa, v1 supplement, humaneval) — writes to `__model/results_TIMESTAMP.json`
- Phase 2: gsm8k only — writes to another `__model/results_TIMESTAMP.json`

The canonical `lm_eval_<model>.json` is only produced by the collection/merge code (lines 550-604) AFTER the container exits. The runner's `SKIP_EXISTING` check only looks for this canonical file. If the container is stopped mid-Phase-2:
1. Phase 1's `results_TIMESTAMP.json` IS on disk in `__model/`
2. The collection code may or may not run (depends on how the container was killed)
3. Even if it runs and creates `lm_eval_<model>.json` with only Phase 1 data, the runner sees it lacks gsm8k and re-runs everything
4. The `skip_phase1` variable (line 211) exists but is only set in `GSM8K_RERUN` mode, not on normal restart

The runner has **no mechanism to detect that Phase 1 was already completed** outside of GSM8K_RERUN mode. It either has the full canonical file (skip the model) or doesn't (re-run everything).

**Contributing factor (cache misunderstanding):** Lesson #14's `--cache_requests` was believed to cache API responses for fast recovery. It does NOT — it caches preprocessing only. See the corrected lesson #14. Even with cache, Phase 1 re-executes all 214,904 API calls from scratch (~3h).

**Confirmed empirically (2026-07-24, nullpo model, gemma4-e4b comparison):**
- Phase 1 completed at ~09:09 UTC, wrote `results_2026-07-24T09-12-37.867131.json` (157KB) to `__model/`
- Container stopped manually at ~09:13 (during Phase 2, request 17/1319)
- New container started at 10:22 — Phase 1 restarted from 0
- Phase 1 result file was on disk the entire time but was ignored
- At 12:18, Phase 1 was at 32% (70,482/214,904) running at full API speed

**The fix:**
1. **Immediate (documentation):** Never advise stopping a container mid-run. Only stop BETWEEN models (after `lm_eval_<model>.json` is written, before the next model's container starts). The safe stop check:
   ```bash
   ls results/lm_eval/lm_eval_<model_name>.json
   # If exists → safe to stop (model fully done)
   # If missing → DO NOT STOP (Phase 1 or Phase 2 in progress)
   ```
2. **Short-term (runner patch):** After Phase 1 completes inside the container, write a checkpoint marker (e.g., `touch /results/.phase1_done_<model>`). On restart, the runner checks for this marker and sets `skip_phase1=1` if it exists, skipping Phase 1 and only running Phase 2.
3. **Long-term (response caching):** Add `--use_cache /lm_eval_cache/responses.sqlite` to both Phase 1 and Phase 2 lm_eval invocations. This SQLite cache stores model responses keyed by request hash, enabling true mid-Phase resume (not just Phase-level skip). TODO — not yet implemented.

**Detection rule:** If about to advise stopping a running lm-eval container:
```
STOP. Read lesson #18. The runner has NO Phase-level checkpointing.
Stopping mid-Phase-1 OR mid-Phase-2 costs a full Phase 1 rerun (~3h).
Only safe to stop AFTER lm_eval_<model>.json is written.
```

---

## Benchmark Tasks

The default suite is the **Open LLM Leaderboard v2** set + GSM8K + TruthfulQA, supplemented with **v1 forensic tasks** and **HumanEval** for abliteration damage detection. The v1 tasks were originally dropped for "saturation" (models score too high for OLL ranking) but **saturation is irrelevant for delta-detection** — a benchmark at 80% → 77% still shows a 3pp damage signal, which is exactly what we need.

### Core tasks (OLL v2 + generative)

| Task | Type | Metric | Shots | Notes |
|---|---|---|---:|---|
| `leaderboard_mmlu_pro` | Loglikelihood (MC) | acc | 5 | 12k questions, 14 disciplines, 10 choices. Replaces regular MMLU. |
| `leaderboard_gpqa_diamond` | Loglikelihood (MC) | acc | 0 | 198 PhD-level science questions. Gated dataset — needs HF_TOKEN. |
| `leaderboard_bbh` | Loglikelihood (MC) | acc | 3 | 23 Big-Bench Hard subtasks. Replaces HellaSwag/ARC/PIQA. |
| `leaderboard_musr` | Loglikelihood (MC) | acc | 0 | 3 subtasks: murder_mysteries, object_placements, team_allocation. |
| `leaderboard_ifeval` | Generate (verifiable) | prompt/inst acc | 0 | 541 prompts, 25 instruction types. Critical for abliteration damage. |
| `gsm8k` | Generate | exact_match | 0 | Math reasoning. Needs high max_gen_toks for thinking models. |
| `truthfulqa` | Mixed (gen + mc1 + mc2) | bleu/rouge/acc | 0 | 817 questions. Sensitive to capability damage. |

### Forensic supplement (v1 delta-detection + code)

Added 2026-07-20 (journal #20). These tasks fill capability gaps that the v2 suite cannot detect:

| Task | Type | Metric | Shots | What it detects |
|---|---|---|---:|---|
| `hellaswag` | Loglikelihood (MC) | acc_norm | 0 | Physical/social common sense (scene completion) |
| `winogrande` | Loglikelihood (MC) | acc | 0 | Pronoun resolution / Winograd schemas |
| `arc_challenge` | Loglikelihood (MC) | acc_norm | 0 | Grade-school science reasoning |
| `piqa` | Loglikelihood (MC) | acc_norm | 0 | Physical interaction QA (world model damage) |
| `lambada_openai` | Loglikelihood | perplexity + acc | 0 | Narrative language modeling fluency |
| `humaneval` | Generate (code exec) | pass@1 | 0 | Coding ability damage (requires `HF_ALLOW_CODE_EVAL=1`) |

**Why keep "saturated" tasks?** For abliteration forensics, we're not ranking models — we're measuring deltas. The relevant criterion is signal-to-noise on per-model score changes, not absolute discrimination between models. A task where base scores 80% and ablated scores 77% gives the same 3pp damage signal as a task where base scores 50% and ablated scores 47%.

### Validation targets for Gemma 4 E4B

Google's published numbers use **generative evaluation with thinking enabled**. Our loglikelihood MC (LL MC) methodology is fundamentally different — see journal #19 for the full methodology analysis. **LL MC scores are NOT comparable to Google's generative numbers** (~25pp gap by design for thinking models). Valid comparison targets:

| Task | Our methodology | Google's methodology | Match? |
|---|---|---|---|
| GSM8K | Generative + thinking | Same | ✅ Directly comparable (~85-88%) |
| IFEval | Generative | Same | ✅ Directly comparable (~80-90%) |
| MMLU-Pro | LL MC (no reasoning) | Generative + thinking | ❌ NOT comparable (gap ~25pp by design) |
| GPQA Diamond | LL MC | Generative + thinking | ❌ NOT comparable (gap ~25pp by design) |

Pipeline stack validated via generative tasks matching community (journal #19). Gemma 4 is NOT on OLL v2 as of 2026-07-20 — no public LL MC reference exists. Document methodology in report.

### Timing estimates (per model, single 5090, bf16)

| Phase | Time | Tasks |
|---|---:|---|
| Core LL MC (Phase 1) | ~3h | mmlu_pro, gpqa_diamond, bbh, musr |
| Forensic supplement (Phase 1) | ~30-45 min | hellaswag, winogrande, arc, piqa, lambada |
| HumanEval (Phase 1) | ~30-60 min | humaneval (code generation + execution) |
| Generative (Phase 1) | ~30 min | ifeval, truthfulqa |
| Generative (Phase 2) | ~55 min | gsm8k (thinking, high max_gen_toks) |
| **Total per model** | **~5h** | |
| **Total per model** | **~6-7h** | |

**Total generate requests:** gsm8k ~1,300 + ifeval ~540 + truthfulqa_gen ~330
**Total loglikelihood requests:** mmlu_pro ~60k (12k × 5-shot × partial) + bbh ~20k + musr ~250 + gpqa ~200

### Dataset pre-download (one-time, required)

Before the runner will work, all benchmark datasets must be pre-downloaded to `.cache/hf/datasets/`. The runner sets `HF_DATASETS_OFFLINE=1` and `HF_HUB_OFFLINE=1` and will fail fast if the cache is empty.

```bash
bash runners/predownload_datasets.sh   # one-time, run from host
```

GPQA requires HF authentication + terms accepted at https://huggingface.co/datasets/Idavidrein/gpqa.

---

## File Locations

```
results/lm_eval/              # Result JSON files
  lm_eval_<model>.json        # Canonical result files
  __model/                    # Raw output from lm-eval (samples, timestamps)

results/lm_eval_logs/         # All run logs
  <model>_vllm_server.log     # vLLM server log
  <model>_lm_eval_combined.log # lm-eval output
  <model>_combined_nohup.log  # nohup wrapper log

results/lm_eval_glm47/        # Archived GLM-4.7-Flash results (old)
comparison.json                # Model configs (paths, names)
```

---

## Automation Pipeline

The `./abliterlitics.sh` CLI orchestrates all runners. The recommended workflow:

```bash
# Full pipeline for any comparison
./abliterlitics.sh auto ./comparisons/<name>/

# Or individual phases
./abliterlitics.sh weights ./comparisons/<name>/
./abliterlitics.sh kl ./comparisons/<name>/
./abliterlitics.sh lm-eval ./comparisons/<name>/
./abliterlitics.sh harmbench ./comparisons/<name>/
./abliterlitics.sh graphs ./comparisons/<name>/

# Useful flags
./abliterlitics.sh --dry-run auto ./comparisons/<name>/   # Preview what runs
./abliterlitics.sh --force lm-eval ./comparisons/<name>/   # Rerun all models
./abliterlitics.sh --backend llamacpp harmbench ./comp/    # Use GGUF fallback
./abliterlitics.sh status ./comparisons/<name>/            # Check result counts
./abliterlitics.sh validate ./comparisons/<name>/          # Verify model paths
```

### Settings Plumbing

`runners/_load_comparison.py` reads `comparison.json` and exports all settings as shell env vars:
- `BASE_DIR`, `VARIANT_NAMES`, `VARIANT_PATH_*` — model paths
- `LM_EVAL_TASKS`, `LM_EVAL_MAX_MODEL_LEN`, `LM_EVAL_MAX_GEN_TOKS` — lm-eval settings
- `HARMBENCH_MAX_TOKENS` — HarmBench generation budget
- `REASONING_PARSER`, `REASONING_CONFIG_JSON` — auto-detected from model `config.json`
- `VARIANT_SKIP_HB_*`, `VARIANT_SKIP_LM_*` — per-variant skip flags

Reasoning auto-detection: reads model's `config.json` architectures field. Qwen3/Qwen3.5 → `qwen3` parser with `<think/>` tags.

### Key Defaults

| Setting | Default | Notes |
|---------|---------|-------|
| `harmbench_max_tokens` | 8096 | Reasoning models need thinking + response room |
| `lm_eval_max_model_len` | 16384 | Default for v2 suite (BBH 3-shot needs more context than v1 0-shot) |
| `lm_eval_max_gen_toks` | 4096 | Phase 1 (generative tasks: ifeval + truthfulqa_gen) |
| GSM8K Phase 2 gen_toks | 7168 | Capped to `max_model_len - 1024` automatically |
| `lm_eval_tasks` | (13 tasks: 7 OLL v2 + 5 v1 supplement + humaneval) | See "Benchmark Tasks" table above |

---

## GPU Hardware

- GPU 0: NVIDIA RTX 5090 (32GB) — used for all eval runs
- GPU 1: NVIDIA RTX 4090 (24GB) — leave alone unless told otherwise
- Docker images need `--runtime=nvidia --shm-size=16g --ipc=host`
- BNB4 model uses ~31.6GB of the 32GB card
