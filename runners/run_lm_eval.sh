#!/usr/bin/env bash
# lm-eval runner — vLLM server + local-completions backend
#
# Architecture: Both vLLM server and lm-eval run inside the SAME Docker
# container. This avoids the tokenizer download hang (AGENTS.md #6).
#
# Two-phase execution per model:
#   Phase 1: Loglikelihood + generative tasks with --apply_chat_template (max_gen_toks=2048)
#   Phase 2: GSM8K with max_gen_toks=7168 (reasoning models need thinking room)
#   Phase 3: Raw-completion tasks (HumanEval, LAMBADA) WITHOUT --apply_chat_template
#            (AGENTS.md #17 — chat template kills these: pass@1=0, perplexity ~1e11)
#
# For large MoE models (GLM, >50GB): supports TP=2 across dual GPUs.
# For Qwen3.5 family: single-GPU BNB4 (Qwen3.5 can't do BNB4+TP).
#
# =============================================================================
# CRITICAL: --apply_chat_template is MANDATORY for instruction-tuned models
# =============================================================================
# Per lm-eval docs (https://lm-evaluation-harness.readthedocs.io/running_evals/chat_templates/)
# and AGENTS.md lesson #9: instruction-tuned models (any model with a chat
# template in tokenizer_config.json or a chat_template.jinja file) MUST be
# evaluated with --apply_chat_template on loglikelihood tasks. Without it, the
# model receives raw "Question: ... Answer:" text instead of its trained chat
# format, causing systematically depressed loglikelihood scores (-30 to -60
# percentage points on MMLU/HellaSwag/ARC vs community references).
#
# This runner auto-detects chat templates and applies --apply_chat_template
# to Phase 1 (loglikelihood) for any model that has one. Phase 2 (GSM8K)
# already applies it unconditionally.
#
# Diagnostic confirming this (Gemma 4 E4B HellaSwag, 200 samples):
#   Without --apply_chat_template: 0.39 acc_norm  (matches our 34.88% on full set)
#   With --apply_chat_template:    0.555           (+16.5 pp)
#   With + 10-shot:                0.695           (+30.5 pp)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] <comparison_dir>

Run lm-evaluation-harness via vLLM server + local-completions backend.

Resource modes:
  --gpus single    Single GPU (auto-detected best GPU, default)
  --gpus dual      Both GPUs with tensor_parallel_size=2 (for large MoE models)
  --gpus 0         Use only GPU index 0
  --gpus 1         Use only GPU index 1
  --gpus 0,1       Use both GPUs (same as --gpus dual)

Quantization:
  All models use native bf16 for fair, consistent comparison conditions.
  BNB4 is disabled by default (it damages scores and makes cross-model
  comparison unfair). Only override with --quantization bitsandbytes if
  reviewing persistent OOM errors.
  For models whose BF16 weights exceed a single GPU (e.g. 27B-class, >50GB
  on disk), --quantization fp8 enables dynamic online FP8 (quantized at
  load, no calibration) which halves the on-GPU footprint. All arms MUST
  use the same quantization path for fair deltas (AGENTS.md lesson #53).
  Defaults can come from comparison.json settings via _load_comparison.py
  (lm_eval_quantization / lm_eval_kv_cache_dtype / lm_eval_language_model_only
  / lm_eval_gpus); CLI flags override.

Options:
   --gpus GPUS           GPU selection (default: auto, or settings.lm_eval_gpus)
   --tasks TASKS         Comma-separated task list (default: from comparison.json)
   --quantization QUANT  Override quantization (bitsandbytes/fp8/none). Default: bf16 native.
   --limit N             Cap samples per task (diagnostic/smoke-test mode, AGENTS.md #10).
                         Default: none (full task sets). With --limit, lm-eval still
                         writes a canonical lm_eval_<model>.json result file.
   --models LIST         Comma-separated model filter (default: all). Examples:
                           --models base              → only the base model
                           --models base,huihui       → base + one variant
                         Useful for base-model pre-flight diagnostics.
   --skip-existing       Skip models with existing results (default)
   --force               Overwrite existing results
   --dry-run             Show what would run without executing
   -h, --help            Show this message
EOF
    exit 0
}

# Defaults from comparison.json settings (exported by _load_comparison.py);
# CLI flags below override. LM_EVAL_GPUS pins GPU selection per comparison —
# REQUIRED for models >50GB on disk that serve quantized on one GPU (the
# >50GB auto-heuristic would otherwise pick dual TP2 and touch GPU 1).
GPUS_ARG="${LM_EVAL_GPUS:-}"
TASKS=""
QUANTIZATION="${LM_EVAL_QUANTIZATION:-}"
DRY_RUN=0
SKIP_EXISTING=1
LM_EVAL_LIMIT=""     # diagnostic cap on samples per task (AGENTS.md #10)
MODELS_FILTER=""     # comma-separated model name filter; "base" = BASE_NAME only

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)           GPUS_ARG="$2"; shift 2 ;;  # alias for --gpus
        --gpus)          GPUS_ARG="$2"; shift 2 ;;
        --backend)       shift 2 ;;  # consumed for CLI compat, lm-eval always uses vLLM
        --tasks)         TASKS="$2"; shift 2 ;;
        --quantization)  QUANTIZATION="$2"; shift 2 ;;
        --limit)         LM_EVAL_LIMIT="$2"; shift 2 ;;
        --models)        MODELS_FILTER="$2"; shift 2 ;;
        --skip-existing) SKIP_EXISTING=1; shift ;;
        --force)         SKIP_EXISTING=0; shift ;;
        --dry-run)       DRY_RUN=1; shift ;;
        -h|--help)       usage ;;
        *)               break ;;
    esac
done

COMP_DIR="${1:?Error: comparison_dir required. Use --help for usage.}"

load_comparison "$COMP_DIR"
# Re-apply settings-sourced defaults AFTER load_comparison: the top-level
# captures of GPUS_ARG/QUANTIZATION above run BEFORE load_comparison exports
# LM_EVAL_GPUS/LM_EVAL_QUANTIZATION, so they were permanently empty and every
# model fell back to the raw-size heuristic (51GB on disk -> dual TP2 on
# GPUs 0+1, max_len clobbered to 4096) instead of the comparison's FP8
# single-GPU recipe (lesson #64). CLI flags still win via the :- chain.
GPUS_ARG="${GPUS_ARG:-${LM_EVAL_GPUS:-}}"
QUANTIZATION="${QUANTIZATION:-${LM_EVAL_QUANTIZATION:-}}"
COMPARISON_DIR="$(cd "${COMPARISON_DIR}" && pwd)"  # resolve to absolute for Docker mounts
TASKS="${TASKS:-$LM_EVAL_TASKS}"
RESULTS_DIR="${COMPARISON_DIR}/results/lm_eval"
LOG_DIR="${COMPARISON_DIR}/results/lm_eval_logs"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

# Shared cache directories
HF_CACHE="${ABL_ROOT}/.cache/hf"
VLLM_CACHE="${ABL_ROOT}/.cache/vllm"
NLTK_CACHE="${ABL_ROOT}/.cache/nltk_data"
mkdir -p "$HF_CACHE" "$VLLM_CACHE" "$NLTK_CACHE"

# Detect HF token on host (needed for gated datasets like GPQA).
# Pre-download via runners/predownload_datasets.sh is required before this
# runner will work — see AGENTS.md lesson #9 and the pre-download script.
# hf CLI stores token at $HF_HOME/token (default ~/.cache/huggingface/token).
HF_TOKEN_MOUNT=()
HF_TOKEN_ENV=()
HOST_HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
if [[ -f "${HOST_HF_HOME}/token" ]]; then
    HF_TOKEN_MOUNT=(-v "${HOST_HF_HOME}/token:/root/.cache/huggingface/token:ro")
elif [[ -f "$HOME/.cache/huggingface/token" ]]; then
    HF_TOKEN_MOUNT=(-v "$HOME/.cache/huggingface/token:/root/.cache/huggingface/token:ro")
elif [[ -n "${HF_TOKEN:-}" ]]; then
    HF_TOKEN_ENV=(-e "HF_TOKEN=${HF_TOKEN}")
fi

# Verify datasets cache is populated — fail fast instead of hanging on
# network retries inside the container. See runners/predownload_datasets.sh
# for the one-time pre-download step.
DATASETS_CACHE="${HF_CACHE}/datasets"
if [[ ! -d "${DATASETS_CACHE}" ]] || [[ -z "$(ls -A "${DATASETS_CACHE}" 2>/dev/null)" ]]; then
    error "HF datasets cache is empty at ${DATASETS_CACHE}"
    error "Run pre-download first: bash runners/predownload_datasets.sh"
    error "See AGENTS.md for context."
    exit 1
fi
log "Datasets cache: $(ls "${DATASETS_CACHE}" | wc -l) datasets at ${DATASETS_CACHE}"

# Settings from _load_comparison.py (with defaults for standalone use)
MAX_MODEL_LEN="${LM_EVAL_MAX_MODEL_LEN:-8192}"
MAX_GEN_TOKS="${LM_EVAL_MAX_GEN_TOKS:-2048}"
TOKENIZER_DIR="${TOKENIZER_DIR:-${BASE_DIR}}"
TOKENIZER_DIR="$(realpath -m "${TOKENIZER_DIR}")"  # lesson #55a: absolute for docker -v
# Number of concurrent HTTP requests to vLLM (local-completions backend).
# Higher = better GPU batch utilization for loglikelihood tasks. Default 4 is
# conservative; 16-32 recommended for models <30GB on a 32GB+ GPU. Does NOT
# affect results (scores are identical regardless of concurrency).
NUM_CONCURRENT="${LM_EVAL_NUM_CONCURRENT:-4}"

# GSM8K generation budget.
# Gemma4 IS a thinking model — it generates long reasoning chains before
# answering. Empirical data shows responses up to 7000+ words. Budget must
# be high enough for thinking + answer. 7168 = max_model_len(8192) - 1024 prompt.
GSM8K_GEN_TOKS="${GSM8K_GEN_TOKS:-7168}"

# Resolve GPU configuration.
# Outputs: NVIDIA_VIS, CUDA_VIS, TP_SIZE, gpu_label
resolve_gpus() {
    local model_size_gb="$1"
    local mode="${GPUS_ARG}"

    # Default: dual for large models, single otherwise
    if [[ -z "$mode" ]]; then
        if [[ ${model_size_gb} -gt 50 ]]; then
            mode="dual"
        else
            mode="single"
        fi
    fi

    case "$mode" in
        single)
            local best
            best=$(detect_best_gpu)
            echo "${best} ${best} 1 single"
            ;;
        dual)
            # Both GPUs, 5090 as cuda:0, TP=2
            echo "0,1 1,0 2 dual"
            ;;
        0)
            echo "0 0 1 gpu0"
            ;;
        1)
            echo "1 1 1 gpu1"
            ;;
        0,1|1,0)
            echo "${mode} ${mode} 2 explicit"
            ;;
        *)
            error "Invalid --gpus value: ${mode}. Use single/dual/0/1/0,1/1,0"
            exit 1
            ;;
    esac
}

run_lm_eval() {
    local model_name="$1"
    local model_path="$2"
    # Resolve to absolute BEFORE docker -v (lesson #55a applies to this runner
    # too): _load_comparison.py string-concats comp_dir + "../../models/...",
    # so a relative invocation yields relative bind sources -> docker rejects
    # them ("includes invalid characters for a local volume name").
    model_path="$(realpath -m "${model_path}")"
    local out_file="${RESULTS_DIR}/lm_eval_${model_name}.json"

    # Check per-variant skip flag
    local var_slug
    # tr '.-' '__': also map dots — base names like "Qwen3.8-27B" produce
    # invalid bash identifiers for ${!indirect} expansion (lesson #64)
    var_slug="$(echo "$model_name" | tr '.-' '__')"
    
    # LoRA adapter detection (AGENTS.md lesson #35). When a LoRA adapter path is
    # set for this variant, vLLM uses --enable-lora mode instead of loading a
    # merged model. The variant's `path` points to the BASE model; the adapter
    # is mounted separately and applied dynamically at FP32 precision.
    local lora_adapter_env="VARIANT_LORA_ADAPTER_${var_slug}"
    local lora_adapter_path="${!lora_adapter_env:-}"
    local vllm_model_id="/model"
    local lora_mount=()
    local lora_vllm_flags=()
    if [[ -n "$lora_adapter_path" ]]; then
        log "  LoRA mode: adapter=${lora_adapter_path}"
        vllm_model_id="${model_name}"
        lora_mount=(-v "${lora_adapter_path}:/lora_adapter:ro")
        lora_vllm_flags=(--enable-lora --max-loras 1 --max-lora-rank 8)
        lora_vllm_flags+=(--lora-modules "${model_name}=/lora_adapter")
    fi
    
    local skip_flag="VARIANT_SKIP_LM_${var_slug}"
    if [[ "${!skip_flag:-0}" == "1" ]]; then
        log "[skip] ${model_name}: skip_lm_eval=true"
        if [[ $DRY_RUN -eq 0 ]]; then
            db_stage_skip "$COMPARISON_NAME" "$model_name" "lm_eval"
        fi
        return 0
    fi

    if result_exists "$out_file" && [[ -z "${GSM8K_RERUN:-}" ]]; then
        log "Skipping (exists): ${model_name}"
        if [[ $DRY_RUN -eq 0 ]]; then
            db_stage_skip "$COMPARISON_NAME" "$model_name" "lm_eval"
        fi
        return 0
    fi

    # GSM8K_RERUN mode: re-run only gsm8k Phase 2, keep existing Phase 1 results.
    local skip_phase1=0
    if [[ -n "${GSM8K_RERUN:-}" && -f "$out_file" ]]; then
        skip_phase1=1
        log "GSM8K re-run: ${model_name} (skipping Phase 1, keeping existing results)"
    elif [[ -n "${GSM8K_RERUN:-}" ]]; then
        log "GSM8K_RERUN: no existing results for ${model_name} — running full eval"
    fi

    if [[ $DRY_RUN -eq 0 ]]; then
        db_stage_start "$COMPARISON_NAME" "$model_name" "lm_eval"
    fi

    # Detect model size for resource decisions — count SAFETENSORS ONLY
    # (directory may contain GGUF, cache, or other non-model files)
    local model_size_kb
    model_size_kb=$(find "${model_path}" -maxdepth 1 -name '*.safetensors' -exec du -ck {} + 2>/dev/null | tail -1 | cut -f1)
    model_size_kb=${model_size_kb:-0}
    local model_size_gb=$(( model_size_kb / 1024 / 1024 ))

    # Quantization strategy: ALWAYS native bf16 for fair, consistent comparison.
    # BNB4 quantization damages scores and makes cross-model comparison unfair.
    # Only enable BNB4 with explicit --quantization bitsandbytes if OOM occurs.
    # fp8 = dynamic online FP8 for models whose BF16 weights exceed a single
    # GPU (AGENTS.md lesson #53): quantized at load from the BF16 checkpoint,
    # identical path for base + variants, only Linears quantize (mamba internals
    # stay BF16). MUST be paired with --gpus 0 pinning for >50GB-on-disk models
    # (see GPUS_ARG note above).
    local use_quant=""
    if [[ -n "${QUANTIZATION}" ]]; then
        case "${QUANTIZATION}" in
            bitsandbytes|bnb)     use_quant="bitsandbytes" ;;
            fp8)                  use_quant="fp8" ;;
            none|off|auto|"")     use_quant="" ;;
        esac
    fi

    # Dynamic FP8 halves the on-GPU footprint vs on-disk BF16 size. Resource
    # decisions (GPU count, util floor, max_model_len) must use the EFFECTIVE
    # size — the >50GB-on-disk heuristic would otherwise force dual TP2
    # (touching GPU 1) and clobber max_model_len/gpu_util for a model that
    # actually serves in ~28GiB on one card (AGENTS.md lesson #53).
    local effective_size_gb="${model_size_gb}"
    if [[ "${use_quant}" == "fp8" ]]; then
        effective_size_gb=$(( model_size_gb / 2 ))
    fi

    # Resolve GPU config for this model
    local gpu_config nvidia_vis cuda_vis tp_size gpu_label
    gpu_config=$(resolve_gpus "${effective_size_gb}")
    read -r nvidia_vis cuda_vis tp_size gpu_label <<< "${gpu_config}"

    # Adjust batch size and gpu_util based on model size.
    # LM_EVAL_GPU_UTIL (from comparison.json) overrides the default 0.9 for
    # models that need headroom (e.g. large-vocab prompt_logprobs OOM).
    local gpu_util="${LM_EVAL_GPU_UTIL:-0.9}"
    local batch_size="auto"
    local max_model_len="${MAX_MODEL_LEN}"

    if [[ ${effective_size_gb} -gt 50 ]]; then
        # Large MoE models (e.g. GLM-4.7-Flash 59GB): 0.75 safety floor always wins.
        gpu_util="0.75"
        batch_size="4"
        max_model_len="4096"
    elif [[ ${effective_size_gb} -gt 25 ]]; then
        batch_size="16"
    fi

    # Cap GSM8K gen_toks to max_model_len minus prompt budget (1024 tokens)
    local phase2_gen_toks="${GSM8K_GEN_TOKS}"
    if [[ $((max_model_len - 1024)) -lt ${GSM8K_GEN_TOKS} ]]; then
        phase2_gen_toks=$((max_model_len - 1024))
    fi

    # Detect whether the model has a chat template (instruction-tuned model).
    # If so, --apply_chat_template is MANDATORY for accurate loglikelihood
    # scoring (AGENTS.md lesson #9). Without it, MMLU/HellaSwag/ARC scores
    # are depressed by 30-60 percentage points because the model receives
    # raw text prompts outside its trained chat format.
    local has_chat_template=0
    if [[ -f "${model_path}/chat_template.jinja" ]]; then
        has_chat_template=1
    elif [[ -f "${model_path}/tokenizer_config.json" ]] && \
         grep -q '"chat_template"' "${model_path}/tokenizer_config.json" 2>/dev/null; then
        has_chat_template=1
    elif [[ -f "${TOKENIZER_DIR}/chat_template.jinja" ]]; then
        # Tokenizer mounted separately (e.g. when model dir lacks tokenizer)
        has_chat_template=1
    elif [[ -f "${TOKENIZER_DIR}/tokenizer_config.json" ]] && \
         grep -q '"chat_template"' "${TOKENIZER_DIR}/tokenizer_config.json" 2>/dev/null; then
        has_chat_template=1
    fi

    if [[ $has_chat_template -eq 1 ]]; then
        log "  Instruction-tuned model: applying --apply_chat_template (required for accurate loglikelihood, AGENTS.md #9)"
    else
        log "  No chat template detected: loglikelihood tasks will use raw prompts (correct for base models)"
    fi

    # Phase 1 chat-template flag (MANDATORY for instruct-tuned models).
    # lm-eval applies the tokenizer's chat_template to wrap raw prompts as
    # chat-formatted text before sending to the completions endpoint with
    # echo=True. Without this, instruct models assign depressed probabilities
    # to answer continuations, producing scores 30-60 pp below correct values.
    # See AGENTS.md lesson #9 and the header comment of this script.
    local phase1_chat_template_flag=""
    if [[ $has_chat_template -eq 1 ]]; then
        phase1_chat_template_flag="--apply_chat_template"
    fi

    local quant_label="${use_quant:-native bf16}"
    log "Evaluating ${model_name} (${model_size_gb}GB, ${quant_label}, TP=${tp_size}, gpu_util=${gpu_util}, max_len=${max_model_len}, batch=${batch_size}, GPUs=${gpu_label})"
    if [[ -n "${LM_EVAL_LIMIT:-}" ]]; then
        log "  DIAGNOSTIC MODE: --limit ${LM_EVAL_LIMIT} samples per task (AGENTS.md #10)"
    fi

    if [[ $DRY_RUN -eq 1 ]]; then
        log "  [dry-run] vLLM server + local-completions in same container"
        log "  [dry-run] NVIDIA_VISIBLE_DEVICES=${nvidia_vis} CUDA_VISIBLE_DEVICES=${cuda_vis}"
        log "  [dry-run] chat_template: has_chat_template=${has_chat_template} → phase1 flag='${phase1_chat_template_flag:-<none>}'"
        log "  [dry-run] Phase 1 tasks: $(echo "$TASKS" | tr ',' '\n' | grep -vE '^(gsm8k|humaneval|lambada_openai)$' | tr '\n' ',' | sed 's/,$//')"
        log "  [dry-run] Phase 2 tasks: gsm8k (gen_toks=${phase2_gen_toks}, thinking=ON)"
        local _p3
        _p3=$(echo "$TASKS" | tr ',' '\n' | grep -E '^(humaneval|lambada_openai)$' | tr '\n' ',' | sed 's/,$//')
        [[ -n "$_p3" ]] && log "  [dry-run] Phase 3 tasks: ${_p3} (raw completion, NO chat_template)"
        return 0
    fi

    local start_time
    start_time=$(date +%s)

    # Marker file for result collection — avoids cross-model contamination.
    # Trap ensures cleanup even if docker run fails (set -e exit).
    local marker_file
    marker_file=$(mktemp "${RESULTS_DIR}/.marker_XXXXXX")
    trap 'rm -f "${marker_file}"' RETURN

    # Per-model request cache (AGENTS.md lesson #14). lm-eval's
    # --cache_requests true writes content-addressed pickle files into
    # LM_HARNESS_CACHE_PATH. On resume (e.g. after container crash, OOM,
    # or user interrupt), already-completed requests are skipped — turns
    # a full Phase 1 restart from "lose 3h of work" into "resume in 5min".
    # --force blows the cache for this model so re-runs are truly fresh.
    local model_cache_dir="${RESULTS_DIR}/lm_eval_cache/${model_name}"
    if [[ $SKIP_EXISTING -eq 0 ]] && [[ -d "$model_cache_dir" ]]; then
        log "  --force: clearing request cache for ${model_name}"
        rm -rf "${model_cache_dir:?}"
    fi
    mkdir -p "$model_cache_dir"

    # Build NCCL env vars for TP=2
    local nccl_env=()
    if [[ ${tp_size} -gt 1 ]]; then
        nccl_env=(
            -e "NCCL_P2P_DISABLE=1"
            -e "NCCL_IB_DISABLE=1"
        )
    fi

    # Build vLLM server flags
    local vllm_server_flags=(
        --model /model
        --dtype bfloat16
        --trust-remote-code
        --max-model-len "${max_model_len}"
        --gpu-memory-utilization "${gpu_util}"
        --port 8080
        --host 127.0.0.1
    )
    # --enforce-eager disables CUDA graph optimization (20-40% throughput cost).
    # Default: enabled (safe for all architectures). Override with
    # LM_EVAL_ENFORCE_EAGER=0 for known-good architectures (Gemma4, Qwen3, etc.)
    # where CUDA graphs are supported by vLLM. Does NOT affect results —
    # CUDA graphs only change execution path, not computation.
    if [[ "${LM_EVAL_ENFORCE_EAGER:-1}" != "0" ]]; then
        vllm_server_flags+=(--enforce-eager)
    fi
    if [[ -n "${use_quant}" ]]; then
        if [[ "${use_quant}" == "fp8" ]]; then
            # Dynamic online FP8: default load-format (auto) — do NOT pass
            # --load-format fp8 (that expects a pre-quantized checkpoint).
            # Fp8PerTensorOnlineLinearMethod quantizes Linears at load;
            # DeepGemm auto-disables on Blackwell for qwen3_5_text (harmless).
            vllm_server_flags+=(--quantization fp8)
            # Memory levers for tight budgets (lesson #53 recipe): fp8 KV cache
            # + skipping the vision tower for text-only benchmarks.
            if [[ -n "${LM_EVAL_KV_CACHE_DTYPE:-}" ]]; then
                vllm_server_flags+=(--kv-cache-dtype "${LM_EVAL_KV_CACHE_DTYPE}")
            fi
            if [[ "${LM_EVAL_LANGUAGE_MODEL_ONLY:-0}" == "1" ]]; then
                vllm_server_flags+=(--language-model-only)
            fi
        else
            vllm_server_flags+=(--quantization "${use_quant}" --load-format "${use_quant}")
        fi
    fi
    if [[ ${tp_size} -gt 1 ]]; then
        vllm_server_flags+=(--tensor-parallel-size "${tp_size}")
    fi
    # MTP speculative decoding (inference acceleration ONLY — at temperature=0
    # spec decode is exactly output-preserving, so arms with/without MTP heads
    # remain comparable). Auto-gated on mtp.* tensor presence: variants whose
    # tooling dropped the MTP head (vanilla-heretic re-saves) skip the flag.
    local spec_tokens="${LM_EVAL_SPEC_TOKENS:-0}"
    if [[ "${spec_tokens}" =~ ^[0-9]+$ ]] && [[ "${spec_tokens}" -gt 0 ]]; then
        if python3 -c "
import json, sys
try:
    idx = json.load(open('${model_path}/model.safetensors.index.json'))
    sys.exit(0 if any(k.startswith('mtp.') for k in idx['weight_map']) else 1)
except Exception:
    sys.exit(1)" 2>/dev/null; then
            vllm_server_flags+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${spec_tokens}}")
            log "  MTP speculative decoding: ${spec_tokens} draft tokens"
        else
            log "  MTP requested but no mtp.* tensors in index for ${model_name} — serving without spec decode (output-identical at temp=0)"
        fi
    fi
    # Reasoning parser (for Qwen3.5 and similar)
    if [[ -n "${REASONING_PARSER:-}" ]]; then
        vllm_server_flags+=(--reasoning-parser "${REASONING_PARSER}")
    fi
    if [[ -n "${REASONING_CONFIG_JSON:-}" ]]; then
        vllm_server_flags+=(--reasoning-config "${REASONING_CONFIG_JSON}")
    fi
    # Enable thinking via chat template — vLLM applies this to all chat completion requests
    if [[ -f "${COMPARISON_DIR}/thinking_chat_template.jinja" ]]; then
        vllm_server_flags+=(--chat-template /comparison/thinking_chat_template.jinja)
    fi
    # LoRA flags (if adapter is set for this variant — AGENTS.md lesson #35)
    if [[ ${#lora_vllm_flags[@]} -gt 0 ]]; then
        vllm_server_flags+=("${lora_vllm_flags[@]}")
    fi
    # Extra vLLM flags passthrough (env, host-side). Needed when the FP8 recipe
    # meets a very large vocab: lm-eval LL MC requests prompt_logprobs, whose
    # fp32 log_softmax transient scales with max_num_batched_tokens x vocab —
    # 2048 chunk x 248320 vocab = 1.95 GiB on a card with <0.5 GiB of non-pool
    # headroom at util 0.95 -> engine-death OOM on the FIRST MMLU-Pro request
    # (AGENTS.md lesson #67). Working config: LM_EVAL_EXTRA_VLLM_FLAGS should
    # carry "--gpu-memory-utilization 0.94 --max-num-batched-tokens 512".
    # Unquoted expansion is intentional (word-split the flag string).
    # shellcheck disable=SC2206
    if [[ -n "${LM_EVAL_EXTRA_VLLM_FLAGS:-}" ]]; then
        vllm_server_flags+=(${LM_EVAL_EXTRA_VLLM_FLAGS})
    fi

    # Split tasks into three phases:
    #   Phase 1: LL MC + generative tasks WITH --apply_chat_template (instruct models need it, AGENTS.md #9)
    #   Phase 2: GSM8K via local-chat-completions with thinking enabled (AGENTS.md #1)
    #   Phase 3: Raw-completion tasks WITHOUT --apply_chat_template (AGENTS.md #17 — HumanEval/LAMBADA killed by chat template)
    local phase1_tasks phase2_tasks phase3_tasks
    phase3_tasks=""
    phase2_tasks=""
    phase1_tasks="$TASKS"

    # Extract raw-completion tasks into Phase 3
    for t in humaneval lambada_openai; do
        if echo "$TASKS" | tr ',' '\n' | grep -qx "$t"; then
            phase3_tasks="${phase3_tasks:+${phase3_tasks},}${t}"
            phase1_tasks=$(echo "$phase1_tasks" | tr ',' '\n' | grep -v "^${t}$" | tr '\n' ',' | sed 's/,$//')
        fi
    done

    # Extract GSM8K into Phase 2
    if echo "$phase1_tasks" | tr ',' '\n' | grep -qx "gsm8k"; then
        phase1_tasks=$(echo "$phase1_tasks" | tr ',' '\n' | grep -v "^gsm8k$" | tr '\n' ',' | sed 's/,$//')
        phase2_tasks="gsm8k"
    fi

    # Build the local-completions model_args
    # vllm_model_id is "/model" for normal variants, or the variant name for LoRA variants
    # timeout=600: lm-eval default is 300s, too short for thinking models generating up to
    # max_gen_toks at ~25 tok/s/request (8096/25=324s > 300s). See AGENTS.md lesson #36.
    local lc_model_args="base_url=http://127.0.0.1:8080/v1/completions,model=${vllm_model_id}"
    lc_model_args+=",tokenizer=/tokenizer,tokenizer_backend=huggingface"
    lc_model_args+=",num_concurrent=${NUM_CONCURRENT},max_length=${max_model_len},timeout=600"

    # Build the local-chat-completions model_args (Phase 2 GSM8K with thinking).
    # Defined on HOST side because the bash -c string interpolates ${chat_model_args}.
    local chat_model_args="base_url=http://127.0.0.1:8080/v1/chat/completions,model=${vllm_model_id}"
    chat_model_args+=",tokenizer=/tokenizer,tokenizer_backend=huggingface"
    chat_model_args+=",num_concurrent=${NUM_CONCURRENT},max_length=${max_model_len},timeout=600"

    # Serialize vLLM flags into a single string for safe passing into bash -c.
    # Using @Q inside bash -c '...' breaks because the single quotes from @Q
    # clash with the outer single-quoted heredoc. Instead, we pass the flags
    # as an environment variable and use bash word splitting inside the container.
    local vllm_flags_str=""
    for flag in "${vllm_server_flags[@]}"; do
        vllm_flags_str+=" $(printf '%q' "$flag")"
    done

    # Build --limit flag string for lm-eval (same env-var passthrough pattern as
    # vllm_flags_str). Empty when no limit is set (default, full task sets).
    # Used for the pre-flight diagnostic (AGENTS.md lesson #10): a quick
    # HellaSwag --limit 200 run on the base model validates that the chat
    # template is being applied before kicking off the full pipeline.
    local lm_limit_flags_str=""
    if [[ -n "${LM_EVAL_LIMIT:-}" ]]; then
        lm_limit_flags_str=" --limit $(printf '%q' "${LM_EVAL_LIMIT}")"
    fi

    # Run both vLLM server + lm-eval inside SAME Docker container
    # --entrypoint bash: override vllm-openai's default ENTRYPOINT
    #
    # Phase 1 chat template override (AGENTS.md lesson #16):
    # Thinking models (Gemma 4, Qwen3, DeepSeek-R1) inject <|think|> tags via
    # their chat template by default. For loglikelihood MC tasks (Phase 1),
    # this contaminates scoring — the model expects thinking tokens, not answer
    # letters, at the scoring position. Result: MMLU-Pro depressed by ~25pp.
    # If the comparison provides a no_thinking_chat_template.jinja variant,
    # overlay it on /tokenizer/chat_template.jinja so Phase 1 loglikelihood
    # uses the non-thinking format. Phase 2 (local-chat-completions → vLLM
    # chat endpoint with --chat-template thinking override) is unaffected.
    local no_think_mount=()
    if [[ -f "${COMPARISON_DIR}/no_thinking_chat_template.jinja" ]]; then
        log "  Phase 1: using no_thinking_chat_template.jinja (disables <|think|> injection for loglikelihood)"
        no_think_mount=(-v "${COMPARISON_DIR}/no_thinking_chat_template.jinja:/tokenizer/chat_template.jinja:ro")
    fi

    docker run --rm --runtime=nvidia --shm-size=16g --ipc=host \
        --entrypoint bash \
        -e "NVIDIA_VISIBLE_DEVICES=${nvidia_vis}" \
        -e "CUDA_VISIBLE_DEVICES=${cuda_vis}" \
        -e "PYTORCH_ALLOC_CONF=expandable_segments:True" \
        "${nccl_env[@]+${nccl_env[@]}}" \
        -e "VLLM_FLAGS=${vllm_flags_str}" \
        -e "LM_EVAL_LIMIT_FLAGS=${lm_limit_flags_str}" \
        -v "${model_path}:/model:ro" \
        "${lora_mount[@]+${lora_mount[@]}}" \
        -v "${TOKENIZER_DIR}:/tokenizer:ro" \
        "${no_think_mount[@]+${no_think_mount[@]}}" \
        -v "${RESULTS_DIR}:/results" \
        -v "${LOG_DIR}:/logs" \
        -v "${HF_CACHE}:/root/.cache/huggingface" \
        -v "${VLLM_CACHE}:/root/.cache/vllm" \
        -v "${NLTK_CACHE}:/root/nltk_data:ro" \
        -v "${COMPARISON_DIR}:/comparison:ro" \
        -v "${model_cache_dir}:/lm_eval_cache" \
        "${HF_TOKEN_MOUNT[@]+${HF_TOKEN_MOUNT[@]}}" \
        "${HF_TOKEN_ENV[@]+${HF_TOKEN_ENV[@]}}" \
        -e HF_HOME=/root/.cache/huggingface \
        -e HF_DATASETS_OFFLINE=1 \
        -e HF_HUB_OFFLINE=1 \
        -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
        -e HF_ALLOW_CODE_EVAL=1 \
        -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        -e LM_HARNESS_CACHE_PATH=/lm_eval_cache \
        "${LMEVAL_IMAGE}" -c '
            set -eo pipefail

            # One-time: install lm-eval[ifeval] extra (langdetect) if missing.
            # The base Docker image does not include this; IFEval task imports
            # langdetect at module load. Quick no-op if already installed.
            if ! python3 -c "import langdetect" >/dev/null 2>&1; then
                echo "[container] Installing lm-eval[ifeval] extra (langdetect)..."
                pip install --quiet "lm-eval[ifeval]" 2>&1 | tail -3
            fi

            # One-time: install human_eval for HumanEval pass@k code verification.
            # Only needed when humaneval task is in the task list.
            if echo "${TASKS:-}" | grep -q "humaneval" && ! python3 -c "import human_eval" >/dev/null 2>&1; then
                echo "[container] Installing human_eval package for pass@k verification..."
                pip install --quiet "human_eval" 2>&1 | tail -3
            fi

            # Patch lm-eval 0.4.11 CLI bug: `--cache_requests` defines
            # type=request_caching_arg_to_dict (converts "true" → dict)
            # AND choices=["true","refresh","delete"] (checks the dict
            # against string choices → always fails). Removing the choices=
            # line lets the type= conversion pass through. Idempotent.
            # See AGENTS.md lesson #15.
            LMEVAL_RUN_PY=/usr/local/lib/python3.12/dist-packages/lm_eval/_cli/run.py
            if grep -q "choices=\[\"true\", \"refresh\", \"delete\"\]," "$LMEVAL_RUN_PY" 2>/dev/null; then
                echo "[container] Patching lm-eval 0.4.11 --cache_requests choices= bug"
                sed -i "/choices=\[\"true\", \"refresh\", \"delete\"\],/d" "$LMEVAL_RUN_PY"
            fi

            # Start vLLM server in background
            # Re-parse the %q-escaped flags from the host: unquoted $VAR
            # expansion word-splits but does NOT consume printf %q backslashes,
            # so JSON-valued flags arrived at vLLM with literal backslashes
            # (e.g. \{\"a\":\ \"b\"\}) and crashed pydantic JSON parsing.
            # eval + array expansion is the correct consumer of %q output
            # (lesson #66). VLLM_FLAGS is fully %q-escaped host-side, so the
            # eval re-parse is exact.
            eval "VLLM_FLAGS_ARR=(${VLLM_FLAGS})"
            # shellcheck disable=SC2086
            python3 -m vllm.entrypoints.openai.api_server "${VLLM_FLAGS_ARR[@]}" \
                > /logs/'"${model_name}"'_vllm_server.log 2>&1 &
            SERVER_PID=$!

            # Wait for server ready (up to 600s)
            for i in $(seq 1 120); do
                if curl -sf http://127.0.0.1:8080/v1/models > /dev/null 2>&1; then
                    echo "vLLM ready after ~$((i*5))s"
                    break
                fi
                if ! kill -0 $SERVER_PID 2>/dev/null; then
                    echo "ERROR: vLLM server crashed during startup"
                    cat /logs/'"${model_name}"'_vllm_server.log 2>/dev/null | tail -30
                    exit 1
                fi
                sleep 5
            done

            # Verify server actually started (handles timeout without crash)
            if ! curl -sf http://127.0.0.1:8080/v1/models > /dev/null 2>&1; then
                echo "ERROR: vLLM server failed to start within 600s timeout"
                cat /logs/'"${model_name}"'_vllm_server.log 2>/dev/null | tail -30
                kill $SERVER_PID 2>/dev/null || true
                exit 1
            fi

            # Phase 1: Loglikelihood + truthfulqa (gen_toks=2048)
            # Skip in GSM8K_RERUN mode when existing results are present.
            # NOTE: --apply_chat_template is interpolated from host (empty for
            # base models without chat templates, set for instruct-tuned models).
            #
            # Two caches are in use:
            #   --cache_requests true      → preprocessing cache (pickle files in
            #                                LM_HARNESS_CACHE_PATH). Saves ~2-5 min
            #                                of task/context building on resume.
            #                                Does NOT skip inference (AGENTS.md #14).
            #   --use_cache <sqlite_path>  → response-level cache (SQLite db of
            #                                model responses keyed by request hash).
            #                                THIS is what makes crash/resume cheap —
            #                                on restart, cached responses are returned
            #                                without re-calling the model. AGENTS.md #18.
            if [[ -n "'"${phase1_tasks}"'" && "'"${skip_phase1}"'" != "1" ]]; then
                echo "Phase 1: '"${phase1_tasks}"' (chat_template='"${has_chat_template}"')"
                python3 -m lm_eval --model local-completions \
                    --model_args "'"${lc_model_args}"'" \
                    --tasks "'"${phase1_tasks}"'" \
                    --batch_size '"${batch_size}"' \
                    --gen_kwargs max_gen_toks='"${MAX_GEN_TOKS}"' \
                    "'"${phase1_chat_template_flag}"'" \
                    ${LM_EVAL_LIMIT_FLAGS} \
                    --cache_requests true \
                    --use_cache /lm_eval_cache/responses.sqlite \
                    --output_path /results/ \
                    --log_samples 2>&1 | tee /logs/'"${model_name}"'_phase1.log
            fi

            # Phase 2: GSM8K via chat completions with thinking enabled.
            # vLLM server has --chat-template pointing to thinking_chat_template.jinja
            # which injects <|think|>. lm-eval sends messages, vLLM applies the template.
            #
            # CRITICAL: until=[] overrides the gsm8k task default stop sequences
            # [Question:, </s>, <|im_end|>]. The Question: stop sequence is
            # destructive for thinking models: the model writes Question: during
            # its reasoning trace, vLLM truncates the generation there, and the
            # reasoning parser sees incomplete thinking with no answer, producing
            # content=null. This caused a 73% null-response rate on Gemma 4 12B
            # Unified GSM8K. Setting until=[] effectively disables all stop
            # sequences; the model stops naturally at EOS after completing its
            # answer. AGENTS.md lesson #30, discovered gemma4-12b 2026-08-02.
            #
            # num_fewshot=0: thinking models MUST be 0-shot on GSM8K. The gsm8k
            # task default is 5-shot, but fewshot examples teach the model to
            # produce Answer: <<calc>>... #### ans format. With thinking enabled,
            # the model solves the problem in its reasoning trace, then tries to
            # reformat to match the fewshot examples, entering a thinking loop
            # that consumes all 8096 tokens without converging. This caused a
            # 92% null-content rate. 0-shot lets the model think freely and
            # produce a natural answer. Verified: 0-shot base model produces
            # correct content in ~377 tokens (vs 8096-token loops at 5-shot).
            # AGENTS.md lesson #30.
            if [[ -n "'"${phase2_tasks}"'" ]]; then
                echo "Phase 2: '"${phase2_tasks}"' (gen_toks='"${phase2_gen_toks}"', thinking=ON, chat-completions, 0-shot, until_overridden)"
                python3 -m lm_eval --model local-chat-completions \
                    --model_args "'"${chat_model_args}"'" \
                    --tasks "'"${phase2_tasks}"'" \
                    --num_fewshot 0 \
                    --gen_kwargs max_gen_toks='"${phase2_gen_toks}"',until=[] \
                    --apply_chat_template \
                    --fewshot_as_multiturn false \
                    ${LM_EVAL_LIMIT_FLAGS} \
                    --cache_requests true \
                    --use_cache /lm_eval_cache/responses.sqlite \
                    --output_path /results/ \
                    --log_samples 2>&1 | tee /logs/'"${model_name}"'_phase2.log
            fi

            # Phase 3: Raw-completion tasks WITHOUT --apply_chat_template (AGENTS.md #17).
            # HumanEval wants raw Python function continuation; LAMBADA wants narrative
            # continuation. With chat template, instruct models respond conversationally
            # → pass@1=0, perplexity ~1e11. Running raw gives real signal.
            if [[ -n "'"${phase3_tasks}"'" ]]; then
                echo "Phase 3: '"${phase3_tasks}"' (raw completion, NO chat_template)"
                UNSAFE_CODE_FLAG=""
                if echo "'"${phase3_tasks}"'" | grep -q "humaneval"; then
                    UNSAFE_CODE_FLAG="--confirm_run_unsafe_code"
                    echo "[container] HumanEval detected: enabling --confirm_run_unsafe_code"
                fi
                python3 -m lm_eval --model local-completions \
                    --model_args "'"${lc_model_args}"'" \
                    --tasks "'"${phase3_tasks}"'" \
                    --batch_size '"${batch_size}"' \
                    --gen_kwargs max_gen_toks='"${MAX_GEN_TOKS}"' \
                    ${LM_EVAL_LIMIT_FLAGS} \
                    --cache_requests true \
                    --use_cache /lm_eval_cache/responses.sqlite \
                    ${UNSAFE_CODE_FLAG} \
                    --output_path /results/ \
                    --log_samples 2>&1 | tee /logs/'"${model_name}"'_phase3.log
            fi

            # Tear down vLLM server
            kill $SERVER_PID 2>/dev/null || true
        ' || true   # tolerate per-model vLLM/eval failures so the batch continues

    # Collect all result files created during this run (both Phase 1 and Phase 2).
    # lm-eval writes to /results/__model/results_TIMESTAMP.json — find files
    # newer than our marker file to avoid cross-model contamination.
    local result_files=()
    while IFS= read -r -d '' f; do
        result_files+=("$f")
    done < <(find "${RESULTS_DIR}" -maxdepth 3 -name "results_*.json" \
        -newer "${marker_file}" -not -name "lm_eval_*.json" -print0 2>/dev/null | sort -z)

    rm -f "${marker_file}"

    # In GSM8K_RERUN mode, include existing results for merging (Phase 1 data).
    # PREPEND the existing file so new Phase 2 (thinking GSM8K) is processed LAST
    # by merged_results.update() — otherwise old non-thinking GSM8K overwrites new.
    if [[ -n "${GSM8K_RERUN:-}" && -f "${out_file}" ]]; then
        result_files=("${out_file}" "${result_files[@]}")
    fi

    if [[ ${#result_files[@]} -eq 0 ]]; then
        log "  ERROR: no results found for ${model_name}"
        if [[ $DRY_RUN -eq 0 ]]; then
            db_stage_fail "$COMPARISON_NAME" "$model_name" "lm_eval" "no results found"
        fi
    elif [[ ${#result_files[@]} -eq 1 ]]; then
        cp "${result_files[0]}" "${out_file}"
        local end_time elapsed
        end_time=$(date +%s)
        elapsed=$(( end_time - start_time ))
        log "  Saved: ${out_file} (${elapsed}s)"
    else
        # Multiple result files (Phase 1 + Phase 2) — merge into canonical file.
        # lm-eval results JSON has a top-level "results" dict keyed by task name.
        # Merge all "results" dicts and "configs" dicts into the first file.
        local merged
        merged=$(python3 -c "
import json, sys
merged_results = {}
merged_configs = {}
source = None
for f in sys.argv[1:]:
    with open(f) as fh:
        data = json.load(fh)
    if source is None:
        source = data
    merged_results.update(data.get('results', {}))
    merged_configs.update(data.get('configs', {}))
if source is not None:
    source['results'] = merged_results
    source['configs'] = merged_configs
    json.dump(source, sys.stdout)
" "${result_files[@]}")
        echo "${merged}" > "${out_file}"
        local end_time elapsed
        end_time=$(date +%s)
        elapsed=$(( end_time - start_time ))
        log "  Saved: ${out_file} (${elapsed}s, merged ${#result_files[@]} phase results)"
    fi

    # Rename per-run samples files to include the model name, so future imports
    # can find them by filename pattern rather than mtime matching.
    # lm-eval writes to /results/__model/samples_<task>_<timestamp>.jsonl — these
    # are anonymous (no model name in filename) and accumulate across model runs.
    # Renaming to <model>__samples_<task>_<timestamp>.jsonl makes the importer's
    # job trivial and prevents cross-model contamination on re-runs.
    # Only rename files newer than the original marker (this run's output).
    if [[ $DRY_RUN -eq 0 ]] && [[ ${#result_files[@]} -gt 0 ]]; then
        while IFS= read -r -d '' sf; do
            sf_name="$(basename "$sf")"
            # Skip if already renamed (idempotent)
            if [[ "${sf_name}" == *__* ]]; then
                continue
            fi
            sf_dir="$(dirname "$sf")"
            new_name="${model_name}__${sf_name}"
            mv "$sf" "${sf_dir}/${new_name}" 2>/dev/null && \
                log "  Renamed samples: ${sf_name} → ${new_name}"
        done < <(find "${RESULTS_DIR}" -maxdepth 3 -name "samples_*.jsonl" \
            -newermt "@${start_time}" -print0 2>/dev/null)
    fi

    # Mark stage complete in DB
    if [[ $DRY_RUN -eq 0 ]] && [[ ${#result_files[@]} -gt 0 ]]; then
        db_stage_complete "$COMPARISON_NAME" "$model_name" "lm_eval" --result-path "$out_file"
    fi
}

# --models filter: scope which models run (default: all). "base" matches BASE_NAME.
# Used by the pre-flight diagnostic (AGENTS.md #10) to evaluate only the base
# model with a sample cap before committing to the full multi-day pipeline.
should_run_model() {
    local name="$1"
    [[ -z "${MODELS_FILTER:-}" ]] && return 0          # no filter → run all
    [[ "$name" == "${BASE_NAME}" ]] && echo ",${MODELS_FILTER}," | grep -qF ",base," && return 0
    echo ",${MODELS_FILTER}," | grep -qF ",${name}," && return 0
    return 1
}

# Run base model
if should_run_model "${BASE_NAME}"; then
    run_lm_eval "${BASE_NAME}" "${BASE_DIR}" || true
else
    log "[filter] skipping base model ${BASE_NAME} (--models '${MODELS_FILTER}')"
fi

# Run each variant
for variant in $VARIANT_NAMES; do
    if ! should_run_model "$variant"; then
        log "[filter] skipping variant ${variant} (--models '${MODELS_FILTER}')"
        continue
    fi
    variant_path_var="VARIANT_PATH_$(echo "$variant" | tr '.-' '__')"
    run_lm_eval "${variant}" "${!variant_path_var}" || true
done

# Import results to DB
if [[ $DRY_RUN -eq 0 ]]; then
    db_import_results "$COMP_DIR"
fi

log "lm-eval complete"
