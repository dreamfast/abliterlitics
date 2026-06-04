#!/usr/bin/env bash
# lm-eval runner — vLLM server + local-completions backend
#
# Architecture: Both vLLM server and lm-eval run inside the SAME Docker
# container. This avoids the tokenizer download hang (AGENTS.md #6).
#
# Two-phase execution per model:
#   Phase 1: Loglikelihood + truthfulqa with max_gen_toks=2048
#   Phase 2: GSM8K with max_gen_toks=7168 (reasoning models need thinking room)
#
# For large MoE models (GLM, >50GB): supports TP=2 across dual GPUs.
# For Qwen3.5 family: single-GPU BNB4 (Qwen3.5 can't do BNB4+TP).
#
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
  Models >25GB automatically use bitsandbytes 4-bit.
  Override with --quantization.

Options:
   --gpus GPUS           GPU selection (default: auto)
   --tasks TASKS         Comma-separated task list (default: from comparison.json)
   --quantization QUANT  Override quantization (auto/bitsandbytes/none)
   --skip-existing       Skip models with existing results (default)
   --force               Overwrite existing results
   --dry-run             Show what would run without executing
   -h, --help            Show this help
EOF
    exit 0
}

GPUS_ARG=""
TASKS=""
QUANTIZATION=""
DRY_RUN=0
SKIP_EXISTING=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)           GPUS_ARG="$2"; shift 2 ;;  # alias for --gpus
        --gpus)          GPUS_ARG="$2"; shift 2 ;;
        --backend)       shift 2 ;;  # consumed for CLI compat, lm-eval always uses vLLM
        --tasks)         TASKS="$2"; shift 2 ;;
        --quantization)  QUANTIZATION="$2"; shift 2 ;;
        --skip-existing) SKIP_EXISTING=1; shift ;;
        --force)         SKIP_EXISTING=0; shift ;;
        --dry-run)       DRY_RUN=1; shift ;;
        -h|--help)       usage ;;
        *)               break ;;
    esac
done

COMP_DIR="${1:?Error: comparison_dir required. Use --help for usage.}"

load_comparison "$COMP_DIR"
TASKS="${TASKS:-$LM_EVAL_TASKS}"
RESULTS_DIR="${COMPARISON_DIR}/results/lm_eval"
LOG_DIR="${COMPARISON_DIR}/results/lm_eval_logs"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

# Shared cache directories
HF_CACHE="${ABL_ROOT}/.cache/hf"
VLLM_CACHE="${ABL_ROOT}/.cache/vllm"
mkdir -p "$HF_CACHE" "$VLLM_CACHE"

# Settings from _load_comparison.py (with defaults for standalone use)
MAX_MODEL_LEN="${LM_EVAL_MAX_MODEL_LEN:-8192}"
MAX_GEN_TOKS="${LM_EVAL_MAX_GEN_TOKS:-2048}"
TOKENIZER_DIR="${TOKENIZER_DIR:-${BASE_DIR}}"

# GSM8K needs more generation tokens for reasoning models
GSM8K_GEN_TOKS=7168

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
    local out_file="${RESULTS_DIR}/lm_eval_${model_name}.json"

    # Check per-variant skip flag
    local var_slug
    var_slug="$(echo "$model_name" | tr '-' '_')"
    local skip_flag="VARIANT_SKIP_LM_${var_slug}"
    if [[ "${!skip_flag:-0}" == "1" ]]; then
        log "[skip] ${model_name}: skip_lm_eval=true"
        return 0
    fi

    if result_exists "$out_file"; then
        log "Skipping (exists): ${model_name}"
        return 0
    fi

    # Detect model size for resource decisions
    local model_size_kb
    model_size_kb=$(du -sk "${model_path}" 2>/dev/null | cut -f1 || echo 0)
    local model_size_gb=$(( model_size_kb / 1024 / 1024 ))

    # Determine quantization strategy
    local use_quant=""
    if [[ -n "${QUANTIZATION}" ]]; then
        case "${QUANTIZATION}" in
            bitsandbytes|bnb) use_quant="bitsandbytes" ;;
            none|off)         use_quant="" ;;
            auto|"")
                if [[ ${model_size_gb} -gt 25 ]]; then
                    use_quant="bitsandbytes"
                fi
                ;;
        esac
    else
        if [[ ${model_size_gb} -gt 25 ]]; then
            use_quant="bitsandbytes"
        fi
    fi

    # Resolve GPU config for this model
    local gpu_config nvidia_vis cuda_vis tp_size gpu_label
    gpu_config=$(resolve_gpus "${model_size_gb}")
    read -r nvidia_vis cuda_vis tp_size gpu_label <<< "${gpu_config}"

    # Adjust batch size and gpu_util based on model size
    local gpu_util="0.9"
    local batch_size="auto"
    local max_model_len="${MAX_MODEL_LEN}"

    if [[ ${model_size_gb} -gt 50 ]]; then
        # Large MoE models (e.g. GLM-4.7-Flash 59GB)
        gpu_util="0.75"
        batch_size="4"
        max_model_len="4096"
    elif [[ ${model_size_gb} -gt 25 ]]; then
        batch_size="16"
    fi

    # Cap GSM8K gen_toks to max_model_len minus prompt budget (1024 tokens)
    local phase2_gen_toks="${GSM8K_GEN_TOKS}"
    if [[ $((max_model_len - 1024)) -lt ${GSM8K_GEN_TOKS} ]]; then
        phase2_gen_toks=$((max_model_len - 1024))
    fi

    local quant_label="${use_quant:-native bf16}"
    log "Evaluating ${model_name} (${model_size_gb}GB, ${quant_label}, TP=${tp_size}, gpu_util=${gpu_util}, max_len=${max_model_len}, batch=${batch_size}, GPUs=${gpu_label})"

    if [[ $DRY_RUN -eq 1 ]]; then
        log "  [dry-run] vLLM server + local-completions in same container"
        log "  [dry-run] NVIDIA_VISIBLE_DEVICES=${nvidia_vis} CUDA_VISIBLE_DEVICES=${cuda_vis}"
        log "  [dry-run] Phase 1 tasks: $(echo "$TASKS" | tr ',' '\n' | grep -v gsm8k | tr '\n' ',')"
        log "  [dry-run] Phase 2 tasks: gsm8k (gen_toks=${phase2_gen_toks})"
        return 0
    fi

    local start_time
    start_time=$(date +%s)

    # Marker file for result collection — avoids cross-model contamination.
    # Trap ensures cleanup even if docker run fails (set -e exit).
    local marker_file
    marker_file=$(mktemp "${RESULTS_DIR}/.marker_XXXXXX")
    trap 'rm -f "${marker_file}"' RETURN

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
        --enforce-eager
        --port 8080
        --host 127.0.0.1
    )
    if [[ -n "${use_quant}" ]]; then
        vllm_server_flags+=(--quantization "${use_quant}" --load-format "${use_quant}")
    fi
    if [[ ${tp_size} -gt 1 ]]; then
        vllm_server_flags+=(--tensor-parallel-size "${tp_size}")
    fi
    # Reasoning parser (for Qwen3.5 and similar)
    if [[ -n "${REASONING_PARSER:-}" ]]; then
        vllm_server_flags+=(--reasoning-parser "${REASONING_PARSER}")
    fi
    if [[ -n "${REASONING_CONFIG_JSON:-}" ]]; then
        vllm_server_flags+=(--reasoning-config "${REASONING_CONFIG_JSON}")
    fi

    # Split tasks: Phase 1 (everything except gsm8k) and Phase 2 (gsm8k only)
    local phase1_tasks phase2_tasks
    if echo "$TASKS" | grep -q "gsm8k"; then
        phase1_tasks=$(echo "$TASKS" | tr ',' '\n' | grep -v '^gsm8k$' | tr '\n' ',' | sed 's/,$//')
        phase2_tasks="gsm8k"
    else
        phase1_tasks="$TASKS"
        phase2_tasks=""
    fi

    # Build the local-completions model_args
    local lc_model_args="base_url=http://127.0.0.1:8080/v1/completions,model=/model"
    lc_model_args+=",tokenizer=/tokenizer,tokenizer_backend=huggingface"
    lc_model_args+=",num_concurrent=4,max_length=${max_model_len}"

    # Run both vLLM server + lm-eval inside SAME Docker container
    # --entrypoint bash: override vllm-openai's default ENTRYPOINT
    docker run --rm --runtime=nvidia --shm-size=16g --ipc=host \
        --entrypoint bash \
        -e "NVIDIA_VISIBLE_DEVICES=${nvidia_vis}" \
        -e "CUDA_VISIBLE_DEVICES=${cuda_vis}" \
        "${nccl_env[@]+${nccl_env[@]}}" \
        -v "${model_path}:/model:ro" \
        -v "${TOKENIZER_DIR}:/tokenizer:ro" \
        -v "${RESULTS_DIR}:/results" \
        -v "${LOG_DIR}:/logs" \
        -v "${HF_CACHE}:/root/.cache/huggingface" \
        -v "${VLLM_CACHE}:/root/.cache/vllm" \
        -e HF_HOME=/root/.cache/huggingface \
        "${LMEVAL_IMAGE}" -c '
            set -eo pipefail

            # Start vLLM server in background
            python3 -m vllm.entrypoints.openai.api_server '"${vllm_server_flags[@]@Q}"' \
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
            if [[ -n "'"${phase1_tasks}"'" ]]; then
                echo "Phase 1: '"${phase1_tasks}"'"
                python3 -m lm_eval --model local-completions \
                    --model_args "'"${lc_model_args}"'" \
                    --tasks "'"${phase1_tasks}"'" \
                    --batch_size '"${batch_size}"' \
                    --gen_kwargs max_gen_toks='"${MAX_GEN_TOKS}"' \
                    --output_path /results/ \
                    --log_samples 2>&1 | tee /logs/'"${model_name}"'_phase1.log
            fi

            # Phase 2: GSM8K (gen_toks=7168 for reasoning models, capped by max_model_len)
            if [[ -n "'"${phase2_tasks}"'" ]]; then
                echo "Phase 2: '"${phase2_tasks}"' (gen_toks='"${phase2_gen_toks}"')"
                python3 -m lm_eval --model local-completions \
                    --model_args "'"${lc_model_args}"'" \
                    --tasks "'"${phase2_tasks}"'" \
                    --batch_size '"${batch_size}"' \
                    --gen_kwargs max_gen_toks='"${phase2_gen_toks}"' \
                    --output_path /results/ \
                    --log_samples 2>&1 | tee /logs/'"${model_name}"'_phase2.log
            fi

            # Tear down vLLM server
            kill $SERVER_PID 2>/dev/null || true
        '

    # Collect all result files created during this run (both Phase 1 and Phase 2).
    # lm-eval writes to /results/__model/results_TIMESTAMP.json — find files
    # newer than our marker file to avoid cross-model contamination.
    local result_files=()
    while IFS= read -r -d '' f; do
        result_files+=("$f")
    done < <(find "${RESULTS_DIR}" -maxdepth 3 -name "results_*.json" \
        -newer "${marker_file}" -not -name "lm_eval_*.json" -print0 2>/dev/null | sort -z)

    rm -f "${marker_file}"

    if [[ ${#result_files[@]} -eq 0 ]]; then
        log "  ERROR: no results found for ${model_name}"
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
}

# Run base model
run_lm_eval "base" "${BASE_DIR}"

# Run each variant
for variant in $VARIANT_NAMES; do
    variant_path_var="VARIANT_PATH_${variant}"
    run_lm_eval "${variant}" "${!variant_path_var}"
done

log "lm-eval complete"
