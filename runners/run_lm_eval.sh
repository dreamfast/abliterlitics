#!/usr/bin/env bash
# Unified lm-eval runner with Docker + vLLM backend
#
# Supports single-GPU and dual-GPU (tensor_parallel) modes.
# For models >50GB (e.g. GLM-4.7-Flash MoE), automatically uses:
#   - bitsandbytes 4-bit quantization
#   - tensor_parallel_size=2 across both GPUs
#   - CUDA_VISIBLE_DEVICES=1,0 (makes 5090=cuda:0)
#   - NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 (PCIe, no P2P)
#   - gpu_memory_utilization=0.75 (avoids OOM on 4090 during log_softmax)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] <comparison_dir>

Run lm-evaluation-harness on base + all variants via Docker (vLLM backend).

Resource modes:
  --gpus single    Single GPU (auto-detected best GPU, default for models <50GB)
  --gpus dual      Both GPUs with tensor_parallel_size=2 (default for models >50GB)
  --gpus 0         Use only GPU index 0
  --gpus 1         Use only GPU index 1
  --gpus 0,1       Use both GPUs (same as --gpus dual)

Quantization:
  For models >25GB, automatically uses bitsandbytes 4-bit quantization.
  Override with --quantization.

Options:
   --gpus GPUS           GPU selection: single/dual/INDEX/INDEX,INDEX (default: auto)
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
        --gpus)          GPUS_ARG="$2"; shift 2 ;;
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
    local log_file="${LOG_DIR}/${model_name}_lm_eval.log"

    if result_exists "$out_file"; then
        log "Skipping (exists): ${model_name}"
        return 0
    fi

    # Detect model size for resource decisions
    local model_size_kb
    model_size_kb=$(du -sk "${model_path}" 2>/dev/null | cut -f1)
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

    # Build model_args
    local model_args="pretrained=/model,dtype=bfloat16,trust_remote_code=True"
    local gpu_util="0.9"
    local max_model_len="4096"
    local batch_size="32"
    local max_gen_toks=""

    if [[ ${model_size_gb} -gt 50 ]]; then
        # Large MoE models (e.g. GLM-4.7-Flash 59GB):
        #   BNB4 + TP=2 fits ~8.7 GiB per GPU
        #   gpu_util=0.75 critical: 0.85 OOMs on 4090 during MMLU log_softmax
        #   max_model_len=4096 works (38x concurrency, 156K token KV cache)
        #   batch_size=4 for stable performance
        gpu_util="0.75"
        batch_size="4"
        max_gen_toks="2048"  # For GSM8K generate_until
    elif [[ ${model_size_gb} -gt 25 ]]; then
        batch_size="16"
        max_gen_toks="1024"
    fi

    if [[ -n "${use_quant}" ]]; then
        model_args+=",quantization=${use_quant}"
    fi
    if [[ ${tp_size} -gt 1 ]]; then
        model_args+=",tensor_parallel_size=${tp_size}"
    fi
    model_args+=",gpu_memory_utilization=${gpu_util},max_model_len=${max_model_len}"

    local quant_label="${use_quant:-native bf16}"
    log "Evaluating ${model_name} (${model_size_gb}GB, ${quant_label}, TP=${tp_size}, gpu_util=${gpu_util}, max_len=${max_model_len}, batch=${batch_size}, GPUs=${gpu_label})"

    if [[ $DRY_RUN -eq 1 ]]; then
        log "  [dry-run] docker: NVIDIA_VISIBLE_DEVICES=${nvidia_vis} CUDA_VISIBLE_DEVICES=${cuda_vis}"
        log "  [dry-run] model_args: ${model_args}"
        log "  [dry-run] tasks: ${TASKS}"
        return 0
    fi

    local start_time
    start_time=$(date +%s)

    # Build the gen_kwargs flag if needed
    local gen_kwargs_flag=""
    if [[ -n "${max_gen_toks}" ]]; then
        gen_kwargs_flag="--gen_kwargs max_gen_toks=${max_gen_toks}"
    fi

    # Build NCCL env vars for TP=2
    local nccl_env=()
    if [[ ${tp_size} -gt 1 ]]; then
        nccl_env=(
            -e "NCCL_P2P_DISABLE=1"
            -e "NCCL_IB_DISABLE=1"
        )
    fi

    # Run lm-eval inside Docker
    # --entrypoint bash: override vllm-openai's default ENTRYPOINT ["vllm", "serve"]
    # Pass model_args via env to avoid bash -c interpolation issues with commas
    docker run --rm --runtime=nvidia --shm-size=16g --ipc=host \
        --entrypoint bash \
        -e "NVIDIA_VISIBLE_DEVICES=${nvidia_vis}" \
        -e "CUDA_VISIBLE_DEVICES=${cuda_vis}" \
        "${nccl_env[@]+${nccl_env[@]}}" \
        -e "LM_EVAL_MODEL_ARGS=${model_args}" \
        -e "LM_EVAL_TASKS=${TASKS}" \
        -e "LM_EVAL_BATCH_SIZE=${batch_size}" \
        -e "LM_EVAL_GEN_KWARGS=${gen_kwargs_flag}" \
        -v "${model_path}:/model:ro" \
        -v "${RESULTS_DIR}:/results" \
        -v "${LOG_DIR}:/logs" \
        -v "${HF_CACHE}:/root/.cache/huggingface" \
        -v "${VLLM_CACHE}:/root/.cache/vllm" \
        -e HF_HOME=/root/.cache/huggingface \
        "${LMEVAL_IMAGE}" -c '
            python3 -m lm_eval --model vllm \
                --model_args "$LM_EVAL_MODEL_ARGS" \
                --tasks "$LM_EVAL_TASKS" \
                --batch_size "$LM_EVAL_BATCH_SIZE" \
                $LM_EVAL_GEN_KWARGS \
                --output_path /results/ \
                --log_samples 2>&1 | tee "/logs/'"${model_name}"'_lm_eval.log"
        '

    # lm-eval writes results in /results/__model/results_TIMESTAMP.json
    # Find the most recent results file created after this run started and copy to canonical name
    local latest_result
    latest_result=$(find "${RESULTS_DIR}" -maxdepth 3 -name "results_*.json" -newer "${RESULTS_DIR}" -not -name "lm_eval_*.json" 2>/dev/null | sort -r | head -1)
    if [[ -n "${latest_result}" && -f "${latest_result}" ]]; then
        cp "${latest_result}" "${out_file}"
        local end_time elapsed
        end_time=$(date +%s)
        elapsed=$(( end_time - start_time ))
        log "  Saved: ${out_file} (${elapsed}s)"
    else
        log "  WARNING: no results file found for ${model_name}, checking for any recent results..."
        latest_result=$(find "${RESULTS_DIR}" -maxdepth 3 -name "results_*.json" -mmin -10 2>/dev/null | sort -r | head -1)
        if [[ -n "${latest_result}" && -f "${latest_result}" ]]; then
            cp "${latest_result}" "${out_file}"
            local end_time elapsed
            end_time=$(date +%s)
            elapsed=$(( end_time - start_time ))
            log "  Saved: ${out_file} (${elapsed}s, fallback)"
        else
            log "  ERROR: no results found for ${model_name}"
        fi
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
