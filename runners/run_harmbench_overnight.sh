#!/usr/bin/env bash
# Overnight HarmBench + lm-eval runner for Qwen 3.6 27B comparison
#
# Pipeline per model:
#   1. Stop existing vLLM container
#   2. Start vLLM with the model (BNB4, reasoning-config, thinking enabled)
#   3. Run harmbench_generate (4 concurrent, 6144 max_tokens, 4096 thinking budget)
#   4. Run harmbench_classify
#   5. Run harmbench_classify score
#   6. Tear down vLLM
#
# After all HarmBench: run lm-eval for all models
#
set -euo pipefail

ABL_ROOT="/home/maxx/projects/abliterlitics"
cd "$ABL_ROOT"

VLLM_CONTAINER="vllm-harmbench"
LMEVAL_IMAGE="abliterlitics-lmeval:1.0.0"
OUTDIR="results/qwen36-27b/harmbench"
mkdir -p "$OUTDIR"

# Model definitions: name:relative_path
declare -A MODELS=(
    [base]="models/Qwen3.6-27B"
    [huihui]="models/Huihui-Qwen3.6-27B-abliterated"
    [aeon]="models/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16"
    [heretic]="models/Qwen3.6-27B-uncensored-heretic-v2"
    [hauhau]="models/Qwen3.6-27B-HauhauCS-Q8KP-recovered"
    [abliterix]="models/Qwen3.6-27B-abliterated-v2"
)

# Run order (base first to establish CoT baseline)
MODEL_ORDER=(base huihui aeon heretic hauhau abliterix)

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
error() { echo "[$(date '+%H:%M:%S')] ERROR: $*" >&2; }

# ── vLLM helpers ──

stop_vllm() {
    if docker ps -q -f "name=${VLLM_CONTAINER}" | grep -q .; then
        log "Stopping vLLM container..."
        docker stop "$VLLM_CONTAINER" >/dev/null 2>&1 || true
        docker rm "$VLLM_CONTAINER" >/dev/null 2>&1 || true
        sleep 5
    fi
}

start_vllm() {
    local model_path="$1"
    stop_vllm

    log "Starting vLLM with $(basename "$model_path")..."
    docker run -d \
        --name "$VLLM_CONTAINER" \
        --gpus '"device=0"' \
        --env NVIDIA_VISIBLE_DEVICES=0 \
        --env CUDA_VISIBLE_DEVICES=0 \
        -p 8080:8080 \
        -v "${ABL_ROOT}/${model_path}:/model:ro" \
        "$LMEVAL_IMAGE" \
        python3 -m vllm.entrypoints.openai.api_server \
            --model /model \
            --dtype bfloat16 \
            --quantization bitsandbytes \
            --load-format bitsandbytes \
            --trust-remote-code \
            --max-model-len 8192 \
            --gpu-memory-utilization 0.90 \
            --enforce-eager \
            --reasoning-parser qwen3 \
            --reasoning-config '{"reasoning_start_str": "<think}", "reasoning_end_str": "</think"}' \
            --port 8080 \
            --host 0.0.0.0

    # Wait for server ready
    log "Waiting for vLLM server..."
    for i in $(seq 1 120); do
        if curl -sf http://localhost:8080/v1/models >/dev/null 2>&1; then
            log "vLLM ready after ~$((i*5))s"
            return 0
        fi
        sleep 5
    done
    error "vLLM failed to start within 10 minutes"
    docker logs "$VLLM_CONTAINER" 2>&1 | tail -30
    return 1
}

# ── HarmBench pipeline for one model ──

run_harmbench_model() {
    local name="$1"
    local model_path="$2"
    local responses="${OUTDIR}/harmbench_${name}_responses.json"
    local classified="${OUTDIR}/harmbench_${name}_classified.json"
    local scores="${OUTDIR}/harmbench_${name}_scores.json"

    # Skip if already completed
    if [[ -f "$scores" ]]; then
        log "SKIP: ${name} already has scores"
        return 0
    fi

    # Start vLLM for this model
    start_vllm "$model_path"

    # Generate responses (resume if partial exists)
    log "Generating HarmBench responses for ${name}..."
    PYTHONPATH="$ABL_ROOT" python3 src/benchmark/harmbench_generate.py \
        --base-url http://localhost:8080 \
        --output "$responses" \
        --max-tokens 6144 \
        --concurrent 4 \
        --model-name "$name"

    if [[ ! -f "$responses" ]]; then
        error "Generation failed for ${name} — no responses file"
        return 1
    fi

    # Count results
    local total errors
    total=$(PYTHONPATH="$ABL_ROOT" python3 -c "
import json; d=json.load(open('${responses}'));
print(len(d['harmbench']))
")
    errors=$(PYTHONPATH="$ABL_ROOT" python3 -c "
import json; d=json.load(open('${responses}'));
print(sum(1 for r in d['harmbench'] if r and r.get('error')))
")
    log "Generated ${total} responses for ${name} (${errors} errors)"

    # Classify
    log "Classifying ${name}..."
    PYTHONPATH="$ABL_ROOT" python3 src/benchmark/harmbench_classify.py classify \
        --input "$responses" \
        --output "$classified" \
        --review-output "${OUTDIR}/harmbench_${name}_reviewed.json"

    # Score
    log "Scoring ${name}..."
    PYTHONPATH="$ABL_ROOT" python3 src/benchmark/harmbench_classify.py score \
        --classified "$classified" \
        --output "$scores"

    # Print quick summary
    PYTHONPATH="$ABL_ROOT" python3 -c "
import json
s = json.load(open('${scores}'))
print(f'  ASR: {s[\"harmful_count\"]}/{s[\"total\"]} = {float(s[\"asr_pct\"].rstrip(\"%\")):.1f}%')
"

    log "DONE: ${name}"
}

# ── Main ──

log "=== HarmBench Overnight Runner ==="
log "Models: ${MODEL_ORDER[*]}"
log ""

# Run HarmBench for each model
for name in "${MODEL_ORDER[@]}"; do
    path="${MODELS[$name]}"
    log ">>> Starting HarmBench for ${name} ($(basename "$path"))"
    run_harmbench_model "$name" "$path" || error "HarmBench FAILED for ${name}"
    log ""
done

# Tear down vLLM
stop_vllm
log "=== HarmBench complete for all models ==="

# Print summary
log "=== HarmBench Summary ==="
PYTHONPATH="$ABL_ROOT" python3 src/benchmark/harmbench_classify.py summary --dir "$OUTDIR"

# ── Now run lm-eval ──

log ""
log "=== Starting lm-eval benchmarks ==="
bash runners/run_lm_eval.sh --gpus 0 .

log "=== ALL DONE ==="
