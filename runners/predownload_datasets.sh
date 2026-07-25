#!/usr/bin/env bash
# ============================================================================
# Pre-download all lm-eval benchmark datasets to local cache
# ============================================================================
#
# Run this ONCE on the host before doing benchmark re-runs. After this,
# the lm-eval runner can use HF_DATASETS_OFFLINE=1 and never re-download.
#
# USAGE:
#   bash runners/predownload_datasets.sh
#
# REQUIREMENTS:
#   - HF authenticated on host (hf auth login, or token in
#     ~/.cache/huggingface/token)
#   - GPQA terms accepted at https://huggingface.co/datasets/Idavidrein/gpqa
#   - Docker with abliterlitics-lmeval:1.0.0 image
#
# WHAT IT DOES:
#   For each task, runs lm_eval --limit 1 with gpt2 (tiny dummy model) to
#   trigger dataset download via lm-eval's own loader. This guarantees we
#   get the exact revision/preprocessing the task expects.
#
# AFFECTS:
#   - Writes to .cache/hf/ (mounted into container at /root/.cache/huggingface)
#   - Future lm-eval runs will use this cache instead of re-downloading
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE="abliterlitics-lmeval:1.0.0"

cd "${PROJECT_ROOT}"

# ============================================================================
# Ensure host's user-installed binaries are on PATH (hf CLI lives at ~/.local/bin)
# ============================================================================
export PATH="${HOME}/.local/bin:${PATH}"

# ============================================================================
# Verify hf CLI is available and authenticated
# ============================================================================
if ! command -v hf >/dev/null 2>&1; then
    echo "✗ 'hf' CLI not found on PATH (looked in ~/.local/bin and standard locations)."
    echo "  Install with: pip install --user -U 'huggingface_hub[cli]'"
    exit 1
fi

echo "✓ hf CLI: $(which hf)"
HF_WHOAMI=$(hf auth whoami 2>&1) || HF_WHOAMI=""
if [[ -z "${HF_WHOAMI}" ]]; then
    echo "✗ hf CLI is not authenticated. Run: hf auth login"
    exit 1
fi
echo "✓ Authenticated as: ${HF_WHOAMI}"
echo ""

# ============================================================================
# Detect HF token on host (hf CLI stores at $HF_HOME/token, default ~/.cache/huggingface/token)
# ============================================================================
HF_TOKEN_ARG=()
TOKEN_FILE=""
HF_HOME_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
for candidate in \
    "${HF_HOME_DIR}/token" \
    "$HOME/.cache/huggingface/token" \
    "$HOME/.huggingface/token"; do
    if [[ -f "$candidate" ]]; then
        TOKEN_FILE="$candidate"
        break
    fi
done

if [[ -n "$TOKEN_FILE" ]]; then
    echo "✓ Found HF token at: ${TOKEN_FILE}"
    # Mount the token file directly into the container at the path hf CLI expects
    HF_TOKEN_ARG=(-v "${TOKEN_FILE}:/root/.cache/huggingface/token:ro")
elif [[ -n "${HF_TOKEN:-}" ]]; then
    echo "✓ Using HF_TOKEN from environment"
    HF_TOKEN_ARG=(-e "HF_TOKEN=${HF_TOKEN}")
else
    echo "⚠️  hf CLI is authenticated but no token file found at standard locations."
    echo "   Falling back to extracting token via 'hf auth token'."
    TOKEN_VALUE=$(hf auth token 2>/dev/null) || TOKEN_VALUE=""
    if [[ -n "$TOKEN_VALUE" ]]; then
        echo "✓ Extracted token via 'hf auth token'"
        HF_TOKEN_ARG=(-e "HF_TOKEN=${TOKEN_VALUE}")
    else
        echo "✗ Could not extract token. GPQA will fail."
        echo "  Continuing anyway — public datasets will still download."
    fi
fi
echo ""

# ============================================================================
# Tasks to pre-download
# ============================================================================
# Current suite (already cached, but include for verification)
LEGACY_TASKS=(
    "gsm8k"
    "truthfulqa"
)

# New suite (Open LLM Leaderboard v2 + kept tasks)
NEW_TASKS=(
    "leaderboard_mmlu_pro"
    "leaderboard_gpqa_diamond"
    "leaderboard_bbh"
    "leaderboard_musr"
    "leaderboard_ifeval"
)

# Supplementary v1 tasks for delta-detection forensic signal.
# These are "saturated" by OLL v2 ranking standards but still detect
# abliteration damage: LAMBADA (fluency), PIQA (physical common sense),
# Winogrande (pronoun/common-sense), HellaSwag (scene completion),
# ARC-Challenge (grade-school science).
FORENSIC_SUPPLEMENT_TASKS=(
    "hellaswag"
    "winogrande"
    "arc_challenge"
    "piqa"
    "lambada_openai"
)

# HumanEval (generative code task). Requires human_eval package for
# execution-based pass@k verification. Added for coding-damage detection —
# abliteration frequently damages precise instruction-following needed for code.
CODE_TASKS=(
    "humaneval"
)

# Optional: MATH Lvl 5 (skipped from default suite but available)
# NEW_TASKS+=("leaderboard_math_algebra_hard")

ALL_TASKS=("${LEGACY_TASKS[@]}" "${NEW_TASKS[@]}" "${FORENSIC_SUPPLEMENT_TASKS[@]}" "${CODE_TASKS[@]}")

echo "============================================"
echo "Pre-downloading ${#ALL_TASKS[@]} lm-eval datasets"
echo "============================================"
echo "Project: ${PROJECT_ROOT}"
echo "Image:   ${IMAGE}"
echo "Cache:   ${PROJECT_ROOT}/.cache/hf/"
echo "Tasks:   ${ALL_TASKS[*]}"
echo "Started: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo ""

# Pre-create cache dir
mkdir -p "${PROJECT_ROOT}/.cache/hf"

# Track results
declare -a SUCCEEDED=()
declare -a FAILED=()

for task in "${ALL_TASKS[@]}"; do
    echo "--------------------------------------------"
    echo "Downloading dataset for: ${task}"
    echo "--------------------------------------------"

    if docker run --rm \
        --name "predownload-${task}" \
        -e "HF_HOME=/root/.cache/huggingface" \
        -e "HF_HUB_ENABLE_HFTRANSFER=0" \
        -e "HF_ALLOW_CODE_EVAL=1" \
        "${HF_TOKEN_ARG[@]}" \
        -v "${PROJECT_ROOT}/.cache/hf:/root/.cache/huggingface" \
        "${IMAGE}" \
        bash -c "
            set -uo pipefail
            # IFEval task needs langdetect (not bundled in lm-eval base install).
            # HumanEval task needs the human_eval package for execution-based pass@k.
            # Install missing extras on first run. Quick no-op if already present.
            if [[ '${task}' == *'ifeval'* ]]; then
                echo '  [container] Installing lm-eval[ifeval] extra (langdetect)...'
                pip install --quiet 'lm-eval[ifeval]' 2>&1 | tail -3
            fi
            if [[ '${task}' == 'humaneval' ]]; then
                echo '  [container] Installing human_eval package for pass@k verification...'
                pip install --quiet 'human_eval' 2>&1 | tail -3
            fi
            echo '  [container] lm-eval version:' \$(python3 -c 'import lm_eval; print(lm_eval.__version__)' 2>&1)
            echo '  [container] HF token set:' \${HF_TOKEN:+yes}\${HF_TOKEN:-no}
            echo '  [container] Triggering dataset load via lm_eval --limit 1 ...'
            # Build unsafe-code flag for humaneval task
            UNSAFE_FLAG=""
            if [[ '${task}' == 'humaneval' ]]; then
                UNSAFE_FLAG="--confirm_run_unsafe_code"
            fi
            # Use SmolLM2-360M (8192 ctx) instead of gpt2 (1024 ctx) because
            # IFEval needs to generate ~1280 tokens and gpt2 cannot fit that.
            # SmolLM2 is ~720MB, downloads in seconds, works for all tasks.
            python3 -m lm_eval --model hf \
                --model_args pretrained=HuggingFaceTB/SmolLM2-360M,dtype=float32 \
                --tasks '${task}' \
                --limit 1 \
                --batch_size 1 \
                ${UNSAFE_FLAG} \
                --output_path /tmp/results/ 2>&1 | tail -20
        " > "/tmp/predownload_${task}.log" 2>&1; then

        echo "  ✓ SUCCESS: ${task}"
        SUCCEEDED+=("$task")
    else
        echo "  ✗ FAILED: ${task} (see /tmp/predownload_${task}.log)"
        echo ""
        echo "  Last 10 lines of error output:"
        tail -10 "/tmp/predownload_${task}.log" | sed 's/^/    /'
        FAILED+=("$task")
    fi
    echo ""
done

# ============================================================================
# Summary
# ============================================================================
echo "============================================"
echo "PRE-DOWNLOAD SUMMARY"
echo "============================================"
echo "Succeeded (${#SUCCEEDED[@]}):"
for t in "${SUCCEEDED[@]}"; do echo "  ✓ $t"; done
echo ""
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "Failed (${#FAILED[@]}):"
    for t in "${FAILED[@]}"; do echo "  ✗ $t"; done
    echo ""
    echo "Check /tmp/predownload_<task>.log for details."
fi

echo ""
echo "============================================"
echo "Cache contents after pre-download:"
echo "============================================"
ls -la "${PROJECT_ROOT}/.cache/hf/datasets/" 2>/dev/null || echo "No datasets directory"
echo ""
echo "Total cache size:"
du -sh "${PROJECT_ROOT}/.cache/hf/" 2>&1

echo ""
echo "============================================"
echo "Next steps:"
echo "============================================"
echo "1. If any failed, check the log and fix (usually HF auth for gated datasets)"
echo "2. The lm-eval runner will automatically use these cached datasets."
echo "3. For fully offline operation, the runner sets HF_DATASETS_OFFLINE=1"
echo "   (after this pre-download succeeds, no network access needed for datasets)."
echo ""
echo "Completed: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
