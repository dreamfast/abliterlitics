#!/usr/bin/env bash
# KL divergence runner — two-phase: collect base logits, then variant logits + KL
# Uses Docker forensics image for GPU access.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] <comparison_dir>

KL divergence analysis (2 phases):
  1. base      — Collect base model first-token logits
  2. variants  — Collect variant logits and compute KL divergence

Options:
   --gpu GPU          GPU index (default: auto-detect)
   --num-prompts N    Number of prompts from dataset (default: 100)
   --skip-existing    Skip existing results (default)
   --force            Overwrite existing results
   --dry-run          Show what would run without executing
   -h, --help         Show this help
EOF
    exit 0
}

GPU=""
NUM_PROMPTS=""
DRY_RUN=0
SKIP_EXISTING=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)           GPU="$2"; shift 2 ;;
        --backend)       shift 2 ;;  # consumed for CLI compat, KL always uses Docker
        --num-prompts)   NUM_PROMPTS="$2"; shift 2 ;;
        --skip-existing) SKIP_EXISTING=1; shift ;;
        --force)         SKIP_EXISTING=0; shift ;;
        --dry-run)       DRY_RUN=1; shift ;;
        -h|--help)       usage ;;
        *)               break ;;
    esac
done

COMP_DIR="${1:?Error: comparison_dir required. Use --help for usage.}"

# KL_DEVICE=cpu — run the entire KL stage in CPU-only containers: no
# --runtime=nvidia (zero GPU visibility), model loaded as REAL CPU tensors via
# the existing --device-map cpu CLI flag (no accelerate hooks, no meta-device
# streaming, no VRAM math). KL methodology (dataset, 1-token full-vocab
# log_softmax, batchmean, chat-template override) is untouched — only weight
# placement changes. KL_THREADS overrides the thread count (default: all
# physical cores via lscpu, fallback nproc). AGENTS.md #28 pattern: long runs
# must be launchable detached with files-only logging.
KL_DEVICE="${KL_DEVICE:-gpu}"
KL_THREADS="${KL_THREADS:-}"

load_comparison "$COMP_DIR"
GPU="${GPU:-$(detect_best_gpu)}"

if [[ "$KL_DEVICE" == "cpu" ]]; then
    if [[ -z "$KL_THREADS" ]]; then
        # All PHYSICAL cores (mawk-safe sed parse of lscpu; falls back to nproc).
        # NO `local` here — outside a function it crashes under set -e (lesson #37).
        _kl_cores="$(lscpu 2>/dev/null | sed -n 's/^Core(s) per socket:[[:space:]]*\([0-9]*\).*/\1/p' | head -1)"
        _kl_sockets="$(lscpu 2>/dev/null | sed -n 's/^Socket(s):[[:space:]]*\([0-9]*\).*/\1/p' | head -1)"
        KL_THREADS="$(( ${_kl_cores:-0} * ${_kl_sockets:-0} ))"
        [[ "$KL_THREADS" -le 0 ]] && KL_THREADS="$(nproc)"
    fi
    KL_DOCKER=(docker_run_cpu)
    KL_DOCKER_ENV=(-e "OMP_NUM_THREADS=${KL_THREADS}" -e "MKL_NUM_THREADS=${KL_THREADS}" -e "CUDA_VISIBLE_DEVICES=")
    # NOTE: --device-map cpu is appended to EXTRA_ARGS only AFTER its
    # declaration below (EXTRA_ARGS=() would otherwise wipe it under set -u).
    log "KL_DEVICE=cpu: CPU-only containers (no GPU runtime), --device-map cpu, torch threads=${KL_THREADS}"
else
    KL_DOCKER=(docker_run "$GPU")
    # expandable_segments: fights CUDA allocator fragmentation on the offload
    # path (the 30GiB-budget OOM showed 2.39 GiB "reserved but unallocated"
    # while a 2.37 GiB contiguous stream failed). Harmless elsewhere.
    KL_DOCKER_ENV=(-e "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
fi
COMPARISON_DIR="$(cd "$(dirname "${COMPARISON_DIR}")" && pwd)/$(basename "${COMPARISON_DIR}")"
BASE_DIR="$(cd "${BASE_DIR}" && pwd)"
for variant in $VARIANT_NAMES; do
    variant_slug="$(echo "$variant" | tr '.-' '__')"
    variant_path_var="VARIANT_PATH_${variant_slug}"
    eval "VARIANT_PATH_${variant_slug}=\"\$(cd \"${!variant_path_var}\" && pwd)\""
done
RESULTS_DIR="${COMPARISON_DIR}/results/kl"
mkdir -p "$RESULTS_DIR"

# Read the detected response prefix EXACTLY (preserving trailing newlines) —
# sets PREFIX_EXACT as a GLOBAL; does NOT print to stdout.
#
# WHY A GLOBAL: this function originally printed the prefix and was consumed
# via command substitution "$(_read_prefix_exact f)". Command substitution
# strips ALL trailing newlines from the substituted output — so the
# printf-sentinel trick inside the function (which preserves the newlines of
# `cat`) was undone one frame up, and "</think>\n\n" silently became
# "</think>". Variants then measured the newline-formatting position instead
# of the content position (KL ~10 vs ~0.05 — heretic_exact run, 2026-09-05,
# caught by the top-5 first-token sanity line). Never return trailing-newline
# strings through $(); assign a global instead.
_read_prefix_exact() {
    local f="$1" content
    if [[ -f "$f" ]]; then
        content="$(cat "$f"; printf x)"
        PREFIX_EXACT="${content%x}"
    else
        PREFIX_EXACT="none"
    fi
}

# NOTE on prefix passing for variant collects: variants receive
# --response-prefix thinkclose (NOT the literal string from
# response_prefix.txt). thinkclose resolves DETERMINISTICALLY from the render,
# and every arm mounts the BASE tokenizer, so the render — and therefore the
# resolved prefix — is byte-identical across arms (heretic's
# detect-on-base-apply-to-all semantics, preserved without string passing).
# The exact resolved string is recorded in each logits .pt's provenance, and
# the compute phase warns on base/variant prefix mismatch.

# CHAT TEMPLATE: NO override for this run (HERETIC-EXACT protocol,
# user-directed 2026-09-05; supersedes the kl-template-override decision and
# the no_thinking_chat_template.jinja approach, which was aborted before
# producing any artifacts). Spec = heretic v1.2.0 source (ref checkout
# /tmp/opencode/heretic_gh): src/heretic/model.py generate() renders
# [system="You are a helpful assistant.", user=<prompt>] through the
# tokenizer's OWN bundled template (config.py:271 default system prompt).
# The tokenizer is mounted from BASE for every arm below, so the render is
# byte-identical across arms — heretic-faithful AND template-neutral (the
# bundled variant templates never enter the render path). The measurement
# position (first CONTENT token after an empty closed think block) is set by
# the response prefix — see --response-prefix thinkclose in kl_divergence.py.
KL_TEMPLATE_ARGS=()

EXTRA_ARGS=()
# KL_DEVICE=cpu: load the model as real CPU tensors (existing CLI flag on the
# frozen collect path — plain string -> device_map={"": "cpu"}, no accelerate
# dispatch, no meta-device streaming).
if [[ "$KL_DEVICE" == "cpu" ]]; then
    EXTRA_ARGS+=(--device-map cpu)
fi
# CLI --num-prompts overrides comparison.json kl_num_prompts setting
if [[ -n "$NUM_PROMPTS" ]]; then
    EXTRA_ARGS+=(--num-prompts "$NUM_PROMPTS")
elif [[ -n "${KL_NUM_PROMPTS:-}" ]]; then
    EXTRA_ARGS+=(--num-prompts "$KL_NUM_PROMPTS")
fi

# Lesson #83 guard: a no-work skip path must never demote a 'completed' stage.
# Read-only best-effort check — on any error returns "no" (status write proceeds
# exactly as before this guard existed). DB_PATH is relative; requires the
# runner's cwd to be the repo root (same assumption as common.sh db_stage_*).
_kl_stage_completed() {
    python3 - "$1" "$2" "$3" <<'PYEOF'
import sqlite3, sys
try:
    conn = sqlite3.connect("file:abliterlitics.db?mode=ro", uri=True)
    row = conn.execute(
        "SELECT 1 FROM model_stages WHERE comparison=? AND model_name=? AND stage=? AND status='completed'",
        (sys.argv[1], sys.argv[2], sys.argv[3]),
    ).fetchone()
    print("yes" if row else "no")
except Exception:
    print("no")
PYEOF
}

# Phase 1: Collect base model logits
# NOTE: logits_base.pt is a frozen reference filename read by the (frozen) KL
# compute code — do NOT rename it. The DB stage label uses BASE_NAME for
# consistency with the models table (registered under directory basename).
BASE_LOGITS="${RESULTS_DIR}/logits_base.pt"
if result_exists "$BASE_LOGITS"; then
    log "Skipping (exists): base logits"
    if [[ $DRY_RUN -eq 0 ]] && [[ "$(_kl_stage_completed "$COMPARISON_NAME" "${BASE_NAME}" "kl")" != "yes" ]]; then
        db_stage_skip "$COMPARISON_NAME" "${BASE_NAME}" "kl"
    fi
else
    log "Phase 1: Collecting base logits from ${BASE_DIR}"
    if [[ $DRY_RUN -eq 0 ]]; then
        db_stage_start "$COMPARISON_NAME" "${BASE_NAME}" "kl"
    fi
    if [[ $DRY_RUN -eq 1 ]]; then
        log "  [dry-run] ${KL_DOCKER[*]} ${FORENSICS_IMAGE} ${KL_DOCKER_ENV[*]:-} kl_divergence.py collect --model /model --tokenizer /tokenizer --output /results/logits_base.pt ${KL_TEMPLATE_ARGS[*]:-} ${EXTRA_ARGS[*]:-}"
    else
        "${KL_DOCKER[@]}" "$FORENSICS_IMAGE" "${KL_DOCKER_ENV[@]}" \
            -v "${BASE_DIR}:/model:ro" \
            -v "${BASE_DIR}:/tokenizer:ro" \
            -v "${COMPARISON_DIR}:/comparison:ro" \
            -v "${RESULTS_DIR}:/results" \
            -v "${ABL_ROOT}:/app:ro" \
            -e PYTHONPATH=/app \
            -- \
            /app/src/kl/kl_divergence.py collect \
                --model /model \
                --tokenizer /tokenizer \
                --output /results/logits_base.pt \
                --response-prefix thinkclose \
                --save-prefix /results/response_prefix.txt \
                "${KL_TEMPLATE_ARGS[@]}" \
                "${EXTRA_ARGS[@]}"
    fi
    if [[ $DRY_RUN -eq 0 ]]; then
        db_stage_complete "$COMPARISON_NAME" "${BASE_NAME}" "kl" --result-path "$BASE_LOGITS"
    fi
fi

# Phase 2: Collect variant logits + compute KL divergence
for variant in $VARIANT_NAMES; do
    # Lesson #83 guard: only mark 'running' when the stage isn't already
    # 'completed' — the skip paths below never restore a demoted status.
    if [[ $DRY_RUN -eq 0 ]] && [[ "$(_kl_stage_completed "$COMPARISON_NAME" "$variant" "kl")" != "yes" ]]; then
        db_stage_start "$COMPARISON_NAME" "$variant" "kl"
    fi
    variant_path_var="VARIANT_PATH_$(echo "$variant" | tr '.-' '__')"
    variant_path="${!variant_path_var}"
    variant_logits="${RESULTS_DIR}/logits_${variant}.pt"

    # Phase 2a: Collect variant logits
    if result_exists "$variant_logits"; then
        log "Skipping (exists): logits for ${variant}"
    else
        log "Phase 2a: Collecting logits for variant ${variant}"

        # Detect LoRA adapter for this variant (AGENTS.md lesson #35)
        variant_slug="$(echo "$variant" | tr '.-' '__')"
        lora_adapter_env="VARIANT_LORA_ADAPTER_${variant_slug}"
        lora_adapter_path="${!lora_adapter_env:-}"

        lora_mount_args=()
        lora_cmd_args=()
        if [[ -n "$lora_adapter_path" ]]; then
            log "  LoRA mode: adapter=${lora_adapter_path}"
            lora_mount_args=(-v "${lora_adapter_path}:/lora_adapter:ro")
            lora_cmd_args=(--lora-adapter /lora_adapter)
        fi

        if [[ $DRY_RUN -eq 1 ]]; then
            log "  [dry-run] ${KL_DOCKER[*]} ${FORENSICS_IMAGE} ${KL_DOCKER_ENV[*]:-} kl_divergence.py collect --model /model --tokenizer /tokenizer --output /results/logits_${variant}.pt ${KL_TEMPLATE_ARGS[*]:-} ${EXTRA_ARGS[*]:-}"
        else
            # For LoRA variants, peft must be pip-installed at runtime (not in
            # the forensics image). Chain install + run via bash -c.
            if [[ -n "$lora_adapter_path" ]]; then
                "${KL_DOCKER[@]}" "$FORENSICS_IMAGE" "${KL_DOCKER_ENV[@]}" \
                    -v "${variant_path}:/model:ro" \
                    "${lora_mount_args[@]}" \
                    -v "${BASE_DIR}:/tokenizer:ro" \
                    -v "${COMPARISON_DIR}:/comparison:ro" \
                    -v "${RESULTS_DIR}:/results" \
                    -v "${ABL_ROOT}:/app:ro" \
                    -e PYTHONPATH=/app \
                    --entrypoint "" \
                    -- \
                    bash -c \
                    "pip install peft -q 2>&1 | tail -1; exec python3 /app/src/kl/kl_divergence.py collect --model /model --tokenizer /tokenizer --output /results/logits_${variant}.pt --response-prefix thinkclose --lora-adapter /lora_adapter ${KL_TEMPLATE_ARGS[*]:-} ${EXTRA_ARGS[*]:-}"
            else
                "${KL_DOCKER[@]}" "$FORENSICS_IMAGE" "${KL_DOCKER_ENV[@]}" \
                    -v "${variant_path}:/model:ro" \
                    "${lora_mount_args[@]}" \
                    -v "${BASE_DIR}:/tokenizer:ro" \
                    -v "${COMPARISON_DIR}:/comparison:ro" \
                    -v "${RESULTS_DIR}:/results" \
                    -v "${ABL_ROOT}:/app:ro" \
                    -e PYTHONPATH=/app \
                    -- \
                    /app/src/kl/kl_divergence.py collect \
                        --model /model \
                        --tokenizer /tokenizer \
                        --output "/results/logits_${variant}.pt" \
                        --response-prefix thinkclose \
                        "${KL_TEMPLATE_ARGS[@]}" \
                        "${EXTRA_ARGS[@]}"
            fi
        fi
    fi

    # Phase 2b: Compute KL divergence
    out_file="${RESULTS_DIR}/kl_${variant}.json"
    if result_exists "$out_file"; then
        log "Skipping (exists): KL for ${variant}"
        continue
    fi

    log "Phase 2b: Computing KL divergence for variant ${variant}"
    if [[ $DRY_RUN -eq 1 ]]; then
        log "  [dry-run] ${KL_DOCKER[*]} ${FORENSICS_IMAGE} ${KL_DOCKER_ENV[*]:-} kl_divergence.py compute --base-logits /results/logits_base.pt --variant-logits /results/logits_${variant}.pt"
        continue
    fi

    "${KL_DOCKER[@]}" "$FORENSICS_IMAGE" "${KL_DOCKER_ENV[@]}" \
        -v "${RESULTS_DIR}:/results" \
        -v "${ABL_ROOT}:/app:ro" \
        -e PYTHONPATH=/app \
        -- \
        /app/src/kl/kl_divergence.py compute \
            --base-logits /results/logits_base.pt \
            --variant-logits "/results/logits_${variant}.pt" \
            --variant-label "${variant}" \
            --output "/results/kl_${variant}.json"
    if [[ $DRY_RUN -eq 0 ]]; then
        db_stage_complete "$COMPARISON_NAME" "$variant" "kl" --result-path "$out_file"
    fi
done

# Import results to DB
if [[ $DRY_RUN -eq 0 ]]; then
    db_import_results "$COMP_DIR"
fi

log "KL analysis complete"
