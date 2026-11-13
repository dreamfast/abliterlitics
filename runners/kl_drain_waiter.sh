#!/usr/bin/env bash
# kl_drain_waiter.sh — poll until the qwen38-27b lm-eval queue drains, then launch KL.
#
# Launch (detached, lesson #28 pattern — survives agent-session teardown):
#   setsid nohup bash runners/kl_drain_waiter.sh < /dev/null >> comparisons/qwen38-27b/results/kl/kl_waiter.log 2>&1 &
#
# Drain conditions (ALL must hold, verified twice with a grace gap):
#   1. results/lm_eval/lm_eval_apostate.json exists (canonical, last model)
#   2. no `runners/run_lm_eval.sh` process alive
#   3. watchdog reached terminal state (DONE or FAIL marker)
#   4. no lmeval-image docker containers running
#   5. GPU 0 memory < 2000 MiB used
# Blocked guard: watchdog FAIL without apostate canonical => exit 3, never launch.
set -uo pipefail

ABL="/home/maxx/projects/abliterlitics"
COMP_DIR="${ABL}/comparisons/qwen38-27b"
KL_DIR="${COMP_DIR}/results/kl"
POLL=300
GRACE=240
MAX_WAIT_H=36
PIDFILE="${KL_DIR}/kl_waiter.pid"

mkdir -p "$KL_DIR"
log() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*"; }

if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; then
    log "already running (pid $(cat "$PIDFILE")) — exiting"
    exit 0
fi
echo $$ > "$PIDFILE"
log "waiter started (pid $$) — polling every ${POLL}s, max ${MAX_WAIT_H}h"

gpu0_used_mib() {
    docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 -e CUDA_VISIBLE_DEVICES=0 \
        --entrypoint nvidia-smi abliterlitics-forensics:1.0.0 \
        --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' '
}

drained() {
    [[ -f "${COMP_DIR}/results/lm_eval/lm_eval_apostate.json" ]] || { log "wait: apostate canonical missing"; return 1; }
    if pgrep -f 'bash .*runners/run_lm_eval\.sh' >/dev/null 2>&1; then log "wait: run_lm_eval.sh still alive"; return 1; fi
    if [[ ! -f "${COMP_DIR}/results/lm_eval_watchdog.DONE" && ! -f "${COMP_DIR}/results/lm_eval_watchdog.FAIL" ]]; then
        local wpid
        wpid="$(cat "${COMP_DIR}/results/lm_eval_watchdog.pid" 2>/dev/null || true)"
        if [[ -n "$wpid" ]] && kill -0 "$wpid" 2>/dev/null; then
            log "wait: watchdog (pid ${wpid}) active, no terminal marker"; return 1
        fi
        log "wait: watchdog process gone but no DONE/FAIL marker"; return 1
    fi
    if docker ps --format '{{.Image}}' | grep -q 'lmeval'; then log "wait: lmeval container still up"; return 1; fi
    local used
    used="$(gpu0_used_mib)"
    if [[ -z "$used" ]]; then log "wait: GPU query failed"; return 1; fi
    if (( used >= 2000 )); then log "wait: GPU0 holds ${used} MiB"; return 1; fi
    return 0
}

deadline=$(( $(date +%s) + MAX_WAIT_H * 3600 ))
while (( $(date +%s) < deadline )); do
    # Blocked guard: watchdog gave up and apostate never finished — never launch on a broken queue.
    if [[ -f "${COMP_DIR}/results/lm_eval_watchdog.FAIL" && ! -f "${COMP_DIR}/results/lm_eval/lm_eval_apostate.json" ]]; then
        log "BLOCKED: watchdog FAIL marker present and apostate canonical missing — NOT launching KL, manual review required"
        echo blocked > "${KL_DIR}/kl_waiter.exit"
        exit 3
    fi
    if drained; then
        log "QUEUE DRAINED — grace ${GRACE}s then re-verify"
        sleep "$GRACE"
        if drained; then
            log "DRAIN CONFIRMED — launching KL (skip-existing default, NO --force)"
            cd "$ABL" || exit 9
            echo $$ > "${KL_DIR}/kl_run.pid"   # runner runs as this process via exec-in-bg below
            nohup ./abliterlitics.sh kl comparisons/qwen38-27b/ >> "${KL_DIR}/kl_run.log" 2>&1 &
            kpid=$!
            echo "$kpid" > "${KL_DIR}/kl_run.pid"
            log "KL launched pid=${kpid} — babysitting"
            wait "$kpid"; rc=$?
            log "KL finished rc=${rc}"
            echo "$rc" > "${KL_DIR}/kl_waiter.exit"
            exit "$rc"
        fi
        log "drain state reverted during grace — back to polling"
    fi
    sleep "$POLL"
done

log "TIMEOUT after ${MAX_WAIT_H}h — NOT launching KL"
echo timeout > "${KL_DIR}/kl_waiter.exit"
exit 2
