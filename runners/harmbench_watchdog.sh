#!/usr/bin/env bash
# harmbench_watchdog.sh — keeps the harmbench queue alive through silent runner deaths.
# Pattern: runners/lm_eval_watchdog.sh (lesson #69). Launch:
#   setsid nohup bash runners/harmbench_watchdog.sh < /dev/null >> comparisons/<cmp>/results/harmbench_watchdog.log 2>&1 &
# Own file name contains no 'run_harmbench' string -> pgrep below cannot match self (lesson #68).

set -uo pipefail
ABL_ROOT="/home/maxx/projects/abliterlitics"
CMP_DIR="${ABL_ROOT}/comparisons/qwen38-27b"
CMP_NAME="qwen38-27b"
MASTER_LOG="${CMP_DIR}/results/harmbench_followup_master.log"
ARMS="ektome davidau rvn apostate_diode"
INTERVAL=600
MAX_RESTARTS=12
RESTARTS=0

log() { echo "$(date -u '+%F %T') [hb-watchdog] $*"; }

queue_done() {
    local out
    out=$(cd "${ABL_ROOT}" && ./abliterlitics.sh db pipeline-status "${CMP_DIR}/" 2>/dev/null)
    local pending=0 arm
    for arm in ${ARMS}; do
        if ! echo "${out}" | awk -v m="${arm}" '$1==m && $2=="harmbench" { if ($3=="completed") found=1 } END { exit found?0:1 }'; then
            pending=$((pending+1))
        fi
    done
    [ "${pending}" -eq 0 ]
}

runner_pids() { pgrep -f 'run_harmbench\.py' 2>/dev/null || true; }

stop_my_containers() {
    # Only containers named vllm-hb-* AND running our image (lesson #46 discipline).
    local c img
    for c in $(docker ps --format '{{.Names}}' | grep '^vllm-hb-'); do
        img=$(docker inspect "${c}" --format '{{.Config.Image}}' 2>/dev/null || echo unknown)
        if [[ "${img}" == abliterlitics-* ]]; then
            log "stopping ${c} (${img})"
            docker stop "${c}" >/dev/null 2>&1 || true
            docker rm "${c}" >/dev/null 2>&1 || true
        else
            log "REFUSING to stop ${c}: unexpected image ${img}"
        fi
    done
}

relaunch() {
    log "relaunching harmbench queue (restart ${RESTARTS}/${MAX_RESTARTS})"
    (cd "${ABL_ROOT}" && setsid nohup ./abliterlitics.sh harmbench "${CMP_DIR}/" < /dev/null >> "${MASTER_LOG}" 2>&1 &)
    sleep 120   # fresh queue needs minutes to boot vLLM before liveness means anything (lesson #78)
}

log "watchdog started: arms='${ARMS}' interval=${INTERVAL}s"
while true; do
    if queue_done; then
        log "DONE: all arms harmbench complete in DB"
        echo done > "${CMP_DIR}/results/harmbench_watchdog.done"
        exit 0
    fi

    pids=$(runner_pids)
    if [ -z "${pids}" ]; then
        RESTARTS=$((RESTARTS+1))
        if [ "${RESTARTS}" -gt "${MAX_RESTARTS}" ]; then
            log "FAIL: exceeded ${MAX_RESTARTS} restarts"
            echo fail > "${CMP_DIR}/results/harmbench_watchdog.fail"
            exit 1
        fi
        log "runner DEAD (pids empty). cleaning stale containers and relaunching"
        stop_my_containers
        relaunch
    else
        # Wedge note only (v0): heartbeat check on the live vllm container
        hb=$(docker ps --format '{{.Names}}' | grep '^vllm-hb-' | head -1)
        if [ -n "${hb}" ]; then
            beats=$(docker logs --since 25m "${hb}" 2>&1 | grep -c "Avg generation" || true)
            if [ "${beats}" -eq 0 ]; then
                log "NOTE: ${hb} has no engine heartbeats in 25m while runner alive (pids ${pids}) — possible wedge, not auto-restarting (classify phases are CPU-quiet)"
            fi
        fi
        log "ok: runner alive (pids ${pids})"
    fi
    sleep "${INTERVAL}"
done
