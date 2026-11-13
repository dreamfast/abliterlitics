#!/usr/bin/env bash
# lm_eval_watchdog.sh — stability watchdog for long lm-eval queues.
#
# Born from the qwen38-27b incident (2026-08-31): base Phase 1 died at 03:57 UTC
# to a native vLLM EngineCore SEGFAULT and sat unnoticed for hours because
# nothing was monitoring the queue. This watchdog is the "never again" layer.
#
# What it does (every CHECK_INTERVAL seconds):
#   1. Queue-alive check (runner + wrapper PIDs, scoped to this comparison).
#   2. Stall detection:
#        - server log (<model>_vllm_server.log) silent > SERVER_STALL_SEC  → engine dead/hung
#          (vLLM logs a heartbeat line every ~10s while alive, so silence = dead)
#        - client log (<model>_phase*.log) silent > CLIENT_STALL_SEC while server alive
#          → lm-eval client hung
#      On stall: graceful kill of OUR runner processes + OUR containers
#      (inspect-verified: image contains 'lmeval' AND mounts this comparison's
#      results/lm_eval — never touches anything else, lesson #46), then relaunch.
#   3. Crash-scan: watches the newest vllm server log for new
#      EngineDead/Segfault lines. The runner self-heals past a single model
#      crash (|| true in the model loop), so we LOG these and let the queue
#      continue; the crashed model is retried by the final pass below.
#   4. Final pass (the actual point): when the queue exits, any model without a
#      canonical results/lm_eval/lm_eval_<model>.json is re-run by relaunching
#      `./abliterlitics.sh lm-eval` (skip-existing default ON → finished models
#      skip in seconds; crashed models resume FROM THE RESPONSE CACHE,
#      lessons #14/#31 — content-hashed keys, config-change tolerant).
#   5. Budgets: max MODEL_RESTART_MAX relaunches per model (then it is excluded
#      via --models filter), max QUEUE_RESTART_MAX queue relaunches, hard
#      GLOBAL_LAUNCH_MAX flap guard. Exhaustion → FAIL marker + exit.
#
# Survival: launched with setsid+nohup, all output to FILES (never stdout
# pipes — lesson #28b SIGPIPE death), stdin from /dev/null. Survives
# agent-session teardown. State in plain files under lm_eval_watchdog_state/.
#
# Safety rails (lesson #46): every docker stop is preceded by a successful
# docker inspect proving image + mount identity. docker stop (graceful) only;
# docker kill escalation only after a re-inspect still verifies ownership.
# Never touches GPU 1, llmdev, comfyui-4090, llama-cpp-4090, or any other
# container. pgrep/pkill patterns are scoped with the comparison slug and can
# never match the watchdog itself (different name; pgrep excludes self).
#
# Usage:  setsid nohup bash runners/lm_eval_watchdog.sh >> <log> 2>&1 < /dev/null &
# Config via env: CHECK_INTERVAL, SERVER_STALL_SEC, CLIENT_STALL_SEC.

set -uo pipefail

# Repo root: derived from this script's location (never hardcode an absolute
# home dir — audit 2026-10-14 GitHub-readiness). ABL_ROOT env override wins.
REPO="${ABL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
COMP_DIR="${REPO}/comparisons/qwen38-27b"
COMP_SLUG="qwen38-27b"
RESULTS_LMEVAL="${COMP_DIR}/results/lm_eval"
LOG_DIR="${COMP_DIR}/results/lm_eval_logs"
STATE_DIR="${COMP_DIR}/results/lm_eval_watchdog_state"
WATCHDOG_LOG="${COMP_DIR}/results/lm_eval_watchdog.log"
FAIL_MARKER="${COMP_DIR}/results/lm_eval_watchdog.FAIL"
DONE_MARKER="${COMP_DIR}/results/lm_eval_watchdog.DONE"
PID_FILE="${COMP_DIR}/results/lm_eval_watchdog.pid"

CHECK_INTERVAL="${CHECK_INTERVAL:-600}"
SERVER_STALL_SEC="${SERVER_STALL_SEC:-1200}"   # 20 min
CLIENT_STALL_SEC="${CLIENT_STALL_SEC:-2700}"   # 45 min
QUEUE_RESTART_MAX="${QUEUE_RESTART_MAX:-3}"
MODEL_RESTART_MAX="${MODEL_RESTART_MAX:-3}"
GLOBAL_LAUNCH_MAX="${GLOBAL_LAUNCH_MAX:-6}"
LAUNCH_FAILURE_MAX=3

mkdir -p "$STATE_DIR/failed"

log() { echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') [watchdog $$] $*" >> "$WATCHDOG_LOG"; }

# Duplicate-launch guard: two watchdogs would both kill/relaunch the queue and
# race vLLM servers onto GPU 0. Check the pid file BEFORE writing it; refuse if
# it names a LIVE process running THIS script (stale/dead pid or foreign pid →
# take over). Also stops a plain `source` of this file from clobbering the
# real instance's pid file.
if [[ -f "$PID_FILE" ]]; then
    oldpid=$(cat "$PID_FILE" 2>/dev/null)
    if [[ -n "$oldpid" && "$oldpid" != "$$" ]] && tr '\0' ' ' < "/proc/${oldpid}/cmdline" 2>/dev/null | grep -q "lm_eval_watchdog\.sh"; then
        echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') [watchdog $$] REFUSED: watchdog already running as pid ${oldpid} (pid file $PID_FILE)" >> "$WATCHDOG_LOG"
        exit 0
    fi
fi
echo $$ > "$PID_FILE"
log "watchdog started (pid $$, interval ${CHECK_INTERVAL}s, server_stall ${SERVER_STALL_SEC}s, client_stall ${CLIENT_STALL_SEC}s)"

# ---------------------------------------------------------------- helpers ---

queue_alive() {
    pgrep -f "run_lm_eval\.sh.*${COMP_SLUG}" >/dev/null 2>&1 && return 0
    pgrep -f "abliterlitics\.sh lm-eval.*${COMP_SLUG}" >/dev/null 2>&1 && return 0
    return 1
}

our_containers() {
    # Only containers that PASS inspection: image contains 'lmeval' AND mounts
    # this comparison's results/lm_eval. Anything else is never touched.
    local cid img
    for cid in $(docker ps -q 2>/dev/null); do
        img=$(docker inspect -f '{{.Config.Image}}' "$cid" 2>/dev/null) || continue
        [[ "$img" == *lmeval* ]] || continue
        if docker inspect -f '{{range .Mounts}}{{.Source}}
{{end}}' "$cid" 2>/dev/null | grep -qF "${COMP_DIR}/results/lm_eval"; then
            echo "$cid"
        fi
    done
}

stop_queue() {
    local reason="$1"
    log "STOP: $reason — terminating queue"
    # Scoped patterns: comparison slug included; watchdog cmdline can't match.
    pkill -TERM -f "abliterlitics\.sh lm-eval.*${COMP_SLUG}" 2>/dev/null || true
    pkill -TERM -f "run_lm_eval\.sh.*${COMP_SLUG}" 2>/dev/null || true
    sleep 10
    pkill -KILL -f "run_lm_eval\.sh.*${COMP_SLUG}" 2>/dev/null || true
    local cid n=0
    for cid in $(our_containers); do
        n=$((n+1))
        log "STOP: docker stop (verified ours: $(docker inspect -f '{{.Name}} {{.Config.Image}}' "$cid" 2>/dev/null))"
        docker stop -t 60 "$cid" >/dev/null 2>&1 || true
    done
    [[ $n -gt 0 ]] && sleep 5
    # Escalate only after re-verify (lesson #46)
    for cid in $(our_containers); do
        log "STOP: escalating to docker kill (re-verified ours: $cid)"
        docker kill "$cid" >/dev/null 2>&1 || true
    done
    sleep 5
}

model_list() {
    python3 - "$COMP_DIR/comparison.json" <<'PY' 2>/dev/null
import json, sys, os
c = json.load(open(sys.argv[1]))
print(os.path.basename(c["base"]))
for v in c["variants"]:
    print(v)
PY
}

missing_models() {
    local m
    while IFS= read -r m; do
        [[ -z "$m" ]] && continue
        [[ -f "${RESULTS_LMEVAL}/lm_eval_${m}.json" ]] || echo "$m"
    done < <(model_list)
}

attempt_file() { echo "${STATE_DIR}/attempts.$(echo "$1" | tr -c 'A-Za-z0-9' '_')"; }
attempt_get()  { local f; f=$(attempt_file "$1"); [[ -f "$f" ]] && cat "$f" || echo 0; }
attempt_inc()  { local f; f=$(attempt_file "$1"); echo $(( $(attempt_get "$1") + 1 )) > "$f"; }

is_failed_model() { [[ -f "${STATE_DIR}/failed/$1" ]]; }

newest_log() {  # $1 = glob pattern → newest path or ""
    find "$LOG_DIR" -name "$1" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-
}
age_of() {  # $1 = path → age seconds, or -1 if missing
    [[ -n "$1" && -f "$1" ]] || { echo -1; return; }
    echo $(( $(date +%s) - $(stat -c %Y "$1") ))
}

active_model() {  # from newest phase log name
    local p; p=$(newest_log '*_phase*.log')
    [[ -z "$p" ]] && { echo "?"; return; }
    basename "$p" | sed -E 's/_phase[0-9]+\.log$//'
}

crash_scan() {
    # Count NEW EngineDead/Segfault lines in the newest server log since last scan.
    local f; f=$(newest_log '*_vllm_server.log')
    [[ -z "$f" ]] && return
    local key off_file off size
    key=$(basename "$f" | tr -c 'A-Za-z0-9' '_')
    off_file="${STATE_DIR}/offset.${key}"
    off=0; [[ -f "$off_file" ]] && off=$(cat "$off_file")
    size=$(stat -c %s "$f" 2>/dev/null) || return
    [[ $size -lt $off ]] && off=0   # log truncated/rotated
    if [[ $size -gt $off ]]; then
        local hits
        hits=$(tail -c +"$((off+1))" "$f" 2>/dev/null | grep -cE "EngineDeadError|Segfault encountered" || true)
        if [[ ${hits:-0} -gt 0 ]]; then
            log "MODEL_CRASH: ${hits} EngineDead/Segfault line(s) in $(basename "$f") — runner self-progresses; crashed model retried at final pass (response-cache resume)"
        fi
    fi
    echo "$size" > "$off_file"
}

progress_line() {
    # Last tqdm-ish progress token from the newest phase log (monitoring aid).
    local f; f=$(newest_log '*_phase*.log')
    [[ -z "$f" ]] && return
    python3 - "$f" <<'PY' 2>/dev/null
import re, sys
data = open(sys.argv[1], 'rb').read()[-6000:].decode('utf-8', 'ignore')
pats = [r'Requesting API:\s+\d+%[^\r\n]*', r'Checking cached requests:\s+\d+%[^\r\n]*']
for p in pats:
    m = re.findall(p, data)
    if m:
        print(m[-1][:120]); break
PY
}

launch_queue() {
    local models_arg="$1"
    (
        cd "$REPO" || exit 1
        if [[ -n "$models_arg" ]]; then
            setsid nohup ./abliterlitics.sh lm-eval --models "$models_arg" "$COMP_DIR" >> "$WATCHDOG_LOG" 2>&1 < /dev/null &
        else
            setsid nohup ./abliterlitics.sh lm-eval "$COMP_DIR" >> "$WATCHDOG_LOG" 2>&1 < /dev/null &
        fi
    )
    sleep 20
    if queue_alive; then
        log "LAUNCH: queue relaunched (scope: ${models_arg:-all models})"
        return 0
    fi
    log "LAUNCH: queue NOT detected 20s after relaunch"
    return 1
}

write_fail() {
    local msg="$1"
    cat > "$FAIL_MARKER" <<EOF
$(date -u '+%Y-%m-%dT%H:%M:%SZ') — lm-eval watchdog gave up.

$msg

Manual recovery:
  cd $REPO
  ./abliterlitics.sh lm-eval $COMP_DIR
(response cache makes finished models skip in seconds; crashed models resume
 from results/lm_eval/lm_eval_cache/<model>/ — see AGENTS.md lessons #14/#31)
EOF
    log "FAIL: watchdog exiting. $msg"
}

# ------------------------------------------------------------------- main ---

while true; do
    if queue_alive; then
        srv=$(newest_log '*_vllm_server.log')
        cli=$(newest_log '*_phase*.log')
        srv_age=$(age_of "$srv"); cli_age=$(age_of "$cli")
        act=$(active_model)
        prog=$(progress_line)
        crash_scan

        # Stale-log guard (2026-09-06 incident): a watchdog started (or a queue
        # relaunched) while container setup has not yet created fresh logs sees
        # the PREVIOUS run's logs as newest — ages > stall thresholds → false
        # STALL kill of a perfectly healthy fresh queue. If the newest log's
        # mtime predates the current queue process start, it carries no signal
        # about THIS queue incarnation; skip stall judgment this cycle.
        stale_note=""
        qpid=$(pgrep -f "run_lm_eval\.sh.*${COMP_SLUG}" 2>/dev/null | head -1)
        qstart=0
        [[ -n "$qpid" ]] && qstart=$(stat -c %Y "/proc/$qpid" 2>/dev/null || echo 0)
        srv_stale=0; cli_stale=0
        if [[ "$srv_age" -ge 0 && "$qstart" -gt 0 ]] && [[ "$(stat -c %Y "$srv" 2>/dev/null || echo 0)" -lt "$qstart" ]]; then
            srv_stale=1; stale_note="server-log"
        fi
        if [[ "$cli_age" -ge 0 && "$qstart" -gt 0 ]] && [[ "$(stat -c %Y "$cli" 2>/dev/null || echo 0)" -lt "$qstart" ]]; then
            cli_stale=1; stale_note="${stale_note:+${stale_note}+}phase-log"
        fi
        [[ -n "$stale_note" ]] && log "NOTE: newest ${stale_note} predates queue launch (pid ${qpid:-?}) — stale from previous run; skipping stall judgment"

        if [[ $srv_age -ge 0 && $srv_age -gt $SERVER_STALL_SEC && $srv_stale -eq 0 ]]; then
            a=$(attempt_get "$act"); attempt_inc "$act"
            log "STALL(server): ${act} server log silent ${srv_age}s (> ${SERVER_STALL_SEC}s) — restart #$((a+1))"
            stop_queue "server heartbeat silent ${srv_age}s while runner alive (model: ${act})"
            if launch_queue ""; then :; else
                log "LAUNCH failure after stall (will retry next cycle)"
            fi
        elif [[ $cli_age -ge 0 && $cli_age -gt $CLIENT_STALL_SEC && $cli_stale -eq 0 ]]; then
            a=$(attempt_get "$act"); attempt_inc "$act"
            log "STALL(client): ${act} phase log silent ${cli_age}s (> ${CLIENT_STALL_SEC}s) while server alive — restart #$((a+1))"
            stop_queue "client progress silent ${cli_age}s while server alive (model: ${act})"
            if launch_queue ""; then :; else
                log "LAUNCH failure after stall (will retry next cycle)"
            fi
        else
            log "OK: queue alive, active=${act}, server-log age=${srv_age}s, phase-log age=${cli_age}s ${prog:+| ${prog}}"
        fi

    else
        # ---- queue not alive: orphan cleanup + final-pass accounting ----
        orphans=$(our_containers)
        if [[ -n "$orphans" ]]; then
            log "ORPHAN: runner dead but lm-eval container(s) alive — stopping (verified ours)"
            for cid in $orphans; do
                log "ORPHAN: docker stop $(docker inspect -f '{{.Name}} {{.Config.Image}}' "$cid" 2>/dev/null)"
                docker stop -t 60 "$cid" >/dev/null 2>&1 || true
            done
            sleep 5
            for cid in $(our_containers); do
                log "ORPHAN: escalating to docker kill (re-verified ours: $cid)"
                docker kill "$cid" >/dev/null 2>&1 || true
            done
        fi

        missing=$(missing_models | sort -u)
        if [[ -z "$missing" ]]; then
            log "DONE: all models have canonical results — queue complete"
            echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') — all models complete (canonical files present)" > "$DONE_MARKER"
            exit 0
        fi

        pending=""
        failed_now=""
        for m in $missing; do
            if is_failed_model "$m"; then continue; fi
            a=$(attempt_get "$m")
            if [[ $a -ge $MODEL_RESTART_MAX ]]; then
                touch "${STATE_DIR}/failed/$m"
                failed_now="$failed_now $m"
                log "MODEL_FAIL: ${m} still missing after ${a} relaunch attempts — excluded from future scope"
            else
                pending="$pending $m"
            fi
        done

        pending=$(echo $pending | xargs)
        if [[ -z "$pending" ]]; then
            write_fail "models without canonical results: ${missing}; all exhausted ${MODEL_RESTART_MAX} restart attempts${failed_now:+ (newly failed:${failed_now})}."
            exit 1
        fi

        qr_file="${STATE_DIR}/queue_restarts"
        qr=0; [[ -f "$qr_file" ]] && qr=$(cat "$qr_file")
        if [[ $qr -ge $QUEUE_RESTART_MAX ]]; then
            write_fail "queue relaunch budget exhausted (${qr}/${QUEUE_RESTART_MAX}); still missing:${pending}."
            exit 1
        fi
        if [[ $qr -ge $GLOBAL_LAUNCH_MAX ]]; then
            write_fail "global relaunch flap-guard hit (${qr}); still missing:${pending}."
            exit 1
        fi

        # Increment attempt budget for every model included in this relaunch.
        scope_count=0; total_count=0
        for m in $pending; do attempt_inc "$m"; scope_count=$((scope_count+1)); done
        for m in $(model_list); do total_count=$((total_count+1)); done

        models_arg=""
        if [[ $scope_count -lt $total_count ]]; then
            models_arg=$(echo $pending | tr ' ' ',')
        fi

        echo $((qr+1)) > "$qr_file"
        log "FINAL-PASS: queue exited; missing:${pending}; relaunch #$((qr+1)) (scope: ${models_arg:-all})"
        if launch_queue "$models_arg"; then
            :
        else
            lf_file="${STATE_DIR}/launch_failures"
            lf=0; [[ -f "$lf_file" ]] && lf=$(cat "$lf_file")
            echo $((lf+1)) > "$lf_file"
            if [[ $((lf+1)) -ge $LAUNCH_FAILURE_MAX ]]; then
                write_fail "${lf} consecutive launch failures; still missing:${pending}."
                exit 1
            fi
        fi
    fi

    sleep "$CHECK_INTERVAL"
done
