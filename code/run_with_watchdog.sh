#!/usr/bin/env bash
# Run a resumable training command under a stall watchdog.
#
# Why this exists (2026-08-03): EXP-064 hung at epoch 18/150 with the process
# still alive, 5,076 MiB of GPU memory held, and 0% utilisation.  It sat that way
# for TWELVE HOURS before anyone looked, because a hang is silent -- the process
# is up, the log just stops.  A crash announces itself; a deadlock does not.
# Suspected cause is the usual DataLoader fork+CUDA deadlock at --workers 2.
#
# The watchdog kills and resumes on two independent stall signals, so it does not
# depend on the log format alone:
#   * log file has not grown for $STALL_LOG_SECS
#   * GPU utilisation has been 0% for $STALL_GPU_CHECKS consecutive samples
#     while the trainer is still alive
#
# The wrapped command MUST be resumable (--resume) or a restart loses progress.
#
# Usage:
#   code/run_with_watchdog.sh <logfile> <done-marker-regex> <command...>
# Example:
#   code/run_with_watchdog.sh logs/x.log 'OUTER-ONCE' python3 -u code/train.py ...
set -uo pipefail

LOG="${1:?logfile required}"
DONE_RE="${2:?done-marker regex required}"
shift 2

STALL_LOG_SECS="${STALL_LOG_SECS:-5400}"   # 90 min; epochs log every ~56 min here
GPU_POLL_SECS="${GPU_POLL_SECS:-60}"
STALL_GPU_CHECKS="${STALL_GPU_CHECKS:-15}" # 15 min of 0% GPU with a live trainer
MAX_RESTARTS="${MAX_RESTARTS:-8}"

cd "$(dirname "$0")/.."
restarts=0

while :; do
  if [ -f "$LOG" ] && grep -qE "$DONE_RE" "$LOG"; then
    echo "[watchdog] done marker '$DONE_RE' present; exiting" >&2
    exit 0
  fi

  echo "[watchdog] launching (restart $restarts) $(date -Is)" >&2
  "$@" >> "$LOG" 2>&1 &
  pid=$!
  idle=0

  while kill -0 "$pid" 2>/dev/null; do
    sleep "$GPU_POLL_SECS"

    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -z "$util" ] && util=100
    if [ "$util" -eq 0 ]; then idle=$((idle + 1)); else idle=0; fi

    now=$(date +%s)
    mtime=$(stat -c %Y "$LOG" 2>/dev/null || echo "$now")
    age=$((now - mtime))

    if [ "$idle" -ge "$STALL_GPU_CHECKS" ] || [ "$age" -ge "$STALL_LOG_SECS" ]; then
      echo "[watchdog] STALL: gpu-idle=${idle} log-age=${age}s -- killing $pid" >&2
      pkill -9 -P "$pid" 2>/dev/null
      kill -9 "$pid" 2>/dev/null
      sleep 10
      break
    fi
  done

  wait "$pid" 2>/dev/null
  if [ -f "$LOG" ] && grep -qE "$DONE_RE" "$LOG"; then
    echo "[watchdog] completed $(date -Is)" >&2
    exit 0
  fi

  restarts=$((restarts + 1))
  if [ "$restarts" -gt "$MAX_RESTARTS" ]; then
    echo "[watchdog] giving up after $MAX_RESTARTS restarts" >&2
    exit 1
  fi
  echo "[watchdog] resuming in 20s" >&2
  sleep 20
done
