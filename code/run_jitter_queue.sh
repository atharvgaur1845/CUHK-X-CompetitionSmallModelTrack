#!/usr/bin/env bash
# EXP-110 close-out, now UNDER THE WATCHDOG.
#
# The previous version of this script was not, and k224_mvitwristjit_all hung entering
# epoch 20 with the process alive, 6,424 MiB held and the GPU at 9%. It sat that way for
# 18.4 hours. That is EXP-064's exact signature, and run_with_watchdog.sh exists solely
# to catch it -- CLAUDE.md says long runs go under it. They now do.
#
# Epochs here log every ~5 min, so a 30-minute silence is already pathological; the
# watchdog's 90-minute default was tuned for a slower trainer.
set -u
cd "$(dirname "$0")/.."
export STALL_LOG_SECS=1800 STALL_GPU_CHECKS=10

wd() {   # tag, then trainer args
  local T="$1"; shift
  code/run_with_watchdog.sh "logs/jit_${T}.log" 'OUTER-ONCE' \
    python3 -u kaggle/cuhkx_224_kaggle.py --stage train --frames 32 "$@" --tag "$T" || return 1
  python3 -u kaggle/cuhkx_224_kaggle.py --stage infer --frames 32 "$@" --tag "$T" || return 1
  mv -f "oof_${T}.npz" research/artifacts/ 2>/dev/null
  mv -f testprobs_${T}*.npz research/artifacts/ 2>/dev/null
  mv -f "${T}.pt" checkpoints/ 2>/dev/null
  rm -f "${T}_resume.pt" sub_${T}*_argmax.csv
  echo "=== $T harvested $(date) ==="
}
# resumes at epoch 19/20, ~5 min
wd k224_mvitwristjit_all --all-train --crop wrist
# the only runs that can settle the jitter verdict: folds 1 and 3 complete the 4-fold mean
wd k224_mvitjit_f1 --fold 1
wd k224_mvitjit_f3 --fold 3
echo "=== QUEUE COMPLETE $(date) ==="
