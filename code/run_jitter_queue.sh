#!/usr/bin/env bash
# EXP-110: temporal jitter gave +2.45 micro on fold 2 (0.71472 -> 0.73926) at zero
# package bytes. Order below is deliberate: cheapest replication check first, because
# fold-2-only adoption is exactly what produced n7's -3 (B-029).
#   1. fold 0 jitter          — does the gain replicate on a different fold?
#   2. person all-train jitter — shippable
#   3. wrist  all-train jitter — shippable
#   4. folds 1,3 jitter        — completes the honest pooled estimate
set -u
cd "$(dirname "$0")/.."
run() {
  local T="$1"; shift
  python3 -u kaggle/cuhkx_224_kaggle.py --stage train --frames 32 "$@" --tag "$T" || return 1
  python3 -u kaggle/cuhkx_224_kaggle.py --stage infer --frames 32 "$@" --tag "$T" || return 1
  mv -f "oof_${T}.npz" research/artifacts/ 2>/dev/null
  mv -f testprobs_${T}*.npz research/artifacts/ 2>/dev/null
  mv -f "${T}.pt" checkpoints/ 2>/dev/null
  rm -f "${T}_resume.pt" sub_${T}*_argmax.csv
  echo "=== $T harvested $(date) ==="
}
run k224_mvitjit_f0        --fold 0
run k224_mvitjit_all       --all-train
run k224_mvitwristjit_all  --all-train --crop wrist
run k224_mvitwristjit_f2   --fold 2 --crop wrist
run k224_mvitjit_f1        --fold 1
run k224_mvitjit_f3        --fold 3
echo "=== QUEUE COMPLETE $(date) ==="
