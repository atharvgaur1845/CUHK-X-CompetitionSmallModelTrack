#!/usr/bin/env bash
# EXP-105 GPU queue: back the wrist view with the same 4 folds the person view has,
# then train the shippable all-train wrist half of the Stage-2 package.
set -u
cd "$(dirname "$0")/.."
run() {  # tag, then the stage args
  local T="$1"; shift
  python3 -u kaggle/cuhkx_224_kaggle.py --stage train "$@" --tag "$T" || return 1
  python3 -u kaggle/cuhkx_224_kaggle.py --stage infer "$@" --tag "$T" || return 1
  mv -f "oof_${T}.npz" "testprobs_${T}.npz" research/artifacts/ 2>/dev/null
  mv -f "${T}.pt" checkpoints/ 2>/dev/null
  rm -f "${T}_resume.pt" "sub_${T}_argmax.csv"
  echo "=== $T harvested $(date) ==="
}
for F in 0 1 3; do run "k224_mvitwrist_f${F}" --fold "$F" --crop wrist || exit 1; done
run k224_mvitwrist_all --all-train --crop wrist
echo "=== QUEUE COMPLETE $(date) ==="
