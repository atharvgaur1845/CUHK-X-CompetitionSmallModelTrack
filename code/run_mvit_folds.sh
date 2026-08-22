#!/usr/bin/env bash
# Give mvit the same 4-fold coverage the CNN bag already has. One fold at a time;
# each is resumable, so a kill mid-run costs at most one epoch.
set -u
cd "$(dirname "$0")/.."
for F in 0 1 3; do
  T="k224_mvit_f${F}"
  if [ -f "research/artifacts/oof_${T}.npz" ]; then echo "SKIP $T"; continue; fi
  echo "=== $T ==="
  python3 -u kaggle/cuhkx_224_kaggle.py --stage train --fold "$F" --tag "$T" || exit 1
  python3 -u kaggle/cuhkx_224_kaggle.py --stage infer --fold "$F" --tag "$T" || exit 1
  mv -f "oof_${T}.npz" "testprobs_${T}.npz" research/artifacts/ 2>/dev/null
  mv -f "${T}.pt" checkpoints/ 2>/dev/null
  mv -f "sub_${T}_argmax.csv" submissions/ 2>/dev/null
done
echo "MVIT-FOLDS-DONE"
