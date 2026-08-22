#!/usr/bin/env bash
# EXP-105 GPU queue: waits for the in-flight all-train run, then backs the wrist view
# with the same 4 folds the person view has, then trains the shippable all-train wrist.
set -u
cd "$(dirname "$0")/.."
while pgrep -f "cuhkx_224_kaggle.py --stage train --all-train --tag k224_mvit_all" >/dev/null; do sleep 60; done
echo "=== k224_mvit_all done, starting wrist folds $(date) ==="
for F in 0 1 3; do
  T="k224_mvitwrist_f${F}"
  python3 -u kaggle/cuhkx_224_kaggle.py --stage train --fold "$F" --crop wrist --tag "$T" || exit 1
  python3 -u kaggle/cuhkx_224_kaggle.py --stage infer --fold "$F" --crop wrist --tag "$T" || exit 1
  mv -f "oof_${T}.npz" "testprobs_${T}.npz" research/artifacts/ 2>/dev/null
  mv -f "${T}.pt" checkpoints/ 2>/dev/null
  rm -f "sub_${T}_argmax.csv"
done
echo "=== wrist folds done, starting all-train wrist $(date) ==="
python3 -u kaggle/cuhkx_224_kaggle.py --stage train --all-train --crop wrist --tag k224_mvitwrist_all \
 && python3 -u kaggle/cuhkx_224_kaggle.py --stage infer --all-train --crop wrist --tag k224_mvitwrist_all
echo "=== QUEUE COMPLETE $(date) ==="
