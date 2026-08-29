#!/usr/bin/env bash
# EXP-113: 288 px passed the bar (4/4 folds, +1.27 micro, sd 0.79). Train the shippable
# all-train person model at 288 so the package can use it. Wrist stays at 224 -- its crops
# are median 147 px and already 1.52x UPsampled there, so 288 would invent pixels.
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/.."
export STALL_LOG_SECS=3600 STALL_GPU_CHECKS=12
T=k224_mvit288_all
code/run_with_watchdog.sh logs/px288_all.log 'OUTER-ONCE' \
  python3 -u kaggle/cuhkx_224_kaggle.py --stage train --all-train --image-size 288 \
    --batch-size 2 --accum 8 --tag "$T" || exit 1
python3 -u kaggle/cuhkx_224_kaggle.py --stage infer --all-train --image-size 288 \
    --batch-size 2 --tag "$T" || exit 1
mv -f "oof_${T}.npz" testprobs_${T}*.npz research/artifacts/ 2>/dev/null
mv -f "${T}.pt" checkpoints/ 2>/dev/null
rm -f "${T}_resume.pt" sub_${T}*_argmax.csv
echo "=== QUEUE COMPLETE $(date) ==="
