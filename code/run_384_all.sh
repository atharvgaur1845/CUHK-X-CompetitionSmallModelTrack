#!/usr/bin/env bash
# EXP-116b: all-train 384 px person model -> the only READABLE 384 submission.
#
# The 4-fold 384 candidate (sub_t1) is rowdiff 2 against sub_r1 -- swapping one stable
# bag for another is invisible after a 0.2925-weight video slot plus the decoder. The
# 288 swap moved 21 rows because it replaced a single all-train model, which is far less
# stable. So a readable 384 read needs an all-train model.
#
# Expectation is low and stated up front: 288 was 4/4 locally and LOST 2 clips on public,
# and 384 is only 2/4 over 288. But B-032 says member strength cannot be screened
# locally, so the answer has to be bought with a submission.
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/.."
export STALL_LOG_SECS=5400 STALL_GPU_CHECKS=15
T=k224_mvit384_all
code/run_with_watchdog.sh logs/px384_all.log 'OUTER-ONCE' \
  python3 -u kaggle/cuhkx_224_kaggle.py --stage train --all-train --image-size 384 \
    --batch-size 2 --accum 8 --grad-checkpoint --workers 0 --tag "$T" || exit 1
python3 -u kaggle/cuhkx_224_kaggle.py --stage infer --all-train --image-size 384 \
    --batch-size 2 --workers 0 --tag "$T" || exit 1
mv -f "oof_${T}.npz" testprobs_${T}*.npz research/artifacts/ 2>/dev/null
mv -f "${T}.pt" checkpoints/ 2>/dev/null
rm -f "${T}_resume.pt" sub_${T}*_argmax.csv
echo "=== 384 ALL-TRAIN COMPLETE $(date) ==="
