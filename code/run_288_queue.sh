#!/usr/bin/env bash
# EXP-113: person crop at 288 px. The one lever with a track record here.
#
# 128 -> 224 gave +3.83 micro, the largest member gain of the campaign. Person crops are
# median 416 px / p75 480, so 224 still throws away 1.86x linear (~65% of pixels); at 320
# only 30% of crops are upsampled at all. Wrist stays at 224 -- its crops are median
# 147 px, already 1.52x UPsampled, so more pixels there are invented, not recovered.
#
# MViT hardcodes spatial_size=(224,224) and sizes its relative-position tables to the
# 56x56 grid. build_model() now rebuilds at the target size and linearly interpolates
# rel_pos_h/w (111 -> 159), the standard MViT/ViTDet resize. Verified: 397/397 tensors
# ported, 32 tables resized, none left at init, and the 224 path is argmax-identical.
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/.."
export STALL_LOG_SECS=3600 STALL_GPU_CHECKS=12
for F in 2 0 1 3; do
  T="k224_mvit288_f${F}"
  code/run_with_watchdog.sh "logs/px288_f${F}.log" 'OUTER-ONCE' \
    python3 -u kaggle/cuhkx_224_kaggle.py --stage train --fold "$F" --image-size 288 \
      --batch-size 2 --accum 8 --tag "$T" || exit 1
  mv -f "oof_${T}.npz" research/artifacts/ 2>/dev/null
  mv -f "${T}.pt" checkpoints/ 2>/dev/null
  rm -f "${T}_resume.pt"
  echo "=== $T done $(date) ==="
done
echo "=== QUEUE COMPLETE $(date) ==="
