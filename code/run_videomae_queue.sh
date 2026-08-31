#!/usr/bin/env bash
# EXP-115: VideoMAE-B / K400 as the video member, replacing MViTv2-S.
#
# Rationale: B-032 says member strength cannot be screened locally, so the local run is a
# SANITY GATE only (is this member competitive at all, or does it collapse the way Swin3D
# did?). The decision is a public submission.
#
# Why this backbone: K400 ~86% vs MViTv2-S's 80.3%, and it is natively 16-frame with
# tubelet 2 -- matching our clip format exactly. Swin3D was refuted precisely because its
# 32-frame pretrain mismatched our 16.
#
# Budget: 86.7M params -> 65.0 MB at int6. Package = 65.0 + world25 p4 22.8 + imu 9.0
# = 96.8 MB, legal, but it costs us the WRIST view (a second video trunk does not fit).
#
# batch 4 x accum 4 = effective 16, matching every MViT run. --workers 0 is mandatory:
# system RAM is 15 GB and the default of 2 silently OOM-killed the 384 run with no
# traceback.
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/home/atharv/.cache/huggingface
cd "$(dirname "$0")/.."
export STALL_LOG_SECS=1800 STALL_GPU_CHECKS=12
for F in 2 0 1 3; do
  T="vmae_f${F}"
  code/run_with_watchdog.sh "logs/vmae_f${F}.log" 'OUTER-ONCE' \
    python3 -u kaggle/cuhkx_224_kaggle.py --stage train --fold "$F" --arch videomae_b \
      --batch-size 4 --accum 4 --workers 0 --tag "$T" || exit 1
  mv -f "oof_${T}.npz" research/artifacts/ 2>/dev/null
  mv -f "${T}.pt" checkpoints/ 2>/dev/null
  rm -f "${T}_resume.pt"
  echo "=== $T done $(date) ==="
done
echo "=== QUEUE COMPLETE $(date) ==="
