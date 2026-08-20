#!/usr/bin/env bash
# EXP-092: 32 frames instead of 16, at 128x128.
#
# Motivated by our OWN measurement, not a guess: EXP-068 found that dropping
# 16 -> 8 frames costs 4.1 points. That is a steep gradient on the temporal axis,
# and we are still discarding roughly half the frames in every clip. Resolution
# (EXP-091, 128 -> 160) came back flat: +1.23 micro, p=0.346, and bagging the two
# resolutions gained nothing over 160 alone -- so the spatial axis is closed and
# the temporal one is the live sibling.
#
# Single change against EXP-086 fold 2 (micro 0.63957 / object 0.54906): frames only.
# Screened on OOF. The pre-registered gate is +3 micro before any submission -- the
# public set has a +/-6 clip SD and cannot resolve less than ~12 clips.
#
# Memory: the 32-frame cache is 6.15 GB against ~4 GB free. workers=0 and lazy
# memmap paging make this survivable but slower per epoch; do NOT raise workers.
set -uo pipefail
cd "$(dirname "$0")/.."
while [ ! -f cache/crop_v1f32/test_index.json ]; do sleep 60; done
sleep 30
echo "=== EXP-092 f32 fold 2 $(date -Is) ==="
# The 32-frame memmap is 6.15 GB against ~4 GB free, so the OS pages from disk every
# batch and epochs run well over the 128-frame run's 305s. The watchdog logs every 3
# epochs and defaults to a 90-minute stall threshold; at >30 min/epoch that would fire
# on healthy progress and kill a working job. Give it 3 hours.
STALL_LOG_SECS=10800 code/run_with_watchdog.sh logs/vid_f32_f2.log 'OUTER-ONCE' \
  python3 -u code/train_video_crop.py --tag vid_f32_f2 \
    --fold-oof oof_visual_mil_v1_f2.npz --cache crop_v1f32 \
    --arch r2plus1d_18 --epochs 30 --workers 0 --resume
echo "=== EXP-092 done $(date -Is) ==="
