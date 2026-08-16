#!/usr/bin/env bash
# EXP-091: does spatial resolution bind AFTER the person crop?
#
# EXP-083 measured resolution as binding on FULL frames (96x128 cost -3.34 object vs
# 192x256). But that was before the crop: on a full 640x480 frame the subject occupies a
# small fraction of the pixels, so downsampling destroys them. After cropping to the
# person, 128x128 is already mostly subject. The two facts are not in conflict and the
# question is genuinely open.
#
# Single change against EXP-086 fold 2 (micro 0.63957, object 0.54906): 128 -> 160.
# Measured on OOF, not on the leaderboard -- the public set has a +/-6 clip SD and cannot
# resolve anything under 12 clips, so member work is screened on 2,933 OOF clips where one
# clip is 0.034%.
set -uo pipefail
cd "$(dirname "$0")/.."
while [ ! -f cache/crop_v160/test_index.json ]; do sleep 60; done
sleep 30
echo "=== EXP-091 res160 fold 2 $(date -Is) ==="
code/run_with_watchdog.sh logs/vid_res160_f2.log 'OUTER-ONCE' \
  python3 -u code/train_video_crop.py --tag vid_res160_f2 \
    --fold-oof oof_visual_mil_v1_f2.npz --cache crop_v160 \
    --arch r2plus1d_18 --epochs 30 --workers 0 --resume
echo "=== EXP-091 done $(date -Is) ==="
