#!/usr/bin/env bash
# EXP-097: R(2+1)D-34, IG-65M -> Kinetics-400, on the person-crop cache.
#
# The last identified, untested explanation for the 22-clip gap between our video member
# (121/201 solo) and the published notebook's (143/201). Everything else on that list is
# now measured dead: data volume (all-18, no gain), resolution (4-fold mean -0.19),
# frame count (+1.38, ns), variant substitution (-9 public).
#
# IG-65M is ~65M Instagram videos vs Kinetics-400's ~240k -- roughly 270x the video
# pretraining corpus. Same crop cache, same recipe, same fold, so the only variable is
# the backbone and its pretraining.
#
# 63.5M params: batch 2 with accum 8 keeps the effective batch at 16 inside 8GB.
# Packaging: 127MB fp16 is over budget alone, 63.5MB int8 is not. R-3 also permits
# distillation from any-size teacher, so a win here has two shipping routes.
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== EXP-097 ig65m_34 fold 2 $(date -Is) ==="
code/run_with_watchdog.sh logs/vid_ig65m_f2.log 'OUTER-ONCE' \
  python3 -u code/train_video_crop.py --tag vid_ig65m_f2 \
    --fold-oof oof_visual_mil_v1_f2.npz --arch ig65m_34 \
    --batch-size 2 --accum 8 --epochs 30 --workers 0 --resume
echo "=== EXP-097 done $(date -Is) ==="
