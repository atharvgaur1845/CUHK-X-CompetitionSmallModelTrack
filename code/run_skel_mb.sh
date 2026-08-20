#!/usr/bin/env bash
# EXP-094: MotionBERT skeleton member + its from-scratch control.
#
# Item (1) of the handover: the unfixed half of R-1. The visual branch was rebuilt on
# pretrained weights for +23.9 micro; the skeleton branch was never touched and still
# carries 0.35, the largest single weight in the champion fusion, at 0.594 solo OOF.
#
# The control matters more than usual here. `--scratch` is the identical architecture,
# identical recipe, random init. Without it a good number cannot be attributed to
# PRETRAINING rather than to DSTformer simply being a better architecture than our
# ASTGCN -- and only the first of those is the R-1 story.
#
# 60.4M params = 120.8MB fp16, over the whole packaging budget on its own. That does
# NOT block the experiment: R-3 permits distillation from a teacher of any size, so a
# win here can be distilled into the small from-scratch stack we already ship.
set -uo pipefail
cd "$(dirname "$0")/.."
while pgrep -f '^python3 -u code/train_video_crop.py' >/dev/null; do sleep 120; done
sleep 20
echo "=== EXP-094 MotionBERT pretrained, fold 2 $(date -Is) ==="
code/run_with_watchdog.sh logs/skel_mb_f2.log 'OUTER-ONCE' \
  python3 -u code/train_skel_motionbert.py --tag skel_mb_f2 \
    --fold-oof oof_visual_mil_v1_f2.npz --epochs 30 --workers 0 --resume
echo "=== EXP-094b from-scratch control, fold 2 $(date -Is) ==="
code/run_with_watchdog.sh logs/skel_mb_scratch_f2.log 'OUTER-ONCE' \
  python3 -u code/train_skel_motionbert.py --tag skel_mb_scratch_f2 --scratch \
    --fold-oof oof_visual_mil_v1_f2.npz --epochs 30 --workers 0 --resume
echo "=== EXP-094 done $(date -Is) ==="
