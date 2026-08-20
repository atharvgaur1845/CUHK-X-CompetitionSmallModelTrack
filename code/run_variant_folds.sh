#!/usr/bin/env bash
# EXP-095: the variant trick applied INSIDE every fold, which is the only deployable form.
#
# EXP-091/092/093/094 each produced a sub-threshold member (+1.2 to +1.8). Bagged together
# on fold 2 they reach +2.90 micro at p=0.0047 (net +16: 24 rescued, 8 broken) -- the first
# significant result since the video member itself. The winning set DROPS the 16f@128
# baseline: 160px + 32f + MotionBERT.
#
# But that measurement is fold-2 variants vs a fold-2 baseline, while deployment bags FOUR
# subject-disjoint folds. `var2` already showed what happens if you confuse the two: it
# looked +3.83 on OOF and lost 9 public clips by trading 3 folds for 2 same-fold variants.
# So the variants have to exist on every fold before they can replace anything.
#
# 160px first (~3h/fold); 32f is ~6h/fold because its 6.15GB memmap pages against 4GB free.
set -uo pipefail
cd "$(dirname "$0")/.."
for FOLD in 0 1 3; do
  echo "=== EXP-095 res160 fold ${FOLD} $(date -Is) ==="
  code/run_with_watchdog.sh "logs/vid_res160_f${FOLD}.log" 'OUTER-ONCE' \
    python3 -u code/train_video_crop.py --tag "vid_res160_f${FOLD}" \
      --fold-oof "oof_visual_mil_v1_f${FOLD}.npz" --cache crop_v160 \
      --arch r2plus1d_18 --epochs 30 --workers 0 --resume
  rm -f "checkpoints/vid_res160_f${FOLD}_resume.pt"
done
echo "=== EXP-095 done $(date -Is) ==="
