#!/usr/bin/env bash
# EXP-086b: complete the person-crop video member on the remaining three folds.
#
# Fold 2 is measured, single-change against every earlier visual member on the
# identical outer subjects:
#   from-scratch MIL trunk : micro 0.35123  object 0.24843 (119/479)  motion 0.63584
#   ImageNet ResNet18      : micro 0.34816  object 0.26931 (129/479)  motion 0.56647
#   ImageNet ResNet50      : micro 0.40031  object 0.34238 (164/479)  motion 0.56069
#   crop + Kinetics R(2+1)D: micro 0.63957  object 0.54906 (263/479)  motion 0.89017
# +23.9 micro / +99 object clips over the best prior member -- three times the
# largest effect this campaign had produced, and it lands on the object classes
# holding 75% of the error.
#
# Two independent causes are confounded in that number (person crop, and Kinetics
# video pretraining); separating them is NOT worth GPU time before the bag is on
# the leaderboard, because neither half is actionable alone.
#
# workers=0 and --resume are load-bearing: this job was OOM-killed at epoch 24/30
# once already, and 15GB of system RAM against a 3GB memmap leaves no margin.
set -uo pipefail
cd "$(dirname "$0")/.."

for FOLD in 0 1 3; do
  echo "=== EXP-086b vid_r2p1d fold ${FOLD} $(date -Is) ==="
  code/run_with_watchdog.sh "logs/vid_r2p1d_f${FOLD}.log" 'OUTER-ONCE' \
    python3 -u code/train_video_crop.py \
      --tag "vid_r2p1d_f${FOLD}" \
      --fold-oof "oof_visual_mil_v1_f${FOLD}.npz" \
      --arch r2plus1d_18 --epochs 30 --workers 0 --resume
  # 250MB of optimizer-free resume state per fold; 65GB free is not a lot of margin.
  rm -f "checkpoints/vid_r2p1d_f${FOLD}_resume.pt"
done

echo "=== EXP-086b done $(date -Is) ==="
