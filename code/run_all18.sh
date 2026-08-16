#!/usr/bin/env bash
# EXP-089: all-18-user video members, two seeds.
#
# Every fold member trains on ~2,200 of 2,933 clips. The published 0.711 notebook
# trains on everything, and its single model scores 143/201 against our best single
# fold's 121 solo. Part of that 22-clip gap is simply +33% training data.
#
# These carry NO honest validation by construction -- that is the point, and it is
# why the recipe was measured on folds first. The printed outer score is train-on-test
# and must never be compared against a fold member's OOF.
set -uo pipefail
cd "$(dirname "$0")/.."
for SEED in 20260730 20260816; do
  echo "=== EXP-089 all18 seed ${SEED} $(date -Is) ==="
  code/run_with_watchdog.sh "logs/vid_all18_s${SEED}.log" 'OUTER-ONCE' \
    python3 -u code/train_video_crop.py --tag "vid_all18_s${SEED}" \
      --fold-oof oof_visual_mil_v1_f2.npz --all-train --seed "${SEED}" \
      --arch r2plus1d_18 --epochs 30 --workers 0 --resume
  rm -f "checkpoints/vid_all18_s${SEED}_resume.pt"
done
echo "=== EXP-089 done $(date -Is) ==="
