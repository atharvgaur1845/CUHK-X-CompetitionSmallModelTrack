#!/usr/bin/env bash
# EXP-071: single-frame appearance member on the remaining folds.
# Fold 2 gave the best sedentary accuracy of any member (0.2402 vs visual MIL's
# 0.2039) and the highest unique contribution (4.75% of sedentary clips right
# where the stack is wrong), but it is overfit and uncalibrated, so weighted
# fusion extracts only +1.5 OOF clips. A 4-fold ensemble is the cheap test of
# whether calibration is what is blocking it: ~10 min per fold.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for FOLD in 0 1 3; do
  echo "=== frame_app_mm fold ${FOLD} $(date -Is) ==="
  python3 -u code/train_frame_appearance.py train --fold "${FOLD}" --epochs 40 \
    --batch-size 256 --lr 2e-3 --memmap --workers 5 --tag frame_app_mm
done
echo "=== all folds done $(date -Is) ==="
