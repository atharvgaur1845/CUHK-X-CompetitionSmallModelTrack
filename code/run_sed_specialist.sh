#!/usr/bin/env bash
# EXP-072: single-frame appearance model trained ONLY on the 26 sedentary
# hand-object classes (EXP-067: 75% of all error, and solving them alone would
# score 0.9104).  The coarse sedentary/motion split is already 96% solved, so the
# 14 whole-body classes are wasted capacity in a generalist member.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for FOLD in 0 1 2 3; do
  echo "=== sed specialist fold ${FOLD} $(date -Is) ==="
  python3 -u code/train_frame_appearance.py train --fold "${FOLD}" --epochs 40 \
    --batch-size 256 --lr 2e-3 --memmap --workers 5 --sedentary-only --tag sed_spec
done
echo "=== sed specialist done $(date -Is) ==="
