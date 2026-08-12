#!/usr/bin/env bash
# EXP-083: what is the binding constraint on visual member quality?
#
# Ensembling is now closed as a lever (all measured on public, vs the 131 champion):
#   weight 0.45 -> 0.55 ................ 130  (-1, noise)
#   +3 per-modality members ............ 130  (-1, noise)
#   +SSL(1-fold) + app_all ............. 126  (-5)
#   all 7 visual members ............... 124  (-7)
# Two members is saturation, and incomplete members poison the mix.  The remaining
# +20-29 clips must come from MEMBER QUALITY: the bag is ~0.31-0.39 solo where the
# fusion arithmetic wants ~0.70.
#
# The model overfits hard (train loss 0.95 on 2281 clips), so the three candidate
# constraints are capacity, spatial resolution, and temporal resolution.  Each arm
# below changes exactly ONE thing against the fold-2 reference:
#   r18 / 192x256 / stride 2 / BN frozen -> micro 0.34816  object 0.26931 (129/479)
#
# Arm A is deliberately a DOWNWARD resolution test.  Building a higher-resolution
# cache costs hours of CPU and ~50 GB, so measure the slope with a free downsample
# first: if 96x128 matches 192x256, resolution is not binding and that cache must not
# be built.  EXP-068 found exactly that flatness for the from-scratch trunk -- but
# that trunk was motion-only, and ImageNet features may use detail it never learned.
set -uo pipefail
cd "$(dirname "$0")/.."

COMMON="--fold-oof oof_visual_mil_v1_f2.npz --batch-size 4 --accum 2 --epochs 30 --freeze-bn"

echo "=== ARM A: half resolution (96x128) $(date -Is) ==="
python3 -u code/train_pretrained_visual.py --tag pre_r18_f2_lowres \
  --resize 96 128 --stride 2 ${COMMON}

echo "=== ARM B: resnet50 capacity $(date -Is) ==="
python3 -u code/train_pretrained_visual.py --tag pre_r50_f2 \
  --arch resnet50 --stride 2 --batch-size 2 --accum 4 \
  --fold-oof oof_visual_mil_v1_f2.npz --epochs 30 --freeze-bn

echo "=== ARM C: all 16 frames, BN frozen $(date -Is) ==="
python3 -u code/train_pretrained_visual.py --tag pre_r18_f2_s1bn \
  --stride 1 --batch-size 2 --accum 4 \
  --fold-oof oof_visual_mil_v1_f2.npz --epochs 30 --freeze-bn

echo "=== EXP-083 done $(date -Is) ==="
