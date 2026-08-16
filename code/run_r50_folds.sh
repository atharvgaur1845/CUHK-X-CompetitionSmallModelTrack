#!/usr/bin/env bash
# EXP-084: 4-fold ImageNet-ResNet50 visual member -- capacity is the binding constraint.
#
# EXP-083 fold 2, single variable (backbone), everything else identical:
#   from-scratch MIL trunk : micro 0.35123  object 0.24843 (119/479)  motion 0.63584
#   ImageNet ResNet18      : micro 0.34816  object 0.26931 (129/479)  motion 0.56647
#   ImageNet ResNet50      : micro 0.40031  object 0.34238 (164/479)  motion 0.56069
# +7.3 object over r18 (+35 clips), motion flat -- the entire gain lands on the 26
# classes holding 75% of the error.  Largest single-change effect measured in this
# campaign.
#
# Also from EXP-083: resolution IS binding (96x128 costs -3.34 object vs 192x256),
# contradicting EXP-068's flatness -- that trunk was from-scratch and motion-only, so
# it had no fine detail to lose.  A higher-resolution cache is therefore justified, but
# capacity is the cheaper lever and needs no cache rebuild, so it goes first.
#
# PACKAGING WARNING: ResNet50 is 96.0 MB fp32, which alone nearly exhausts the 100 MB
# single-file budget for ALL inference weights.  R-6 explicitly permits and encourages
# fp16/int8, and fp16 puts it at 48.0 MB -- but quantized packaging is now mandatory
# rather than optional and must be verified before any final submission.
#
# workers=2: arm C died to the OOM killer at workers=4 with stride 1.
set -uo pipefail
cd "$(dirname "$0")/.."

for FOLD in 0 1 3; do
  echo "=== EXP-084 resnet50 fold ${FOLD} $(date -Is) ==="
  python3 -u code/train_pretrained_visual.py \
    --tag "pre_r50_f${FOLD}" \
    --fold-oof "oof_visual_mil_v1_f${FOLD}.npz" \
    --arch resnet50 --stride 2 --batch-size 2 --accum 4 \
    --epochs 30 --freeze-bn --workers 2
done

echo "=== EXP-083 ARM C retry (OOM at workers=4) $(date -Is) ==="
python3 -u code/train_pretrained_visual.py --tag pre_r18_f2_s1bn \
  --stride 1 --batch-size 2 --accum 4 --workers 2 \
  --fold-oof oof_visual_mil_v1_f2.npz --epochs 30 --freeze-bn

echo "=== EXP-084 done $(date -Is) ==="
