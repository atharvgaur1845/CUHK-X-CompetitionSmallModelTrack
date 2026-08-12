#!/usr/bin/env bash
# EXP-082: per-modality ImageNet-ResNet18 members (BN frozen), 4 folds each.
#
# Motivation.  Our visual models have always fed IR+depth+thermal through ONE shared
# backbone under one set of weights.  The paper trains each modality separately and
# reports Thermal 92.57 > Depth 90.5 > IR 90.2 -- thermal is the single best modality
# and we have never trained a model on it alone.
#
# Why now.  EXP-081 measured the mechanism that matters: the pretrained member scores
# WORSE solo than the from-scratch member (0.31343 vs 0.38805) yet ADDING it to the
# ensemble gained +5 public clips (126 -> 131), while REPLACING with it lost 2.  Member
# diversity, not member rank, is what the fusion pays for -- and per-modality members
# are diverse by construction (each sees a strictly different input).
#
# Recipe is held identical to EXP-081 (stride 2, BN frozen, 30 epochs) so --modalities
# is the ONLY variable.  Deliberately NOT also raising stride to 1 despite the 3x
# compute headroom: multi-change runs have produced three uninterpretable results in
# this campaign already.
set -uo pipefail
cd "$(dirname "$0")/.."

for MOD in thermal depth ir; do
  for FOLD in 0 1 2 3; do
    echo "=== EXP-082 ${MOD} fold ${FOLD} $(date -Is) ==="
    python3 -u code/train_pretrained_visual.py \
      --tag "pre_${MOD}_f${FOLD}" \
      --fold-oof "oof_visual_mil_v1_f${FOLD}.npz" \
      --modalities "${MOD}" \
      --stride 2 --batch-size 4 --accum 2 --epochs 30 --freeze-bn
  done
done
echo "=== all per-modality folds done $(date -Is) ==="
