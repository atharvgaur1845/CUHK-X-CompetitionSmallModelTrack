#!/usr/bin/env bash
# EXP-081: 4-fold ImageNet-ResNet18 visual member (BN frozen).
#
# Why a 4-fold bag and not another single-fold submission.  The 0.38805 public solo
# reference is `testprobs_visual_mil_f0123` -- verified at 405/405 argmax parity with
# sub_visual_mil_v1.csv -- i.e. a FOUR-FOLD BAG.  Both single-fold members were scored
# against it and called failures (SSL 0.28358, pretrained 0.24378); those comparisons
# were confounded.  `sub_visual_mil_v1_folds2.csv` exists on disk but was never
# submitted, so no single-fold from-scratch control exists, and LEADERBOARD.md's
# "bagging is worth zero" was measured on the FUSED submission where the skeleton
# stack dominates -- it does not transfer to a solo member.
#
# Fold-2 evidence that this recipe is worth 4 folds (BN frozen was the single change):
#   from-scratch MIL trunk : micro 0.35123  object 0.24843 (119/479)  motion 0.63584
#   ResNet18 BN live       : micro 0.33282  object 0.24843 (119/479)  motion 0.56647
#   ResNet18 BN frozen     : micro 0.34816  object 0.26931 (129/479)  motion 0.56647
# First member to beat 0.24843 on the object classes that hold 75% of the error, while
# losing only on motion -- which skeleton already covers in fusion.
set -uo pipefail
cd "$(dirname "$0")/.."

for FOLD in 0 1 3; do
  echo "=== EXP-081 fold ${FOLD} $(date -Is) ==="
  python3 -u code/train_pretrained_visual.py \
    --tag "pre_r18_bn_f${FOLD}" \
    --fold-oof "oof_visual_mil_v1_f${FOLD}.npz" \
    --stride 2 --batch-size 4 --accum 2 --epochs 30 --freeze-bn
done
echo "=== all folds done $(date -Is) ==="
