#!/usr/bin/env bash
# EXP-065: regularized visual MIL recipe (EXP-062 overrides, 60 epochs) on folds 0,1,3.
#
# Fold 2 already exists as `visual_mil_reg1`.  EXP-064 established that 150 epochs
# buys nothing over 60 (micro identical to 5 decimals), so this uses 60.
#
# Motivation: on the fold-2 overlap the regularized member contributes +2.90 points
# to the full stack versus the fixed recipe's +1.27.  A 4-fold regularized member is
# the direct test of whether that survives into the deployed submission.
#
# Run under tmux; the watchdog handles stalls, tmux handles session teardown:
#   tmux new-session -d -s regfolds "bash code/run_reg_folds.sh >> logs/regfolds_watchdog.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/.."

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUHKX_GATE_OVERRIDE_LEADERBOARD=1
export CUHKX_VMIL_CROP_MIN=0.60
export CUHKX_VMIL_DROPOUT=0.35
export CUHKX_VMIL_MDROP_IR=0.25
export CUHKX_VMIL_MDROP_DEPTH=0.25
export CUHKX_VMIL_MDROP_THERMAL=0.35
export CUHKX_VMIL_LABEL_SMOOTHING=0.10

for FOLD in 0 1 3; do
  if [ -f "checkpoints/visual_mil_reg1_f${FOLD}.pt" ]; then
    echo "fold ${FOLD}: checkpoint exists, skipping"
    continue
  fi
  echo "=== reg fold ${FOLD} starting $(date -Is) ==="
  code/run_with_watchdog.sh "logs/visual_mil_reg1_f${FOLD}.log" 'OUTER-ONCE' \
    python3 -u code/train_visual_mil.py train --tag visual_mil_reg1 \
      --fold "${FOLD}" --device cuda --batch-size 1 --workers 2 --resume
  echo "=== reg fold ${FOLD} exited $? at $(date -Is) ==="
done
echo "=== all regularized folds done $(date -Is) ==="
