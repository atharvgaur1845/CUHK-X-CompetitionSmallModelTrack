#!/usr/bin/env bash
# EXP-112: layer-wise LR decay, 4 folds, under the watchdog.
#
# Why this and not another single-fold punt: seed sigma on this partition is 2.80, so a
# one-fold result is uninformative -- that is what made jitter look like +2.45 when its
# 4-fold mean was +0.14. Four folds or nothing.
#
# The recipe every member so far used is light for a 34M pretrained transformer on 2,281
# clips: uniform lr 1e-4 on all 397 tensors, weight decay 0.05 on norms and biases too.
# --llrd 0.8 gives earlier blocks a smaller step and lifts wd off norms/biases/tokens.
# The 4-channel stem is exempt from the decay: it was rebuilt by surgery and is
# effectively untrained, so decaying it would freeze the layer that most needs to move.
set -u
cd "$(dirname "$0")/.."
export STALL_LOG_SECS=1800 STALL_GPU_CHECKS=10
for F in 2 0 1 3; do
  T="k224_mvitlr_f${F}"
  code/run_with_watchdog.sh "logs/llrd_f${F}.log" 'OUTER-ONCE' \
    python3 -u kaggle/cuhkx_224_kaggle.py --stage train --fold "$F" --llrd 0.8 --tag "$T" || exit 1
  mv -f "oof_${T}.npz" research/artifacts/ 2>/dev/null
  mv -f "${T}.pt" checkpoints/ 2>/dev/null
  rm -f "${T}_resume.pt"
  echo "=== $T done $(date) ==="
done
echo "=== QUEUE COMPLETE $(date) ==="
