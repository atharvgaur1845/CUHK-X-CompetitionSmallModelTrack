#!/usr/bin/env bash
# EXP-114: person crop at 384 px with gradient checkpointing.
#
# Resolution is the only member axis still paying: 288 px was +1.27 micro on 4/4 folds
# (EXP-113), the first change to clear the bar since 224 itself. The cap was VRAM --
# 320/bs2 OOMs on 8 GB -- so --grad-checkpoint recomputes each block in backward and
# buys the next step. Measured: 384/bs2 peaks at 5.67 GiB, 1437 ms/step, ~9.1 h/fold.
#
# batch 2 x accum 8 keeps the effective batch at 16, identical to the 288 run, so
# resolution stays the single change.
#
# Person crops are median 416 px, so 384 discards ~32% of pixels against 224's 65% and
# 288's 52%. Wrist stays at 224 (median 147 px, already 1.52x upsampled there).
#
# Folds 2 and 0 run FIRST as the screen. Do not adopt on one fold: sigma is 2.80 and
# both jitter and LLRD threw a spurious +2.45 that way.
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/.."
export STALL_LOG_SECS=5400 STALL_GPU_CHECKS=15
# fold 2 is already in flight from a standalone launch; wait on its PID, never on a
# pgrep pattern -- `pgrep -f` matches the waiter's own command line and deadlocks (EXP-106
# cost 7 idle GPU-hours to exactly that, and a stray `pkill -f` later killed this script's
# own shell for the same reason).
WAIT_PID=${WAIT_PID:-0}
while [ "$WAIT_PID" != 0 ] && kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
if [ -f oof_k224_mvit384_f2.npz ]; then
  mv -f oof_k224_mvit384_f2.npz research/artifacts/ 2>/dev/null
  mv -f k224_mvit384_f2.pt checkpoints/ 2>/dev/null
  rm -f k224_mvit384_f2_resume.pt
  echo "=== k224_mvit384_f2 harvested $(date) ==="
fi
for F in 0 1 3; do
  T="k224_mvit384_f${F}"
  code/run_with_watchdog.sh "logs/px384_f${F}.log" 'OUTER-ONCE' \
    python3 -u kaggle/cuhkx_224_kaggle.py --stage train --fold "$F" --image-size 384 \
      --batch-size 2 --accum 8 --grad-checkpoint --workers 0 --tag "$T" || exit 1
  mv -f "oof_${T}.npz" research/artifacts/ 2>/dev/null
  mv -f "${T}.pt" checkpoints/ 2>/dev/null
  rm -f "${T}_resume.pt"
  echo "=== $T done $(date) ==="
done
echo "=== QUEUE COMPLETE $(date) ==="
