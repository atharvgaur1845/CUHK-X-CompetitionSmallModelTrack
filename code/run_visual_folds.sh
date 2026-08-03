#!/usr/bin/env bash
# Sequentially train the remaining visual MIL folds so the GPU never idles.
# One visual model (fold 2, 14 users) produced +11 public clips as a fused
# member. Folds 0/1/3 cover different user subsets, so the ensemble broadens
# user coverage - the same diversity mechanism that produced today's gains.
# Each run is resumable: a kernel OOM-kill costs one epoch, not the run.
set -uo pipefail
cd "$(dirname "$0")/.."
export CUHKX_GATE_OVERRIDE_LEADERBOARD=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for FOLD in 0 1 3; do
  # Skip a fold that already produced its checkpoint.
  if [ -f "checkpoints/visual_mil_v1_f${FOLD}.pt" ]; then
    echo "fold ${FOLD}: checkpoint exists, skipping"
    continue
  fi
  # Wait for any current training to release the GPU.
  # Anchor on the python process: an unanchored -f pattern also matches any
  # SHELL whose command line merely mentions the trainer (a monitoring command,
  # or this script's own launcher), which blocks the loop forever.  Seen live
  # 2026-08-01: a leftover launcher wrapper held the driver idle with the GPU
  # free.  '^python3' cannot match "/bin/bash -c ...".
  while pgrep -f "^python3 .*train_visual_mil\.py train" > /dev/null; do
    sleep 180
  done
  echo "=== visual fold ${FOLD} starting $(date -Is) ==="
  python3 -u code/train_visual_mil.py train --tag visual_mil_v1 \
    --fold "${FOLD}" --device cuda --batch-size 1 --workers 2 --resume \
    >> "logs/visual_mil_f${FOLD}.log" 2>&1
  echo "=== visual fold ${FOLD} exited $? at $(date -Is) ==="
done
echo "=== all visual folds done $(date -Is) ==="
