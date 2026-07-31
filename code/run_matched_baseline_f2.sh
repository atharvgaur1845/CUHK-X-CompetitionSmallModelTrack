#!/usr/bin/env bash
# Partition-matched all-18 fold-2 baseline for the EXP-050 visual MIL screen.
#
# Waits for the running visual screen to release the GPU, then trains the
# Adaptive ST-GCN core on exactly the all-18 fold-2 training partition
# (14 users, 2281 clips) and dumps validation probabilities on the 652
# held-out clips of users 5, 6, 22, and 24.
#
# Recipe is EXP-040 verbatim: adaptive skelg, jv features, width 64, 100
# epochs, bs 64, lr 1e-3, label smoothing 0.1, trunc augmentation, seed 0.
# Only the fold protocol changes.
#
# Two baselines are dumped on purpose:
#   *_best  - epoch selected on fold-2 val micro accuracy. Optimistic, and
#             therefore the CONSERVATIVE control for adopting a visual member.
#   *_last  - final epoch, no selection at all. Symmetric with the visual
#             model's final-EMA, outer-once evaluation.
# The visual member is adopted only on a positive fusion delta against the
# optimistic *_best baseline.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
export CUHKX_FOLD_FILE="$ROOT/research/artifacts/cv_folds_all18.json"

LOG="$ROOT/logs/matched_baseline_f2.log"
mkdir -p "$ROOT/logs" "$ROOT/checkpoints"

WAIT_PID="${1:-}"

{
  echo "=== waiting for visual screen to finish (pid=${WAIT_PID:-none}) ==="
  # Wait on the real training PID. A -f pattern would also match unrelated
  # shells whose command line mentions the script, so it is not used here.
  if [ -n "$WAIT_PID" ]; then
    while kill -0 "$WAIT_PID" 2>/dev/null; do
      sleep 120
    done
  fi
  # Do not start until the GPU is actually free, whatever released it.
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${used:-9999}" -lt 1000 ] && break
    sleep 120
  done
  echo "GPU released at $(date -Is)"

  echo "=== training matched all-18 fold-2 baseline ==="
  python3 -u code/train.py \
    --modality skelg --arch adaptive --tag astgcn_all18 --exp P-09 \
    --feat jv --width 64 --epochs 100 --bs 64 --lr 1e-3 --ls 0.1 \
    --aug-spec trunc --n-frames 8 --t-skel 32 --skel-time stretch \
    --person first --seed 0 --folds 2 --workers 6

  for variant in best last; do
    if [ "$variant" = "best" ]; then
      ckpt="$ROOT/checkpoints/astgcn_all18_f2.pt"
    else
      ckpt="$ROOT/checkpoints/astgcn_all18_f2_last.pt"
    fi
    echo "=== dumping $variant probabilities ==="
    python3 -u code/baseline_all18_probs.py \
      --checkpoint "$ckpt" --fold 2 \
      --modality skelg --arch adaptive --width 64 --feat jv \
      --n-frames 8 --t-skel 32 --skel-time stretch --person first \
      --output "$ROOT/research/artifacts/oof_astgcn_all18_f2_${variant}.npz"
  done

  echo "=== matched baseline complete at $(date -Is) ==="
} >> "$LOG" 2>&1
