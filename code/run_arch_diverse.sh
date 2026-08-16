#!/usr/bin/env bash
# EXP-090: architectural diversity in the video bag.
#
# The four r2plus1d_18 folds agree with each other on only 62.5-68.6% of test clips
# and bagging them was worth +5 public (151 -> 156). torchvision ships two more
# Kinetics-400 video backbones with genuinely different inductive biases -- mc3_18
# (3D early, 2D late) and r3d_18 (full 3D) -- so architecture diversity should
# decorrelate further than seed diversity does, at the same cost per member.
#
# Trained in --all-train mode: the recipe is already measured on folds, so these
# exist to score, not to validate.
set -uo pipefail
cd "$(dirname "$0")/.."
# NOTE: do NOT gate this on `pgrep -f run_all18.sh`. That matched the tool-invocation
# shell whose command line still contained the script's own text from the heredoc that
# created it, so the wait loop never exited and the GPU sat idle for two hours. Gate on a
# result marker in the log instead -- a fact about the work, not about the process table.
while [ -f logs/vid_all18_s20260816.log ] && ! grep -q 'OUTER-ONCE' logs/vid_all18_s20260816.log; do sleep 120; done
echo "[chain] all18 finished, starting architecture sweep $(date -Is)"
for ARCH in mc3_18 r3d_18; do
  echo "=== EXP-090 ${ARCH} all18 $(date -Is) ==="
  code/run_with_watchdog.sh "logs/vid_${ARCH}_all18.log" 'OUTER-ONCE' \
    python3 -u code/train_video_crop.py --tag "vid_${ARCH}_all18" \
      --fold-oof oof_visual_mil_v1_f2.npz --all-train \
      --arch "${ARCH}" --epochs 30 --workers 0 --resume
  rm -f "checkpoints/vid_${ARCH}_all18_resume.pt"
done
echo "=== EXP-090 done $(date -Is) ==="
