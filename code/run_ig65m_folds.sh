#!/usr/bin/env bash
# EXP-097b: IG-65M on the remaining folds, to replace the video bag wholesale.
#
# Fold 2, paired against the Kinetics-400 baseline on identical clips:
#   overall  +68/-44  net +24  p=0.0233   SIGNIFICANT
#   object   +63/-37  net +26  p=0.0093   SIGNIFICANT
#   motion    +5/- 7  net  -2  p=0.5637   noise
# micro 0.67638 vs 0.63957, object 0.60334 vs 0.54906. The pretraining corpus was the
# gap: ~65M Instagram videos vs Kinetics-400's ~240k.
#
# BUT swapping it into the fusion in place of the fold-2 baseline gives +0.00, and
# bagging the two gives +1.45. The champion's video slot is a FOUR-FOLD bag, so a single
# IG-65M fold cannot replace it -- that is exactly the confound that made `var2` look
# +3.83 on OOF and lose 9 public clips. All four folds have to exist first.
set -uo pipefail
cd "$(dirname "$0")/.."
for FOLD in 0 1 3; do
  echo "=== EXP-097b ig65m fold ${FOLD} $(date -Is) ==="
  code/run_with_watchdog.sh "logs/vid_ig65m_f${FOLD}.log" 'OUTER-ONCE' \
    python3 -u code/train_video_crop.py --tag "vid_ig65m_f${FOLD}" \
      --fold-oof "oof_visual_mil_v1_f${FOLD}.npz" --arch ig65m_34 \
      --batch-size 2 --accum 8 --epochs 30 --workers 0 --resume
  rm -f "checkpoints/vid_ig65m_f${FOLD}_resume.pt"
done
echo "=== EXP-097b done $(date -Is) ==="
