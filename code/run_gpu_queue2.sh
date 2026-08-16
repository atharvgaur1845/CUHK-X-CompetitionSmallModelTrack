#!/usr/bin/env bash
# GPU queue, second pass. Thermal first, then the remaining video folds.
#
# The first attempt died: build_model() in train_video_crop.py unconditionally
# rebuilt the stem as 4-channel, so the 3-channel thermal tensor failed every
# forward pass and the watchdog burned its 8 restarts. build_model now takes
# in_channels (default 4, so every existing checkpoint and in-flight fold rebuilds
# identically) and skips the surgery entirely at 3 channels.
#
# Fold 1 was interrupted at epoch 9/30 to let thermal go first; --resume picks it
# back up from its per-epoch state, so the interruption cost one partial epoch.
#
# Order rationale: bagging folds is a known +2-5 clips (fold0 and fold2 agree on
# only 65.7% of test clips, so the bag is genuinely additive), while thermal is an
# unmeasured new modality the dataset paper ranks first of six sensors and both
# public notebooks discard. Unknown-with-large-upside goes first.
set -uo pipefail
cd "$(dirname "$0")/.."

echo "=== EXP-088 thermal video member, fold 2 $(date -Is) ==="
code/run_with_watchdog.sh logs/vidth_f2.log 'OUTER-ONCE' \
  python3 -u code/train_video_thermal.py --tag vidth_f2 \
    --fold-oof oof_visual_mil_v1_f2.npz --epochs 30 --workers 0 --resume
rm -f checkpoints/vidth_f2_resume.pt

for FOLD in 1 3; do
  echo "=== EXP-086b vid_r2p1d fold ${FOLD} $(date -Is) ==="
  code/run_with_watchdog.sh "logs/vid_r2p1d_f${FOLD}.log" 'OUTER-ONCE' \
    python3 -u code/train_video_crop.py \
      --tag "vid_r2p1d_f${FOLD}" \
      --fold-oof "oof_visual_mil_v1_f${FOLD}.npz" \
      --arch r2plus1d_18 --epochs 30 --workers 0 --resume
  rm -f "checkpoints/vid_r2p1d_f${FOLD}_resume.pt"
done

echo "=== GPU queue 2 done $(date -Is) ==="
