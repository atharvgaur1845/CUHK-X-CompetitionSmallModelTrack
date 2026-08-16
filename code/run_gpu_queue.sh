#!/usr/bin/env bash
# GPU queue reorder: thermal jumps ahead of video folds 1/3.
#
# Rationale. Bagging folds 1/3 onto the existing fold-2 member is a KNOWN quantity
# (+2-5 clips, the usual bag gain). The thermal member is UNKNOWN and potentially
# much larger: the dataset paper ranks thermal first of six sensors (92.57 > depth
# 90.5 > IR 90.2), both public notebooks discard it, and EXP-086 measured that a
# decorrelated member earns fusion weight far above its solo strength (IMU: solo
# 0.36, weight 0.45, +3.4 micro to the fusion). Information per GPU-hour favours
# thermal, so it runs first.
#
# Waits for fold 0 to finish on its own rather than killing it mid-epoch -- it is
# at epoch 24/30 and its resume state is intact, but a restart under a watchdog
# would re-run the tail for nothing.
set -uo pipefail
cd "$(dirname "$0")/.."

echo "[queue] waiting for vid_r2p1d_f0 to finish $(date -Is)"
while ! grep -q 'OUTER-ONCE' logs/vid_r2p1d_f0.log 2>/dev/null; do
  if ! pgrep -f 'train_video_crop.py --tag vid_r2p1d_f0' >/dev/null; then
    echo "[queue] fold 0 process gone without a result marker; continuing anyway"
    break
  fi
  sleep 60
done
echo "[queue] fold 0 done $(date -Is)"
grep 'OUTER-ONCE' logs/vid_r2p1d_f0.log 2>/dev/null

# Stop the old driver so it cannot start fold 1 underneath us.
pkill -f 'bash code/run_vid_folds.sh' 2>/dev/null
sleep 5
pkill -f 'train_video_crop.py --tag vid_r2p1d_f1' 2>/dev/null
sleep 5

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

echo "=== GPU queue done $(date -Is) ==="
