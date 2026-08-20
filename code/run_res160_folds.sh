#!/usr/bin/env bash
# EXP-091b: 160px on the remaining folds.
#
# EXP-091/092 measured that 32f@128 and 16f@160 are each ~+1.3 solo (neither
# significant alone) but BAG to +3.83 micro / +4.38 object on fold 2, clearing the
# pre-registered +3 gate. They agree only 75.9% with each other: decorrelated AND
# individually strong, which is exactly what thermal lacked (thermal was
# decorrelated but uncalibrated, and contributed zero).
#
# Adding the 16f@128 baseline to that pair makes it WORSE (+2.61 vs +3.83), so the
# original variant is the one to drop, not to keep.
#
# 160px is the cheaper of the two variants (~3h vs ~6h for 32f, which pages a 6.15GB
# memmap against 4GB of free RAM), so it goes first. Target: a 4-fold 160px bag plus
# the 32f fold-2 member, giving BOTH variant and subject diversity.
set -uo pipefail
cd "$(dirname "$0")/.."
for FOLD in 0 1 3; do
  echo "=== EXP-091b res160 fold ${FOLD} $(date -Is) ==="
  code/run_with_watchdog.sh "logs/vid_res160_f${FOLD}.log" 'OUTER-ONCE' \
    python3 -u code/train_video_crop.py --tag "vid_res160_f${FOLD}" \
      --fold-oof "oof_visual_mil_v1_f${FOLD}.npz" --cache crop_v160 \
      --arch r2plus1d_18 --epochs 30 --workers 0 --resume
  rm -f "checkpoints/vid_res160_f${FOLD}_resume.pt"
done
echo "=== EXP-091b done $(date -Is) ==="
