#!/usr/bin/env bash
# EXP-093: upper-body crop -- targeted at the 86.4% of errors that are object->object.
#
# Measured, not guessed. The champion's OOF confusion matrix over 2,700 clips:
#   object->object 677 (86.4%) | object->motion 51 (6.5%) | motion->* 56 (7.1%)
# and the mass sits in symmetric adjacent pairs that share a posture and differ only
# in a small held object or a fine hand trajectory:
#   21 Read_documents <-> 22 Turn_pages          49
#   8/9/10/11 tableware/pour/stir/peel          ~60
#   6 Drink_water    <-> 7 Eat_food              34
#   12 Sweep_floor   <-> 13 Mop_floor            31
# In a 128x128 whole-body crop the hands are ~15x15 px. Cropping to the top 62% of the
# person box magnifies them ~3x for the same compute.
#
# Corroboration on the same axis: EXP-091 (128->160, a mere 1.56x pixel increase) moved
# object +13 clips (p=0.096) while motion went -5. Right direction, too small a lever.
#
# Waits for the res160 folds so the two jobs never share the GPU.
set -uo pipefail
cd "$(dirname "$0")/.."
while pgrep -f '^python3 -u code/train_video_crop.py' >/dev/null; do sleep 120; done
# Gate on the build driver's DONE marker, not an index file: a 6-clip smoke test
# had already written cache/crop_upper/test_index.json, which satisfied an
# index-file gate and launched training against a train split that did not exist.
while ! grep -q DONE logs/upper_driver.log 2>/dev/null; do sleep 60; done
sleep 30
echo "=== EXP-093 upper-body crop, fold 2 $(date -Is) ==="
code/run_with_watchdog.sh logs/vid_upper_f2.log 'OUTER-ONCE' \
  python3 -u code/train_video_crop.py --tag vid_upper_f2 \
    --fold-oof oof_visual_mil_v1_f2.npz --cache crop_upper \
    --arch r2plus1d_18 --epochs 30 --workers 0 --resume
echo "=== EXP-093 done $(date -Is) ==="
