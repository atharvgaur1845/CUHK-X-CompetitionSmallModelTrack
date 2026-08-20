#!/usr/bin/env bash
# EXP-096: sample the FINAL 55% of each clip, not the whole clip.
#
# EXTERNAL EVIDENCE, not our own ideation (research/DIRECTIVE.md STEP 2). Competition
# forum topic 735601, 2026-08-17, a participant describing the corpus:
#   "some activity clips contain other activities (such as writing might contain writing,
#    turning pages, and reading a document; or ... lying down containing walking, sitting
#    down, and then lying down)"
# The second example is the mechanism: the label is the activity the clip ENDS on.
#
# This predicts our error structure rather than being fitted to it. Measured champion
# confusions over 2,700 OOF clips: 86.4% object->object, largest cluster
# 21 Read_documents <-> 22 Turn_pages at 49 errors -- the exact pair that participant
# names. Uniform sampling + global average pooling averages the labelled activity with
# lead-in activities that are themselves valid classes.
#
# Single change vs EXP-086 fold 2 (micro 0.63957 / object 0.54906): the temporal window.
# Falsifiable and it forbids something: if the label were the DOMINANT activity rather
# than the FINAL one, restricting to the tail must LOSE accuracy.
set -uo pipefail
cd "$(dirname "$0")/.."
while ! grep -q DONE logs/late_driver.log 2>/dev/null; do sleep 60; done
sleep 20
echo "=== EXP-096 late-window fold 2 $(date -Is) ==="
code/run_with_watchdog.sh logs/vid_late_f2.log 'OUTER-ONCE' \
  python3 -u code/train_video_crop.py --tag vid_late_f2 \
    --fold-oof oof_visual_mil_v1_f2.npz --cache crop_late \
    --arch r2plus1d_18 --epochs 30 --workers 0 --resume
echo "=== EXP-096 done $(date -Is) ==="
