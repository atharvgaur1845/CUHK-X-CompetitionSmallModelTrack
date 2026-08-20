#!/usr/bin/env bash
# Queue: upper-body crop, then the two MotionBERT knobs most likely to close its gap.
#
# EXP-094 measured MotionBERT pretrained 0.57515 vs identical-architecture random init
# 0.35276 -- pretraining is worth +22.24 micro, reproducing the visual branch's +23.9 and
# confirming R-1 on the second branch. But a single MotionBERT (0.58514 on the fusion
# subset) only TIES our 48-member from-scratch world25 stack (0.59420), and it adds just
# +1.45 to the fusion at w=0.05-0.15 -- under the +3 gate.
#
# Its calibration signature is the thermal one again: uniquely correct on 37 clips the
# champion misses, but confidence 0.335 there vs 0.299 when wrong. A 0.036 separation is
# too small to gate on. What it has that thermal lacked is genuine decorrelation from the
# base (argmax agreement 0.5906) plus real strength, which is why it gives +1.45 not 0.00.
#
# Two knobs are untested and both are principled rather than a sweep:
#   n-frames 243 -- the length MotionBERT was pretrained and NTU-finetuned at. We used 64.
#                   A transformer with learned positional embeddings is sensitive to this.
#   depth-channel -- we have genuine 3D where NTU HRNet had only a detector score, so the
#                   third channel is free information rather than a constant 1.0.
set -uo pipefail
cd "$(dirname "$0")/.."

echo "=== EXP-093 upper-body crop, fold 2 $(date -Is) ==="
code/run_with_watchdog.sh logs/vid_upper_f2.log 'OUTER-ONCE' \
  python3 -u code/train_video_crop.py --tag vid_upper_f2 \
    --fold-oof oof_visual_mil_v1_f2.npz --cache crop_upper \
    --arch r2plus1d_18 --epochs 30 --workers 0 --resume

echo "=== EXP-094c MotionBERT, 243 frames $(date -Is) ==="
code/run_with_watchdog.sh logs/skel_mb_f243.log 'OUTER-ONCE' \
  python3 -u code/train_skel_motionbert.py --tag skel_mb_f243 --n-frames 243 \
    --batch-size 2 --accum 8 --fold-oof oof_visual_mil_v1_f2.npz --epochs 30 --workers 0 --resume

echo "=== EXP-094d MotionBERT, depth in the confidence channel $(date -Is) ==="
code/run_with_watchdog.sh logs/skel_mb_depth.log 'OUTER-ONCE' \
  python3 -u code/train_skel_motionbert.py --tag skel_mb_depth --depth-channel \
    --fold-oof oof_visual_mil_v1_f2.npz --epochs 30 --workers 0 --resume

echo "=== queue3 done $(date -Is) ==="
