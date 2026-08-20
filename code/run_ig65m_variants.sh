#!/usr/bin/env bash
# EXP-098: IG-65M on the variant caches, and a second IG-65M pretrain family.
#
# Two mechanisms have paid and nothing else has:
#   (a) a genuinely stronger member class -- IG-65M beat K400 on 4/4 folds, pooled
#       +4.36 micro, paired p=1.04e-08.
#   (b) DECORRELATED members ADDED to the bag (never replacing) -- g4only, which
#       replaced the K400 family, scored 157 against h8all's 162.
# Everything else is measured dead.
#
# The variants were each worth ~+1.4 on the WEAK backbone and together added +2 to h8
# (h8 160 -> h8all 162). Retrained on IG-65M they sit on a backbone that is +4.24 better,
# and the upper-body crop is aimed squarely at the 86.4% of errors that are object->object
# (Read_documents<->Turn_pages 49, tableware cluster ~60, Drink<->Eat 34).
#
# clip8 is a genuinely different pretrain of the same corpus, so it should decorrelate
# from clip32 the way IG-65M decorrelated from K400 (family agreement 0.7150).
set -uo pipefail
cd "$(dirname "$0")/.."

echo "=== EXP-098a ig65m on upper-body crop, fold 2 $(date -Is) ==="
code/run_with_watchdog.sh logs/vid_ig65m_upper_f2.log 'OUTER-ONCE' \
  python3 -u code/train_video_crop.py --tag vid_ig65m_upper_f2 \
    --fold-oof oof_visual_mil_v1_f2.npz --cache crop_upper --arch ig65m_34 \
    --batch-size 2 --accum 8 --epochs 30 --workers 0 --resume

echo "=== EXP-098b ig65m on 32-frame cache, fold 2 $(date -Is) ==="
code/run_with_watchdog.sh logs/vid_ig65m_f32_f2.log 'OUTER-ONCE' \
  python3 -u code/train_video_crop.py --tag vid_ig65m_f32_f2 \
    --fold-oof oof_visual_mil_v1_f2.npz --cache crop_v1f32 --arch ig65m_34 \
    --batch-size 2 --accum 8 --epochs 30 --workers 0 --resume
echo "=== EXP-098 done $(date -Is) ==="
