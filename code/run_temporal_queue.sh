#!/usr/bin/env bash
# EXP-110: wrist 32-frame cache + TTA, then a fold-2 retrain that uses the deeper cache
# as TEMPORAL JITTER augmentation (make_dataset picks a random phase when train=True).
# TTA reuses existing checkpoints; the retrain tests whether the discarded 50.9% of
# frames are worth more as training signal than as test-time averaging.
set -u
cd "$(dirname "$0")/.."
python3 -u kaggle/cuhkx_224_kaggle.py --stage cache --frames 32 --crop wrist || exit 1
echo "=== wrist t32 cache built $(date) ==="
python3 -u code/temporal_tta_oof.py --prefix k224_mvitwrist --crop wrist \
        --out oof_k224_mvitwrist_t32 || exit 1
echo "=== wrist TTA measured $(date) ==="
T=k224_mvitjit_f2
python3 -u kaggle/cuhkx_224_kaggle.py --stage train --fold 2 --frames 32 --tag "$T" || exit 1
mv -f "oof_${T}.npz" research/artifacts/ 2>/dev/null
mv -f "${T}.pt" checkpoints/ 2>/dev/null
rm -f "${T}_resume.pt"
echo "=== QUEUE COMPLETE $(date) ==="
