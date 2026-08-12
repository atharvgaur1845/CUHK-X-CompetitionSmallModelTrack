#!/usr/bin/env bash
# EXP-076 gate: does the SSL init beat an identical-recipe scratch run?
#
# This is the decisive test of the whole SSL direction.  Predeclared gate (X-03):
# >= +2.0 micro on fold 2, with sedentary accuracy reported as the primary metric
# (EXP-067: overall accuracy is 39% dominated by an already-solved sub-problem).
# The seed floor on this partition is sigma = 2.80 (DA-006-A), so a single-fold
# +2 is DIRECTIONAL ONLY -- replicate on fold 0 before adopting anything.
#
# Both arms use identical recipe, data, folds and epoch budget; the ONLY
# difference is --init-trunk.  Anything else and the comparison is worthless.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUHKX_CACHE_SEGMENTS=32
export CUHKX_CACHE_HEIGHT=96
export CUHKX_CACHE_WIDTH=128
# EXP-062/065 regularized constants (replicated +2.5 micro / +3.0 object)
export CUHKX_VMIL_CROP_MIN=0.60
export CUHKX_VMIL_DROPOUT=0.35
export CUHKX_VMIL_MDROP_IR=0.25
export CUHKX_VMIL_MDROP_DEPTH=0.25
export CUHKX_VMIL_MDROP_THERMAL=0.35
export CUHKX_VMIL_LABEL_SMOOTHING=0.10
export CUHKX_GATE_OVERRIDE_LEADERBOARD=1

COMMON="--fold 2 --device cuda --batch-size 1 --workers 2 --cache-dir cache/visual_mil_v2 --resume"

echo "=== ARM A: scratch $(date -Is) ==="
code/run_with_watchdog.sh logs/ssl_screen_scratch.log 'OUTER-ONCE' \
  python3 -u code/train_visual_mil.py train --tag v2_scratch_f2 ${COMMON}

echo "=== ARM B: SSL init $(date -Is) ==="
code/run_with_watchdog.sh logs/ssl_screen_ssl.log 'OUTER-ONCE' \
  python3 -u code/train_visual_mil.py train --tag v2_ssl_f2 ${COMMON} \
    --init-trunk checkpoints/ssl_trunk_v2.pt

echo "=== screen done $(date -Is) ==="
