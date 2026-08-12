#!/usr/bin/env bash
# EXP-078: SSL init on the geometry that is KNOWN to transfer (cache v1).
#
# Why the v2 deployment job was killed mid-run.  Public results:
#   v1 visual member (4-fold, 16f, 192x256, scratch)  OOF 0.379 -> public 0.38805
#   v2 SSL member    (1-fold, 32f,  96x128, SSL)      OOF 0.394 -> public 0.28358
# and LEADERBOARD.md already measured that 1-fold -> 4-fold bagging is worth
# ZERO on public (123 = 123, 59 rows changed, net nil).  So the -10.4 is caused by
# SSL init and/or cache v2 geometry -- bagging cannot absorb any of it, and folds
# 0/1/3 would only have finished a member whose fold-2 solo already scored 0.284.
#
# Eliminated as the cause (measured, not assumed): the v2 thermal span gate fires
# on 16.2% of train and 16.0% of test clips -- symmetric, so it cannot produce a
# one-sided transfer drop (it is still a real capability loss, 16% vs v1's 3%, on
# the paper's best single modality).
#
# This run separates the two remaining candidates with ONE job, and unlike a
# scratch control it can also win: SSL init on v1's 16-frame 192x256 cache, the
# geometry whose member transfers at +0.9.  The SSL trunk is fully convolutional,
# so 96x128-pretrained weights load into a 192x256 model unchanged; --init-trunk
# raises rather than silently falling back if any expected tensor fails to transfer.
#
#   beats 0.38805 public solo -> SSL is good, cache v2 geometry was the problem
#   lands near 0.284          -> SSL init itself does not survive new subjects; kill it
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# cache v1 geometry: 16 frames at 192x256 (the module defaults)
export CUHKX_CACHE_SEGMENTS=16
export CUHKX_CACHE_HEIGHT=192
export CUHKX_CACHE_WIDTH=256
# EXP-062/065 regularized constants, identical to the v1 member's recipe
export CUHKX_VMIL_CROP_MIN=0.60
export CUHKX_VMIL_DROPOUT=0.35
export CUHKX_VMIL_MDROP_IR=0.25
export CUHKX_VMIL_MDROP_DEPTH=0.25
export CUHKX_VMIL_MDROP_THERMAL=0.35
export CUHKX_VMIL_LABEL_SMOOTHING=0.10
export CUHKX_GATE_OVERRIDE_LEADERBOARD=1

echo "=== EXP-078 fold 2: SSL init on cache v1 $(date -Is) ==="
code/run_with_watchdog.sh logs/ssl_v1geo_f2.log 'OUTER-ONCE' \
  python3 -u code/train_visual_mil.py train --tag v1geo_ssl_f2 --fold 2 \
    --device cuda --batch-size 1 --workers 2 --cache-dir cache/visual_mil_v1 --resume \
    --init-trunk checkpoints/ssl_trunk_v2.pt
echo "=== done $(date -Is) ==="
