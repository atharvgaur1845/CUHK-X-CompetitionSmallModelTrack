#!/usr/bin/env bash
# EXP-076: cross-modal masked SSL on the visual trunk, cache v2 (32 frames, 96x128).
#
# The legal substitute for the pretrained init the paper's 90-92% visual baselines
# depend on: EXP-015 measured ImageNet init at roughly 2x from-scratch here, and
# pretrained weights are banned.  EXP-021 measured this objective at +3.0 but ran
# it on a transformer trunk that lost for unrelated reasons; trunk identity with
# train_visual_mil.py is the whole point of this re-attachment.
#
# Corpus is all 3,338 clips including unlabelled test (transduction ruled legal).
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUHKX_CACHE_SEGMENTS=32
export CUHKX_CACHE_HEIGHT=96
export CUHKX_CACHE_WIDTH=128
python3 -u code/ssl_pretrain_visual.py \
  --cache-dir cache/visual_mil_v2 --epochs 40 --batch-size 4 --frames 12 --workers 4 \
  --tag ssl_trunk_v2
echo "=== ssl pretrain done $(date -Is) ==="
