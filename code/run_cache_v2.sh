#!/usr/bin/env bash
# Cache v2: 32 frames at 96x128 with the thermal span-reliability gate.
#
# EXP-068 measured accuracy flat down to 96x128 (sedentary +0.45) while 16->8
# frames costs 4.1 points -- v2 spends the freed 4x pixel budget on 2x frames,
# and still costs less per clip than v1.  The thermal gate clears the modality
# mask for the 23.6% of clips whose thermal/IR frame ratio says the two streams
# do not span the same window (median 2.43, std 0.70, min 0.02, max 7.60).
set -uo pipefail
cd "$(dirname "$0")/.."
export CUHKX_CACHE_SEGMENTS=32
export CUHKX_CACHE_HEIGHT=96
export CUHKX_CACHE_WIDTH=128
export CUHKX_THERMAL_RATIO_MIN=1.5
export CUHKX_THERMAL_RATIO_MAX=3.5
python3 -u code/visual_mil_cache.py build --split both --confirm-full-build \
  --output-dir cache/visual_mil_v2
echo "=== cache v2 build done $(date -Is) ==="
