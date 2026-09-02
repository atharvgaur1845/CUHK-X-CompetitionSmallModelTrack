#!/usr/bin/env bash
# EXP-120: is the EXP-118/119 full-frame thermal gain real, or one fold's seed draw?
#
# Fold 2 alone gave full-frame 0.55828 vs cropped 0.54448 (+1.38 micro). That is
# below this partition's adoption bar and EXP-119 said so. Three of the campaign's
# retracted results were exactly this shape: one fold, one draw, +2.45.
#
# Only fold 2 exists for EITHER thermal variant, so folds 0/1/3 need BOTH arms or
# the comparison is unpaired and unreadable. Runs are ordered fold-major so that
# stopping this queue at any point still leaves complete PAIRS on disk -- an
# interrupted fold-major queue is analysable, an interrupted arm-major queue is not.
#
# ~2.6 h per run, 6 runs, ~16 h. Both arms are resumable; the watchdog restarts a
# stall (EXP-064 hung 12 h with the process alive and the GPU idle).
set -uo pipefail
cd "$(dirname "$0")/.."

run () {                       # run <cache> <tag> <fold>
  local cache="$1" tag="$2" fold="$3"
  if [ -f "research/artifacts/oof_${tag}.npz" ]; then
    echo "== ${tag}: already on disk, skipping"; return 0
  fi
  echo "== ${tag} (cache=${cache}, fold=${fold})"
  code/run_with_watchdog.sh "logs/${tag}.log" 'OUTER-ONCE' \
    python3 -u code/train_video_thermal.py \
      --cache "$cache" --tag "$tag" \
      --fold-oof "oof_visual_mil_v1_f${fold}.npz" --resume
}

for F in 0 1 3; do
  run thermal_full "vidth_full_f${F}" "$F"   # the hypothesis
  run thermal_v1   "vidth_f${F}"       "$F"  # its paired control
done
echo "== thermal pair queue complete"
