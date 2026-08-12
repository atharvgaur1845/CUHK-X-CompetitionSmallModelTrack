#!/usr/bin/env bash
# EXP-077: promote the SSL init from a fold-2 screen result to a deployable member.
#
# EXP-076 gate on fold 2 (identical recipe, only --init-trunk differs):
#   micro   0.33282 -> 0.39417  (+6.14, gate was +2.0)
#   object  0.23591 -> 0.32359  (+8.77; 113 -> 155 of 479 sedentary clips)
#   motion  0.60116 -> 0.58960  (-1.16, noise)
# The gain loads entirely onto the 26 sedentary classes that hold 75% of the
# error mass, and leaves motion flat.  A seed fluctuation of +6.14 micro would
# not respect that partition, which is why this is treated as mechanism rather
# than noise -- but sigma = 2.80 on this partition still makes fold 2 alone
# DIRECTIONAL.  Folds 0/1/3 below are simultaneously the replication and the
# deployment artifact: four fold checkpoints bag into the test member, and the
# concatenated outer predictions give an honest 4-fold OOF.
#
# Fold 2 is already trained (checkpoints/v2_ssl_f2_f2.pt) and is skipped.
set -uo pipefail
cd "$(dirname "$0")/.."
source code/ssl_v2_env.sh

COMMON="--device cuda --batch-size 1 --workers 2 --cache-dir cache/visual_mil_v2 --resume"

for FOLD in 0 1 3; do
  echo "=== fold ${FOLD}: SSL init $(date -Is) ==="
  code/run_with_watchdog.sh "logs/ssl_deploy_f${FOLD}.log" 'OUTER-ONCE' \
    python3 -u code/train_visual_mil.py train --tag v2_ssl_f2 --fold "${FOLD}" ${COMMON} \
      --init-trunk checkpoints/ssl_trunk_v2.pt
done

echo "=== all folds done $(date -Is) ==="
