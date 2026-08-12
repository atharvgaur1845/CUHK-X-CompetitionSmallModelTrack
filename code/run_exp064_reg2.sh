#!/usr/bin/env bash
# EXP-064: regularized visual MIL recipe (EXP-062 overrides) at 150 epochs, fold 2.
#
# Lives in the repo rather than /tmp because /tmp is cleared between sessions --
# a restart on 2026-08-07 failed for exactly that reason after the job had already
# lost three idle days.
#
# Run it under tmux so it survives shell teardown:
#   tmux new-session -d -s reg2 "bash code/run_exp064_reg2.sh >> logs/reg2_watchdog.log 2>&1"
# The watchdog handles stalls; tmux handles session teardown.  Both are needed:
# setsid alone did not survive teardown on 2026-08-04.
set -uo pipefail
cd "$(dirname "$0")/.."

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUHKX_VMIL_CROP_MIN=0.60
export CUHKX_VMIL_DROPOUT=0.35
export CUHKX_VMIL_MDROP_IR=0.25
export CUHKX_VMIL_MDROP_DEPTH=0.25
export CUHKX_VMIL_MDROP_THERMAL=0.35
export CUHKX_VMIL_LABEL_SMOOTHING=0.10
export CUHKX_VMIL_EPOCHS=150

exec code/run_with_watchdog.sh logs/visual_mil_reg2_e150.log 'OUTER-ONCE' \
  python3 -u code/train_visual_mil.py train --tag visual_mil_reg2_e150 --fold 2 \
    --device cuda --batch-size 1 --workers 2 --resume
