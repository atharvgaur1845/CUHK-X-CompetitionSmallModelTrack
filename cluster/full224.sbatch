#!/usr/bin/env bash
#SBATCH --job-name=cuhkx-full224
#SBATCH --partition=gpu_a100_8
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --array=0-4
#SBATCH --output=/scratch/pabitra/cuhkx/logs/full224_%A_%a.out
#
# EXP-143 -- IR+Depth on the FULL FRAME at 224px, the framing nobody has tried.
#
# The thermal gain (EXP-139) arrived at 224px on full frames, and the EXP-140
# leave-one-out then found the person CROP to be the most droppable member of the
# ensemble -- consistent with thermal supplying the scene/object context that cropping
# deletes. 75% of the residual error is OBJECT classes and EXP-068 measured that the
# objects are in the pixels. IR+Depth is our strongest modality (0.7116 cropped) and has
# never been seen uncropped at 224px.
#
# The cache was built WITHOUT touching the trainer: code/make_fullframe_windows.py writes
# a windows.json mapping every sid to [], which _render treats as "no box" (full frame)
# and compute_windows treats as "already done" (no YOLO). Adding a --crop choice would
# have changed vars(args) and invalidated --resume for the LOSO arrays in flight.
#
# Task 4 is the all-train model: a view is only shippable if that exists (EXP-142 needed
# an unplanned extra run because the thermal all-train model did not).
set -euo pipefail
export CUHKX_ROOT=/home/pabitra/cuhkx
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/home/pabitra/.conda/envs/physmon/bin/python
mkdir -p /scratch/pabitra/cuhkx/runs
cd /scratch/pabitra/cuhkx/runs
F=$SLURM_ARRAY_TASK_ID
if [ "$F" = "4" ]; then
  ARGS="--all-train --tag k224_mvit_full_all"
else
  ARGS="--fold $F --tag k224_mvit_full_f$F"
fi
echo "=== full224 task $F on $(hostname) $(date) ==="
$P -u $CUHKX_ROOT/kaggle/cuhkx_224_kaggle.py --stage train $ARGS \
   --cache-dir /scratch/pabitra/cuhkx/cache/full224 \
   --batch-size 16 --accum 1 --workers 6 --seed 1
echo "=== task $F done $(date) ==="
