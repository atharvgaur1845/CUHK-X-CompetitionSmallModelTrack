#!/usr/bin/env bash
#SBATCH --job-name=cuhkx-r288
#SBATCH --partition=gpu_h100_4
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00:00
#SBATCH --array=0-9
#SBATCH --output=/scratch/pabitra/cuhkx/logs/r288_%A_%a.out
#
# EXP-154 -- INPUT RESOLUTION 224 -> 288 on the two views that have never had it.
#
# Measured, not speculative. Person is the ONLY view ever trained at 288, and it beat its
# own 224 model on 4 of 4 folds: +0.86 / +0.86 / +0.92 / +2.45, pooled 0.71156 -> 0.72383.
# Swapped into the shipped fusion it is +11 clips of 2700 and negative on no fold.
#
# The reason to expect more, not less, from the other two views:
#   wrist   -- a small hand region upscaled to the input, so resolution is the binding
#              constraint more than it is for a full-body crop
#   thermal -- native frames are 320x240, so 288 is NEARER NATIVE than 224 and nothing is
#              invented by upsampling
#
# And the reason this is worth the GPU-hours over anything else left: it costs ZERO package
# bytes. Same MViTv2-S, same 25.9 MB at int5 -- only inference compute changes. Every other
# surviving lever needed bytes we do not have.
set -euo pipefail
export CUHKX_ROOT=/home/pabitra/cuhkx
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/home/pabitra/.conda/envs/physmon/bin/python
C=/scratch/pabitra/cuhkx
mkdir -p $C/runs; cd $C/runs
I=$SLURM_ARRAY_TASK_ID
V=$(( I / 5 ))      # 0 wrist, 1 thermal
S=$(( I % 5 ))      # 0-3 folds, 4 all-train
case $V in
  0) CACHE=$C/cache/crop_wrist288; BASE=k224_mvitwrist288 ;;
  1) CACHE=$C/cache/thermal_288;   BASE=k224_mvit_th288 ;;
esac
if [ "$S" = "4" ]; then ARGS="--all-train --tag ${BASE}_all"; else ARGS="--fold $S --tag ${BASE}_f$S"; fi
echo "=== r288 task $I: view=$V slot=$S on $(hostname) $(date) ==="
$P -u $CUHKX_ROOT/kaggle/cuhkx_224_kaggle.py --stage train $ARGS \
   --cache-dir $CACHE --image-size 288 --batch-size 8 --accum 2 --workers 3 --seed 1
echo "=== task $I done $(date) ==="
