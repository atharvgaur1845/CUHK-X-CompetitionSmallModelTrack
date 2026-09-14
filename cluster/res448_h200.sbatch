#!/usr/bin/env bash
#SBATCH --job-name=cuhkx-p448
#SBATCH --partition=gpu_h200_8
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --array=2,3,4
#SBATCH --output=/scratch/pabitra/cuhkx/logs/p448h_%A_%a.out
#
# EXP-160 -- person crop at 448, the last step of the resolution curve (224 -> 288 -> 384
# was +1.5 / +0.4 on the member, EXP-154/158). Fire-and-forget: expected <= +1 clip, ships
# only if the pooled fusion OOF beats 384 by 2 SE AND the all-train run lands before the
# freeze. Same crop windows as crop_384 (windows.json copied), rendered at 448.
# Tasks 0,1 = folds 0,1; task 4 = all-train. Folds 2,3 run on the H200 partition
# (res448_h200.sbatch) so the five runs finish in two rounds instead of three.
set -euo pipefail
export CUHKX_ROOT=/home/pabitra/cuhkx
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/home/pabitra/.conda/envs/physmon/bin/python
C=/scratch/pabitra/cuhkx
mkdir -p $C/runs; cd $C/runs
S=$SLURM_ARRAY_TASK_ID
BASE=k224_mvit448
if [ "$S" = "4" ]; then ARGS="--all-train --tag ${BASE}_all"; else ARGS="--fold $S --tag ${BASE}_f$S"; fi
echo "=== p448 task $S on $(hostname) $(date) ==="
$P -u $CUHKX_ROOT/kaggle/cuhkx_224_kaggle.py --stage train $ARGS \
   --cache-dir $C/cache/crop_448 --image-size 448 --data-root /scratch/pabitra/cuhkx \
   --batch-size 4 --accum 4 --workers 3 --seed 1
echo "=== task $S done $(date) ==="
