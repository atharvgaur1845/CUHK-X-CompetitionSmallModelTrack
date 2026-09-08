#!/usr/bin/env bash
#SBATCH --job-name=cuhkx-th224
#SBATCH --partition=gpu_h100_4
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --array=0-3
#SBATCH --output=/scratch/pabitra/cuhkx/logs/th224_%A_%a.out
#
# L3a — thermal through the 224px/MViT recipe, 4 subject folds.
# Gate (EXP-123): member accuracy is the binding quantity, NOT calibration. Pooled must
# reach ~0.66 before the fusion question is even meaningful, and ~0.68 before a global
# weight can harvest it against 247 errors. Judge on micro first; do not fuse below that.
set -euo pipefail
export CUHKX_ROOT=/home/pabitra/cuhkx
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/home/pabitra/.conda/envs/physmon/bin/python
cd $CUHKX_ROOT
F=$SLURM_ARRAY_TASK_ID
echo "=== thermal224 fold $F on $(hostname) $(date) ==="
$P -u kaggle/cuhkx_224_kaggle.py --stage train --fold $F --tag k224_mvit_th_f$F \
   --cache-dir /scratch/pabitra/cuhkx/cache/thermal_224 \
   --batch-size 16 --accum 1 --workers 6 --seed 1
echo "=== fold $F done $(date) ==="
