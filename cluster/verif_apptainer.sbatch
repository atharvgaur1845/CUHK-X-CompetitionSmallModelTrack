#!/usr/bin/env bash
#SBATCH --job-name=cuhkx-verif
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/pabitra/cuhkx/logs/verif_%x_%j.out
#
# EXP-161 / gate G8 -- run the verification IMAGE (the docker save archive converted to a SIF)
# on a cluster GPU with Apptainer: --nv for the GPU, --containall so nothing from the host
# leaks in, writable tmpfs for the run's scratch. Partition and GRES come from sbatch flags:
#   sbatch -p gpu_a100_8 --gres=gpu:nvidia_a100-sxm4-80gb:1 cluster/verif_apptainer.sbatch
set -euo pipefail
V=/scratch/pabitra/cuhkx/verif
DATA=/scratch/pabitra/cuhkx/Small-Model-Track/Testing
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_' | tr -d '/')
OUT=$V/out_${GPU}_$SLURM_JOB_ID; mkdir -p $OUT
echo "=== $GPU  job $SLURM_JOB_ID  $(date -u) ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
for SUB in 56213683 56212349; do
  echo "--- submission $SUB"
  apptainer exec --nv --containall --no-home --writable-tmpfs --pwd /package \
      --env CUHKX_WORK=/output/work_$SUB --env WORKERS=8 \
      -B $DATA:/data:ro -B $OUT:/output $V/cuhkx-small-agx-v1.sif \
      bash inference.sh /data /output/submission_$SUB.csv $SUB 2>&1 | grep -E "^== |WARNING|ERROR|FAILED|windows crop_384|Traceback" 
  sha256sum $OUT/submission_$SUB.csv
done
echo "=== done $(date -u) ==="
