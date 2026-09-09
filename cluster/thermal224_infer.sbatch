#!/usr/bin/env bash
#SBATCH --job-name=cuhkx-thinf
#SBATCH --partition=gpu_h100_4
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/pabitra/cuhkx/logs/thinfer_%j.out
#
# Rebuild the thermal TEST cache (the first data sync delivered 334 of 395 thermal test
# directories despite reporting success), then infer with all four fold models.
# Outputs go to SCRATCH: /home is quota-capped at 40G and truncated writes are what
# killed 7 of the 18 LOSO tasks.
set -euo pipefail
export CUHKX_ROOT=/home/pabitra/cuhkx
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/home/pabitra/.conda/envs/physmon/bin/python
C=/scratch/pabitra/cuhkx
cd $CUHKX_ROOT
echo "=== rebuild thermal test cache $(date) ==="
$P -u code/build_thermal224_cache.py --split test --workers 8 \
   --ref-cache $C/cache/crop_224 --out $C/cache/thermal_224
cd $C/checkpoints          # stage_infer reads {tag}.pt from the CWD and writes beside it
for F in 0 1 2 3; do
  echo "=== infer fold $F $(date) ==="
  $P -u $CUHKX_ROOT/kaggle/cuhkx_224_kaggle.py --stage infer --tag k224_mvit_th_f$F \
     --cache-dir $C/cache/thermal_224 --data-root $C --batch-size 32 --workers 6
done
echo "=== done $(date) ==="; ls -la testprobs_k224_mvit_th_f*.npz
