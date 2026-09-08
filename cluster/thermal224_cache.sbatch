#!/usr/bin/env bash
#SBATCH --job-name=cuhkx-thcache
#SBATCH --partition=compute
#SBATCH --cpus-per-task=48
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/pabitra/cuhkx/logs/thcache_%j.out
#
# Build the 224px full-frame thermal cache in the JPEG format the MViT trainer reads.
# CPU-only (no YOLO: full frame), so it belongs on a compute node, not a GPU one.
set -euo pipefail
export CUHKX_ROOT=/home/pabitra/cuhkx
P=/home/pabitra/.conda/envs/physmon/bin/python
cd $CUHKX_ROOT
echo "=== thermal cache build on $(hostname) $(date) ==="
$P -u code/build_thermal224_cache.py --split train --workers 48 --ref-cache /scratch/pabitra/cuhkx/cache/crop_224 \
   --out /scratch/pabitra/cuhkx/cache/thermal_224
$P -u code/build_thermal224_cache.py --split test  --workers 48 --ref-cache /scratch/pabitra/cuhkx/cache/crop_224 \
   --out /scratch/pabitra/cuhkx/cache/thermal_224
echo "=== cache done $(date) ==="
ls -la /scratch/pabitra/cuhkx/cache/thermal_224
