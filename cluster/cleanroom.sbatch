#!/usr/bin/env bash
#SBATCH --job-name=cuhkx-cleanroom
#SBATCH --partition=gpu_a100_8
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/pabitra/cuhkx/logs/cleanroom_%j.out
#
# EXP-159 -- clean-room, cross-device reproduction of the Stage-2 package.
#
# The organisers will run inference.sh on THEIR machine. It had only ever run on the
# laptop (RTX, torch 2.12+cu130, py3.12). This job runs it on an A100 with
# torch 2.7.1+cu126 / py3.11 from a code-only copy of the repo: no caches, no training
# data, no checkpoints, no research/ tree -- just code/, kaggle/, inference_assets/,
# inference.sh and the 96 MB package. The work dir is wiped first so every cache is
# rebuilt from raw frames with the detectors extracted from the package.
#
# Gate: rowdiff 0 against submissions/sub_final_ship7.csv (the laptop's output from raw).
set -euo pipefail
C=/scratch/pabitra/cuhkx/cleanroom
TEST=/scratch/pabitra/cuhkx/Small-Model-Track/Testing/data/small_model_track_test
export PYTHON=$C/venv/bin/python           # physmon torch 2.7.1+cu126 + ultralytics 8.4.118
export WORKERS=${WORKERS:-6}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=$C/out_${SLURM_JOB_ID}.csv

cd $C/repo
rm -rf $C/work cache research Small-Model-Track        # clean room: nothing derived survives
nvidia-smi -L; $PYTHON -c "import torch,torchvision,ultralytics;print('torch',torch.__version__,'tv',torchvision.__version__,'ultralytics',ultralytics.__version__)"
echo "=== start $(date -u) ==="
time ./inference.sh $C/stage2_ship7.pth $TEST $OUT $C/work
echo "=== end $(date -u) ==="
echo "=== rowdiff vs laptop sub_final_ship7.csv ==="
$PYTHON code/rowdiff.py $C/ref_sub_final_ship7.csv $OUT
