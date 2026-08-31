#!/usr/bin/env bash
#SBATCH --job-name=cuhkx-parity
#SBATCH --partition=gpu_h100_4
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/parity_%j.out
#
# ⚠ THE GATE. Nothing else runs until this passes.
#
# A silent environment difference (cuDNN algo choice, torchvision version, JPEG decoder)
# would make every cluster-vs-laptop number incomparable, and we would not notice for
# days. Two checks:
#   1. k224_mvit_f2 retrains to micro = 0.71472  (laptop reference, EXP-102)
#   2. build_video_slot reproduces testprobs_r2.npz to max abs diff 0.0
# Check 2 is exact because it is pure NumPy over stored probabilities. Check 1 involves
# fp16 autocast, which is not bit-deterministic across GPUs -- accept |delta| <= 0.005
# micro (about 3 clips of 652) and FAIL loudly outside that.
set -euo pipefail
source "$(dirname "$0")/activate.sh"
mkdir -p logs/slurm

echo "=== rebuilding the 224 person cache (96-core node) ==="
python3 -u kaggle/cuhkx_224_kaggle.py --stage cache --image-size 224 --cache-workers 64

echo "=== check 1: retrain k224_mvit_f2, expect micro = 0.71472 ==="
python3 -u kaggle/cuhkx_224_kaggle.py --stage train --fold 2 --tag parity_mvit_f2 \
        --batch-size 16 --accum 1 --workers 8
python3 - <<'PY'
import numpy as np, sys
d = np.load("oof_parity_mvit_f2.npz", allow_pickle=True)
micro = float((d["probs"].argmax(1) == d["labels"]).mean())
ref = 0.71472
print(f"  cluster micro={micro:.5f}  laptop ref={ref:.5f}  delta={micro-ref:+.5f}")
if abs(micro - ref) > 0.005:
    sys.exit(f"PARITY FAIL: |delta| {abs(micro-ref):.5f} > 0.005 — investigate before trusting any cluster number")
print("  check 1 PASS")
PY

echo "=== check 2: exact NumPy fusion parity ==="
python3 code/build_video_slot.py --tag parity_r2 --skel w25_p4 --imu imu_stats_t200_d12 \
        --no-motionbert --view person=k224_mvit_all_q6:0.5 --view wrist=k224_mvitwrist_all_q6:0.5
python3 - <<'PY'
import numpy as np, sys
a = np.load("research/artifacts/testprobs_r2.npz")["probs"].astype(float)
b = np.load("research/artifacts/testprobs_parity_r2.npz")["probs"].astype(float)
d = abs(a-b).max()
print(f"  max abs diff = {d:.3e}")
if d != 0.0:
    sys.exit("PARITY FAIL: NumPy fusion is not bit-identical — the artifacts differ")
print("  check 2 PASS")
PY
echo "=== PARITY GATE PASSED $(date) ==="
