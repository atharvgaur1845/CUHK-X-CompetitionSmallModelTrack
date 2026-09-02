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
#   1. k224_mvit_f2 seed-replicates and the laptop 0.71472 falls inside the spread
#   2. build_video_slot reproduces testprobs_r2.npz to max abs diff 0.0
# Check 2 is exact because it is pure NumPy over stored probabilities. Check 1 is NOT
# exact and must never be written as an equality: the trainer seeds nothing, so 0.71472
# is a single draw of a random process and the laptop cannot reproduce it either. The
# old |delta| <= 0.005 tolerance was fiction -- Kaggle came back 0.03220 low and the
# gate could not say whether that meant a broken environment (EXP-120). Replicate under
# --seed and test whether the reference lies inside the spread.
set -euo pipefail
source "$(dirname "$0")/activate.sh"
mkdir -p logs/slurm

echo "=== rebuilding the 224 person cache (96-core node) ==="
python3 -u kaggle/cuhkx_224_kaggle.py --stage cache --image-size 224 --cache-workers 64

# check 1 used to demand micro == 0.71472 within 0.005. That gate was unmeetable:
# the trainer seeds nothing, so 0.71472 is one draw and the LAPTOP fails it too.
# Kaggle returned 0.68252 and the gate could not tell a broken environment from an
# ordinary draw (EXP-120). Replicate instead of demanding equality, and require the
# laptop reference to fall inside the spread. --batch-size 16 --accum 1 keeps
# steps = len(tl)//accum = 142, identical to the laptop's bs4 x accum4 schedule.
echo "=== check 1: 3 seeded replicates of k224_mvit_f2; laptop 0.71472 must sit inside ==="
for S in 1 2 3; do
  python3 -u kaggle/cuhkx_224_kaggle.py --stage train --fold 2 --tag "parity_mvit_f2_s${S}" \
          --seed "$S" --batch-size 16 --accum 1 --workers 8
done
python3 - <<'PYCHK'
import numpy as np, sys
m = []
for s in (1, 2, 3):
    d = np.load(f"oof_parity_mvit_f2_s{s}.npz", allow_pickle=True)
    m.append(float((d["probs"].argmax(1) == d["labels"]).mean()))
mu, sd = float(np.mean(m)), float(np.std(m, ddof=1))
ref = 0.71472                      # laptop, EXP-102 -- one unseeded draw, not a constant
lo, hi = mu - 2 * sd, mu + 2 * sd
print("  replicates: " + "  ".join(f"{x:.5f}" for x in m))
print(f"  mean={mu:.5f}  seed sd={sd:.5f}  2sd=[{lo:.5f}, {hi:.5f}]  laptop ref={ref:.5f}")
print(f"  >>> VISUAL SEED SIGMA = {100*sd:.2f} points -- first measurement (LOG.md:2069)")
if not (lo <= ref <= hi):
    sys.exit("PARITY FAIL: laptop reference lies outside the replicate spread -- that is "
             "an environment difference, not a draw. Investigate before trusting any "
             "number from this machine.")
print("  check 1 PASS")
PYCHK

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
