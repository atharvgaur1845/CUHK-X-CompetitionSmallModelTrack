#!/usr/bin/env bash
# Build the `cuhkx` conda env on sharanga. Idempotent; safe to re-run.
#
# LAYOUT (Atharv's instruction 2026-09-08): code in user space, data on scratch.
#   /home/pabitra/cuhkx              code, docs, research/  (Lustre /home, 266 TB free)
#   /scratch/pabitra/cuhkx/cache     caches                 (Lustre /scratch, 266 TB free)
#   /scratch/pabitra/cuhkx/checkpoints, logs
# The repo expects cache/ and checkpoints/ beside the code, so those are SYMLINKS from
# the home tree into scratch -- the scripts stay unmodified and the bytes stay on scratch.
#
# Driver is 580.126.20 on the H100 nodes, so a cu124 wheel is safe; torch ships its own
# CUDA runtime, which is why the spack cuda-11.8 module is irrelevant here.
set -euo pipefail

HOME_ROOT=/home/pabitra/cuhkx
SCRATCH=/scratch/pabitra/cuhkx

mkdir -p "$SCRATCH"/{cache,checkpoints,logs,submissions}
cd "$HOME_ROOT"
for d in cache checkpoints logs submissions; do
  [ -e "$d" ] || ln -s "$SCRATCH/$d" "$d"
done
echo "layout:"; ls -la "$HOME_ROOT" | grep -E "^l|^d" | sed 's/^/  /'

source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | grep -qE "^cuhkx\s"; then
  conda create -y -n cuhkx python=3.11
fi
conda activate cuhkx
pip install --upgrade pip -q
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -q numpy scipy scikit-learn pillow

python - <<'PY'
import torch, torchvision, numpy, sklearn
print(f"  torch {torch.__version__}  torchvision {torchvision.__version__}  cuda {torch.version.cuda}")
print(f"  numpy {numpy.__version__}  sklearn {sklearn.__version__}")
PY
echo "ENV READY"
