#!/usr/bin/env bash
# One-time environment bootstrap on the cluster login node.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
python3 -m venv --system-site-packages .venv 2>/dev/null || python3 -m venv .venv
source .venv/bin/activate
python -m pip install -q --upgrade pip
python -m pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu124 || \
  python -m pip install -q torch torchvision
python -m pip install -q transformers safetensors ultralytics scikit-learn joblib pillow numpy
cat > "$ROOT/cluster/activate.sh" <<EOF
export CUHKX_ROOT="$ROOT"
export CUHKX_FOLD_FILE="\$CUHKX_ROOT/research/artifacts/cv_folds_all18.json"
export HF_HOME="\$CUHKX_ROOT/.hf"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source "$ROOT/.venv/bin/activate"
cd "\$CUHKX_ROOT"
EOF
echo "env ready. Jobs should 'source cluster/activate.sh'."
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
