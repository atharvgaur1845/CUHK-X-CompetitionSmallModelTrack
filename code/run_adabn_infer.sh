#!/usr/bin/env bash
# Re-run test inference for every deployed video member with AdaBN (EXP-099).
# Writes research/artifacts/testprobs_<TAG>_adabn.npz; the non-adabn files are untouched,
# so the champion remains reproducible byte for byte.
set -u
cd "$(dirname "$0")/.."
run () { # tag cache
  if [ -f "research/artifacts/testprobs_$1_adabn.npz" ]; then echo "SKIP $1 (exists)"; return; fi
  echo "=== $1 (cache=$2) ==="
  python3 -u code/infer_video_crop.py --tag "$1" --cache "$2" --adabn || echo "FAILED $1"
}
run vid_r2p1d_f0 crop_v1
run vid_r2p1d_f1 crop_v1
run vid_r2p1d_f2 crop_v1
run vid_r2p1d_f3 crop_v1
run vid_ig65m_f0 crop_v1
run vid_ig65m_f1 crop_v1
run vid_ig65m_f2 crop_v1
run vid_ig65m_f3 crop_v1
run vid_f32_f2   crop_v1f32
run vid_res160_f2 crop_v160
run vid_upper_f2 crop_upper
echo "ADABN-INFER-DONE"
