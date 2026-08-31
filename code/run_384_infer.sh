#!/usr/bin/env bash
# EXP-116: test probabilities for the four 384 px fold models.
#
# Purpose is a single-change public read: 4-fold 384 person + 4-fold 224 wrist against
# sub_r1 (4-fold 224 person + 4-fold wrist) = 166. Illegal to ship (4 x 34.4 MB int8),
# but it answers whether 384 pays on the TEST distribution before we spend ~11 h on an
# all-train 384 model. B-032: member strength cannot be screened locally, so this has to
# be bought with a submission.
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(dirname "$0")/.."
WAIT_PID=${WAIT_PID:-0}
while [ "$WAIT_PID" != 0 ] && kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
for F in 0 1 2 3; do
  T="k224_mvit384_f${F}"
  cp -f "checkpoints/${T}.pt" "./${T}.pt" || exit 1
  python3 -u kaggle/cuhkx_224_kaggle.py --stage infer --fold "$F" --image-size 384 \
      --batch-size 2 --workers 0 --tag "$T" || exit 1
  mv -f testprobs_${T}*.npz research/artifacts/ 2>/dev/null
  rm -f "./${T}.pt" sub_${T}*_argmax.csv
  echo "=== $T inferred $(date) ==="
done
echo "=== 384 INFER COMPLETE $(date) ==="
