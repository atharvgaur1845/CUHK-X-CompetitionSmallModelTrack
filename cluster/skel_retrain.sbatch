#!/usr/bin/env bash
#SBATCH --job-name=cuhkx-skel
#SBATCH --partition=gpu_h100_4
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --array=0-11
#SBATCH --output=logs/slurm/skel_%A_%a.out
#
# T1.5 — retrain the world25 skeleton/IMU stack fixing TWO recorded defects:
#
#   1. 32 of 48 packaged members were trained with aug=0 on a CROSS-SUBJECT problem
#      (skel_jvb_big x5, stgcn_v1/s1/w96, imu_inv4). Flagged at research/LOG.md:1360,
#      never acted on. --aug-spec trunc,rot,scale,jit,jdrop fixes it.
#   2. The default fold file research/artifacts/cv_folds.json is a SIXTEEN-user protocol
#      (always_train: [user5, user21]) -- the packaged members never trained on two of
#      our subjects. activate.sh sets CUHKX_FOLD_FILE to cv_folds_all18.json.
#
# The branch is worth +6.0 public clips (EXP-108 marginal value) and the whole stack
# retrains in under a GPU-hour, so this is the best cost/benefit available.
#
# One task per TAG (train.py does all 4 folds internally via --folds), not per tag x fold:
# these runs are 4-11 minutes and per-fold tasks would be all scheduler overhead.
set -euo pipefail
source "$(dirname "$0")/activate.sh"
mkdir -p logs/slurm

# Aug spec is PER-MODALITY: har_data checks names against the modality, so passing
# skeleton augs to the IMU member would be a silent no-op.
AUG_SKEL="trunc,rot,scale,jit,jdrop"
AUG_IMU="trunc,iscale,inoise,chan"
# tag|modality|extra args   (from code/fuse_submit.py CFG, the canonical registry)
SPECS=(
  "skel_jvb_big|skel|--feat jvb --width 256 --epochs 150 --seed 0"
  "skel_jvb_big_s1|skel|--feat jvb --width 256 --epochs 150 --seed 1"
  "skel_jvb_big_s2|skel|--feat jvb --width 256 --epochs 150 --seed 2"
  "skel_jvb_big_s3|skel|--feat jvb --width 256 --epochs 150 --seed 3"
  "skel_jvb_big_s4|skel|--feat jvb --width 256 --epochs 150 --seed 4"
  "stgcn_v1|skelg|--epochs 100 --seed 0"
  "stgcn_s1|skelg|--epochs 100 --seed 1"
  "stgcn_w96|skelg|--width 96 --epochs 100 --seed 0"
  "skel_multitcn|skel|--arch multitcn --width 128 --feat jvb --epochs 150 --seed 0"
  "astgcn_v1|skelg|--arch adaptive --width 64 --feat jv --epochs 100 --t-skel 32 --skel-time stretch --person first --seed 0"
  "imu_inv4|imu|--imu-feat inv --epochs 60 --seed 0"
  "imu_world|__standalone__|"
)
IFS='|' read -r TAG MOD EXTRA <<< "${SPECS[$SLURM_ARRAY_TASK_ID]}"
echo "=== task $SLURM_ARRAY_TASK_ID: $TAG (aug=$AUG, folds=$CUHKX_FOLD_FILE) ==="

if [ "$MOD" = "__standalone__" ]; then
  python3 -u code/train_imu_world.py train --folds 0,1,2,3 --amp
else
  case "$MOD" in
    imu) AUG="$AUG_IMU" ;;
    *)   AUG="$AUG_SKEL" ;;
  esac
  # --bs stays at the recorded 64. Raising it would change the OneCycleLR schedule
  # (max_lr=args.lr) and confound the two fixes under test with a recipe change; these
  # runs take 4-11 minutes anyway, so there is nothing to gain by going faster.
  python3 -u code/train.py --exp T15 --modality "$MOD" --tag "${TAG}_a18" \
    --aug-spec "$AUG" --folds 0,1,2,3 --bs 64 --workers 12 $EXTRA
fi
echo "=== $TAG done $(date) ==="
