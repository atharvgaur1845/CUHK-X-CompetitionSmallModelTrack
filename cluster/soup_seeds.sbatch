#!/usr/bin/env bash
#SBATCH --job-name=cuhkx-soup
#SBATCH --partition=gpu_h100_4
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00:00
#SBATCH --array=0-29
#SBATCH --output=/scratch/pabitra/cuhkx/logs/soup_%A_%a.out
#
# EXP-148 -- SEED SOUP. Average the WEIGHTS of models that differ only by seed, ship one.
#
# Why now, and why this is not the same question the adoption bar was rejecting. Private
# advancement is a step function at rank 15, and the simulation puts P(top 15) at 0.70 with
# neighbours 1-4 clips away against a 4.4-clip binomial noise floor. +3 clips moves that to
# 0.85. A change worth +0.2 points is uninteresting as "an improvement" and very
# interesting as "survival probability".
#
# Why seeds and not folds. A soup of the four FOLD models has collectively seen every
# clip, so no honest OOF evaluation of it exists -- the only models that never saw subject
# u are fold_f(u) and loso_u. Seeds are different: three models trained on the SAME
# training subjects with different seeds are all fold-safe for that fold's held-out
# subjects, so souping them is measurable under the existing protocol with no new leakage.
# EXP-120a measured visual seed sigma at 1.16, so there is real seed variance to average.
#
# 3 views x (4 folds + all-train) x 2 extra seeds = 30 tasks. Seed 1 already exists.
set -euo pipefail
export CUHKX_ROOT=/home/pabitra/cuhkx
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/home/pabitra/.conda/envs/physmon/bin/python
C=/scratch/pabitra/cuhkx
mkdir -p $C/runs; cd $C/runs
I=$SLURM_ARRAY_TASK_ID
VIEW=$(( I / 10 ))          # 0 person, 1 wrist, 2 thermal
R=$(( I % 10 ))
SEED=$(( 2 + R / 5 ))       # 2 or 3
SLOT=$(( R % 5 ))           # 0-3 = folds, 4 = all-train
case $VIEW in
  0) CACHE=$C/cache/crop_224;       BASE=k224_mvit ;;
  1) CACHE=$C/cache/crop_wrist224;  BASE=k224_mvitwrist ;;
  2) CACHE=$C/cache/thermal_224;    BASE=k224_mvit_th ;;
esac
if [ "$SLOT" = "4" ]; then
  ARGS="--all-train --tag ${BASE}_all_s${SEED}"
else
  ARGS="--fold $SLOT --tag ${BASE}_f${SLOT}_s${SEED}"
fi
echo "=== soup task $I: view=$VIEW seed=$SEED slot=$SLOT on $(hostname) $(date) ==="
$P -u $CUHKX_ROOT/kaggle/cuhkx_224_kaggle.py --stage train $ARGS \
   --cache-dir $CACHE --batch-size 16 --accum 1 --workers 3 --seed $SEED
echo "=== task $I done $(date) ==="
