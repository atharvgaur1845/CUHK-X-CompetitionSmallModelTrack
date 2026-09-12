#!/usr/bin/env bash
#SBATCH --job-name=cuhkx-pl
#SBATCH --partition=gpu_h100_4
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00:00
#SBATCH --array=0-4
#SBATCH --output=/scratch/pabitra/cuhkx/logs/pl_%A_%a.out
#
# EXP-156 -- R-4 self-training on the TEST clips (transductive), with a fold-safe control.
#
# Why this is not EXP-136 again. EXP-136 asked "do pseudo-labelled UNSEEN subjects improve
# generalisation to OTHER unseen subjects?" and found the subject curve flat past 6. This
# asks a different question: does training on pseudo-labelled clips of the SUBJECTS BEING
# SCORED help on those subjects? The research note (09-10 s6) found the Kaggle test set is
# four subjects and private shares them, so transductive adaptation is aimed at exactly the
# board that decides advancement. R-4 permits it explicitly.
#
# Tasks 0-1 are the control: fold-2 subjects pseudo-labelled by the DECODED OOF (models that
# never saw fold 2; 97.0% label accuracy in the top-50%), added to a 13-user training set,
# scored on fold 2 against the identical run without them. Tasks 2-4 are the real thing.
set -euo pipefail
export CUHKX_ROOT=/home/pabitra/cuhkx
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/home/pabitra/.conda/envs/physmon/bin/python
C=/scratch/pabitra/cuhkx
mkdir -p $C/runs; cd $C/runs
TR13=user19,user21,user8,user9,user18,user20,user23,user3,user7,user16,user17,user2,user4
F2=user22,user24,user5,user6
PL=$CUHKX_ROOT/research/artifacts
case $SLURM_ARRAY_TASK_ID in
  0) ARGS="--train-users $TR13 --eval-users user1 --tag sim_base   --cache-dir $C/cache/crop_224 --batch-size 16 --accum 1" ;;
  1) ARGS="--train-users $TR13 --eval-users user1 --tag sim_pseudo --cache-dir $C/cache/crop_224 --batch-size 16 --accum 1 --pseudo $PL/pseudo_sim_oof.npz --pseudo-users $F2 --pseudo-top-frac 0.5" ;;
  2) ARGS="--all-train --tag k224_mvit384_pl   --cache-dir $C/cache/crop_384      --image-size 384 --batch-size 4 --accum 4 --pseudo $PL/pseudo_test_soup5.npz --pseudo-top-frac 0.5" ;;
  3) ARGS="--all-train --tag k224_mvitwrist_pl --cache-dir $C/cache/crop_wrist224 --batch-size 16 --accum 1 --pseudo $PL/pseudo_test_soup5.npz --pseudo-top-frac 0.5" ;;
  4) ARGS="--all-train --tag k224_mvit_th_pl   --cache-dir $C/cache/thermal_224   --batch-size 16 --accum 1 --pseudo $PL/pseudo_test_soup5.npz --pseudo-top-frac 0.5" ;;
esac
echo "=== pseudo task $SLURM_ARRAY_TASK_ID on $(hostname) $(date) ==="
$P -u $CUHKX_ROOT/kaggle/cuhkx_224_kaggle.py --stage train $ARGS --workers 3 --seed 1
echo "=== done $(date) ==="
