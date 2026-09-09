#!/usr/bin/env bash
#SBATCH --job-name=cuhkx-loso
#SBATCH --partition=gpu_a100_8
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --cpus-per-task=12
#SBATCH --time=04:00:00
#SBATCH --array=0-17   # override with -a on resubmit
#SBATCH --output=/scratch/pabitra/cuhkx/logs/loso_%A_%a.out
#
# Leave-one-SUBJECT-out for all 18 training subjects, person view.
#
# WHY, and it is not about the mean. EXP-124 measured between-subject sd at 5.26 points
# against a seed sigma of 1.16, and EXP-132 re-measured it at 5.09 with a worst-4 mean of
# 0.6810 against an overall 0.7546. Kaggle private is 8 unseen subjects and the on-site
# stage is 8 more at 30% of the grade, so both are draws from a distribution with a
# 16-point spread and the WORST subjects decide them. The 4-fold protocol gives 4 numbers;
# this gives 18, which is what a lower-quartile selection statistic needs.
#
# Each model trains on 17 subjects, which is also closer to the deployed all-18 model than
# the 14-subject fold models are.
set -euo pipefail
export CUHKX_ROOT=/home/pabitra/cuhkx
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/home/pabitra/.conda/envs/physmon/bin/python
# Outputs go to SCRATCH, not $CUHKX_ROOT. /home is quota-capped at 40 GB and each run
# writes a 274 MB resume plus a 137 MB model; 7 of the first 18 tasks died with
# "unexpected pos ... vs ..." from torch.save, which is a TRUNCATED WRITE, not a bug in
# the trainer. find_paths() puts outputs in the CWD, so cd is the whole fix.
mkdir -p /scratch/pabitra/cuhkx/runs
cd /scratch/pabitra/cuhkx/runs
USERS=(user1 user2 user3 user4 user5 user6 user7 user8 user9 \
       user16 user17 user18 user19 user20 user21 user22 user23 user24)
U=${USERS[$SLURM_ARRAY_TASK_ID]}
echo "=== LOSO holdout=$U on $(hostname) $(date) ==="
$P -u $CUHKX_ROOT/kaggle/cuhkx_224_kaggle.py --stage train --tag loso_${U} --eval-users "$U" \
   --cache-dir /scratch/pabitra/cuhkx/cache/crop_224 \
   --batch-size 16 --accum 1 --workers 8 --seed 1
echo "=== $U done $(date) ==="
