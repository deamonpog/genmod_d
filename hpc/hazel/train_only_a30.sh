#!/bin/bash
#SBATCH --job-name=genmod-train-a30
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:a30:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=results/logs/genmod-train-a30.%j.out
#SBATCH --error=results/logs/genmod-train-a30.%j.err

# Hazel HPC: train the rule-tree classifier (base: time_space) on an A30.
#
# Fallback for when L40 is unavailable. During the LSF->Slurm migration
# the Slurm pool exposes only one (fully-booked) L40, and the large
# L40/L40S pools are LSF-owned or 2h-capped. The `gpu` QOS DOES allow
# a30 (cap 4, 3-day wall) and A30 nodes are up in the Slurm pool, so
# this is the realistic way to run the full experiment now.
#
# A30 is 24 GB VRAM (vs 48 GB on L40), so we halve the batch size to 8
# (--batch_size 8) to fit. --mem=32G is host RAM for the tokenized
# dataset and is independent of GPU memory.
#
# NOTE: batch_size 8 vs the L40 default of 16 is a training-dynamics
# difference; note it when comparing against any L40 run.
#
# This writes to results/logs/ruletree_base/ -- the SAME place as
# train_only.sh (L40) and the LSF base job. Only let ONE base-train
# job run at a time; cancel the others once this starts.
#
# Submission (data must already exist in GENERATED_DATA):
#   sbatch hpc/hazel/train_only_a30.sh

# Activate conda BEFORE `set -e`; the activation chain emits internal
# non-zero exits that `set -e` would catch and abort on, even though
# the overall activation succeeds.
source ~/.bashrc
module load conda
conda activate /usr/local/usrapps/cads/$USER/genmod-env

set -e

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

mkdir -p results/logs results/checkpoints results/figures

echo "=== Train rule-tree classifier (base: time_space) [A30, batch 8] ==="
echo "Host:    $(hostname)"
echo "JobID:   $SLURM_JOB_ID"
echo "Started: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo

python scripts/train.py --config configs/ruletree_base.yaml --batch_size 8

echo
echo "Finished: $(date)"
