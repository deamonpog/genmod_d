#!/bin/bash
#SBATCH --job-name=genmod-train-rowcol-a30
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:a30:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=results/logs/genmod-train-rowcol-a30.%j.out
#SBATCH --error=results/logs/genmod-train-rowcol-a30.%j.err

# Hazel HPC: train the DECOMPOSED row/col embedding variant
# (positional_encoding: time_row_col) on an A30.
#
# Fallback for when L40 is unavailable (see train_only_a30.sh for the
# rationale). The `gpu` QOS allows a30 (cap 4, 3-day wall) and A30
# nodes are up in the Slurm pool.
#
# Reuses the SAME GENERATED_DATA as the base run. The config sets
# name=ruletree_rowcol, so logs / checkpoints / figures land in their
# own subdirectories.
#
# A30 is 24 GB VRAM, so --batch_size 8 (vs the L40 default of 16).
# --mem=32G is host RAM for the tokenized dataset.
#
# NOTE: batch_size 8 vs the L40 default of 16 is a training-dynamics
# difference; keep both variants on the same batch size when comparing
# rowcol vs base (this A30 rowcol pairs with train_only_a30.sh, not the
# L40 train_only.sh).
#
# This writes to results/logs/ruletree_rowcol/ -- the SAME place as
# train_only_rowcol.sh (L40) and the LSF rowcol job. Only let ONE
# rowcol-train job run at a time; cancel the others once this starts.
#
# Submission (data must already exist in GENERATED_DATA):
#   sbatch hpc/hazel/train_only_rowcol_a30.sh

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

echo "=== Train rule-tree classifier (rowcol: time_row_col) [A30, batch 8] ==="
echo "Host:    $(hostname)"
echo "JobID:   $SLURM_JOB_ID"
echo "Started: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo

python scripts/train.py --config configs/ruletree_rowcol.yaml --batch_size 8

echo
echo "Finished: $(date)"
