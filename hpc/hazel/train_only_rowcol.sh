#!/bin/bash
#SBATCH --job-name=genmod-train-rowcol
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:l40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=results/logs/genmod-train-rowcol.%j.out
#SBATCH --error=results/logs/genmod-train-rowcol.%j.err

# Hazel HPC: train the rule-tree classifier with the DECOMPOSED
# row/column spatial embedding (positional_encoding: time_row_col).
#
# This is the ablation counterpart to train_only.sh. It reuses the SAME
# generated data (GENERATED_DATA/rule_trees), so you do NOT need to
# rebuild the library or re-simulate -- just run this after stage 2 has
# populated GENERATED_DATA once.
#
# The config sets name=ruletree_rowcol, so logs / checkpoints / figures
# land in their own subdirectories and do not overwrite the base run.
#
# Same GPU (L40, 48 GB) and resources as train_only.sh for a fair
# apples-to-apples comparison. Parameter count is essentially identical
# to the base model (two small row/col embedding tables replace one flat
# space table), so any accuracy delta reflects inductive bias, not
# capacity.
#
# Submission (after GENERATED_DATA exists):
#   sbatch hpc/hazel/train_only_rowcol.sh
# Or chained on stage 2 like the base run:
#   sbatch --dependency=afterok:$SIM_JID hpc/hazel/train_only_rowcol.sh

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

echo "=== Train rule-tree classifier (rowcol: time_row_col) ==="
echo "Host:    $(hostname)"
echo "JobID:   $SLURM_JOB_ID"
echo "Started: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo

python scripts/train.py --config configs/ruletree_rowcol.yaml

echo
echo "Finished: $(date)"
