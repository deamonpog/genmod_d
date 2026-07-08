#!/bin/bash
#BSUB -J genmod-train-rowcol
#BSUB -q gpu
#BSUB -R "select[l40] span[hosts=1]"
#BSUB -gpu "num=1"
#BSUB -n 8
#BSUB -W 12:00
#BSUB -o results/logs/genmod-train-rowcol-lsf.%J.out
#BSUB -e results/logs/genmod-train-rowcol-lsf.%J.err

# TRANSITIONAL LSF script -- train the DECOMPOSED row/col embedding
# variant (positional_encoding: time_row_col) on an L40.
#
# Hazel is migrating from LSF to Slurm. LSF still owns most of the GPU
# fleet during the transition, so submitting training through LSF is
# more likely to land an L40 quickly than the small Slurm test pool.
#
# Reuses the SAME GENERATED_DATA as the base run (no rebuild or
# re-simulation needed). The config sets name=ruletree_rowcol, so logs
# / checkpoints / figures land in their own subdirectories.
#
# GPU selection: L40 (48 GB). To also accept H100, change the select
# line to:  #BSUB -R "select[h100||l40] span[hosts=1]"
#
# IMPORTANT: this writes results to results/logs/ruletree_rowcol/, the
# same place as the Slurm train_only_rowcol.sh. Do NOT let both the
# Slurm and LSF rowcol-train jobs RUN simultaneously -- submit to both,
# then cancel the loser (bkill / scancel) as soon as one starts.
#
# Submission (LSF uses a redirect):
#   bsub < hpc/hazel_lsf/train_only_rowcol.sh

# Activate conda BEFORE `set -e`; the activation chain emits internal
# non-zero exits that `set -e` would catch and abort on, even though
# the overall activation succeeds.
source ~/.bashrc
module load conda
conda activate /usr/local/usrapps/cads/$USER/genmod-env

set -e

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

mkdir -p results/logs results/checkpoints results/figures

echo "=== Train rule-tree classifier (rowcol: time_row_col) [LSF] ==="
echo "Host:    $(hostname)"
echo "JobID:   $LSB_JOBID"
echo "Started: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo

python scripts/train.py --config configs/ruletree_rowcol.yaml

echo
echo "Finished: $(date)"
