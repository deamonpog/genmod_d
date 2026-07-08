#!/bin/bash
#SBATCH --job-name=genmod-train-rowcol-smoke
#SBATCH --partition=gpu_partners
#SBATCH --qos=short_gpu
#SBATCH --gres=gpu:a30:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=results/logs/genmod-train-rowcol-smoke.%j.out
#SBATCH --error=results/logs/genmod-train-rowcol-smoke.%j.err

# Hazel HPC smoke: tiny model end-to-end training with the DECOMPOSED
# row/column spatial embedding (positional_encoding: time_row_col).
#
# Exercises the time_row_col code path end-to-end (build -> factory ->
# train -> conformal) on a tiny model before committing to the full
# rowcol run. Reuses the smoke data in GENERATED_DATA_smoke.
#
# GPU routing: uses the partner short-GPU pool
# (`--partition=gpu_partners --qos=short_gpu --gres=gpu:a30:1`) for the
# same reasons as train_only_smoke.sh -- the standard `gpu` QOS caps
# a30 group-wide and there is only 1 physical L40, so smoke jobs there
# stall on QOSGrpGRES. short_gpu (open to all users, 2h max) has idle
# GPUs and no a30 group cap, so smoke jobs start quickly.
#
# The config sets name=ruletree_rowcol_smoke so it does not overwrite
# the base smoke results.
#
# Submission with a dependency on the smoke array:
#   sbatch --dependency=afterok:$SMOKE_SIM_JID hpc/hazel/train_only_rowcol_smoke.sh

# Activate conda BEFORE `set -e`; the activation chain emits internal
# non-zero exits that `set -e` would catch and abort on, even though
# the overall activation succeeds.
source ~/.bashrc
module load conda
conda activate /usr/local/usrapps/cads/$USER/genmod-env

set -e

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

mkdir -p results/logs results/checkpoints results/figures

echo "=== Smoke train (rowcol: time_row_col) ==="
echo "Host:    $(hostname)"
echo "JobID:   $SLURM_JOB_ID"
echo "Started: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo

python scripts/train.py \
    --config configs/ruletree_rowcol_smoke.yaml \
    --name ruletree_rowcol_smoke

echo
echo "Finished: $(date)"
