#!/bin/bash
#SBATCH --job-name=genmod-train-smoke
#SBATCH --partition=gpu_partners
#SBATCH --qos=short_gpu
#SBATCH --gres=gpu:a30:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=results/logs/genmod-train-smoke.%j.out
#SBATCH --error=results/logs/genmod-train-smoke.%j.err

# Hazel HPC smoke stage 3 of 3: tiny model end-to-end training on a GPU.
#
# Uses the tiny model (~147K params) and short sequences from
# configs/ruletree_smoke.yaml -- it fits in any Hazel GPU.
#
# GPU routing: the standard `gpu` QOS caps a30 at 4 GPUs group-wide
# (gpu:a30=4 in `sqos -v`) and forbids l40s/h200, and there is only 1
# physical L40 in the `gpu` partition -- so smoke jobs there queue on
# QOSGrpGRES or wait for the single L40. Instead we use the partner
# short-GPU pool: `--partition=gpu_partners --qos=short_gpu` (open to
# all users for jobs under 2h). short_gpu has no group cap on a30, and
# gpu_partners has more idle GPUs, so smoke jobs start quickly. The
# 2h short_gpu limit is fine for a <30 min smoke run.
#
# The full/production runs (train_only.sh, train_only_rowcol.sh) stay
# on `--partition=gpu --qos=gpu --gres=gpu:l40:1` because they exceed
# the 2h short_gpu limit.
#
# Submission with a dependency on the smoke array:
#   sbatch --dependency=afterok:$SMOKE_SIM_JID hpc/hazel/train_only_smoke.sh

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

echo "=== Smoke train (base: time_space) ==="
echo "Host:    $(hostname)"
echo "JobID:   $SLURM_JOB_ID"
echo "Started: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo

python scripts/train.py \
    --config configs/ruletree_smoke.yaml \
    --name ruletree_smoke

echo
echo "Finished: $(date)"
