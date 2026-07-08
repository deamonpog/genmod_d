#!/bin/bash
#SBATCH --job-name=genmod-train-smoke
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:l40:1
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
# configs/ruletree_smoke.yaml. The smoke model fits in any Hazel GPU,
# but the new Slurm system REQUIRES a GPU type in --gres, so we request
# an L40 (widely available, 48 GB). To grab whatever is free fastest,
# switch to a smaller type such as --gres=gpu:a10:1 or --gres=gpu:a30:1.
#
# For a higher-priority short queue (idle partner GPUs, 2h max), swap:
#   #SBATCH --partition=gpu       -> #SBATCH --partition=gpu_partners
#   #SBATCH --qos=gpu             -> #SBATCH --qos=short_gpu
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
