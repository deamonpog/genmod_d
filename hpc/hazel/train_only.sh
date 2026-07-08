#!/bin/bash
#SBATCH --job-name=genmod-train
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:l40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=results/logs/genmod-train.%j.out
#SBATCH --error=results/logs/genmod-train.%j.err

# Hazel HPC stage 3 of 3: train the rule-tree classifier on a GPU.
#
# GPU: 1x NVIDIA L40 (48 GB). The base model at batch_size=16 fits
# comfortably in 48 GB at fp32. On the new Slurm system the GPU type is
# REQUIRED in --gres (there is no untyped "any GPU" request).
#
# To request a different GPU type, edit the --gres line above. Valid
# types (see https://hpc.ncsu.edu/RunningJobs/Resources.php):
#   --gres=gpu:h200:1    141 GB
#   --gres=gpu:h100:1     80 GB
#   --gres=gpu:l40s:1     48 GB
#   --gres=gpu:l40:1      48 GB   (default here)
#   --gres=gpu:a100:1     40 GB
#
# Partner projects with contributed GPUs can trade the two lines
#   #SBATCH --partition=gpu
#   #SBATCH --qos=gpu
# for
#   #SBATCH --partition=gpu_partners
#   #SBATCH --qos=p_cads_gpu
# to land on the partner allocation with higher priority.
#
# Submission with a dependency on stage 2:
#   sbatch --dependency=afterok:$SIM_JID hpc/hazel/train_only.sh

# Activate conda BEFORE `set -e`; the activation chain emits internal
# non-zero exits that `set -e` would catch and abort on, even though
# the overall activation succeeds.
source ~/.bashrc
module load conda
conda activate /usr/local/usrapps/cads/$USER/genmod-env

set -e

# Reasonable thread counts for the dataloader.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

mkdir -p results/logs results/checkpoints results/figures

echo "=== Train rule-tree classifier (base: time_space) ==="
echo "Host:    $(hostname)"
echo "JobID:   $SLURM_JOB_ID"
echo "Started: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo

python scripts/train.py --config configs/ruletree_base.yaml

echo
echo "Finished: $(date)"
