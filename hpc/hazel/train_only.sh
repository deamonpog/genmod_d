#!/bin/bash
#BSUB -J genmod-train
#BSUB -q gpu
#BSUB -R "select[h100||l40] span[hosts=1]"
#BSUB -gpu "num=1"
#BSUB -n 8
#BSUB -W 12:00
#BSUB -o results/logs/genmod-train.%J.out
#BSUB -e results/logs/genmod-train.%J.err

# Hazel HPC stage 3 of 3: train the rule-tree classifier on a GPU.
#
# GPU selection: H100 (80 GB) OR L40 (48 GB).
#
# Why both?  The Hazel `gpu` queue routes to:
#   gpu_xtx, gpu_p100, gpu_a10, gpu_a30, gpu_a100, gpu_l40, gpu_h100
# (confirmed via `bqueues -r | grep gpu`). It does NOT route to
# gpu_l40s or gpu_h200 -- those live in group-private queues or in
# short_gpu/multi_gpu.
#
# H100 and L40 are the two models in the `gpu` queue with >= 48 GB
# VRAM, which is what our base model at batch_size=16 comfortably
# needs. OR-ing them (`select[h100||l40]`) gives 6 candidate hosts
# (gpu14, gpu15, gpu16, gpu17, gpu32, gpu33) instead of 4, shortening
# queue wait on average.
#
# If you want to force a specific model:
#   #BSUB -R "select[h100] span[hosts=1]"    H100 only (4 cards, 80 GB)
#   #BSUB -R "select[l40] span[hosts=1]"     L40 only  (2 cards, 48 GB)
#   #BSUB -R "select[a100] span[hosts=1]"    A100 only (2 cards, 40 or 80 GB)
#
# GPU models NOT reachable via the `gpu` queue (as of our check):
# l40s, h200. If you want those, submit via `-q short_gpu` (2 h time
# limit) or one of the group-private queues (e.g. pfaendtner_gpu has
# l40s+h100+h200, if you have access).
#
# Submission with a dependency on stage 2:
#   bsub -w "done($SIM_JID)" < hpc/hazel/train_only.sh

# Activate conda BEFORE `set -e`; the activation chain emits internal
# non-zero exits that `set -e` would catch and abort on, even though
# the overall activation succeeds.
source ~/.bashrc
module load conda
conda activate /usr/local/usrapps/cads/$USER/genmod-env

set -e

# Reasonable thread counts for the dataloader.
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

mkdir -p results/logs results/checkpoints results/figures

echo "=== Train rule-tree classifier ==="
echo "Host:    $(hostname)"
echo "JobID:   $LSB_JOBID"
echo "Started: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo

python scripts/train.py --config configs/ruletree_base.yaml

echo
echo "Finished: $(date)"
