#!/bin/bash
#BSUB -J genmod-train
#BSUB -q gpu
#BSUB -m "gpu_l40s"
#BSUB -gpu "num=1"
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=64000]"
#BSUB -W 12:00
#BSUB -o results/logs/genmod-train.%J.out
#BSUB -e results/logs/genmod-train.%J.err

# Stage 3 of 3: train the rule-tree classifier on Hazel HPC.
#
# Constrained to the gpu_l40s host group (NVIDIA L40S, 48GB VRAM).
# At batch_size=16 (configs/ruletree_base.yaml) the model fits with
# room to spare. To use H100 instead, change `-m "gpu_l40s"` to
# `-m "gpu_h100"`. To use H200, change to `-m "gpu_h200"`. Both have
# 80GB+ VRAM and can use batch_size=32.
#
# Submit with a dependency on stage 2:
#   bsub -w "done($SIM_JID)" < slurm/train_only_hazel.sh

set -e

source ~/.bashrc
module load conda
conda activate /usr/local/usrapps/cads/cdondim/genmod-env

# Reasonable thread counts for the dataloader
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
