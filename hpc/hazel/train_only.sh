#!/bin/bash
#BSUB -J genmod-train
#BSUB -q gpu
#BSUB -gpu "num=1:type=l40s"
#BSUB -n 8
#BSUB -W 12:00
#BSUB -o results/logs/genmod-train.%J.out
#BSUB -e results/logs/genmod-train.%J.err

# Hazel HPC stage 3 of 3: train the rule-tree classifier on a GPU.
#
# Constrained to L40S (48 GB VRAM) via `-gpu "num=1:type=l40s"`. At
# batch_size=16 (configs/ruletree_base.yaml) the model fits with room
# to spare. To use H100 instead, change `type=l40s` to `type=h100`.
# To use H200, change to `type=h200`. Both have 80 GB+ VRAM and can
# use batch_size=32.
#
# Available Hazel GPU types (per Hazel docs): a10, a30, a100, gtx1080,
# h100, h200, l40, l40s, p100, rtx_2080.
#
# Submission with a dependency on stage 2:
#   bsub -w "done($SIM_JID)" < hpc/hazel/train_only.sh

set -e

source ~/.bashrc
module load conda
conda activate /usr/local/usrapps/cads/$USER/genmod-env

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
