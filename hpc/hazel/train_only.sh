#!/bin/bash
#BSUB -J genmod-train
#BSUB -q gpu
#BSUB -R "select[l40s]"
#BSUB -gpu "num=1"
#BSUB -n 8
#BSUB -W 12:00
#BSUB -o results/logs/genmod-train.%J.out
#BSUB -e results/logs/genmod-train.%J.err

# Hazel HPC stage 3 of 3: train the rule-tree classifier on a GPU.
#
# GPU model selection per Hazel LSF resources docs
# (https://hpc.ncsu.edu/Documents/LSFResources.php): each GPU host
# carries its model name as a boolean LSF resource, and the canonical
# way to request a specific model is:
#
#   -R "select[<model>]"
#
# Available models: rtx2080, gtx1080, p100, a10, a30, a100, l40, l40s,
# h100, h200. To use H100 instead of L40S, change to `select[h100]`.
# To use H200, change to `select[h200]`. Both have 80 GB+ VRAM and can
# use batch_size=32. L40S (48 GB) is enough at the default batch_size
# of 16 in configs/ruletree_base.yaml.
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
