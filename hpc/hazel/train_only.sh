#!/bin/bash
#BSUB -J genmod-train
#BSUB -q gpu
#BSUB -R "select[l40s]"
#BSUB -gpu "num=1:type=l40s"
#BSUB -n 8
#BSUB -W 12:00
#BSUB -o results/logs/genmod-train.%J.out
#BSUB -e results/logs/genmod-train.%J.err

# Hazel HPC stage 3 of 3: train the rule-tree classifier on a GPU.
#
# Constrained to L40S (48 GB VRAM) via two redundant filters:
#   -R "select[l40s]"      -- boolean host resource (matches the
#                             l40s tag on each L40S host in lshosts)
#   -gpu "num=1:type=l40s" -- IBM LSF GPU type field (mirrors the
#                             Hazel Slurm equivalent --gres=gpu:l40s:1)
# Belt and suspenders: if either is the wrong syntax for our LSF
# version, the other should still resolve to an L40S host. If LSF
# rejects either, drop the bad one.
#
# At batch_size=16 (configs/ruletree_base.yaml) the model fits with
# room to spare on a 48 GB L40S. To use H100 instead, replace `l40s`
# with `h100` in BOTH the -R and -gpu lines. To use H200, use `h200`.
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
