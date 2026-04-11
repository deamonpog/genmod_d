#!/bin/bash
#BSUB -J genmod-train-smoke
#BSUB -gpu "num=1"
#BSUB -n 4
#BSUB -W 00:30
#BSUB -o results/logs/genmod-train-smoke.%J.out
#BSUB -e results/logs/genmod-train-smoke.%J.err

# Hazel HPC smoke stage 3 of 3: tiny model end-to-end training on a GPU.
#
# Uses the tiny model (147K params) and short sequences from
# configs/ruletree_smoke.yaml. The smoke model is small enough to fit
# in any Hazel GPU including the older 8 GB cards, so we do not
# constrain GPU type and let LSF assign whichever is free fastest.
# To force a specific type, add `:type=l40s` (etc.) to the -gpu line.
#
# Submission with a dependency on the smoke array:
#   bsub -w "done($SMOKE_SIM_JID)" < hpc/hazel/train_only_smoke.sh

set -e

source ~/.bashrc
module load conda
conda activate /usr/local/usrapps/cads/$USER/genmod-env

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

mkdir -p results/logs results/checkpoints results/figures

echo "=== Smoke train ==="
echo "Host:    $(hostname)"
echo "JobID:   $LSB_JOBID"
echo "Started: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo

python scripts/train.py \
    --config configs/ruletree_smoke.yaml \
    --name ruletree_smoke

echo
echo "Finished: $(date)"
