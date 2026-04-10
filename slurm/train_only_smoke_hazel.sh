#!/bin/bash
#BSUB -J genmod-train-smoke
#BSUB -q gpu
#BSUB -m "gpu_l40s gpu_a30 gpu_a10"
#BSUB -gpu "num=1"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16000]"
#BSUB -W 00:30
#BSUB -o results/logs/genmod-train-smoke.%J.out
#BSUB -e results/logs/genmod-train-smoke.%J.err

# Smoke stage 3 of 3: tiny model end-to-end training on Hazel HPC.
#
# Uses the tiny model (147K params) and short sequences from
# configs/ruletree_smoke.yaml. Allows L40S, A30, or A10 since the smoke
# model fits in any of them and a30 is currently 0% utilized (shortest
# queue wait).
#
# Submit with a dependency on the smoke array:
#   bsub -w "done($SMOKE_SIM_JID)" < slurm/train_only_smoke_hazel.sh

set -e

source ~/.bashrc
module load conda
conda activate /usr/local/usrapps/cads/cdondim/genmod-env

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
