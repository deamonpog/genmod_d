#!/bin/bash
#BSUB -J genmod-train-smoke
#BSUB -q short_gpu
#BSUB -gpu "num=1"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -W 00:30
#BSUB -o results/logs/genmod-train-smoke.%J.out
#BSUB -e results/logs/genmod-train-smoke.%J.err

# Hazel HPC smoke stage 3 of 3: tiny model end-to-end training on a GPU.
#
# Uses the tiny model (147K params) and short sequences from
# configs/ruletree_smoke.yaml. The smoke model is small enough to fit
# in any Hazel GPU including older 8 GB cards, so we do NOT constrain
# GPU type and let LSF assign whichever is free fastest.
#
# Note: unlike the CPU smoke jobs, this one cannot rely on Hazel's
# default-queue fallback chain (debug -> serial -> short -> ...) since
# none of those accept GPU jobs. `short_gpu` is the GPU counterpart of
# `debug` -- priority 62, 64 slots per user, 2h max runtime -- and is
# the right queue for short/interactive GPU smoke tests.
#
# To force a specific GPU type on Hazel, use `-R "select[<model>]"`
# e.g. `-R "select[l40s] span[hosts=1]"`.
#
# Submission with a dependency on the smoke array:
#   bsub -w "done($SMOKE_SIM_JID)" < hpc/hazel/train_only_smoke.sh

# Activate conda BEFORE `set -e`; the activation chain emits internal
# non-zero exits that `set -e` would catch and abort on, even though
# the overall activation succeeds.
source ~/.bashrc
module load conda
conda activate /usr/local/usrapps/cads/$USER/genmod-env

set -e

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
