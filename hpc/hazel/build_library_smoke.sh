#!/bin/bash
#BSUB -J genmod-build-smoke
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -W 00:30
#BSUB -o results/logs/genmod-build-smoke.%J.out
#BSUB -e results/logs/genmod-build-smoke.%J.err

# Hazel HPC smoke stage 1 of 3: build a tiny rule-tree library.
#
# Generates 100 candidate trees at depth 2 (instead of 10000 at depth
# 3 in the production stage) and runs short 200-step probes for dedup.
# Output goes to GENERATED_DATA_smoke so it does not collide with the
# production GENERATED_DATA directory.

set -e

source ~/.bashrc
module load conda
conda activate /usr/local/usrapps/cads/$USER/genmod-env

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mkdir -p results/logs results/checkpoints results/figures

echo "=== Smoke build tree library ==="
echo "Host:    $(hostname)"
echo "JobID:   $LSB_JOBID"
echo "Started: $(date)"
echo

python scripts/generate_ruletrees.py \
    --library_only \
    --n_candidates 100 \
    --max_depth 2 \
    --method mixed \
    --grid_size 50 \
    --max_steps 200 \
    --output_dir GENERATED_DATA_smoke \
    --seed 42 \
    --workers 16

echo
echo "Finished: $(date)"
