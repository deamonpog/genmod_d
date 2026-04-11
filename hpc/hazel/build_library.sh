#!/bin/bash
#BSUB -J genmod-build
#BSUB -q standard
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -W 04:00
#BSUB -o results/logs/genmod-build.%J.out
#BSUB -e results/logs/genmod-build.%J.err

# Hazel HPC stage 1 of 3: build the rule-tree library.
#
# Generates 10000 candidate trees, deduplicates structurally and
# behaviorally via probe simulations, and writes the result to
# GENERATED_DATA/rule_trees/tree_library.json.
#
# Submission:
#   bsub < hpc/hazel/build_library.sh
#
# To chain stages 2 and 3 automatically:
#   LIB=$(bsub < hpc/hazel/build_library.sh   | awk '{print $2}' | tr -d '<>')
#   SIM=$(bsub -w "done($LIB)" < hpc/hazel/simulate_array.sh | awk '{print $2}' | tr -d '<>')
#   TR=$(bsub  -w "done($SIM)" < hpc/hazel/train_only.sh    | awk '{print $2}' | tr -d '<>')

set -e

source ~/.bashrc
module load conda
conda activate /usr/local/usrapps/cads/$USER/genmod-env

# Prevent BLAS / OpenMP threads from oversubscribing while our Python
# multiprocessing pool is running. The simulator is pure Python so this
# only affects numpy / torch helpers.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mkdir -p results/logs results/checkpoints results/figures

echo "=== Build tree library ==="
echo "Host:    $(hostname)"
echo "JobID:   $LSB_JOBID"
echo "CPUs:    16"
echo "Started: $(date)"
echo

python scripts/generate_ruletrees.py \
    --library_only \
    --n_candidates 10000 \
    --max_depth 3 \
    --method mixed \
    --grid_size 50 \
    --max_steps 500 \
    --output_dir GENERATED_DATA \
    --seed 42 \
    --workers 16

echo
echo "Finished: $(date)"
