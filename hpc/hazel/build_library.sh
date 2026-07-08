#!/bin/bash
#SBATCH --job-name=genmod-build
#SBATCH --partition=compute
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=results/logs/genmod-build.%j.out
#SBATCH --error=results/logs/genmod-build.%j.err

# Hazel HPC stage 1 of 3: build the rule-tree library.
#
# Generates 10000 candidate trees, deduplicates structurally and
# behaviorally via probe simulations, and writes the result to
# GENERATED_DATA/rule_trees/tree_library.json.
#
# Submission:
#   sbatch hpc/hazel/build_library.sh
#
# To chain stages 2 and 3 automatically (afterok = only if prior job
# succeeds; --parsable makes sbatch print just the numeric job id):
#   LIB=$(sbatch --parsable hpc/hazel/build_library.sh)
#   SIM=$(sbatch --parsable --dependency=afterok:$LIB hpc/hazel/simulate_array.sh)
#   TR=$(sbatch  --parsable --dependency=afterok:$SIM hpc/hazel/train_only.sh)

# Activate conda BEFORE `set -e`; the activation chain emits internal
# non-zero exits that `set -e` would catch and abort on, even though
# the overall activation succeeds.
source ~/.bashrc
module load conda
conda activate /usr/local/usrapps/cads/$USER/genmod-env

set -e

# Prevent BLAS / OpenMP threads from oversubscribing while our Python
# multiprocessing pool is running. The simulator is pure Python so this
# only affects numpy / torch helpers.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mkdir -p results/logs results/checkpoints results/figures

echo "=== Build tree library ==="
echo "Host:    $(hostname)"
echo "JobID:   $SLURM_JOB_ID"
echo "CPUs:    $SLURM_CPUS_PER_TASK"
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
    --workers ${SLURM_CPUS_PER_TASK:-16}

echo
echo "Finished: $(date)"
