#!/bin/bash
#SBATCH --job-name=genmod-ruletrees
#SBATCH --output=results/logs/genmod-ruletrees_%j.out
#SBATCH --error=results/logs/genmod-ruletrees_%j.err
#SBATCH --partition=standard
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=128G
#SBATCH --time=48:00:00

# Full Option D experiment: generate rule-tree library, simulate, train,
# evaluate, and run conformal prediction.
#
# Data generation uses multiprocessing across 128 CPU workers on the
# Pasteur fat node (384 cores total, ~33% of node). The simulator is
# pure Python and CPU-bound, so this gives a near-linear speedup over
# the serial path. Training runs on a single H200 GPU.
# Estimated wall-clock: ~12h data gen + 1-2h training = ~14h total
# (well under the 48h standard partition limit).

set -e

source /opt/miniforge3/etc/profile.d/conda.sh
conda activate /data/apps/casl/arachchige/genmod-env

# Prevent BLAS / OpenMP threads from oversubscribing when many Python
# workers are running in parallel. The simulator itself is pure Python
# so this only affects numpy/torch helpers.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mkdir -p results/logs results/checkpoints results/figures

echo "=== Job info ==="
echo "Host:    $(hostname)"
echo "JobID:   $SLURM_JOB_ID"
echo "CPUs:    ${SLURM_CPUS_PER_TASK}"
echo "Started: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Step 1: data generation (skip if already present)
LIB_FILE="GENERATED_DATA/rule_trees/tree_library.json"
if [ ! -f "$LIB_FILE" ]; then
    echo "=== Generating rule-tree dataset ==="
    python scripts/generate_ruletrees.py \
        --n_candidates 10000 \
        --max_depth 3 \
        --method mixed \
        --num_runs 100 \
        --grid_size 50 \
        --max_steps 500 \
        --snapshot_ticks 100,200,300,400,500 \
        --output_dir GENERATED_DATA \
        --seed 42 \
        --workers ${SLURM_CPUS_PER_TASK}
else
    echo "Found existing tree library at $LIB_FILE, skipping generation."
fi

# Step 2: training + conformal evaluation. Restore reasonable thread
# counts so PyTorch can use a few CPU workers for the dataloader.
echo ""
echo "=== Training rule-tree classifier ==="
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
python scripts/train.py --config configs/ruletree_base.yaml

echo ""
echo "Finished: $(date)"
