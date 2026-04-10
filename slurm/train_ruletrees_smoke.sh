#!/bin/bash
#SBATCH --job-name=genmod-ruletrees-smoke
#SBATCH --output=results/logs/genmod-ruletrees-smoke_%j.out
#SBATCH --error=results/logs/genmod-ruletrees-smoke_%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

# Smoke test: generate ~50 rule trees, 5 runs each, train tiny model 3 epochs.
# Confirms data generation, dataloader, training loop, and conformal step
# all work end-to-end before launching the full job.

set -e

source /opt/miniforge3/etc/profile.d/conda.sh
conda activate /data/apps/casl/arachchige/genmod-env

mkdir -p results/logs results/checkpoints results/figures

echo "=== Smoke test ==="
echo "Host:    $(hostname)"
echo "JobID:   $SLURM_JOB_ID"
echo "Started: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Use a separate output directory to avoid clobbering the real dataset
SMOKE_DATA_DIR="GENERATED_DATA_smoke"

if [ ! -f "$SMOKE_DATA_DIR/rule_trees/tree_library.json" ]; then
    echo "=== Generating smoke dataset ==="
    python scripts/generate_ruletrees.py \
        --n_candidates 100 \
        --max_depth 2 \
        --method mixed \
        --num_runs 5 \
        --grid_size 50 \
        --max_steps 200 \
        --snapshot_ticks 50,100,150,200 \
        --output_dir "$SMOKE_DATA_DIR" \
        --seed 42
fi

echo ""
echo "=== Training smoke model ==="
python scripts/train.py \
    --config configs/ruletree_smoke.yaml \
    --name ruletree_smoke

echo ""
echo "Finished: $(date)"
