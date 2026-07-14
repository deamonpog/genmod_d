#!/bin/bash
#SBATCH --job-name=genmod-fish-smoke
#SBATCH --output=results/logs/genmod-fish-smoke_%j.out
#SBATCH --error=results/logs/genmod-fish-smoke_%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00

# Fish case study smoke test: prove the whole pipeline runs on the cluster
# before committing the ablation array to the queue.
#
# Runs the CPU preprocessing (01-06) if its artifacts are missing, then a tiny
# model for 2 epochs on one fold, then RAPS. It is NOT expected to produce a
# good accuracy: it is expected to produce ARTIFACTS without crashing.

set -e

source /opt/miniforge3/etc/profile.d/conda.sh
conda activate /data/apps/casl/arachchige/genmod-env

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

mkdir -p results/logs results/fish

echo "=== Job info ==="
echo "Host:    $(hostname)"
echo "Started: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# The verification scripts are cheap and catch the failures that matter:
# a broken symmetry, or a model that cannot track an agent through time.
echo "=== Verifying feature symmetries and permutation invariance ==="
python scripts/fish/verify_augmentation.py
python scripts/fish/verify_permutation_invariance.py

if [ ! -f "data/fish/runs.npz" ]; then
    echo ""
    echo "=== Preprocessing (01-06) ==="
    python scripts/fish/01_inspect_agent_fish_data.py
    python scripts/fish/02_create_rule_lookup.py
    python scripts/fish/03_preprocess_agent_trajectories.py
    python scripts/fish/04_create_grouped_splits.py
    python scripts/fish/05_create_trajectory_windows.py
    python scripts/fish/06_create_summary_features.py
else
    echo "Found data/fish/runs.npz, skipping preprocessing."
fi

echo ""
echo "=== Smoke training ==="
python scripts/fish/09_train_fish_transformer.py --config configs/fish_smoke.yaml

echo ""
echo "=== Smoke conformal ==="
python scripts/fish/10_calibrate_fish_raps.py --tag smoke_T64

echo ""
echo "Finished: $(date)"
