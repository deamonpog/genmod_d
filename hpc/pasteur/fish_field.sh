#!/bin/bash
#SBATCH --job-name=genmod-fish-field
#SBATCH --output=results/logs/genmod-fish-field_%j.out
#SBATCH --error=results/logs/genmod-fish-field_%j.err
#SBATCH --partition=standard
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# Fish field-representation ablation: the lossy theta_obs control.
#
# Trains the field Transformer on 32x32 density/momentum renders of the fish
# runs, at the same T=64 / grouped 5-fold / 120-epoch protocol as the
# trajectory headline (configs/fish_transformer_inv.yaml), then runs the SAME
# RAPS and evaluation steps. The comparison of interest is field_T64 vs
# transformer_inv_T64: how much the conformal set size inflates once the
# observation discards per-agent identity and sub-cell geometry.
#
# PREREQUISITE: steps 01-06 must have produced
#   data/fish/runs.npz, splits.json, windows_T64.npz
# (pure numpy, run on the login node or in fish_smoke.sh).

set -e

source /opt/miniforge3/etc/profile.d/conda.sh
conda activate /data/apps/casl/arachchige/genmod-env

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

mkdir -p results/logs results/fish

echo "=== Job info ==="
echo "Host:    $(hostname)"
echo "JobID:   ${SLURM_JOB_ID}"
echo "Started: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

if [ ! -f "data/fish/runs.npz" ]; then
    echo "ERROR: data/fish/runs.npz not found."
    echo "Run scripts/fish/01..06 first (CPU-only and quick)."
    exit 1
fi
if [ ! -f "data/fish/windows_T64.npz" ]; then
    echo "ERROR: data/fish/windows_T64.npz not found (run scripts/fish/05)."
    exit 1
fi

echo "=== Training field Transformer (5 folds, 120 epochs) ==="
python scripts/fish/15_train_fish_field.py --config configs/fish_field_T64.yaml

echo ""
echo "=== RAPS conformal ==="
python scripts/fish/10_calibrate_fish_raps.py --tag field_T64

echo ""
echo "=== Rule evaluation (compared against the trajectory transformer) ==="
python scripts/fish/11_evaluate_fish_rules.py --tag field_T64 --compare transformer_inv_T64

echo ""
echo "Finished: $(date)"
