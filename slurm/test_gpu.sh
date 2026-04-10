#!/bin/bash
#SBATCH --job-name=genmod-test
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:2
#SBATCH --qos=elevated
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err

# ============================================
# Quick GPU test: 10 ECA rules, Tiny model, 3 epochs
# Tests: conda env, GPU access, full pipeline
# ============================================

set -e

echo "=== Environment ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "GPUs:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Setup conda
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate /data/apps/casl/arachchige/genmod-env

echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo ""

cd "$SLURM_SUBMIT_DIR"

# Generate small test data (10 ECA rules, 20 runs each)
echo "=== Generating test data ==="
python scripts/generate_all_256.py --start 0 --end 10 --num_runs 20 --steps 50

# Train tiny model for 3 epochs on CPU-available GPU
echo ""
echo "=== Training (10 rules, Tiny model, 3 epochs) ==="
python scripts/train.py \
    --config configs/smoke_test.yaml \
    --device cuda \
    --name hpc_test \
    --conformal

echo ""
echo "=== Test complete! ==="
echo "Results at: results/logs/hpc_test/results.json"
cat results/logs/hpc_test/results.json
