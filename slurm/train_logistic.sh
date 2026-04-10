#!/bin/bash
#SBATCH --job-name=genmod-logistic
#SBATCH --partition=standard
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err

# ============================================
# Train Logistic Map regime classifier
# ============================================

source /opt/miniforge3/etc/profile.d/conda.sh
ENV_DIR="/data/apps/casl/arachchige/genmod-env"
conda activate "$ENV_DIR"

cd "$SLURM_SUBMIT_DIR"

# Generate data if not already done
if [ ! -d "GENERATED_DATA/logistic_map" ]; then
    echo "Generating logistic map data..."
    python scripts/generate_logistic.py
fi

# Train
echo "Starting logistic map training..."
python scripts/train.py --config configs/logistic_base.yaml

echo "Logistic map training complete."
