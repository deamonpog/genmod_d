#!/bin/bash
#SBATCH --job-name=genmod-schelling
#SBATCH --partition=standard
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err

# ============================================
# Train Schelling segregation threshold classifier
# ============================================

source /opt/miniforge3/etc/profile.d/conda.sh
ENV_DIR="/data/apps/casl/arachchige/genmod-env"
conda activate "$ENV_DIR"

cd "$SLURM_SUBMIT_DIR"

# Generate data if not already done
if [ ! -d "GENERATED_DATA/schelling" ]; then
    echo "Generating Schelling data..."
    python scripts/generate_schelling.py
fi

# Train
echo "Starting Schelling training..."
python scripts/train.py --config configs/schelling_base.yaml

echo "Schelling training complete."
