#!/bin/bash
#SBATCH --job-name=genmod-eca
#SBATCH --partition=standard
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err

# ============================================
# Train ECA rule classifier (256 rules, Base model)
# ============================================

# Setup conda
source /opt/miniforge3/etc/profile.d/conda.sh
ENV_DIR="/data/apps/casl/arachchige/genmod-env"
conda activate "$ENV_DIR"

# Navigate to project
cd "$SLURM_SUBMIT_DIR"

# Generate data if not already done
if [ ! -f "GENERATED_DATA/rule_255.json" ]; then
    echo "Generating ECA data for all 256 rules..."
    python scripts/generate_all_256.py --start 0 --end 256 --num_runs 100 --steps 100
fi

# Train
echo "Starting ECA training..."
python scripts/train.py --config configs/all_256_base.yaml

echo "ECA training complete."
