#!/bin/bash
# Setup conda environment on Pasteur HPC
# Run this ONCE on the login node to create the environment
#
# Usage: bash slurm/setup_env.sh

set -e

# Use the group apps directory for environments
ENV_DIR="/data/apps/casl/arachchige/genmod-env"

echo "Creating conda environment at: $ENV_DIR"

source /opt/miniforge3/etc/profile.d/conda.sh

mamba create --prefix "$ENV_DIR" python=3.11 -y

conda activate "$ENV_DIR"

# Install PyTorch with CUDA via pip (more reliable for GPU builds)
# H200 requires CUDA 12.x. Using the PyTorch index URL ensures we get the GPU build.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install other dependencies
pip install pyyaml umap-learn seaborn scikit-learn matplotlib

echo ""
echo "Environment created at: $ENV_DIR"
echo "To activate: conda activate /data/apps/casl/arachchige/genmod-env"
echo ""
echo "Next steps:"
echo "  1. Copy the GenMod project to /data/scratch/casl/$USER/ or /data/shared/"
echo "  2. Submit jobs with: sbatch slurm/train_eca.sh"
