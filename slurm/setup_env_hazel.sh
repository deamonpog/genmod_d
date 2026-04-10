#!/bin/bash
# One-time conda env setup on Hazel HPC for the cads/cdondim project.
# Run interactively from a Hazel login node:
#
#   bash slurm/setup_env_hazel.sh
#
# This script:
#   1. Writes ~/.condarc to redirect the conda packages cache to
#      /share/cads/$USER/conda/pkgs (so it does NOT fill the 15 GB home
#      quota).
#   2. Creates a conda env at /usr/local/usrapps/cads/$USER/genmod-env
#   3. Installs PyTorch (CUDA 12.4) + project dependencies via pip.
#
# If you are a different user in the cads group, just change GROUP / the
# usrapps path below. If you belong to a different group entirely, change
# both.

set -e

GROUP=cads
ENV_DIR="/usr/local/usrapps/${GROUP}/${USER}/genmod-env"
PKGS_DIR="/share/${GROUP}/${USER}/conda/pkgs"

echo "Hazel conda env setup"
echo "  group   : ${GROUP}"
echo "  user    : ${USER}"
echo "  env dir : ${ENV_DIR}"
echo "  pkgs    : ${PKGS_DIR}"
echo

# Step 1: redirect conda pkgs cache so the 15 GB home quota stays safe.
if [ ! -f "$HOME/.condarc" ] || ! grep -q "$PKGS_DIR" "$HOME/.condarc"; then
    echo "Writing ~/.condarc to redirect pkgs_dirs to $PKGS_DIR"
    cat >> "$HOME/.condarc" <<EOF
pkgs_dirs:
  - $PKGS_DIR
EOF
fi

mkdir -p "$PKGS_DIR"

# Step 2: create the conda env
module load conda
conda create --prefix "$ENV_DIR" python=3.11 -y

# Hazel docs require source ~/.bashrc before conda activate in jobs.
source ~/.bashrc
conda activate "$ENV_DIR"

# Step 3: install PyTorch (CUDA 12.4 build works on H100, L40S, A100, etc)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install pyyaml umap-learn seaborn scikit-learn matplotlib

echo
echo "Done. Environment created at: $ENV_DIR"
echo "To activate manually:"
echo "  source ~/.bashrc"
echo "  module load conda"
echo "  conda activate $ENV_DIR"
