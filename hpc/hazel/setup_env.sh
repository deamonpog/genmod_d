#!/bin/bash
# One-time conda env setup on Hazel HPC.
#
# Run interactively from a Hazel login node:
#
#   bash hpc/hazel/setup_env.sh
#
# What it does:
#   1. Writes ~/.condarc to redirect the conda packages cache to
#      /share/cads/$USER/conda/pkgs (otherwise the 15 GB home quota
#      fills up on the first PyTorch install).
#   2. Creates a fresh conda env at /usr/local/usrapps/cads/$USER/genmod-env
#      (REMOVES any existing env at that path first).
#   3. Installs all project dependencies in the correct order:
#      - PyTorch (CUDA 12.4) from the PyTorch wheel index.
#      - Everything else (numpy, pyyaml, scikit-learn, matplotlib,
#        umap-learn, seaborn) from regular PyPI.
#   4. Runs a lightweight import-level verification to catch obvious
#      packaging breakage before you submit any real job. GPU-level
#      verification requires a compute node; see hpc/hazel/verify_env.sh
#      for that (submit via bsub).
#
# Hardcoded for the cads project group. Edit GROUP below if you belong
# to a different group.
#
# IMPORTANT: conda activation emits non-zero internal status codes even
# when successful. Do NOT enable 'set -e' before the conda activate
# chain; this script activates env-free-of-set-e on purpose.

GROUP=cads
ENV_DIR="/usr/local/usrapps/${GROUP}/${USER}/genmod-env"
PKGS_DIR="/share/${GROUP}/${USER}/conda/pkgs"

echo "========================================================"
echo "Hazel conda env fresh install for the cads project"
echo "========================================================"
echo "  group   : ${GROUP}"
echo "  user    : ${USER}"
echo "  env dir : ${ENV_DIR}"
echo "  pkgs    : ${PKGS_DIR}"
echo

# ---- Step 1: redirect conda pkgs cache ----
if [ ! -f "$HOME/.condarc" ] || ! grep -q "$PKGS_DIR" "$HOME/.condarc"; then
    echo "[1/5] Writing ~/.condarc to redirect pkgs_dirs to $PKGS_DIR"
    cat >> "$HOME/.condarc" <<EOF
pkgs_dirs:
  - $PKGS_DIR
EOF
else
    echo "[1/5] ~/.condarc already redirects pkgs_dirs (OK)"
fi
mkdir -p "$PKGS_DIR"

# ---- Step 2: remove any existing env, create fresh ----
source ~/.bashrc
module load conda

if [ -d "$ENV_DIR" ]; then
    echo
    echo "[2/5] Removing existing env at $ENV_DIR"
    conda env remove --prefix "$ENV_DIR" --yes || rm -rf "$ENV_DIR"
fi

echo
echo "[2/5] Creating fresh conda env at $ENV_DIR (python=3.11)"
conda create --prefix "$ENV_DIR" python=3.11 --yes

# ---- Step 3: activate and install ----
# Do NOT turn on 'set -e' here. The conda activation chain emits
# non-zero internal substeps that would kill the script.
conda activate "$ENV_DIR"

echo
echo "[3/5] Installing PyTorch (CUDA 12.4)"
# PyTorch's custom wheel index has restricted dependency resolution
# and will NOT pull regular PyPI deps (like numpy) transitively. We
# install torch here and the rest from PyPI in the next step.
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo
echo "[4/5] Installing remaining dependencies from PyPI"
pip install \
    numpy \
    pyyaml \
    scikit-learn \
    matplotlib \
    seaborn \
    umap-learn

# ---- Step 4: verify imports ----
echo
echo "[5/5] Verifying imports"
python <<'PY'
import importlib, sys

required = [
    "torch",
    "numpy",
    "yaml",
    "sklearn",
    "matplotlib",
    "seaborn",
    "umap",
]

print()
print(f"  python    : {sys.version.split()[0]} @ {sys.executable}")
fail = False
for name in required:
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "unknown")
        print(f"  {name:10s}: {version}")
    except Exception as e:
        print(f"  {name:10s}: FAIL ({e.__class__.__name__}: {e})")
        fail = True

# Also check submodules train.py actually imports
print()
print("Checking sklearn submodules used by the project:")
for sub in [
    "sklearn.manifold",
    "sklearn.linear_model",
    "sklearn.preprocessing",
    "sklearn.model_selection",
]:
    try:
        importlib.import_module(sub)
        print(f"  {sub:35s}: OK")
    except Exception as e:
        print(f"  {sub:35s}: FAIL ({e})")
        fail = True

print()
print("Checking torch submodules:")
for sub in ["torch.nn", "torch.nn.functional", "torch.utils.data"]:
    try:
        importlib.import_module(sub)
        print(f"  {sub:35s}: OK")
    except Exception as e:
        print(f"  {sub:35s}: FAIL ({e})")
        fail = True

# CUDA availability will be False on login nodes; do not fail on that.
print()
print("Torch CUDA status (False on login nodes is expected):")
print(f"  torch.cuda.is_available() = {__import__('torch').cuda.is_available()}")

if fail:
    print()
    print("VERIFICATION FAILED -- env is incomplete")
    sys.exit(1)
print()
print("Import-level verification passed.")
PY

echo
echo "========================================================"
echo "Done. Environment created at: $ENV_DIR"
echo
echo "To activate manually:"
echo "  source ~/.bashrc"
echo "  module load conda"
echo "  conda activate $ENV_DIR"
echo
echo "For a full health check on a GPU node:"
echo "  bsub -Is -q short_gpu -gpu \"num=1\" -n 2 -R \"span[hosts=1]\" \\"
echo "       -W 00:10 bash hpc/hazel/verify_env.sh"
echo "========================================================"
