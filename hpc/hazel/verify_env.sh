#!/bin/bash
# Hazel HPC env health check (Slurm).
#
# Can be run two ways:
#
# 1. Interactive on a GPU node (recommended, tests CUDA too). Grab an
#    allocation, then run this script inside it:
#
#      salloc --partition=gpu --qos=gpu --gres=gpu:l40:1 -n 1 \
#             --cpus-per-task=2 --mem=8G --time=00:15:00
#      # inside the allocation:
#      bash hpc/hazel/verify_env.sh
#
# 2. Directly as a bash script (without an allocation) on whatever node
#    you are on. CUDA checks will return False on login nodes, which is
#    expected and NOT a failure -- login nodes have no GPU.
#
# This script does NOT use `set -e`. The conda activation chain emits
# non-zero internal substeps that would kill it.

GROUP=cads
ENV_DIR="/usr/local/usrapps/${GROUP}/${USER}/genmod-env"

echo "========================================================"
echo "Hazel env verification"
echo "========================================================"
echo "  host    : $(hostname)"
echo "  user    : ${USER}"
echo "  env dir : ${ENV_DIR}"
echo

# ---- Activate the env ----
if [ ! -d "$ENV_DIR" ]; then
    echo "FAIL: env not found at $ENV_DIR"
    echo "Run hpc/hazel/setup_env.sh first."
    exit 1
fi

source ~/.bashrc
module load conda
conda activate "$ENV_DIR"

# ---- Basic env info ----
echo "--- Environment info ---"
echo "  python : $(which python)"
echo "  python version : $(python --version)"
echo

# ---- nvidia-smi (only if we're on a GPU node) ----
echo "--- GPU hardware (nvidia-smi) ---"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader 2>/dev/null \
        || echo "  nvidia-smi present but failed (are we on a GPU node?)"
else
    echo "  nvidia-smi not found (expected on login nodes)"
fi
echo

# ---- Full import + CUDA check in Python ----
python <<'PY'
import importlib, sys, traceback

print("--- Python package imports ---")

required = [
    "torch",
    "numpy",
    "yaml",
    "sklearn",
    "matplotlib",
    "seaborn",
    "umap",
]
fail = False
for name in required:
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "unknown")
        print(f"  {name:12s}: {version}")
    except Exception as e:
        print(f"  {name:12s}: FAIL ({e.__class__.__name__}: {e})")
        fail = True

print()
print("--- Torch / CUDA ---")
import torch
print(f"  torch.__version__             : {torch.__version__}")
print(f"  torch.version.cuda            : {torch.version.cuda}")
print(f"  torch.cuda.is_available()     : {torch.cuda.is_available()}")
print(f"  torch.cuda.device_count()     : {torch.cuda.device_count()}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  torch.cuda.get_device_name({i}) : {torch.cuda.get_device_name(i)}")

    # Tiny compute test: a matmul on GPU to confirm CUDA works end-to-end.
    try:
        x = torch.randn(512, 512, device="cuda")
        y = x @ x
        s = float(y.sum().cpu().item())
        print(f"  512x512 matmul on GPU         : OK (sum={s:.2f})")
    except Exception as e:
        print(f"  512x512 matmul on GPU         : FAIL")
        traceback.print_exc()
        fail = True
else:
    print("  (no CUDA device visible -- this is expected on login nodes)")

print()
print("--- Project imports ---")
sys.path.insert(0, ".")
project_modules = [
    "genmod.data.rule_trees",
    "genmod.data.tree_generators",
    "genmod.data.schelling_ruletree",
    "genmod.data.ruletree_dataset",
    "genmod.data.ruletree_metadata",
    "genmod.data.schelling_dataset",
    "genmod.data.splits",
    "genmod.models.factory",
    "genmod.models.transformer",
    "genmod.evaluation.metrics",
    "genmod.evaluation.calibration",
    "genmod.evaluation.conformal",
    "genmod.evaluation.embeddings",
    "genmod.utils.config",
]
for mod_name in project_modules:
    try:
        importlib.import_module(mod_name)
        print(f"  {mod_name:45s}: OK")
    except Exception as e:
        print(f"  {mod_name:45s}: FAIL ({e.__class__.__name__}: {e})")
        fail = True

print()
if fail:
    print("VERIFICATION FAILED")
    sys.exit(1)
print("ALL CHECKS PASSED")
PY

rc=$?
exit $rc
