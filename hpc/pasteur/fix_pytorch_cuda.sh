#!/bin/bash
# Fix PyTorch CUDA in existing environment
#
# Step 1: Run the pip install from the login node (no GPU needed for install):
#     bash hpc/pasteur/fix_pytorch_cuda.sh install
#
# Step 2: Test GPU access in an interactive job:
#     srun --partition=debug --gres=gpu:1 --time=00:10:00 --mem=8G --pty \
#         bash hpc/pasteur/fix_pytorch_cuda.sh test

set -e

source /opt/miniforge3/etc/profile.d/conda.sh
conda activate /data/apps/casl/arachchige/genmod-env

ACTION="${1:-install}"

if [ "$ACTION" = "install" ]; then
    echo "Reinstalling PyTorch with CUDA 12.4..."
    pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    echo ""
    echo "Done. Now test GPU access with:"
    echo "  srun --partition=debug --gres=gpu:1 --time=00:10:00 --mem=8G --pty bash hpc/pasteur/fix_pytorch_cuda.sh test"

elif [ "$ACTION" = "test" ]; then
    echo "=== GPU Test ==="
    echo "Host: $(hostname)"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
    python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
    x = torch.randn(1000, 1000, device='cuda')
    y = x @ x
    print(f'Matrix multiply on GPU: OK ({y.shape})')
else:
    print('ERROR: CUDA not available. Reinstall PyTorch with: bash hpc/pasteur/fix_pytorch_cuda.sh install')
"
fi
