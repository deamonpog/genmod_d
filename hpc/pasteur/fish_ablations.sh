#!/bin/bash
#SBATCH --job-name=genmod-fish-abl
#SBATCH --output=results/logs/genmod-fish-abl_%A_%a.out
#SBATCH --error=results/logs/genmod-fish-abl_%A_%a.err
#SBATCH --partition=standard
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-13

# Fish case study: the ablation grid, one array task per ablation.
#
#   0-5    window length   T in {16, 32, 64, 128, 256, 512}
#   6-7    features        all (20) vs basic (11)
#   8-11   augmentation    none / rot / rot+refl / all
#   12-13  GRU baseline    T=64 and T=128
#
# Each task trains 5 grouped folds and then calibrates RAPS. The model is
# small (about 1M params) and the windows number only tens of thousands, so
# a single GPU per task is plenty; the array is for throughput, not size.
#
# The dataloader, not the GPU, is the bottleneck: augmentation (rotation,
# reflection, agent permutation) runs in numpy per window. Local profiling
# showed about 13% GPU utilisation with 4 workers, so we ask for 16 CPUs and
# raise num_workers accordingly.
#
# PREREQUISITE: steps 01-06 must have been run once to produce
#   data/fish/runs.npz, splits.json, windows_T*.npz, summary_T*.npz
# They are pure numpy and take a few minutes; run them on the login node or
# in an interactive session.

set -e

source /opt/miniforge3/etc/profile.d/conda.sh
conda activate /data/apps/casl/arachchige/genmod-env

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

mkdir -p results/logs results/fish

echo "=== Job info ==="
echo "Host:    $(hostname)"
echo "JobID:   ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Started: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

if [ ! -f "data/fish/runs.npz" ]; then
    echo "ERROR: data/fish/runs.npz not found."
    echo "Run scripts/fish/01..06 first (they are CPU-only and quick)."
    exit 1
fi

TASK=${SLURM_ARRAY_TASK_ID}

case $TASK in
  0)  python scripts/fish/12_run_fish_ablations.py --only window   --which T16   ;;
  1)  python scripts/fish/12_run_fish_ablations.py --only window   --which T32   ;;
  2)  python scripts/fish/12_run_fish_ablations.py --only window   --which T64   ;;
  3)  python scripts/fish/12_run_fish_ablations.py --only window   --which T128  ;;
  4)  python scripts/fish/12_run_fish_ablations.py --only window   --which T256  ;;
  5)  python scripts/fish/12_run_fish_ablations.py --only window   --which T512  ;;
  6)  python scripts/fish/12_run_fish_ablations.py --only features --which all   ;;
  7)  python scripts/fish/12_run_fish_ablations.py --only features --which basic ;;
  8)  python scripts/fish/12_run_fish_ablations.py --only augment  --which none     ;;
  9)  python scripts/fish/12_run_fish_ablations.py --only augment  --which rot      ;;
  10) python scripts/fish/12_run_fish_ablations.py --only augment  --which rot_refl ;;
  11) python scripts/fish/12_run_fish_ablations.py --only augment  --which all      ;;
  12) python scripts/fish/08_train_gru_baseline.py --window 64  ;;
  13) python scripts/fish/08_train_gru_baseline.py --window 128 ;;
  *)  echo "unknown array task $TASK"; exit 1 ;;
esac

echo ""
echo "Finished: $(date)"
