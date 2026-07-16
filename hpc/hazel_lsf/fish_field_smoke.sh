#!/bin/bash
#BSUB -J fish-fsmoke
#BSUB -q short_gpu
#BSUB -R "select[l40s||l40||a100||h100] rusage[mem=16] span[hosts=1]"
#BSUB -gpu "num=1:mode=shared:j_exclusive=no:gmem=20G"
#BSUB -n 2
#BSUB -W 0:30
#BSUB -o results/logs/fish-fsmoke.%J.out
#BSUB -e results/logs/fish-fsmoke.%J.err

# Fish field-ablation SMOKE test: prove the renderer -> field transformer ->
# probs_fold contract -> RAPS chain runs on a Hazel GPU before committing the
# full 5-fold job. One fold, two epochs (configs/fish_field_smoke.yaml), batch
# 16. It is NOT expected to be accurate; it is expected to produce artifacts
# without crashing.
#
# WHY short_gpu. Unlike the `gpu` queue (which reaches only the contended H100
# cards and no L40S), `short_gpu` routes to every GPU model including the large
# gpu_l40s pool, so this dispatches almost immediately. Its 2 h limit is no
# constraint for a two-epoch run. batch 16 needs ~12 GB, so gmem=20G fits any
# of l40s/l40/a100/h100.
#
# Submission:
#   bsub < hpc/hazel_lsf/fish_field_smoke.sh
#
# PREREQUISITE: data/fish/{runs.npz,windows_T64.npz}.

source ~/.bashrc
module load conda
conda activate /usr/local/usrapps/cads/$USER/genmod-env

set -e

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

mkdir -p results/logs results/fish

echo "=== Fish field smoke [LSF, short_gpu] ==="
echo "Host:    $(hostname)"
echo "JobID:   ${LSB_JOBID}"
echo "Started: $(date)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader || true
echo

if [ ! -f "data/fish/runs.npz" ]; then
    echo "ERROR: data/fish/runs.npz not found (run scripts/fish/01..06)."
    exit 1
fi

echo "=== Train (1 fold, 2 epochs) ==="
python scripts/fish/15_train_fish_field.py --config configs/fish_field_smoke.yaml

echo
echo "=== RAPS on the smoke fold ==="
python scripts/fish/10_calibrate_fish_raps.py --tag field_smoke_T64

echo
echo "Smoke finished: $(date)"
echo "If this produced results/fish/field_smoke_T64/{probs_fold0.npz,conformal.json}"
echo "without error, the full job (fish_field_shared.sh) is trustworthy to run."
