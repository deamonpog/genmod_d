#!/bin/bash
#BSUB -J fish-fields[1-5]
#BSUB -q gpu
#BSUB -R "select[h100||h200] rusage[mem=32] span[hosts=1]"
#BSUB -gpu "num=1:mode=shared:j_exclusive=no:gmem=60G"
#BSUB -n 8
#BSUB -W 10:00
#BSUB -o results/logs/fish-fields.%J.%I.out
#BSUB -e results/logs/fish-fields.%J.%I.err

# Field ablation on H100/H200 via SHARED mode with a GPU-memory RESERVATION,
# instead of j_exclusive=yes.
#
# WHY. Under current load, exclusive GPU requests barely match any host on
# Hazel: `bjobs -p` on the exclusive job reported "1 of 214 candidate hosts",
# and it pended for six hours. A shared request with `gmem` scheduled INSTANTLY
# onto an idle 80 GB H100 (an interactive `nvidia-smi` probe landed on gpu16
# with 0 MiB used). The reservation is enforced, so the shared-GPU OOM that
# fish_long_windows.sh documents -- which was shared WITHOUT a reservation --
# cannot recur.
#
# gmem=60G on an 80 GB card fits batch 64 (~49 GB) with margin and leaves too
# little for a heavy co-tenant, so each fold effectively gets a dedicated H100
# while still scheduling like a shared job. This lets us use the FULL batch 64 /
# 120-epoch config (configs/fish_field_T64.yaml), i.e. the batch-identical,
# step-identical match to transformer_inv_T64 -- the cleanest comparison.
#
# One grouped fold per array task, each to its own directory (field_T64_f{k}),
# no per-task RAPS. Array index i trains fold k = i - 1. If fewer than five
# H100s have 60 GB free, some folds queue briefly and start as cards free up.
#
# PREREQUISITE: data/fish/{runs.npz,splits.json,windows_T64.npz} and the
# trajectory baseline transformer_inv_T64 (for --compare in finalize).
#
# Submission:
#   bsub < hpc/hazel_lsf/fish_field_shared.sh
#
# FINALIZE, on a login node, once all five folds finish (same tags as the other
# variants):
#   mkdir -p results/fish/field_T64
#   for k in 0 1 2 3 4; do
#       cp results/fish/field_T64_f${k}/probs_fold${k}.npz results/fish/field_T64/
#   done
#   python scripts/fish/10_calibrate_fish_raps.py --tag field_T64
#   python scripts/fish/11_evaluate_fish_rules.py --tag field_T64 \
#       --compare transformer_inv_T64

source ~/.bashrc
module load conda
conda activate /usr/local/usrapps/cads/$USER/genmod-env

set -e

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

mkdir -p results/logs results/fish configs/fish_ablations

echo "=== Fish field ablation, shared+gmem on H100/H200 [LSF] ==="
echo "Host:    $(hostname)"
echo "JobID:   ${LSB_JOBID}  index ${LSB_JOBINDEX}"
echo "Started: $(date)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader || true
echo

if [ ! -f "data/fish/runs.npz" ]; then
    echo "ERROR: data/fish/runs.npz not found."
    echo "Run scripts/fish/01..06 first (CPU only, a few minutes)."
    exit 1
fi
if [ ! -f "data/fish/windows_T64.npz" ]; then
    echo "ERROR: data/fish/windows_T64.npz not found (run scripts/fish/05)."
    exit 1
fi

K=$(( LSB_JOBINDEX - 1 ))
CFG="configs/fish_ablations/field_T64_shared_f${K}.yaml"
sed -e "s/^tag:.*/tag: field_T64_f${K}/" \
    -e "s/^folds:.*/folds: [${K}]/" \
    configs/fish_field_T64.yaml > "$CFG"

echo "=== Training field fold ${K} (config ${CFG}, batch 64) ==="
python scripts/fish/15_train_fish_field.py --config "$CFG"

echo
echo "Finished fold ${K}: $(date)"
echo "When all five folds are done, run the FINALIZE block from this script's"
echo "header on a login node to gather probs, calibrate RAPS, and evaluate."
