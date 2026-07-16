#!/bin/bash
#BSUB -J fish-fieldl[1-5]%3
#BSUB -q gpu
#BSUB -R "select[h100||h200||l40s] rusage[mem=32] span[hosts=1]"
#BSUB -gpu "num=1:j_exclusive=yes"
#BSUB -n 8
#BSUB -W 10:00
#BSUB -o results/logs/fish-fieldl.%J.%I.out
#BSUB -e results/logs/fish-fieldl.%J.%I.err

# L40S-schedulable variant of hpc/hazel_lsf/fish_field.sh, for when the 80 GB
# H100/H200 cards are saturated (bjobs -p shows no free exclusive H100/H200 but
# gpu_l40s has hundreds of free slots). Uses configs/fish_field_T64_l40s.yaml:
# batch 32 fits the 48 GB L40S, and 60 epochs keeps the 48,000-step budget of
# transformer_inv_T64, so the comparison stays step-matched. If it lands on an
# H100/H200 instead, batch 32 still runs fine (just below capacity).
#
# Same structure as fish_field.sh: one grouped fold per array task, each written
# to its own directory (field_T64_f{k}), no per-task RAPS. Array index i trains
# fold k = i - 1.
#
# PREREQUISITE: data/fish/{runs.npz,splits.json,windows_T64.npz} and the
# trajectory baseline transformer_inv_T64 (for --compare in finalize).
#
# Submission:
#   bsub < hpc/hazel_lsf/fish_field_l40s.sh
#
# FINALIZE, on a login node, once all five folds finish (identical to
# fish_field.sh -- both variants use the field_T64_f{k} tags):
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

echo "=== Fish field ablation, L40S-schedulable [LSF] ==="
echo "Host:    $(hostname)"
echo "JobID:   ${LSB_JOBID}  index ${LSB_JOBINDEX}"
echo "Started: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
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
CFG="configs/fish_ablations/field_T64_l40s_f${K}.yaml"
sed -e "s/^tag:.*/tag: field_T64_f${K}/" \
    -e "s/^folds:.*/folds: [${K}]/" \
    configs/fish_field_T64_l40s.yaml > "$CFG"

echo "=== Training field fold ${K} (config ${CFG}) ==="
python scripts/fish/15_train_fish_field.py --config "$CFG"

echo
echo "Finished fold ${K}: $(date)"
echo "When all five folds are done, run the FINALIZE block from this script's"
echo "header on a login node to gather probs, calibrate RAPS, and evaluate."
