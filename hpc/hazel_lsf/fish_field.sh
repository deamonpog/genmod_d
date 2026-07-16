#!/bin/bash
#BSUB -J fish-field[1-5]%3
#BSUB -q gpu
#BSUB -R "select[h100||h200] rusage[mem=32] span[hosts=1]"
#BSUB -gpu "num=1:j_exclusive=yes"
#BSUB -n 8
#BSUB -W 10:00
#BSUB -o results/logs/fish-field.%J.%I.out
#BSUB -e results/logs/fish-field.%J.%I.err

# Fish FIELD-representation ablation (lossy theta_obs control), as an LSF array:
# one grouped fold per task. The field Transformer is ~10x heavier per step than
# the trajectory model, so a single 5-fold job would risk the walltime cap; one
# fold per task keeps each run to ~5 h and lets a failed fold rerun on its own.
#
# WHY h100||h200 ONLY (not l40s). Batch 64 -- chosen to match the trajectory
# baseline's 48,000-step budget exactly -- needs about 49 GB. The L40S has only
# 48 GB and would OOM; the 80 GB H100/H200 fit it with room to spare.
# j_exclusive=yes + the %3 throttle avoid the shared-GPU OOM that cost
# fish_long_windows.sh 6 of 9 tasks (see that script's note).
#
# Each task trains ONE fold into its own directory (field_T64_f{k}) so the five
# tasks never collide on the aggregate log/summary files. RAPS needs all five
# folds in one place, so it is NOT run per task; run the finalize block below on
# a login node (CPU only) AFTER all five array tasks finish.
#
# Array index i (LSF is 1-based) trains fold k = i - 1.
#
# PREREQUISITE: data/fish/{runs.npz,splits.json,windows_T64.npz} must exist
# (scripts/fish/01..06, CPU only). The trajectory baseline transformer_inv_T64
# must already be present for the --compare in the finalize block.
#
# Submission (LSF uses a redirect):
#   bsub < hpc/hazel_lsf/fish_field.sh
#
# FINALIZE, on a login node, once all five folds have finished:
#   mkdir -p results/fish/field_T64
#   for k in 0 1 2 3 4; do
#       cp results/fish/field_T64_f${k}/probs_fold${k}.npz results/fish/field_T64/
#   done
#   python scripts/fish/10_calibrate_fish_raps.py --tag field_T64
#   python scripts/fish/11_evaluate_fish_rules.py --tag field_T64 \
#       --compare transformer_inv_T64
# (Per-fold convergence curves live in results/fish/field_T64_f{k}/training_log.csv.)

# Activate conda BEFORE `set -e`: the activation chain emits internal non-zero
# exits that `set -e` would abort on, even though activation itself succeeds.
source ~/.bashrc
module load conda
conda activate /usr/local/usrapps/cads/$USER/genmod-env

set -e

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

mkdir -p results/logs results/fish configs/fish_ablations

echo "=== Fish field-representation ablation [LSF] ==="
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

# Per-fold config: same field settings, one fold, its own tag/directory.
K=$(( LSB_JOBINDEX - 1 ))
CFG="configs/fish_ablations/field_T64_f${K}.yaml"
sed -e "s/^tag:.*/tag: field_T64_f${K}/" \
    -e "s/^folds:.*/folds: [${K}]/" \
    configs/fish_field_T64.yaml > "$CFG"

echo "=== Training field fold ${K} (config ${CFG}) ==="
python scripts/fish/15_train_fish_field.py --config "$CFG"

echo
echo "Finished fold ${K}: $(date)"
echo "When all five folds are done, run the FINALIZE block from this script's"
echo "header on a login node to gather probs, calibrate RAPS, and evaluate."
