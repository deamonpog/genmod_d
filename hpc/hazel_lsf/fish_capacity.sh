#!/bin/bash
#BSUB -J fish-cap[1-2]%2
#BSUB -q gpu
#BSUB -R "select[h100||h200||l40s] rusage[mem=32] span[hosts=1]"
#BSUB -gpu "num=1:j_exclusive=yes"
#BSUB -n 8
#BSUB -W 6:00
#BSUB -o results/logs/fish-cap.%J.%I.out
#BSUB -e results/logs/fish-cap.%J.%I.err

# Fish capacity/depth control runs (Phase 1b of the capacity reframe), as an
# LSF job array. Two T=64 transformer runs, both to a 120-epoch budget, five
# grouped folds, so the capacity comparison rests on converged models with a
# committed per-epoch validation curve (scripts/fish/09 now writes
# training_log.csv and run_summary.json into each run directory).
#
# Array indices (LSF is 1-based):
#   1  VERIFY the 148k / 2-layer model into a NEW directory
#      (transformer_inv_T64_small_verify), leaving the original
#      transformer_inv_T64_small untouched for provenance. This is the run
#      that turns the "0.803, converged" claim from a commit-message assertion
#      into an inspectable plateau.
#   2  NEW ~183k / 4-layer model (transformer_inv_T64_depth): full depth of the
#      1.8M model at a comparable budget, to PARTIALLY separate depth from
#      width/capacity. Supports two labelled comparisons only:
#        148k-2L vs ~183k-4L  comparable-budget depth/architecture
#        ~183k-4L vs 1.8M-4L  width/parameter-budget at fixed depth
#
# AFTER both tasks finish, recompute the capacity statistics on a login node
# (CPU only): the paired diffs, the gap-recovery g, and the delta-equivalence
# verdict, with exp_id-clustered bootstrap as primary and run-level as
# sensitivity:
#
#   python scripts/fish/12_capacity_stats.py \
#       --small transformer_inv_T64_small_verify \
#       --depth transformer_inv_T64_depth \
#       --large transformer_inv_T64 \
#       --gru   gru_T64
#
# PREREQUISITE: data/fish/{runs.npz,splits.json,windows_T64.npz} must exist
# (scripts/fish/01..06, CPU only). The 1.8M (transformer_inv_T64) and GRU
# (gru_T64) probability files must already be present for step 12.
#
# Submission (LSF uses a redirect):
#   bsub < hpc/hazel_lsf/fish_capacity.sh

# Activate conda BEFORE `set -e`: the activation chain emits internal non-zero
# exits that `set -e` would abort on, even though activation itself succeeds.
source ~/.bashrc
module load conda
conda activate /usr/local/usrapps/cads/$USER/genmod-env

set -e

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

mkdir -p results/logs results/fish

echo "=== Fish capacity/depth controls [LSF] ==="
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

case ${LSB_JOBINDEX} in
  # Verify the 148k model into a NEW directory (do not overwrite the original).
  1)  python scripts/fish/09_train_fish_transformer.py \
          --config configs/fish_transformer_inv_small.yaml \
          --tag transformer_inv_T64_small_verify
      python scripts/fish/10_calibrate_fish_raps.py \
          --tag transformer_inv_T64_small_verify ;;

  # New full-depth, comparable-budget model (~183k, 4 layers).
  2)  python scripts/fish/09_train_fish_transformer.py \
          --config configs/fish_transformer_inv_depth.yaml
      python scripts/fish/10_calibrate_fish_raps.py \
          --tag transformer_inv_T64_depth ;;

  *)  echo "unknown array index ${LSB_JOBINDEX}"; exit 1 ;;
esac

echo
echo "Finished: $(date)"
