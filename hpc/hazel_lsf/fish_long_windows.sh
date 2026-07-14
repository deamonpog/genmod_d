#!/bin/bash
#BSUB -J fish-long[1-9]
#BSUB -q gpu
#BSUB -R "select[h100||h200||l40s] rusage[mem=32] span[hosts=1]"
#BSUB -gpu "num=1"
#BSUB -n 8
#BSUB -W 24:00
#BSUB -o results/logs/fish-long.%J.%I.out
#BSUB -e results/logs/fish-long.%J.%I.err

# Fish case study: the LONG-WINDOW and ablation runs, as an LSF job array.
#
# Split of labour with the local RTX 4090:
#
#   local   T=16, T=32, T=64   -- cheap, on the critical path for the paper,
#                                 and guaranteed to finish before the deadline
#   Hazel   T=128, 256, 512    -- expensive (T=512 alone is ~12 h on a 4090),
#           + the ablations       and NOT on the critical path. If they land in
#                                 time they go in the paper; if not, future work.
#
# Cost scales roughly linearly in T: attention is O(A*T^2 + T*A^2) per window
# and the window count falls as 1/T. T=512 is about 7x a T=64 epoch, but it has
# only ~3k training windows, so the epoch budget is cut accordingly (a 120-epoch
# schedule there would overfit, not help).
#
# Array indices (LSF is 1-based):
#   1  transformer, invariant features, T=128
#   2  transformer, invariant features, T=256
#   3  transformer, invariant features, T=512
#   4  GRU baseline, T=128
#   5  GRU baseline, T=256
#   6  GRU baseline, T=512
#   7  ablation: raw ("all") features at T=64      -- the invariance comparison
#   8  ablation: basic (solo kinematics) at T=64   -- the relational comparison
#   9  ablation: no augmentation at T=64
#
# PREREQUISITE: data/fish/{runs.npz,splits.json,windows_T*.npz} must exist.
# They are produced by scripts/fish/01..06, which are CPU-only and take a few
# minutes. Run them once on a login node or in an interactive session:
#
#   python scripts/fish/01_inspect_agent_fish_data.py
#   python scripts/fish/02_create_rule_lookup.py
#   python scripts/fish/03_preprocess_agent_trajectories.py
#   python scripts/fish/04_create_grouped_splits.py
#   python scripts/fish/05_create_trajectory_windows.py
#   python scripts/fish/06_create_summary_features.py
#
# Submission (LSF uses a redirect):
#   bsub < hpc/hazel_lsf/fish_long_windows.sh

# Activate conda BEFORE `set -e`: the activation chain emits internal non-zero
# exits that `set -e` would abort on, even though activation itself succeeds.
source ~/.bashrc
module load conda
conda activate /usr/local/usrapps/cads/$USER/genmod-env

set -e

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

mkdir -p results/logs results/fish configs/fish_ablations

echo "=== Fish case study, long windows / ablations [LSF] ==="
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

# Write a config on the fly by overriding the two fields that change.
# The base config already pins the model, the folds, and the conformal alphas.
make_cfg () {   # $1 = tag, $2 = window, $3 = features, $4 = epochs, $5 = augment(true/false)
    local out="configs/fish_ablations/$1.yaml"
    sed -e "s/^tag:.*/tag: $1/" \
        -e "s/^window:.*/window: $2/" \
        -e "s/^features:.*/features: $3/" \
        -e "s/^epochs:.*/epochs: $4/" \
        configs/fish_transformer_inv.yaml > "$out"
    if [ "$5" = "false" ]; then
        # Turn every symmetry off. Only the first 'enabled:' belongs to augment.
        sed -i -e "s/^  enabled: true/  enabled: false/" \
               -e "s/^  rotate: true/  rotate: false/" \
               -e "s/^  reflect: true/  reflect: false/" \
               -e "s/^  permute: true/  permute: false/" "$out"
    fi
    echo "$out"
}

case ${LSB_JOBINDEX} in
  1)  CFG=$(make_cfg fish_inv_T128 128 invariant 120 true)
      python scripts/fish/09_train_fish_transformer.py --config "$CFG"
      python scripts/fish/10_calibrate_fish_raps.py --tag fish_inv_T128 ;;

  2)  CFG=$(make_cfg fish_inv_T256 256 invariant  80 true)
      python scripts/fish/09_train_fish_transformer.py --config "$CFG"
      python scripts/fish/10_calibrate_fish_raps.py --tag fish_inv_T256 ;;

  # T=512 keeps only ~3k training windows, so a long schedule overfits rather
  # than helps. 60 epochs, and it is still the longest job in the array.
  3)  CFG=$(make_cfg fish_inv_T512 512 invariant  60 true)
      python scripts/fish/09_train_fish_transformer.py --config "$CFG"
      python scripts/fish/10_calibrate_fish_raps.py --tag fish_inv_T512 ;;

  4)  python scripts/fish/08_train_gru_baseline.py --window 128
      python scripts/fish/10_calibrate_fish_raps.py --tag gru_T128 ;;
  5)  python scripts/fish/08_train_gru_baseline.py --window 256
      python scripts/fish/10_calibrate_fish_raps.py --tag gru_T256 ;;
  6)  python scripts/fish/08_train_gru_baseline.py --window 512
      python scripts/fish/10_calibrate_fish_raps.py --tag gru_T512 ;;

  # The invariance comparison: same architecture, same budget, raw coordinates.
  7)  CFG=$(make_cfg fish_raw_T64 64 all 120 true)
      python scripts/fish/09_train_fish_transformer.py --config "$CFG"
      python scripts/fish/10_calibrate_fish_raps.py --tag fish_raw_T64 ;;

  # The relational comparison: solo kinematics only, no neighbor information.
  8)  CFG=$(make_cfg fish_basic_T64 64 basic 120 true)
      python scripts/fish/09_train_fish_transformer.py --config "$CFG"
      python scripts/fish/10_calibrate_fish_raps.py --tag fish_basic_T64 ;;

  9)  CFG=$(make_cfg fish_noaug_T64 64 invariant 120 false)
      python scripts/fish/09_train_fish_transformer.py --config "$CFG"
      python scripts/fish/10_calibrate_fish_raps.py --tag fish_noaug_T64 ;;

  *)  echo "unknown array index ${LSB_JOBINDEX}"; exit 1 ;;
esac

echo
echo "Finished: $(date)"
