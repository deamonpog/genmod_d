#!/bin/bash
# Capacity/depth control runs, LOCAL on the RTX 4090.
#
# Both runs are T=64, which the fish pipeline runs locally by convention (see
# hpc/hazel_lsf/fish_long_windows.sh: T=16/32/64 are cheap and on the critical
# path, so they run locally where completion is guaranteed; only T>=128 goes to
# Hazel). Sequential, not parallel: one 4090 shared between two runs would only
# slow both down.
#
# Produces, for each run, a committed per-epoch training_log.csv and a
# run_summary.json with the exact parameter count, then recomputes the capacity
# statistics (exp_id-clustered primary, run-level sensitivity, gap-recovery g,
# delta-equivalence verdict) into results/fish/capacity_stats.json.

set -e
cd "$(dirname "$0")/../.."

# Absolute interpreter: Git Bash re-initializes PATH from the user profile, so
# the env's `python` is not on PATH here even under `conda run`. Override $PY
# if the env moves.
PY="${PY:-C:/Users/pog66/.conda/envs/ml2/python.exe}"
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

echo "=================================================="
echo "=== [1/2] VERIFY 148k / 2-layer -> transformer_inv_T64_small_verify"
echo "===       (original transformer_inv_T64_small is left untouched)"
echo "=================================================="
"$PY" -u scripts/fish/09_train_fish_transformer.py \
    --config configs/fish_transformer_inv_small.yaml \
    --tag transformer_inv_T64_small_verify
"$PY" -u scripts/fish/10_calibrate_fish_raps.py \
    --tag transformer_inv_T64_small_verify

echo "=================================================="
echo "=== [2/2] NEW ~183k / 4-layer -> transformer_inv_T64_depth"
echo "=================================================="
"$PY" -u scripts/fish/09_train_fish_transformer.py \
    --config configs/fish_transformer_inv_depth.yaml
"$PY" -u scripts/fish/10_calibrate_fish_raps.py \
    --tag transformer_inv_T64_depth

echo "=================================================="
echo "=== capacity statistics (exp_id primary, run sensitivity)"
echo "=================================================="
"$PY" -u scripts/fish/12_capacity_stats.py \
    --small transformer_inv_T64_small_verify \
    --depth transformer_inv_T64_depth \
    --large transformer_inv_T64 \
    --gru   gru_T64

echo "CAPACITY CONTROL RUNS COMPLETE"
