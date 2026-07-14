#!/bin/bash
# Short-window suite: the regime where equifinality is real.
#
# At T=64 the models already reach ~0.93 and the conformal sets have collapsed
# to singletons, so the coverage table there measures nothing (Section 4.3 of
# the paper says exactly this). The substantive result -- prediction sets that
# actually CONTAIN several rules, shrinking as observation grows -- lives at
# T=16 (7 s) and T=32 (14 s), where the relational random forest scores only
# 0.55 and 0.62.
#
# This is the paper's core figure, so it runs locally where completion is
# guaranteed, rather than behind an HPC queue.

set -e
cd "$(dirname "$0")/../.."

# Absolute interpreter: Git Bash re-initializes PATH from the user profile, so
# the conda env's `python` is not on PATH inside this shell even when the
# script is launched through `conda run`. Override with $PY if the env moves.
PY="${PY:-C:/Users/pog66/.conda/envs/ml2/python.exe}"
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

for T in 16 32; do
    echo "=================================================="
    echo "=== T=$T  transformer (invariant features)"
    echo "=================================================="
    "$PY" -u scripts/fish/09_train_fish_transformer.py \
        --config "configs/fish_inv_T${T}.yaml"

    echo "=== T=$T  GRU baseline"
    "$PY" -u scripts/fish/08_train_gru_baseline.py --window "$T"

    echo "=== T=$T  RAPS"
    "$PY" -u scripts/fish/10_calibrate_fish_raps.py --tag "transformer_inv_T${T}"
    "$PY" -u scripts/fish/10_calibrate_fish_raps.py --tag "gru_T${T}"
done

echo "=== regenerating figures"
"$PY" -u scripts/fish/13_make_figures.py
echo "SHORT WINDOW SUITE COMPLETE"
