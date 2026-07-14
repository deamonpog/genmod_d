#!/bin/bash
# The conformal-vs-observation-length sweep. This is the paper's core figure.
#
# WHY THE GRU AND NOT THE TRANSFORMER.
#
# Section 4.1 establishes that model choice does not matter once relational and
# temporal structure are present: a 156k-parameter GRU over group averages
# matches a 1.8M-parameter axial transformer over per-agent tokens (paired
# difference -0.0015, 95% CI [-0.008, +0.005]). Using the GRU for the sweep is
# therefore not a shortcut, it is the finding applied: the two models are
# interchangeable for this measurement, and the GRU trains roughly 40x faster,
# which is what makes a six-point sweep with five folds and full conformal
# calibration feasible at all.
#
# The transformer is still reported at T=64, so the equivalence is shown rather
# than merely asserted.
#
# WHAT THE SWEEP IS FOR.
#
# At T=64 the model is already 93% accurate and the conformal sets have
# collapsed to singletons -- there is no equifinality left to measure, and a
# coverage table there demonstrates nothing. The substantive regime is short
# windows: at T=16 (7 s) the relational random forest scores 0.549. That is
# where prediction sets genuinely contain several rules, and where their
# shrinkage with observation length becomes the paper's argument in a picture.

# NOT `set -e`. On Windows, 08_train_gru_baseline.py runs to completion, prints
# its results and writes its .npz -- and then exits 127, apparently while tearing
# down the cuDNN RNN state. (09_train_fish_transformer.py, which has no RNN,
# exits 0 cleanly.) The results are valid; only the exit code lies. Under
# `set -e` that bogus code aborted the whole sweep after the first window
# length, so we check for the OUTPUT FILE instead of trusting the exit status.
cd "$(dirname "$0")/../.."

# Git Bash re-initializes PATH from the user profile, so the conda env's python
# is not on PATH here even under `conda run`. Use the interpreter directly.
PY="${PY:-C:/Users/pog66/.conda/envs/ml2/python.exe}"
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

for T in 16 32 128 256 512; do
    echo "=================================================="
    echo "=== T=$T  GRU"
    echo "=================================================="

    if [ -f "results/fish/gru_T${T}/probs_fold4.npz" ]; then
        echo "already trained, skipping"
    else
        "$PY" -u scripts/fish/08_train_gru_baseline.py --window "$T" --epochs 60 || true
        if [ ! -f "results/fish/gru_T${T}/probs_fold4.npz" ]; then
            echo "ERROR: T=$T produced no fold-4 output; this one really did fail"
            continue
        fi
    fi

    echo "=== T=$T  RAPS"
    "$PY" -u scripts/fish/10_calibrate_fish_raps.py --tag "gru_T${T}" || \
        echo "ERROR: RAPS failed for T=$T"
done

echo "=== regenerating figures"
"$PY" -u scripts/fish/13_make_figures.py || echo "ERROR: figures failed"
echo "CONFORMAL SWEEP COMPLETE"
