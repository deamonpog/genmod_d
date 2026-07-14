#!/bin/bash
#BSUB -J genmod-recal
#BSUB -q gpu
#BSUB -R "select[h100||h200||l40s] rusage[mem=32] span[hosts=1]"
#BSUB -gpu "num=1"
#BSUB -n 8
#BSUB -W 2:00
#BSUB -o results/logs/genmod-recal.%J.out
#BSUB -e results/logs/genmod-recal.%J.err

# Recompute the Schelling conformal results with the corrected RAPS code.
#
# NO RETRAINING. This loads results/checkpoints/<name>/best.pt and does a
# single forward pass over the held-out data. It needs a GPU only because the
# login nodes have none and a 2089-class forward pass over the test set is
# slow on CPU; the work itself is minutes.
#
# Three defects are corrected relative to the numbers currently in
# results/logs/<name>/results.json:
#
#   1. The APS randomization was skipped when the true class ranked first, so
#      for an accurate model most calibration scores were inflated to the full
#      p_max instead of u * p_max. q_hat inflated with them and every
#      prediction set came out too large.
#
#   2. conformal_predict built sets deterministically while calibration
#      randomized them, so the two used different rules.
#
#   3. The conformal calibration set was drawn from the VALIDATION split --
#      the same data used for early stopping and temperature fitting. Conformal
#      requires the calibration scores to be exchangeable with the test scores,
#      and a split that drove model SELECTION is not. The coverage guarantee as
#      originally computed is therefore not valid, merely plausible.
#
#      Fixed by carving the calibration set out of TEST, which the model never
#      saw. The weights do not depend on which held-out data we calibrate on,
#      so this needs no retraining.
#
# Output: results/logs/<name>/conformal_recalibrated.json
# The accuracy numbers in the original results.json remain valid; only the
# conformal block is superseded.
#
# Submission:
#   bsub < hpc/hazel_lsf/recalibrate_conformal.sh

source ~/.bashrc
module load conda
conda activate /usr/local/usrapps/cads/$USER/genmod-env

set -e

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

mkdir -p results/logs

echo "=== Recalibrate Schelling conformal [LSF] ==="
echo "Host:    $(hostname)"
echo "JobID:   $LSB_JOBID"
echo "Started: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo

for CFG in configs/ruletree_base.yaml configs/ruletree_rowcol.yaml; do
    NAME=$(basename "$CFG" .yaml)
    CKPT="results/checkpoints/${NAME}/best.pt"
    if [ ! -f "$CKPT" ]; then
        echo "SKIP $CFG -- no checkpoint at $CKPT"
        continue
    fi
    echo "=== $CFG ==="
    python scripts/recalibrate_conformal.py --config "$CFG"
    echo
done

echo "Finished: $(date)"
