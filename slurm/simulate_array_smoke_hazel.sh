#!/bin/bash
#BSUB -J "genmod-sim-smoke[1-3]%2"
#BSUB -q standard
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8000]"
#BSUB -W 00:30
#BSUB -o results/logs/genmod-sim-smoke.%J.%I.out
#BSUB -e results/logs/genmod-sim-smoke.%J.%I.err

# Smoke stage 2 of 3: 3-task LSF array on Hazel HPC.
#
# This deliberately mirrors the production array submission (just smaller)
# so we exercise:
#   - LSF array syntax  -J "name[1-N]%K"
#   - LSB_JOBINDEX in shell
#   - --n_tasks / --task_id flags in generate_ruletrees.py
#   - Slice math: with ~30-50 unique trees (after dedup), 3 slices give
#     each task ~10-17 trees to simulate. None should be empty.
#   - %J.%I in output filenames
#
# Submit with a dependency on the smoke build:
#   bsub -w "done($SMOKE_LIB_JID)" < slurm/simulate_array_smoke_hazel.sh

set -e

source ~/.bashrc
module load conda
conda activate /usr/local/usrapps/cads/cdondim/genmod-env

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

TASK_ID=$((LSB_JOBINDEX - 1))
N_TASKS=3

echo "=== Smoke simulate array task ${TASK_ID}/${N_TASKS} ==="
echo "Host:     $(hostname)"
echo "JobID:    $LSB_JOBID"
echo "ArrayIdx: $LSB_JOBINDEX"
echo "Started:  $(date)"
echo

python scripts/generate_ruletrees.py \
    --simulate_only \
    --n_tasks $N_TASKS \
    --task_id $TASK_ID \
    --num_runs 5 \
    --grid_size 50 \
    --max_steps 200 \
    --snapshot_ticks 50,100,150,200 \
    --output_dir GENERATED_DATA_smoke \
    --seed 42 \
    --workers 4

echo
echo "Finished: $(date)"
