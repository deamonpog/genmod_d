#!/bin/bash
#BSUB -J "genmod-sim[1-30]%8"
#BSUB -q standard
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32000]"
#BSUB -W 04:00
#BSUB -o results/logs/genmod-sim.%J.%I.out
#BSUB -e results/logs/genmod-sim.%J.%I.err

# Stage 2 of 3: simulate trees on Hazel HPC.
#
# LSF job array: 30 tasks total, max 8 running concurrently. Each task
# loads the previously-built tree library and simulates the slice
# library[task_id*total/30 : (task_id+1)*total/30]. The slicing math
# guarantees every label is covered exactly once.
#
# At 8 concurrent tasks * 16 cores = 128 cores at peak, well within the
# standard queue's 1024 per-user limit.
#
# Submit with a dependency on stage 1:
#   bsub -w "done($LIB_JID)" < slurm/simulate_array_hazel.sh

set -e

source ~/.bashrc
module load conda
conda activate /usr/local/usrapps/cads/cdondim/genmod-env

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# LSB_JOBINDEX is 1-based; convert to 0-based for the python slicing math
TASK_ID=$((LSB_JOBINDEX - 1))
N_TASKS=30

echo "=== Simulate array task ${TASK_ID}/${N_TASKS} ==="
echo "Host:     $(hostname)"
echo "JobID:    $LSB_JOBID"
echo "ArrayIdx: $LSB_JOBINDEX"
echo "CPUs:     16"
echo "Started:  $(date)"
echo

python scripts/generate_ruletrees.py \
    --simulate_only \
    --n_tasks $N_TASKS \
    --task_id $TASK_ID \
    --num_runs 100 \
    --grid_size 50 \
    --max_steps 500 \
    --snapshot_ticks 100,200,300,400,500 \
    --output_dir GENERATED_DATA \
    --seed 42 \
    --workers 16

echo
echo "Finished: $(date)"
