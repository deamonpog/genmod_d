#!/bin/bash
#SBATCH --job-name=genmod-sim
#SBATCH --partition=compute
#SBATCH --qos=normal
#SBATCH --array=1-30%8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=results/logs/genmod-sim.%A_%a.out
#SBATCH --error=results/logs/genmod-sim.%A_%a.err

# Hazel HPC stage 2 of 3: simulate trees.
#
# Slurm job array: 30 tasks total (--array=1-30), max 8 running
# concurrently (%8). Each task loads the previously-built tree library
# and simulates the slice
# library[task_id*total/30 : (task_id+1)*total/30]. The slicing math
# guarantees every label is covered exactly once.
#
# Wall-clock budget: 12 hours per task. Under the old LSF run at 4h,
# 13/30 tasks were killed by the run limit exactly at 4 hours, so the
# hardest slices genuinely take a bit more than 4h on 16 cores. 12h is
# a generous ceiling.
#
# Resumability: generate_ruletrees skips trees whose ruletree_NNNN.json
# already exists. If a task is re-run (e.g. after hitting the time
# limit) it will pick up only the trees it did not finish.
#
# %A in the output filename is the parent array job ID, %a is the array
# task index. $SLURM_ARRAY_TASK_ID (1-based) holds the current task
# index inside the script body.
#
# Submission with a dependency on stage 1:
#   sbatch --dependency=afterok:$LIB_JID hpc/hazel/simulate_array.sh

# Activate conda BEFORE `set -e`; the activation chain emits internal
# non-zero exits that `set -e` would catch and abort on, even though
# the overall activation succeeds.
source ~/.bashrc
module load conda
conda activate /usr/local/usrapps/cads/$USER/genmod-env

set -e

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# SLURM_ARRAY_TASK_ID is 1-based; convert to 0-based for the slicing math
TASK_ID=$((SLURM_ARRAY_TASK_ID - 1))
N_TASKS=30

echo "=== Simulate array task ${TASK_ID}/${N_TASKS} ==="
echo "Host:     $(hostname)"
echo "JobID:    $SLURM_ARRAY_JOB_ID"
echo "ArrayIdx: $SLURM_ARRAY_TASK_ID"
echo "CPUs:     $SLURM_CPUS_PER_TASK"
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
    --workers ${SLURM_CPUS_PER_TASK:-16}

echo
echo "Finished: $(date)"
