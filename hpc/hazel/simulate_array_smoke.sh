#!/bin/bash
#SBATCH --job-name=genmod-sim-smoke
#SBATCH --partition=compute
#SBATCH --qos=normal
#SBATCH --array=1-3%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=results/logs/genmod-sim-smoke.%A_%a.out
#SBATCH --error=results/logs/genmod-sim-smoke.%A_%a.err

# Hazel HPC smoke stage 2 of 3: 3-task Slurm array.
#
# Deliberately mirrors the production array submission (just smaller)
# so we exercise the Slurm features the production run depends on:
#   - Array syntax  --array=1-N%K
#   - $SLURM_ARRAY_TASK_ID in the script body
#   - --n_tasks / --task_id flags in generate_ruletrees.py
#   - Slice math: with ~30-50 unique trees (after dedup), 3 slices give
#     each task ~10-17 trees to simulate. None should be empty.
#   - %A_%a in output filenames
#
# Submission with a dependency on the smoke build:
#   sbatch --dependency=afterok:$SMOKE_LIB_JID hpc/hazel/simulate_array_smoke.sh

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

TASK_ID=$((SLURM_ARRAY_TASK_ID - 1))
N_TASKS=3

echo "=== Smoke simulate array task ${TASK_ID}/${N_TASKS} ==="
echo "Host:     $(hostname)"
echo "JobID:    $SLURM_ARRAY_JOB_ID"
echo "ArrayIdx: $SLURM_ARRAY_TASK_ID"
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
    --workers ${SLURM_CPUS_PER_TASK:-16}

echo
echo "Finished: $(date)"
