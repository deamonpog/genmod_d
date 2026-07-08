#!/bin/bash
#SBATCH --job-name=genmod-build-smoke
#SBATCH --partition=compute
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=results/logs/genmod-build-smoke.%j.out
#SBATCH --error=results/logs/genmod-build-smoke.%j.err

# Hazel HPC smoke stage 1 of 3: build a tiny rule-tree library.
#
# Generates 100 candidate trees at depth 2 (instead of 10000 at depth
# 3 in the production stage) and runs short 200-step probes for dedup.
# Output goes to GENERATED_DATA_smoke so it does not collide with the
# production GENERATED_DATA directory.

# Activate the conda env BEFORE turning on `set -e`. The conda
# activation chain (source ~/.bashrc -> module load conda -> conda
# activate) emits non-zero exit codes from internal substeps that
# `set -e` would catch and abort the script on, even though the
# overall activation succeeds. Order matters: env first, then strict.
source ~/.bashrc
module load conda
conda activate /usr/local/usrapps/cads/$USER/genmod-env

set -e

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mkdir -p results/logs results/checkpoints results/figures

echo "=== Smoke build tree library ==="
echo "Host:    $(hostname)"
echo "JobID:   $SLURM_JOB_ID"
echo "Python:  $(which python)"
echo "Started: $(date)"
echo

python scripts/generate_ruletrees.py \
    --library_only \
    --n_candidates 100 \
    --max_depth 2 \
    --method mixed \
    --grid_size 50 \
    --max_steps 200 \
    --output_dir GENERATED_DATA_smoke \
    --seed 42 \
    --workers ${SLURM_CPUS_PER_TASK:-16}

echo
echo "Finished: $(date)"
