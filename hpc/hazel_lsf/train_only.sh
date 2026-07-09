#!/bin/bash
#BSUB -J genmod-train
#BSUB -q gpu
#BSUB -R "select[h100||h200||l40] rusage[mem=4] span[hosts=1]"
#BSUB -gpu "num=1"
#BSUB -n 8
#BSUB -W 12:00
#BSUB -o results/logs/genmod-train-lsf.%J.out
#BSUB -e results/logs/genmod-train-lsf.%J.err

# TRANSITIONAL LSF script -- train the rule-tree classifier on an L40.
#
# Hazel is migrating from LSF to Slurm. During the transition LSF still
# owns most of the GPU fleet (bhosts shows gpu_l40 with ~64 slots and
# gpu_l40s with ~896), while the Slurm test pool currently exposes only
# one fully-booked L40 node. Submitting the training stage through LSF
# is therefore far more likely to land an L40 quickly.
#
# Storage is shared, so this reads the SAME GENERATED_DATA produced by
# the Slurm data-gen stages (hpc/hazel/build_library.sh +
# simulate_array.sh). Run those first (on Slurm), then submit this.
#
# GPU selection: accept H100 (80 GB), H200 (141 GB), or L40 (48 GB) --
# whichever the LSF gpu queue can place first. Per bhosts, gpu_h100
# (~109 free) and gpu_h200 have far more free capacity than gpu_l40
# (closed/full), so this starts much faster than L40-only. Note: the
# LSF `gpu` queue historically routes to h100/l40 but NOT h200 (h200
# lives in other queues), so in practice this lands on H100 or L40.
# All three are >= 48 GB, so the config default batch_size 16 fits.
#
# Host memory: on Hazel LSF, rusage[mem] is in GB (the default was
# mem=2.00/task). rusage[mem=4] requests 4 GB/task x 8 tasks = 32 GB,
# matching the Slurm train_only.sh --mem=32G (the 16 GB default is too
# low for the full tokenized dataset).
#
# IMPORTANT: this writes results to results/logs/ruletree_base/, the
# same place as the Slurm train_only.sh. Do NOT let both the Slurm and
# LSF base-train jobs RUN simultaneously -- submit to both, then cancel
# the loser (bkill / scancel) as soon as one starts.
#
# Submission (LSF uses a redirect):
#   bsub < hpc/hazel_lsf/train_only.sh

# Activate conda BEFORE `set -e`; the activation chain emits internal
# non-zero exits that `set -e` would catch and abort on, even though
# the overall activation succeeds.
source ~/.bashrc
module load conda
conda activate /usr/local/usrapps/cads/$USER/genmod-env

set -e

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

mkdir -p results/logs results/checkpoints results/figures

echo "=== Train rule-tree classifier (base: time_space) [LSF] ==="
echo "Host:    $(hostname)"
echo "JobID:   $LSB_JOBID"
echo "Started: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo

python scripts/train.py --config configs/ruletree_base.yaml

echo
echo "Finished: $(date)"
