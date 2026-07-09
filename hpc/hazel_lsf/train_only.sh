#!/bin/bash
#BSUB -J genmod-train
#BSUB -q gpu
#BSUB -R "select[h100||h200] rusage[mem=32] span[hosts=1]"
#BSUB -gpu "num=1"
#BSUB -n 8
#BSUB -W 48:00
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
# GPU selection: H100 (80 GB) or H200 (141 GB) -- i.e. any GPU with
# >= 80 GB VRAM. The gpu queue routes to a100/l40/h100 (per
# `bqueues -l gpu`); h200 is included in case it becomes reachable,
# though in practice this lands on H100. We require >= 80 GB because it
# comfortably fits batch_size 16 at this long sequence length (L40's
# 48 GB is tight, A100's 40 GB too small), and keeping both base and
# rowcol on the same >=80GB class makes the comparison apples-to-apples.
#
# Host memory: on Hazel LSF, rusage[mem] is per-host GB (mem=4 reserved
# only 4 GB and the job ran at ~625% MEM efficiency). rusage[mem=32]
# reserves 32 GB, matching the Slurm train_only.sh --mem=32G.
#
# Wall time: full training is ~45 min/epoch x 30 epochs + eval ~= 24 h,
# so -W 12:00 is far too short (the job would be killed ~epoch 15,
# before the conformal step, leaving no results.json). The gpu queue
# allows up to 4320 min (72 h); we request 48 h for margin.
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
