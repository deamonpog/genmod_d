#!/bin/bash
# Submit all three training jobs to Slurm
# Usage: bash slurm/run_all.sh

set -e

mkdir -p results/logs

echo "Submitting ECA training..."
JOB1=$(sbatch slurm/train_eca.sh | awk '{print $4}')
echo "  Job ID: $JOB1"

echo "Submitting Logistic Map training..."
JOB2=$(sbatch slurm/train_logistic.sh | awk '{print $4}')
echo "  Job ID: $JOB2"

echo "Submitting Schelling training..."
JOB3=$(sbatch slurm/train_schelling.sh | awk '{print $4}')
echo "  Job ID: $JOB3"

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$(whoami)"
echo "View logs in results/logs/"
