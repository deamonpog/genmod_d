# Running GenMod on Pasteur HPC

## Step 1: Copy project to Pasteur

From your local machine:

```bash
# Copy code (exclude local data, results, and caches)
rsync -av --exclude='GENERATED_DATA' --exclude='results' --exclude='__pycache__' \
    --exclude='.git' --exclude='*.pyc' \
    GenMod/ pasteur-login:/data/scratch/casl/$USER/GenMod/
```

Or if rsync isn't available, use scp:

```bash
# From the parent directory of GenMod
scp -r GenMod pasteur-login:/data/scratch/casl/$USER/
```

## Step 2: Create the conda environment (one-time)

SSH into pasteur-login, then run:

```bash
ssh pasteur-login

cd /data/scratch/casl/$USER/GenMod
bash slurm/setup_env.sh
```

This creates a conda environment at `/data/apps/casl/arachchige/genmod-env` with Python 3.11, PyTorch (CUDA 12.1 for H200), and all dependencies.

**Verify it worked:**

```bash
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate /data/apps/casl/arachchige/genmod-env
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Step 3: Submit all experiments

```bash
cd /data/scratch/casl/$USER/GenMod
mkdir -p results/logs

# Option A: Submit all three systems at once
bash slurm/run_all.sh

# Option B: Submit individually
sbatch slurm/train_eca.sh
sbatch slurm/train_logistic.sh
sbatch slurm/train_schelling.sh
```

Each job:
1. Generates the data if not already present
2. Trains a Base model
3. Runs calibration analysis, conformal prediction, embedding analysis
4. Saves results to `results/`

## Step 4: Monitor jobs

```bash
# Check job status
squeue -u $(whoami)

# Watch a specific job's output in real-time
tail -f results/logs/genmod-eca_<JOBID>.out

# Check cluster status
sinfo

# Cancel a job if needed
scancel <JOBID>
```

## Step 5: Check results

After jobs complete:

```bash
# Results JSON files
cat results/logs/all_256_base/results.json
cat results/logs/logistic_base/results.json
cat results/logs/schelling_base/results.json

# Figures (reliability diagrams, t-SNE plots)
ls results/figures/*/

# Training curves
cat results/logs/*/training_log.csv
```

## Step 6: Copy results back to local machine

From your local machine:

```bash
scp -r pasteur-login:/data/scratch/casl/$USER/GenMod/results/ GenMod/results/
```

---

## Experiment Configs Reference

| Config | System | What it does |
|--------|--------|-------------|
| `configs/all_256_base.yaml` | ECA | 256 rules, Base model, 20 epochs |
| `configs/logistic_base.yaml` | Logistic Map | 40 r-values, Base model, 20 epochs, + conformal |
| `configs/schelling_base.yaml` | Schelling | 9 thresholds, Base model, 20 epochs, + conformal |
| `configs/eca_conformal.yaml` | ECA | Same as all_256_base but with conformal prediction |
| `configs/generalization_75_25.yaml` | ECA | Held-out 25% of rules (generalization test) |
| `configs/ablation_std_pos.yaml` | ECA | Standard positional encoding (ablation) |

## Running Custom Experiments

```bash
# Override config values from command line
srun --partition=debug --gres=gpu:1 --time=01:00:00 --mem=32G --pty bash

# Then inside the interactive session:
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate /data/apps/casl/arachchige/genmod-env
cd /data/scratch/casl/$USER/GenMod

# Quick test
python scripts/train.py --config configs/logistic_base.yaml \
    --model_size tiny --epochs 3 --name quick_test

# ECA with conformal
python scripts/train.py --config configs/eca_conformal.yaml

# Generalization experiment
python scripts/train.py --config configs/generalization_75_25.yaml
```

## Expected Runtimes (H200 GPU)

| Experiment | Estimated Time |
|-----------|---------------|
| ECA Base (256 rules, 20 epochs) | ~4-6 hours |
| Logistic Base (40 classes, 20 epochs) | ~30 min |
| Schelling Base (9 classes, 20 epochs) | ~1-2 hours |
| All three in parallel | ~6 hours total (1 GPU each) |

## Troubleshooting

**"CUDA out of memory"**: Reduce batch size in the config YAML or via `--batch_size 128`.

**"Missing data file"**: The Slurm scripts auto-generate data if not found. If you want to generate separately:
```bash
python scripts/generate_all_256.py
python scripts/generate_logistic.py
python scripts/generate_schelling.py
```

**"Module not found"**: Make sure you activated the conda env:
```bash
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate /data/apps/casl/arachchige/genmod-env
```
