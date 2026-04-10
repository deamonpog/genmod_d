# Running GenMod (ideaD: rule trees) on Pasteur HPC

This branch implements **Option D**: classification of Schelling segregation
dynamics over auto-generated rule trees, with conformal prediction for
uncertainty quantification.

## Step 1: Copy project to Pasteur

From your local machine:

```bash
# Copy code (exclude local data, results, and caches)
rsync -av --exclude='GENERATED_DATA' --exclude='GENERATED_DATA_smoke' \
    --exclude='results' --exclude='__pycache__' \
    --exclude='.git' --exclude='*.pyc' \
    genmod_d/ pasteur-login:/data/scratch/casl/$USER/genmod_d/
```

Or use scp:

```bash
scp -r genmod_d pasteur-login:/data/scratch/casl/$USER/
```

## Step 2: Create the conda environment (one-time)

```bash
ssh pasteur-login

cd /data/scratch/casl/$USER/genmod_d
bash slurm/setup_env.sh
```

This creates a conda environment at `/data/apps/casl/arachchige/genmod-env`
with Python 3.11, PyTorch (CUDA 12.4 for H200), and dependencies.

**Verify it worked.** Login nodes have no GPUs, so `torch.cuda.is_available()`
will always return `False` there. Test on a GPU node via an interactive srun:

```bash
srun --partition=debug --gres=gpu:1 --time=00:10:00 --mem=8G --pty \
    bash slurm/fix_pytorch_cuda.sh test
```

That activates the env, runs `nvidia-smi`, and runs a small CUDA matmul to
confirm everything works. If PyTorch reports CUDA unavailable, reinstall the
GPU build from the login node:

```bash
bash slurm/fix_pytorch_cuda.sh install
```

then re-run the interactive test above.

## Step 3: Submit experiments

```bash
cd /data/scratch/casl/$USER/genmod_d
mkdir -p results/logs

# Smoke test first (debug partition, ~10 minutes)
sbatch slurm/train_ruletrees_smoke.sh

# Once smoke test passes, submit the full job
# (standard partition, 64 CPUs for parallel data gen, ~3-4h wall clock)
sbatch slurm/train_ruletrees.sh
```

The full job:
1. Generates 10000 candidate rule trees, deduplicates structurally and
   behaviorally, saves the resulting library to
   `GENERATED_DATA/rule_trees/tree_library.json`
2. Runs 100 Schelling simulations per tree (50x50 grid, 500 ticks)
3. Trains the base Transformer (4M params) for 30 epochs
4. Runs calibration analysis, RAPS conformal prediction, embedding analysis
5. Saves results to `results/logs/ruletree_base/results.json`

## Step 4: Monitor jobs

```bash
# Check job status
squeue -u $(whoami)

# Watch a specific job's output in real time
tail -f results/logs/genmod-ruletrees_<JOBID>.out

# Check cluster status
sinfo

# Cancel a job
scancel <JOBID>
```

## Step 5: Inspect results

```bash
# Smoke test results
cat results/logs/ruletree_smoke/results.json

# Full job results
cat results/logs/ruletree_base/results.json

# Figures (reliability diagrams, t-SNE)
ls results/figures/ruletree_base/

# Training curves
cat results/logs/ruletree_base/training_log.csv
```

## Step 6: Copy results back

From your local machine:

```bash
scp -r pasteur-login:/data/scratch/casl/$USER/genmod_d/results/ ./results/
```

---

## Configs

| Config | Purpose |
|--------|---------|
| `configs/ruletree_base.yaml` | Full experiment: 50x50 grid, base model, 30 epochs, conformal enabled |
| `configs/ruletree_smoke.yaml` | Smoke test: tiny model, 100 candidates, 5 runs/tree, 3 epochs |

## Slurm scripts

| Script | Partition | Time | Purpose |
|--------|-----------|------|---------|
| `slurm/setup_env.sh` | login | n/a | One-time conda env setup |
| `slurm/fix_pytorch_cuda.sh` | n/a | n/a | Reinstall PyTorch CUDA build |
| `slurm/train_ruletrees_smoke.sh` | debug | 30 min | End-to-end smoke test |
| `slurm/train_ruletrees.sh` | standard | 24 h | Full Option D experiment |

## Custom runs

```bash
# Interactive session for debugging
srun --partition=debug --gres=gpu:1 --time=01:00:00 --mem=32G --pty bash

# Inside the session
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate /data/apps/casl/arachchige/genmod-env
cd /data/scratch/casl/$USER/genmod_d

# Example: override hyperparameters at the CLI
python scripts/train.py --config configs/ruletree_base.yaml \
    --model_size small --epochs 5 --name quick_test
```

## Expected runtimes (1x H200 + 128 CPU workers, Pasteur fat node)

| Stage | Estimate |
|-------|----------|
| Tree fingerprinting (10000 -> ~6500 unique, parallel across 128 cores) | 5-10 min |
| Simulation data (~2000 trees x 100 runs, parallel across 128 cores) | 8-12 hours |
| Training (30 epochs, base model on ~150K samples) on H200 | 2-4 hours |
| Total | ~12-16 hours wall clock |

Note: the first run produced 2089 behaviorally distinct trees from the
10000 candidates (much more than the original 200-500 estimate), which
is why the simulation phase dominates total runtime.

The simulator is pure Python and CPU-bound, so multiprocessing gives a
near-linear speedup over the serial path. Workers are dispatched at the
tree level for the simulation phase and at the candidate level for
fingerprinting; output is bit-identical to the serial path.

## Troubleshooting

**"CUDA out of memory"**: reduce `batch_size` in
`configs/ruletree_base.yaml` or via `--batch_size 16`. The default of 32
assumes an 80GB H200; smaller GPUs will need a smaller batch.

**"Missing tree_library.json"**: run the data-generation script first, or
re-submit `slurm/train_ruletrees.sh` (it will auto-generate if absent).

**"Module not found"**: activate the conda env:
```bash
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate /data/apps/casl/arachchige/genmod-env
```
