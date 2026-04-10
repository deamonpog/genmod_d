# Running GenMod (ideaD: rule trees) on HPC

This branch implements **Option D**: classification of Schelling segregation
dynamics over auto-generated rule trees, with conformal prediction for
uncertainty quantification.

The codebase ships with two sets of HPC scripts side by side in `slurm/`:

| Cluster | Scheduler | Script suffix | Status |
|---|---|---|---|
| **Pasteur** (single 384-core fat node, 8x H200) | Slurm | `*.sh` | Default; use when Pasteur is online |
| **NCSU Hazel** (multi-node, ~14000 cores, mixed GPUs) | LSF | `*_hazel.sh` | Backup during Pasteur maintenance |

If Pasteur is up, use the Slurm scripts (Step 1 onward below). If Pasteur
is in maintenance, jump to **"Running on NCSU Hazel HPC (LSF)"** at the
bottom of this file.

## Configs (shared between both clusters)

| Config | Purpose |
|--------|---------|
| `configs/ruletree_base.yaml` | Full experiment: 50x50 grid, base model, 30 epochs, conformal enabled, batch_size 16 |
| `configs/ruletree_smoke.yaml` | Smoke test: tiny model, ~50-100 trees, 5 runs/tree, 3 epochs |

---

## Running on Pasteur HPC (Slurm)

### Step 1: Copy project to Pasteur

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

### Step 2: Create the conda environment (one-time)

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

### Step 3: Submit experiments

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

### Step 4: Monitor jobs

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

### Step 5: Inspect results

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

### Step 6: Copy results back

From your local machine:

```bash
scp -r pasteur-login:/data/scratch/casl/$USER/genmod_d/results/ ./results/
```

### Pasteur slurm scripts

| Script | Partition | Time | Purpose |
|--------|-----------|------|---------|
| `slurm/setup_env.sh` | login | n/a | One-time conda env setup |
| `slurm/fix_pytorch_cuda.sh` | n/a | n/a | Reinstall PyTorch CUDA build |
| `slurm/train_ruletrees_smoke.sh` | debug | 30 min | End-to-end smoke test |
| `slurm/train_ruletrees.sh` | standard | 48 h | Full Option D experiment |

### Pasteur custom runs

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

### Pasteur expected runtimes (1x H200 + 128 CPU workers, fat node)

| Stage | Estimate |
|-------|----------|
| Tree fingerprinting (10000 -> ~6500 unique, parallel across 128 cores) | 5-10 min |
| Simulation data (~2000 trees x 100 runs, parallel across 128 cores) | 8-12 hours |
| Training (30 epochs, base model on ~150K samples) on H200 | 2-4 hours |
| Total | ~12-16 hours wall clock |

The simulator is pure Python and CPU-bound, so multiprocessing gives a
near-linear speedup over the serial path. Workers are dispatched at the
tree level for the simulation phase and at the candidate level for
fingerprinting; output is bit-identical to the serial path.

### Pasteur troubleshooting

**"CUDA out of memory"**: reduce `batch_size` in
`configs/ruletree_base.yaml` or via `--batch_size 8`. The Pasteur H200
has 143 GB so this should not happen at the default batch size 16.

**"Missing tree_library.json"**: run the data-generation script first, or
re-submit `slurm/train_ruletrees.sh` (it will auto-generate if absent).

**"Module not found"**: activate the conda env:
```bash
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate /data/apps/casl/arachchige/genmod-env
```

---

## Running on NCSU Hazel HPC (LSF)

Use this section when Pasteur is in maintenance. Hazel runs IBM Spectrum
LSF (the cluster is in transition to Slurm but LSF still owns the bulk of
the resources). Submission, monitoring, and array syntax are all
LSF-specific (`bsub`, `bjobs`, `-J "name[1-N]%K"`, `-w "done(JID)"`).

The Hazel pipeline is split into three stages so the GPU does not sit
idle during pure-Python data generation:

1. **build library** -- one CPU job (16 cores, ~15-30 min)
2. **simulate trees** -- LSF job array of 30 tasks (16 cores each, max 8
   concurrent, ~3-4 hours)
3. **train classifier** -- one GPU job on an L40S (~3-5 hours)

The three stages are chained with `bsub -w "done(JID)"` so you can
submit them all at once and walk away.

### Hazel: Step 1 - First-time setup (login node)

```bash
ssh cdondim@login.hpc.ncsu.edu

# Find your project group (cads in our case)
groups

# Clone or rsync the project into the scratch area
cd /share/cads/$USER
git clone <repo-url> genmod_d
# OR rsync -av --exclude='GENERATED_DATA*' --exclude='results' --exclude='.git' \
#     ../from-local/genmod_d/ /share/cads/$USER/genmod_d/

cd /share/cads/$USER/genmod_d

# One-time conda env setup. This script also writes ~/.condarc to
# redirect the conda packages cache to /share/cads/$USER/conda/pkgs
# (otherwise conda will fill the 15 GB home quota).
bash slurm/setup_env_hazel.sh
```

The env lives at `/usr/local/usrapps/cads/cdondim/genmod-env`. To verify
it can see a GPU, request an interactive GPU node:

```bash
bsub -Is -q gpu -m "gpu_l40s gpu_a30 gpu_a10" -gpu "num=1" -n 2 \
    -R "rusage[mem=8000]" -W 00:15 bash
# Inside the interactive session:
source ~/.bashrc
module load conda
conda activate /usr/local/usrapps/cads/$USER/genmod-env
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
exit
```

### Hazel: Step 2 - Smoke test (3-stage chain, ~15 minutes)

The smoke test exercises the same array + dependency machinery as the
real run but with a tiny library and a tiny model. Submit all three
stages with chained dependencies:

```bash
cd /share/cads/$USER/genmod_d
mkdir -p results/logs

LIB=$(bsub < slurm/build_library_smoke_hazel.sh    | awk '{print $2}' | tr -d '<>')
SIM=$(bsub -w "done($LIB)" < slurm/simulate_array_smoke_hazel.sh | awk '{print $2}' | tr -d '<>')
TR=$(bsub  -w "done($SIM)" < slurm/train_only_smoke_hazel.sh    | awk '{print $2}' | tr -d '<>')
echo "smoke: lib=$LIB sim=$SIM train=$TR"
```

When everything finishes, inspect:

```bash
ls results/logs/ | grep smoke
cat results/logs/ruletree_smoke/results.json
```

### Hazel: Step 3 - Full run (3-stage chain)

Same submission pattern as the smoke test, but with the production
scripts:

```bash
cd /share/cads/$USER/genmod_d
mkdir -p results/logs

LIB=$(bsub < slurm/build_library_hazel.sh    | awk '{print $2}' | tr -d '<>')
SIM=$(bsub -w "done($LIB)" < slurm/simulate_array_hazel.sh | awk '{print $2}' | tr -d '<>')
TR=$(bsub  -w "done($SIM)" < slurm/train_only_hazel.sh    | awk '{print $2}' | tr -d '<>')
echo "full: lib=$LIB sim=$SIM train=$TR"
```

### Hazel: Step 4 - Monitor

```bash
bjobs                           # all your jobs
bjobs -l <jobid>                # detailed status of one job
bjobs -A <jobid>                # array task summary
tail -f results/logs/genmod-build.<JID>.out
tail -f results/logs/genmod-sim.<JID>.<TASKID>.out
tail -f results/logs/genmod-train.<JID>.out
bkill <jobid>                   # cancel a job (or array)
bqueues -u $USER                # which queues you have access to
```

### Hazel: LSF scripts

| Script | Queue | Resources | Purpose |
|--------|-------|-----------|---------|
| `slurm/setup_env_hazel.sh` | login | interactive | One-time conda env setup |
| `slurm/build_library_hazel.sh` | standard | 16 CPUs, 32G, 4h | Stage 1: build tree library |
| `slurm/simulate_array_hazel.sh` | standard | array 1-30 %8, 16 CPUs each | Stage 2: simulate trees |
| `slurm/train_only_hazel.sh` | gpu | 1x L40S, 8 CPUs, 64G, 12h | Stage 3: train classifier |
| `slurm/build_library_smoke_hazel.sh` | standard | 4 CPUs, 8G, 30m | Smoke stage 1 |
| `slurm/simulate_array_smoke_hazel.sh` | standard | array 1-3 %2, 4 CPUs each | Smoke stage 2 |
| `slurm/train_only_smoke_hazel.sh` | gpu | 1x small GPU, 4 CPUs, 16G, 30m | Smoke stage 3 |

### Hazel: Expected runtimes

| Stage | Resource | Wall clock |
|-------|----------|------------|
| Stage 1: build library | 16 CPUs | 15-30 min |
| Stage 2: simulate array (30 tasks, 8 concurrent) | up to 128 cores peak | 3-4 hours |
| Stage 3: train | 1 L40S + 8 CPUs | 3-5 hours (slower at batch=16) |
| **End to end (with queue waits)** | | **7-12 hours** |

### Hazel: Troubleshooting

**"User not authorized to use queue"**: try a different queue. The
production scripts default to `-q standard`. If you have access to a
private group queue, edit the `#BSUB -q` line in
`slurm/build_library_hazel.sh` and `slurm/simulate_array_hazel.sh`. To
see your queue access from the login node:
```bash
bqueues -u $USER
```

**"Cannot find tree_library.json" in stage 2**: stage 1 must complete
successfully before stage 2 runs. This should be enforced by the
`-w "done(JID)"` dependency, but if you submitted stages out of order
just resubmit stage 2 after stage 1 finishes.

**"CUDA out of memory" on L40S**: the production config uses
`batch_size=16` which fits comfortably in 48 GB at fp32. If you somehow
still OOM, drop to `--batch_size 8` on the CLI or in the YAML.

**Conda activation fails in batch jobs**: Hazel requires
`source ~/.bashrc` BEFORE `conda activate` in batch scripts. All our
`*_hazel.sh` scripts already do this. If you write a custom script,
do not forget it.

**Pasteur is back online**: switch back to the Pasteur slurm scripts at
the top of this file. The `*_hazel.sh` files can stay in `slurm/` for
the next maintenance cycle.
