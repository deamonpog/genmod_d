# GenMod_D Project Context (branch: ideaD)

## Project goal

Build a framework for **inverse generative modeling** of agent-based
models -- given observed dynamics from a Schelling-style simulation,
identify which **behavioral rule** (a tree expression over the 6
Gunaratne primitive factors) generated them, with **conformal prediction
guarantees** on the resulting prediction set.

This branch implements the approach we call **Option D** in the planning
documents: classification over an auto-generated library of rule trees.
A multi-system version of the framework (ECA, logistic map, Schelling
threshold) lives on `main` and is recoverable from there.

**Target venue:** NeurIPS 2026 (abstract May 4, paper May 6).

## Current state (as of 2026-04-10)

### What is implemented on `ideaD`

- **Rule tree core** ([genmod/data/rule_trees.py](genmod/data/rule_trees.py))
  - Frozen-dataclass tree structures (`LeafNode`, `OpNode`)
  - 6 primitive factors matched to Gunaratne et al. 2023 (including
    asymmetric F_Move and stored-utility F_Neigh)
  - Tree evaluation, canonical infix string, JSON serialization
- **Pluggable tree generation** ([genmod/data/tree_generators.py](genmod/data/tree_generators.py))
  - Pseudo-random, quasi-random (Halton), and mixed strategies
  - Behavioral fingerprinting for deduplication
  - Generator interface allows future GP-based methods to be plugged in
- **Extended Schelling simulator** ([genmod/data/schelling_ruletree.py](genmod/data/schelling_ruletree.py))
  - 50x50 grid, 0.95 density, torus boundary, Moore neighborhood
  - Random-vacancy movement: agent moves if utility there > current
  - Two-pass step (compute all utilities, then move) so F_Neigh reads
    a consistent snapshot of neighbor utilities
  - Per-cell ages and move histories for F_Age and F_Move
- **Dataset and metadata helpers**
  - [genmod/data/ruletree_dataset.py](genmod/data/ruletree_dataset.py)
    reuses `SchellingDataset` for tokenization
  - [genmod/data/ruletree_metadata.py](genmod/data/ruletree_metadata.py)
    groups trees by dominant factor for per-group analysis
- **Training pipeline**
  - [scripts/train.py](scripts/train.py) drives training, calibration,
    RAPS conformal prediction, and embedding analysis
  - [scripts/generate_ruletrees.py](scripts/generate_ruletrees.py)
    generates the tree library and per-tree simulation data
- **Configs**
  - [configs/ruletree_base.yaml](configs/ruletree_base.yaml) -- full
    experiment (10000 candidates, base model, 30 epochs)
  - [configs/ruletree_smoke.yaml](configs/ruletree_smoke.yaml) -- end-to-end
    smoke test with a tiny model
- **Slurm scripts for Pasteur HPC**
  - [slurm/train_ruletrees_smoke.sh](slurm/train_ruletrees_smoke.sh)
  - [slurm/train_ruletrees.sh](slurm/train_ruletrees.sh)

### What was removed from this branch

Recoverable from `main`:
- ECA system (`eca.py`, `dataset.py`, `wolfram.py`,
  `scripts/generate_all_256.py`, ECA configs and slurm scripts)
- Logistic map system (`logistic_map.py`, `logistic_dataset.py`,
  `logistic_metadata.py`, `scripts/generate_logistic.py`)
- Schelling threshold system (`schelling_metadata.py`,
  `scripts/generate_schelling.py`)
- Permutation entropy baseline
- Legacy root-level scripts (`main.py`, `gen_ca_data.py`,
  `train_ca_*.py`)
- All ECA / logistic / threshold-Schelling configs and slurm scripts

### Verification status

Local sanity checks (no heavy compute):
- All files are pure ASCII
- All Python imports succeed
- Tree round-trip through JSON works on hand-built trees
- Six factor functions return expected values on a small test grid
- Both YAML configs parse and produce the planned tokenization params
  (vocab=82, seq_len=3126, time_size=6, space_size=626)

HPC verification: pending. The smoke and full slurm jobs are written
but have not been submitted yet.

## Key parameters (matched to Gunaratne et al. 2023, JASSS)

| Parameter | Value | Source |
|---|---|---|
| Grid | 50 x 50 | adapted from paper's 51x51 |
| Density | 0.95 | paper |
| Agent ratio | 50/50 | paper |
| Neighborhood | Moore, 1 hop | paper |
| Simulation length | 500 ticks | paper |
| Boundary | torus | paper |
| Movement | random vacancy, move if utility > current | paper |
| Snapshot ticks | 100, 200, 300, 400, 500 | our design (skip uninformative initial state) |
| Per-agent threshold | none | our simplification |
| Random relocation (m) | 0 | our simplification |

## Tokenization

| Parameter | Value |
|---|---|
| Patch size | 2x2 |
| Patches per snapshot | 625 |
| Vocab size | 82 (3^4 + CLS) |
| Number of snapshots | 5 |
| Sequence length | 3126 (1 CLS + 5 * 625) |
| Time positions | 6 |
| Space positions | 626 |

## Tree library

| Parameter | Value |
|---|---|
| Max depth | 3 |
| Number of candidates | 10000 |
| Generation method | mixed (50% pseudo + 50% quasi) |
| Dedup probes | 3 fixed seeds, 100 steps each |
| Target after dedup | 200-500 distinct trees |

## Model and training

| Parameter | Value |
|---|---|
| Architecture | Transformer encoder, decomposed time/space embeddings |
| Model size | base (d=256, 8 heads, 4 layers, ~4M params) |
| Batch size | 32 |
| Epochs | 30 |
| LR / scheduler | 3e-4 / cosine + 3 warmup epochs |
| Label smoothing | 0.05 |
| Conformal | RAPS, alpha = [0.01, 0.05, 0.10, 0.20] |

## Primitive factors (per Gunaratne et al. 2023)

| Factor | Definition |
|---|---|
| F_Race(a, i)  | Fraction of same-type agents in Moore neighborhood of i |
| F_Age(a, i)   | Mean tenure of neighbors at i, normalized by max steps |
| F_Dist(a, i)  | Euclidean (torus) distance from agent's home to i, normalized |
| F_Isol(a, i)  | Fraction of vacant cells in Moore neighborhood of i |
| F_Move(a, i)  | Asymmetric: recent moves / 10 if i != home; 1 - that if i == home |
| F_Neigh(a, i) | Mean stored utility of neighbors at i (two-pass, no recursion) |

Operators: +, -, *, / (protected division returns 0 when denominator < 1e-10).

## Compute setup

- **Local (Windows)**: Miniconda at `C:\ProgramData\miniconda3`,
  env at `C:\Users\pog66\.conda\envs\genmod` (Python 3.11, PyTorch
  2.11.0+cu126). Local machine is dev only -- no heavy compute here.
- **Pasteur HPC**: Miniforge at `/opt/miniforge3`, env at
  `/data/apps/casl/arachchige/genmod-env`. PyTorch CUDA 12.4 for H200.
  All experiments run here.

## Key references

### Inverse generative social science
- Epstein (2023). Inverse generative social science: Backward to the
  future. *JASSS* 26(2).
- Gunaratne et al. (2023). Generating mixed patterns of residential
  segregation: an evolutionary approach. *JASSS* 26(2).
- Gunaratne and Garibay (2020). Evolutionary model discovery of causal
  factors. *PLOS ONE*.

### Conformal prediction
- Angelopoulos et al. (2021). Uncertainty sets for image classifiers
  using conformal prediction. *ICLR*.
- Vovk, Gammerman, Shafer (2005). *Algorithmic Learning in a Random
  World*. Springer.

### Cellular automata + neural networks (related work)
- AutomataGPT (2025), LifeGPT (Berkovich and Buehler, 2025), Burtsev
  (2024) -- forward prediction only, no uncertainty quantification.

## Codebase structure (this branch)

```
genmod_d/
+-- genmod/
|   +-- data/
|   |   +-- rule_trees.py            # tree structures, factors, evaluation
|   |   +-- tree_generators.py       # pseudo/quasi/mixed + dedup pipeline
|   |   +-- schelling_ruletree.py    # extended simulator (two-pass, torus)
|   |   +-- schelling.py             # base Schelling primitives (kept as ref)
|   |   +-- schelling_dataset.py     # tokenization (reused for rule trees)
|   |   +-- ruletree_dataset.py      # loader for ruletree_NNNN.json files
|   |   +-- ruletree_metadata.py     # tree grouping by dominant factor
|   |   +-- splits.py                # train/val/test splitting
|   +-- models/
|   |   +-- transformer.py           # Transformer encoder (system-agnostic)
|   |   +-- factory.py               # build_model + tokenization dispatch
|   +-- evaluation/
|   |   +-- conformal.py             # RAPS conformal prediction
|   |   +-- calibration.py           # ECE, temperature scaling
|   |   +-- metrics.py               # top-k, NLL, per-class accuracy
|   |   +-- embeddings.py            # t-SNE, linear probes
|   +-- utils/
|       +-- config.py                # YAML config dataclasses
+-- scripts/
|   +-- generate_ruletrees.py        # data generation entry point
|   +-- train.py                     # training + conformal evaluation
+-- configs/
|   +-- ruletree_base.yaml
|   +-- ruletree_smoke.yaml
+-- slurm/
|   +-- setup_env.sh
|   +-- fix_pytorch_cuda.sh
|   +-- train_ruletrees.sh
|   +-- train_ruletrees_smoke.sh
+-- HPC_GUIDE.md
+-- paper_draft.md
+-- CLAUDE.md
```

## Next steps

1. Submit smoke job on Pasteur (`sbatch slurm/train_ruletrees_smoke.sh`)
2. Submit full job once smoke passes
3. Inspect results, iterate on hyperparameters
4. Fill in numerical results in `paper_draft.md`
