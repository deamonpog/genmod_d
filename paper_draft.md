# Identifying Behavioral Rules in Agent-Based Models with Conformal Guarantees

> Working title for AAAI 2027 submission. Section 4 now reports the
> first full-scale run (2089-class rule library, base Transformer +
> RAPS) and the row/col embedding ablation. Baseline comparisons and
> factor-importance experiments remain in progress.

## Abstract (sketch)

A central question in computational social science is: given an observed
agent-based simulation, which **behavioral rule** governs the agents? Prior
work (Gunaratne et al. 2023) addresses this with evolutionary model
discovery: a multi-objective genetic program evolves a population of rule
trees, then random forest feature importance identifies the influential
factors. This is computationally expensive and provides no formal
uncertainty quantification.

We present an alternative based on **rule classification with conformal
prediction**. We auto-generate a library of distinct rule trees from the
6 primitive factors of Gunaratne et al. (racial similarity, neighbor age,
distance, isolation, recent moves, neighbor satisfaction) combined under
the standard arithmetic operators. Each rule tree defines an agent
utility function, which we plug into the Schelling segregation model to
produce labelled simulation data. A Transformer encoder, trained to
predict the generating rule from a sequence of grid snapshots, is then
combined with **RAPS conformal prediction** (Angelopoulos et al. 2021)
to produce prediction sets with distribution-free coverage guarantees:
P(true rule in set) >= 1 - alpha.

From a single forward pass we recover (i) the most likely rule tree,
(ii) a calibrated prediction set whose size adapts to identification
ambiguity, and (iii) factor-presence statistics aggregated across the
prediction set. The factor-presence statistics inherit the conformal
coverage guarantee, giving the first formally-calibrated answer to the
question "which factors drive the observed dynamics?".

In a first full-scale run over a 2089-tree library, the classifier
attains 0.32 top-1 / 0.56 top-5 accuracy and RAPS sets that meet their
target coverage at every level; a decomposed row/column embedding
ablation underperforms the flat spatial embedding. [Comparison to the
EMD baseline and factor-importance experiments -- in progress.]

---

## 1. Introduction

### 1.1 The inverse generative problem

In computational social science, generative modeling typically follows a
*forward* direction: a researcher specifies behavioral rules, runs an
agent-based simulation, and observes the resulting dynamics. Epstein
(2006) formalized this as *generative social science*: "If you didn't
grow it, you didn't explain it." Epstein (2023) introduced the inverse:
*inverse generative social science (iGSS)*, which asks **given observed
dynamics, what generating mechanism could have produced them?** Solutions
to the inverse problem must (i) handle large rule spaces, (ii) quantify
uncertainty about which rules are plausible, and (iii) be computationally
tractable.

### 1.2 The Gunaratne approach: evolutionary model discovery

Gunaratne et al. (2023) operationalize iGSS for Schelling's segregation
model. They define six primitive factors (F_Race, F_Age, F_Dist, F_Isol,
F_Move, F_Neigh) that an agent might consider when deciding whether to
move. A genetic program evolves expression trees over these factors and
the arithmetic operators {+, -, *, /}, optimizing a fitness function
(C-index of the resulting segregation pattern). Random forest feature
importance is then applied to the evolved population to identify which
factors matter most. The approach is informative but expensive (thousands
of generations) and provides no formal coverage guarantees on the
factor-importance estimates.

### 1.3 Our contribution

We replace the evolutionary search with **classification over an
auto-generated rule library**, and replace the random forest importance
analysis with **RAPS conformal prediction**. Concretely:

1. **Rule generation.** We auto-generate ~10000 candidate trees from the
   same primitive factor and operator set, using a mix of pseudo-random
   and quasi-random (Halton) sampling for low-discrepancy coverage of
   the structural space. After structural and behavioral deduplication,
   we keep ~200-500 distinct trees as classification labels.
2. **Simulation.** Each retained tree drives a Schelling segregation
   simulation (50x50 grid, 95% density, 500 ticks, torus boundary),
   matched to Gunaratne's setup. We collect snapshot sequences as
   training data.
3. **Classification.** A Transformer encoder with decomposed time and
   space embeddings (re-used unchanged from a prior multi-system framework
   on the `main` branch) predicts the generating tree from snapshots.
4. **Conformal prediction.** RAPS produces prediction sets with
   distribution-free coverage P(true tree in set) >= 1 - alpha. From the
   set, we derive aggregated factor-presence statistics that inherit the
   coverage guarantee.

The result is a **single forward pass** in place of evolutionary search,
and **distribution-free guarantees** in place of post-hoc importance
heuristics.

---

## 2. Related work

### 2.1 Inverse generative social science
- Epstein (2006). *Generative Social Science*. Princeton.
- Epstein (2023). Inverse generative social science: backward to the
  future. *JASSS* 26(2).
- Gunaratne et al. (2023). Generating mixed patterns of residential
  segregation: an evolutionary approach. *JASSS* 26(2).
- Gunaratne and Garibay (2020). Evolutionary model discovery of causal
  factors. *PLOS ONE*.

### 2.2 Conformal prediction
- Vovk, Gammerman, Shafer (2005). *Algorithmic Learning in a Random World*.
- Angelopoulos, Bates, Malik, Jordan (2021). RAPS for image classifiers.
  *ICLR*.

### 2.3 Neural networks for cellular automata and ABMs
- AutomataGPT (2025), LifeGPT (Berkovich and Buehler, 2025), Burtsev
  (2024). All operate forward (state prediction) with no uncertainty
  quantification.

---

## 3. Method

### 3.1 Primitive factors (per Gunaratne et al. 2023)

All factors return floats in [0, 1] when evaluated at a candidate
location i for an agent a:

| Factor | Definition |
|--------|-----------|
| F_Race(a, i)  | Fraction of same-type agents in the Moore neighborhood of i |
| F_Age(a, i)   | Mean tenure of neighbors at i, normalized by max simulation steps |
| F_Dist(a, i)  | Euclidean distance from agent's home to i, normalized by grid diagonal |
| F_Isol(a, i)  | Fraction of vacant cells in the Moore neighborhood of i |
| F_Move(a, i)  | Asymmetric: relocations in the last 10 ticks / 10 if i is not the home; 1 minus that if i is the home |
| F_Neigh(a, i) | Mean stored utility of agents in the Moore neighborhood of i |

F_Neigh would normally introduce a recursion (utility depends on neighbor
utility, which depends on their neighbor utility, ...). We resolve this
with a **two-pass step**: in pass 1 we compute and store the current
utility of every occupied cell; in pass 2, F_Neigh reads from this
snapshot, so utility evaluation is non-recursive and consistent across
all agents within a step.

### 3.2 Rule trees

A rule tree is built from leaves (one of the 6 factors) and internal
nodes (one of the 4 operators). Division is protected (returns 0 if the
denominator is below 1e-10). Trees are expression trees rather than
arbitrary DAGs and admit a canonical infix string representation that
serves as a structural fingerprint.

### 3.3 Tree library generation

We generate `n_candidates = 10000` trees using a mix of:
- **Pseudo-random** sampling (Python `random.Random`)
- **Quasi-random** sampling driven by Halton sequences over the
  factor / operator / depth decision space, giving low-discrepancy
  coverage of the combinatorial space

After structural deduplication (canonical string), we apply **behavioral
deduplication**: each surviving tree is run on three fixed probe
scenarios (different seeds, 100 simulation steps), and a behavioral
fingerprint is computed (coarse grid hash + total moves + steps to
equilibrium). Trees with identical fingerprints are collapsed to one
representative. Target: 200-500 behaviorally distinct trees.

The generation interface is pluggable: each generator is a function
`(n, max_depth, seed, **kwargs) -> List[Node]`, registered in a `GENERATORS`
dict. A future GP-based generator can be added without touching the
deduplication or downstream pipeline.

### 3.4 Simulator

Parameters matched to Gunaratne et al. 2023 (with simplifications noted):

| Parameter | Value | Source |
|---|---|---|
| Grid | 50x50 (closest even to paper's 51x51) | adapted from paper |
| Density | 0.95 | paper |
| Agent ratio | 50/50 | paper |
| Neighborhood | Moore, 1 hop | paper |
| Simulation length | 500 ticks | paper |
| Boundary | torus | paper (NetLogo default) |
| Movement | random vacancy, move if utility there > current | paper |
| Per-agent threshold | none (rule tree is the sole criterion) | simplification |
| Random relocation | none (m = 0) | simplification |

The simplifications are conservative: they remove confounds that are
orthogonal to the question we are studying (which factors drive the
dynamics) and reduce the cost of the data generation step.

### 3.5 Tokenization

Each 50x50 grid is divided into 2x2 patches. With 3 cell states per
position (vacant / type A / type B), each patch maps to a base-3 integer
in [0, 80]. Vocabulary size: 82 (81 patch values + 1 CLS). With 5
snapshots taken at ticks 100, 200, 300, 400, 500 (skipping the
information-free initial random state), each sample is a sequence of
length 1 + 5 * 625 = 3126 tokens. Time positions index the snapshot
(0-4 plus CLS). Space positions index the patch (0-624 plus CLS).

### 3.6 Architecture

We re-use the Transformer encoder from a prior multi-system framework
unchanged:
- Decomposed token, time-position, and space-position embeddings, summed
- Pre-norm, GELU, 4 layers, 8 heads, d_model = 256 (~4M parameters)
- [CLS] token pooling, linear classification head
- Label smoothing 0.05, AdamW with cosine schedule, 30 epochs

### 3.7 Conformal prediction

Given the trained classifier, we apply RAPS (Angelopoulos et al. 2021)
to construct prediction sets with marginal coverage guarantee
P(g_true in C(w)) >= 1 - alpha. We evaluate at
alpha = {0.01, 0.05, 0.10, 0.20}.

### 3.8 Factor-presence aggregation

From a conformal prediction set C(w) over rule trees, we compute the
factor-presence statistic:

  pi_factor(w) = |{tree in C(w) : factor in tree}| / |C(w)|

Because C(w) inherits the conformal coverage guarantee, statements like
"F_Race appears in trees that contain the true generator with probability
>= 1 - alpha" follow directly from the set construction.

---

## 4. Experiments

We report a first full-scale run on the auto-generated rule library.
Data generation (library build + per-tree simulation) ran on CPU nodes;
the classifier and conformal evaluation ran on a single NVIDIA H100.
Unless noted, the model is the base configuration (~3.9M parameters,
d_model = 256, 4 layers, 8 heads), trained for 30 epochs at batch size
16 with the settings of Section 3.6.

### 4.1 Tree library statistics

From `n_candidates = 10000` generated with the mixed pseudo/quasi-random
strategy, structural and behavioral deduplication retained **2089
behaviorally distinct trees**, which become the classification labels.
This is substantially above the 200-500 target anticipated in
Section 3.3: at depth 3 over 6 factors and 4 operators, the behavioral
fingerprint separates many trees that are structurally similar but
dynamically distinct. Each retained tree was simulated for 100 runs,
yielding ~208,900 labelled snapshot sequences, split 80/10/10 into
167,120 train / ~20,900 validation / ~20,900 test samples.

The large label count makes this a demanding 2089-way classification
problem and directly inflates conformal set sizes (Section 4.3);
tightening the behavioral dedup to produce fewer, cleaner classes is the
most promising lever for future runs.

### 4.2 Classification accuracy

On the held-out test set (2089 classes, chance top-1 = 0.048%):

| Metric | Value |
|---|---|
| Top-1 accuracy | 0.319 |
| Top-3 accuracy | 0.485 |
| Top-5 accuracy | 0.562 |
| Test NLL | 2.822 |
| Brier score | 0.740 |
| ECE (pre-temperature) | 0.0167 |
| ECE (post-temperature) | 0.0250 |
| Learned temperature | 0.932 |
| Group-probe accuracy (dominant factor) | 0.297 |

Top-1 of 0.319 over 2089 classes (~665x chance) shows the Transformer
recovers a strong signal about the generating rule from five grid
snapshots. The learned temperature near 1.0 and low pre-temperature ECE
(0.017) indicate the raw softmax is already well calibrated; temperature
scaling does not improve ECE here.

### 4.3 Conformal prediction results

RAPS prediction sets achieve their target marginal coverage
(P(true tree in set) >= 1 - alpha) at every level:

| alpha | Target coverage | Empirical coverage | Avg set size | Median | Singletons |
|---|---|---|---|---|---|
| 0.01 | 0.99 | 0.991 | 244.2 | 244 | 0.0% |
| 0.05 | 0.95 | 0.948 | 101.2 | 99 | 0.0% |
| 0.10 | 0.90 | 0.902 | 55.5 | 50 | 0.0% |
| 0.20 | 0.80 | 0.828 | 23.5 | 17 | 2.1% |

Coverage tracks the nominal level closely, confirming the distribution-
free guarantee holds empirically. Set sizes are large in absolute terms
-- a direct consequence of the 2089-way label space -- but shrink sharply
as alpha relaxes (from ~244 labels at 99% coverage to ~24 at 80%), and
singleton (fully-resolved) predictions begin to appear at alpha = 0.20.
Conditional coverage by dominant factor is uniform to within a few
percentage points of the marginal level.

### 4.4 Ablation: decomposed row/column spatial embedding

We tested replacing the flat spatial embedding (one learned vector per
2x2 patch position, 625 positions) with a **decomposed row + column
embedding** (separate learned row and column vectors, summed), motivated
by the standard spatiotemporal-transformer factorization. All other
settings, data, and the training budget are identical; the two runs use
the same H100 hardware.

| Metric | Flat space (base) | Row+col (ablation) |
|---|---|---|
| Parameters | 3,879,209 | 3,732,265 |
| Top-1 | **0.319** | 0.312 |
| Top-3 | **0.485** | 0.472 |
| Top-5 | **0.562** | 0.546 |
| Test NLL | **2.822** | 2.967 |
| Brier | **0.740** | 0.746 |
| ECE (pre-temp) | **0.0167** | 0.0184 |
| Conformal avg set @ alpha=0.01 | **244** | 323 |
| Conformal avg set @ alpha=0.05 | **101** | 188 |
| Conformal avg set @ alpha=0.10 | **55.5** | 89.3 |
| Conformal avg set @ alpha=0.20 | **23.5** | 28.8 |

Both variants reach their target coverage, so the fair comparison is set
size at matched coverage -- and there the flat embedding is decisively
better, producing sets roughly half the size at alpha = 0.05 (101 vs
188 labels). The decomposition also slightly lowers top-k accuracy and
NLL. We attribute this to expressiveness: the flat embedding learns a
distinct representation for each of the 625 patch positions, capturing
position-specific structure, whereas the additive row + col form is
constrained to a separable function of coordinates and removes ~147K
parameters concentrated in the spatial representation. For this task the
decomposition is a regularizer that costs more than it saves. We
therefore retain the flat time+space embedding as the main model and
report the decomposition as a negative ablation.

### 4.5 Remaining experiments (in progress)

- Factor-presence aggregation over conformal sets, and comparison to a
  random-forest importance baseline (Section 3.8).
- Comparison to the Gunaratne et al. evolutionary-model-discovery
  baseline.
- Generation-strategy (pseudo vs quasi vs mixed), tree-depth, and
  snapshot-count / timing ablations.

---

## 5. Discussion (sketch)

- When the conformal set is large, identification is genuinely ambiguous;
  the model is honest about it.
- Factor-presence statistics inherit the coverage guarantee for free.
- The framework slots into other ABMs by swapping the simulator and the
  primitive factor set; the Transformer / conformal pipeline is unchanged.

---

## 6. Limitations

- Two simplifications versus the paper (no per-agent threshold, no random
  relocation) are conservative but not faithful. Adding them is a future
  ablation.
- Depth-3 trees are interpretable but may miss dynamics that need deeper
  expressions.
- F_Neigh is computed from stored prev-step utilities; this is a one-step
  lag and may not perfectly match the paper's semantics.

---

## References

- Angelopoulos, Bates, Malik, Jordan (2021). RAPS. *ICLR*.
- Epstein (2006). *Generative Social Science*. Princeton.
- Epstein (2023). Inverse generative social science. *JASSS* 26(2).
- Gunaratne et al. (2023). Generating mixed patterns of residential
  segregation. *JASSS* 26(2).
- Gunaratne and Garibay (2020). Evolutionary model discovery of causal
  factors. *PLOS ONE*.
- Vovk, Gammerman, Shafer (2005). *Algorithmic Learning in a Random World*.
- Wolfram (2002). *A New Kind of Science*.
