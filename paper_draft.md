# Identifying Behavioral Rules in Agent-Based Models with Conformal Guarantees

> Working title for NeurIPS 2026 submission. Numerical results sections
> are TBD pending HPC experiments. This stub captures the framing,
> contributions, and structure for Option D.

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

[Numerical results, comparison to EMD baseline, ablations -- TBD.]

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

[TBD pending HPC runs.]

### 4.1 Tree library statistics
- Number of structural duplicates removed
- Number of behavioral duplicates removed
- Distribution of tree depths and dominant factors

### 4.2 Classification accuracy
- Top-1 / top-3 / top-5 accuracy on the test split
- Per-group (dominant-factor) accuracy
- Calibration (ECE before / after temperature scaling)

### 4.3 Conformal prediction results
- Empirical coverage at each alpha
- Average / median / max set sizes
- Conditional coverage by dominant factor

### 4.4 Factor importance
- Aggregated factor presence across the conformal prediction sets
- Comparison to a random forest importance baseline
- Conditional importance by simulation regime

### 4.5 Ablations
- Random vs quasi-random vs mixed tree generation
- Tree depth (2 vs 3 vs 4)
- Number of snapshots and observation timing

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
