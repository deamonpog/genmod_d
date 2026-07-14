# Measuring Equifinality: Conformal Rule Identification in Agent-Based Models

*Submission to CSS 2026 (Santa Fe). Full paper.*

---

## Abstract

Agent-based modelers have worried for decades that different behavioral rules can
generate indistinguishable macroscopic patterns. Equifinality is usually stated as
a caveat. We show it can be **measured**.

We recast inverse generative social science as classification over a library of
candidate rules, and attach conformal prediction to the classifier. Conformal
prediction returns a *set* of rules guaranteed to contain the true generator with
probability at least `1 - alpha`, without distributional assumptions and regardless
of how good the classifier is. The size of that set is therefore a calibrated,
observation-conditional measure of equifinality: when the data cannot separate
three mechanisms, the set contains three rules, and the guarantee certifies that
this is a fact about the data rather than a failure of the analyst.

We demonstrate this in two domains that share nothing but the framework.

In **Schelling segregation**, we auto-generate a library of 2,089 behaviorally
distinct utility rules (arithmetic trees over six primitive factors), simulate
each, and classify from five grid snapshots. Top-1 accuracy is 0.319 -- 665 times
chance -- and conformal sets attain nominal coverage while narrowing the plausible
rule space from 2,089 candidates to 53 at 90% confidence, a 40-fold reduction.

In **collective motion**, five agents move under one of eleven known
neighbor-selection rules. Here the measure shows its point: from **7 seconds** of
trajectory the 95%-confidence set contains 3.3 of the eleven rules and only 1.4% of
observations resolve the mechanism uniquely; from **28 seconds** it contains 1.2
rules and 79% resolve. Equifinality is not a fixed property of a rule library. It is
a property of the library *and the observation*, and it decays measurably as
observation accumulates.

The cross-domain comparison yields the paper's most actionable finding.
Identifiability is governed by **what a model can represent**, not by **how large it
is**: single-agent kinematics reach 0.35, adding relational information reaches
0.72, adding the time course reaches 0.93 -- after which a 156k-parameter recurrent
net over group averages matches a 1.8M-parameter transformer over per-agent
trajectories (paired difference -0.002, 95% CI [-0.008, +0.005]), an eleven-fold
capacity increase for nothing. Yet in the segregation study, *changing how spatial
position is encoded* -- at constant capacity -- inflates conformal sets by 68%.
Capacity is not the lever. Representation is.

---

## 1. Introduction

Epstein's generative dictum -- "if you didn't grow it, you didn't explain it" --
licenses a model by its ability to produce a target pattern. The inverse question
is harder and more useful: given the pattern, *which rule grew it?* Inverse
generative social science (IGSS) asks exactly this, and typically answers by search
-- evolving candidate rules and keeping those whose output resembles the target
(Gunaratne and Garibay 2020; Gunaratne et al. 2023; Epstein 2023).

Two problems dog this program. The first is cost. The second, and deeper, is
**equifinality**: several distinct rules may generate output no observer can tell
apart (Beven 2006). A search returns a rule; it does not say how many *other* rules
would have served equally well. Modelers know this and hedge, but the hedge is
rhetorical. There is no number.

This paper supplies one. We recast rule recovery as supervised classification over
a library of candidate rules, then wrap the classifier in **conformal prediction**
(Vovk et al. 2005; Angelopoulos et al. 2021), which converts scores into a set
`C(W)` of candidate rules with a distribution-free guarantee

> `P( g_true in C(W) ) >= 1 - alpha`,

requiring only that calibration and test data be exchangeable. Crucially, **the
guarantee holds whether or not the classifier is any good.** A weak classifier does
not break the coverage; it simply returns larger sets.

That is what makes set size meaningful. Because coverage is guaranteed, `|C(W)|`
cannot be waved away as a modeling artifact: it is a calibrated statement about how
much the *observation* constrains the *rule*. A singleton means the mechanism is
identified. A set of three means three mechanisms remain live at the requested
confidence. **Equifinality, quantified, conditional on the data you actually have.**

We test this in two domains chosen to share nothing except the framework: a
lattice-based segregation model with a library of 2,089 rules, and an off-lattice
collective-motion model with 11 rules. The first probes whether the approach
*scales*; the second, where ground truth is exhaustively known, probes what makes a
rule identifiable at all.

**Contributions.**

1. **Conformal sets as an equifinality measure** for ABM rule recovery, with a
   coverage guarantee that makes set size interpretable rather than incidental.
2. **Demonstration across two unrelated domains** -- 2,089 rules on a lattice, 11
   rules in continuous space -- with nominal coverage in both.
3. **An information ladder**: which *observations* make a rule identifiable
   (relational structure, then temporal structure), and the finding that beyond
   these, model choice contributes nothing measurable. This speaks directly to
   pattern-oriented modeling and to summary-statistic selection in ABM calibration.
4. **A mechanistic reading of residual ambiguity**: in collective motion, *how many*
   neighbors an agent attends to is far more identifiable than *which* neighbors it
   selects -- and the conformal sets say so.

---

## 2. Background

**IGSS.** Evolutionary model discovery (Gunaratne and Garibay 2020) evolves rule
trees and uses random-forest importance to rank influential factors; Gunaratne et
al. (2023) apply it to residential segregation. The output is a rule and a factor
ranking, not a calibrated set of alternatives.

**Equifinality and pattern-oriented modeling.** That many structures reproduce a
pattern is a familiar hazard (Beven 2006). The standard response is
pattern-oriented modeling (Grimm et al. 2005): constrain the model with *multiple*
patterns, on the reasoning that a rule surviving several filters is more likely
correct. This is sound but informal -- *which* patterns, and how much constraint does
each buy? Section 5 answers that empirically for one system; Section 6 turns the
residual ambiguity into a number.

**Calibration and summary statistics.** ABM calibration, including approximate
Bayesian computation, turns on the choice of summary statistics: statistics that
discard the information distinguishing two mechanisms render those mechanisms
inferentially identical. Our information ladder measures this directly.

**Conformal prediction.** Split conformal (Vovk et al. 2005) uses a held-out
calibration set to convert scores into sets with finite-sample coverage. RAPS
(Angelopoulos et al. 2021) is an adaptive variant penalizing long tails. We use RAPS
throughout.

---

## 3. Framework

The pipeline is domain-agnostic and has four stages.

1. **Rule library.** Enumerate or generate a set `G` of candidate behavioral rules,
   and deduplicate them *behaviorally* -- two rules that produce indistinguishable
   dynamics are one rule, and keeping both would manufacture ambiguity rather than
   measure it.
2. **Forward simulation.** Run each rule to produce labelled observations. The label
   is the generating rule; this is what makes recovery measurable.
3. **Recovery.** Train a classifier `P(g | W)` mapping an observation window `W` to
   a distribution over rules.
4. **Conformal layer.** On a held-out calibration partition never used for training
   or model selection, calibrate RAPS to produce `C(W)` with coverage `1 - alpha`.

**The experimental unit is the simulation run, never the observation window.**
Windows from the same run are dependent; treating them as independent inflates
apparent performance and silently voids the conformal guarantee, which needs
calibration and test scores to be exchangeable. In both case studies, whole runs are
assigned to partitions, evaluation windows within a run do not overlap, and all
confidence intervals bootstrap over runs.

---

## 4. Case study 1: Schelling segregation (2,089 rules)

**Rules.** Following Gunaratne et al. (2023), an agent's utility for a location is
an arithmetic expression tree over six primitive factors -- same-type fraction,
neighbor tenure, distance from home, vacancy fraction, recent moves, and neighbor
utility -- combined with `+ - * /`. From 10,000 generated candidates (depth <= 3),
behavioral deduplication against fixed probe simulations retains **2,089
behaviorally distinct trees**, which become the class labels.

**Observations.** A 50x50 torus, density 0.95, Moore neighborhood, 500 ticks, with
snapshots at ticks 100-500. Each snapshot is tokenized into 2x2 patches; a
transformer encoder (3.9M parameters) with decomposed time/space embeddings
classifies the sequence. Each tree is simulated 100 times, giving ~209k labelled
sequences split 80/10/10 by run.

**Results.** Chance top-1 is 1/2089 = 0.048%.

| Metric | Value |
|---|---|
| Top-1 | **0.319** (665x chance) |
| Top-3 | 0.485 |
| Top-5 | 0.562 |
| ECE (pre-temperature) | 0.017 |

The classifier extracts a strong signal about the generating rule from five grid
snapshots, and its raw softmax is already well calibrated (learned temperature
0.93).

**Conformal sets.**

| alpha | target | empirical coverage | mean \|C\| | as % of library |
|---|---|---|---|---|
| 0.01 | 0.99 | 0.989 | 234.3 | 11.2% |
| 0.05 | 0.95 | 0.948 | 104.2 | 5.0% |
| 0.10 | 0.90 | 0.897 | **52.9** | **2.5%** |
| 0.20 | 0.80 | 0.801 | 20.5 | 1.0% |

Coverage tracks the nominal level at every alpha. In absolute terms the sets are
large -- but the right denominator is the library. **At 90% confidence, 2,089
candidate rules are narrowed to 53: a 40-fold reduction, with a guarantee.** That is
a quantitative statement about how much a five-snapshot observation of a segregation
process constrains the underlying utility rule, and it is the kind of statement the
IGSS literature has not previously been able to make.

The remaining ambiguity is not noise but structure: the surviving rules are
behaviorally similar trees. Conditional coverage by dominant factor is uniform to
within a few points of the marginal level, so no factor family is systematically
under-covered.

### 4.1 Representation matters, at constant capacity

We ablate one thing: how spatial position is encoded. The base model learns a
separate embedding for each of the 625 patch positions; the ablation decomposes it
into a learned row vector plus a learned column vector. **This *reduces* parameters
slightly (3.73M vs 3.88M) and leaves the observation, the data, and the training
budget untouched.** Only the model's representational form changes.

| | top-1 | mean \|C\| @ alpha=0.10 | @ alpha=0.05 |
|---|---|---|---|
| flat spatial embedding | **0.319** | **52.9** | **104.2** |
| decomposed row + column | 0.312 | 88.7 | 192.4 |

Accuracy barely moves (0.319 vs 0.312), but **the conformal sets inflate by 68%**:
the row/column model needs 89 candidate rules where the flat model needs 53 to make
the same coverage guarantee. The two models are nearly equally *accurate* and very
unequally *informative*.

This matters for two reasons. First, it shows set size is a more discriminating
diagnostic than accuracy -- a difference invisible in top-1 is stark in the
equifinality measure. Second, read against Section 5.3, it locates precisely what
does and does not matter. There, an eleven-fold increase in *capacity* bought
nothing. Here, a change in *representation* at constant capacity costs 68% of the
resolving power. **What a model can represent matters; how big it is does not.**

---

## 5. Case study 2: collective motion (11 rules)

The segregation study leaves an objection: perhaps the classifier recognizes *grid
textures*, not behavioral rules. So we move to a domain with no grid, no lattice,
and a rule space small enough to be exhaustively understood.

### 5.1 The benchmark

Five agents move in a circular arena (`R = 0.25 m`) under burst-and-coast dynamics.
A rule `g = (s, k)` fixes a neighbor-selection strategy `s` and the number `k` of
neighbors attended to: no interaction (`k=0`); the `k` **nearest**, `k` **random**,
or `k` **most influential** neighbors for `k` in {1,2,3}; or **all** four. Eleven
rules; chance 0.091.

The families differ *only* in which neighbors enter the social sum -- the interaction
functions are identical. This makes the benchmark sharp, and it makes the high-`k`
rules genuinely hard: with `k = 3` of 4 available neighbors, "nearest three",
"random three" and "most influential three" draw from nearly the same pool.

We use the **simulated-agent** trajectories released with Lei et al. (2020), who
build on the burst-and-coast model of Calovi et al. (2018). These are model output,
not empirical fish data, and the point is precisely that: the generating rule is
known for every run. 11 rules x 50 runs = 550 runs, balanced. Time is irregular
**kick time** (median interval 0.434 s), so we index sequences by kick event rather
than resampled clock time -- interpolating onto a uniform grid would smooth away the
discontinuities through which the rule acts.

### 5.2 A leakage hazard worth documenting

Experiment identifiers are **shared across the eleven rule files**, and runs begin
from common initial placements: inspecting the first recorded event of every run,
one experiment is identical across all eleven rules and thirteen more are partially
identical. A classifier could learn the *initial condition* rather than the rule.

We block this twice: the grouping key is the experiment id applied across all eleven
rules at once (if experiment 42 is in test, all eleven of its runs are), and we
discard a 50-event burn-in that removes both the placement transient and the region
where the coupling lives. Grouped 5-fold cross-validation over the 50 seeds; each
experiment is tested exactly once.

We report this because it is the kind of hazard that inflates results silently, and
because it is a property of a widely-used public dataset.

### 5.3 The information ladder [Figure 1]

We compare four **observation designs**, holding the task fixed. `T = 64` kicks
(28 s):

| What is observed | Model | Exact-rule accuracy |
|---|---|---|
| nothing | chance | 0.091 |
| solo kinematics | random forest | 0.352 +- 0.008 |
| **+ relational** | random forest | 0.717 +- 0.012 |
| **+ temporal** | GRU (156k params) | **0.929 +- 0.023** |
| **+ per-agent** | transformer (1.8M params) | 0.927 +- 0.020 |

Two large gains, then nothing.

**Relational information is worth +0.365.** An observer recording only how each fish
moves -- speed, turning, wall distance -- recovers the rule 35% of the time. Letting it
see the neighbors doubles that.

**Temporal information is worth +0.212.** The *identical* relational statistics, fed
to a model that sees their time course rather than their window average, go from
0.717 to 0.929. The rules differ in the *persistence* of the influencing set, and a
window average cannot express persistence.

**Capacity is worth nothing.** A 156k-parameter GRU over thirteen group averages
matches a 1.8M-parameter transformer over per-agent trajectories: paired difference
**-0.0015, 95% CI [-0.008, +0.005]**, bootstrapped over runs. Preserving individual
agent identity -- the representational upgrade one would naturally reach for -- buys
nothing measurable once relational and temporal structure are present, despite an
eleven-fold difference in parameter count.

**For modelers, this is the operative result.** If a rule is not identifiable from
your data, adding model capacity will not rescue it. The lever is the observation
design: whether the data retain the relational and temporal structure through which
the mechanism acts. This is pattern-oriented modeling's intuition, measured.

*A methodological caution, since it nearly misled us.* Our first transformer,
trained for 40 epochs, scored 0.825 and appeared to lose to the GRU by ten points.
Its validation accuracy was still climbing when the epoch budget ran out; early
stopping never fired. At 120 epochs the same architecture reaches 0.927. **A
negative result about architecture is worthless unless every model is trained to
convergence**, and the failure mode is silent -- an undertrained model returns a
perfectly plausible number, and a comparison against it will confidently support the
wrong conclusion.

### 5.4 Identifiability grows with observation, then saturates [Figure 2]

| T (kicks) | seconds | solo kinematics | + relational (RF) | + temporal (GRU) |
|---|---|---|---|---|
| 16 | 7 | 0.254 | 0.549 | **0.771** |
| 32 | 14 | 0.291 | 0.624 | 0.869 |
| 64 | 28 | 0.352 | 0.717 | 0.929 |
| 128 | 56 | 0.434 | 0.809 | 0.967 |
| 256 | 111 | 0.533 | 0.885 | **0.984** |
| 512 | 222 | 0.661 | 0.962 | 0.970 |

Every rung rises with observation, and the *gap* between rungs persists: at every
window length, seeing the neighbors is worth more than seeing an agent alone, and
seeing the time course is worth more again. With two minutes of trajectory the
sequence model is at 0.984 and the task is effectively solved.

Rule identification here is therefore *easy given enough observation*, which means
the scientifically interesting regime is the **short** one, where the mechanism is
genuinely underdetermined. That is also where conformal prediction earns its keep.

### 5.5 Equifinality decays as observation accumulates [Figure 3]

This is the central measurement of the paper. We calibrate RAPS at every
observation length, and report the size of the resulting prediction set -- the count
of rules that survive at the stated confidence.

At 95% confidence:

| T | seconds | acc | empirical coverage (target 0.95) | mean \|C\| | singletons |
|---|---|---|---|---|---|
| **16** | 7 | 0.771 | **0.949** [0.944, 0.955] | **3.28** | 1.4% |
| 32 | 14 | 0.869 | 0.957 | 1.86 | 44% |
| 64 | 28 | 0.929 | 0.968 | 1.22 | 79% |
| 128 | 56 | 0.967 | 0.983 | 1.07 | 94% |
| 256 | 111 | 0.984 | 0.989 | 1.02 | 98% |
| 512 | 222 | 0.970 | 0.986 | 1.06 | 94% |

**At seven seconds of observation, the average 95%-confidence prediction set
contains 3.28 of the eleven rules, and only 1.4% of observations resolve the
mechanism to a single rule. By twenty-eight seconds the set holds 1.22 rules and
79% resolve; by two minutes it is a singleton almost always.**

Equifinality is therefore **not a fixed property of a rule library**. It is a
property of the library *and the observation*, and it decays measurably as
observation accumulates. That is the quantity this paper proposes to report.

The set-size distribution makes it concrete. At `alpha = 0.10`:

| | \|C\|=1 | \|C\|=2 | \|C\|=3 | \|C\|>=4 |
|---|---|---|---|---|
| T = 16 (7 s) | 33% | 40% | 25% | 2% |
| T = 32 (14 s) | 71% | 25% | 3% | 0% |
| T = 64 (28 s) | 91% | 8% | 1% | 0% |
| T = 256 (111 s) | 99% | 1% | 0% | 0% |

At seven seconds, **two-thirds of observations cannot pin the mechanism to a single
rule** -- and the conformal set states exactly how many remain live, with a coverage
guarantee behind the count.

Two further points.

**Coverage is exact where it matters.** At `T = 16`, `alpha = 0.05`, empirical
coverage is 0.949 against a target of 0.950. The guarantee is not merely satisfied
but tight, in precisely the regime where the model is uncertain and the guarantee is
actually being tested.

**The over-coverage at long windows is a ceiling effect, and it is instructive.**
With top-1 accuracy at 0.93 and a convention of never returning an empty set,
coverage cannot fall below ~0.93 whatever `alpha` requests; the sets have collapsed
to singletons and there is nothing left to express. *Measuring equifinality at an
observation length where the task is already solved measures nothing.* This is a
caution for anyone tempted to calibrate on a saturated model and conclude that a
system exhibits no equifinality: they will have measured the ceiling, not the
system.

### 5.6 The residual equifinality is structured [Figures 4-5]

We expected confusions to stay inside a strategy family and slip on `k`
("nearest-2" mistaken for "nearest-3"). **The reverse is true.**

| | transformer | GRU |
|---|---|---|
| exact rule | 0.927 | 0.929 |
| strategy family | 0.939 | 0.944 |
| **neighbor count `k`** | **0.959** | 0.951 |

Of the transformer's errors, 43% preserve the correct `k` while getting the family
wrong; only 16% do the opposite.

**How many neighbors an agent attends to is far easier to recover than which
neighbors it selects.** The mechanism explains it. `k` sets the effective strength of
cohesion, written directly into group geometry -- mean nearest-neighbor distance falls
monotonically with `k` within every family (0.28, 0.22, 0.19 for nearest-1/2/3).
*Which* neighbors are chosen is a subtler, higher-order property, and at `k = 3` of
four available neighbors the three families draw from an almost identical pool. They
converge behaviorally. No observer -- and no amount of model capacity -- can separate
them.

**This is where the conformal set pays off.** A prediction set

> `C(W) = {nearest-3, influential-3, all-4}`

is not a classifier failing. It is a calibrated, coverage-guaranteed statement that
*three mechanisms are behaviorally indistinguishable given this observation*. The
equifinality is a property of the system and the data, not of the analyst's model
choice -- and the guarantee is what licenses that reading. Without it, a large
prediction set would be indistinguishable from a weak classifier.

---

## 6. Discussion

**Equifinality becomes an estimand.** Across both domains, the conformal set
converts a qualitative caveat into a measured quantity with an interpretable unit:
the number of rules surviving at confidence `1 - alpha`, computable per observation.
The two case studies show the measure behaves sensibly at very different scales --
narrowing 2,089 segregation rules to 53 (2.5% of the library), and eleven movement
rules to one or two. One can now ask, quantitatively: *how much data must I collect
before mechanism A and mechanism B become distinguishable?* Section 5.4 answers
exactly that.

**Representation matters; capacity does not.** The two case studies isolate
different variables and, read together, they say something sharper than either does
alone.

*Collective motion varies capacity, holding representation fixed.* A GRU over group
averages and a transformer over per-agent tokens both see the same relational and
temporal structure; one has eleven times the parameters. The difference is
indistinguishable from zero.

*Segregation varies representation, holding capacity fixed.* A flat per-patch
spatial embedding and a decomposed row+column embedding have essentially the same
parameter count and see identical data. Accuracy is nearly identical -- yet the
conformal sets differ by 68%.

The lever is therefore not model size but **what the model is able to represent** --
which includes, upstream of any architectural choice, what the observation itself
retains. The information ladder is the same principle applied to the data: relational
and temporal structure are representational prerequisites, and no amount of capacity
substitutes for them. For the ABM calibration literature, where summary-statistic
selection is the central practical problem, this is a concrete measured instance.

We would not have learned this without the aggregate control. A transformer beating
a random forest looks like a victory for agent-level modeling; the same transformer
beside a GRU reveals the gain came from the time axis, which the forest never had.
Nor without training both models to convergence: our first transformer, stopped at
40 epochs while still improving, scored 0.825 and would have supported the confident
and false conclusion that agent-level representation *hurts*.

**Set size is a sharper diagnostic than accuracy.** The row/column ablation is
invisible in top-1 (0.319 vs 0.312) and unmistakable in the conformal sets (53 vs
89 rules). A model can be equally *right* and considerably less *informative*, and
only the equifinality measure detects it. This is an argument for reporting set
sizes even when a point estimate is all one needs.

**A cautionary note on conformal implementations.** We found a defect in a common
RAPS formulation: if the randomization term in the non-conformity score is applied
only when the true class is *not* ranked first, then for an accurate model most
calibration points receive an inflated score, the calibration quantile inflates with
them, and every prediction set comes out too large. Coverage remains *valid*, so
nothing in the output announces the problem. Notably, **the severity scales with
model accuracy**: our collective-motion model (93% top-1) had ~93% of calibration
points at rank 0 and its coverage sat at 0.97 regardless of `alpha`; our segregation
model (32% top-1) had only ~32%, and its numbers barely moved. The bug is most
dangerous exactly where the model is best -- which is when one is least inclined to
be suspicious. For any work measuring equifinality *by set size*, this is a silent
and material error.

**Limitations.** Both case studies use simulated data with known ground truth; this
is a controlled benchmark for rule recovery, not an empirical inference about
segregation or fish. The Schelling library, though large, is a depth-3 subset of an
infinite rule space. Exchangeability holds at the level of runs rather than windows;
we mitigate structurally and bootstrap over runs. In the collective-motion study, 50
seeds give a fold-to-fold spread of about +-0.02, which is the honest resolution of
those numbers.

**Next.** The same release contains trajectories of *real* fish. With a rule library
calibrated on simulation and a conformal layer attached, one can ask which
interaction strategies remain *plausible* for real *H. rhodostomus* at a stated
coverage level -- an IGSS question with a calibrated answer, which the existing
model-selection machinery cannot give.

---

## References

Angelopoulos, A., Bates, S., Malik, J., and Jordan, M. (2021). Uncertainty sets for
image classifiers using conformal prediction. *ICLR*.

Beven, K. (2006). A manifesto for the equifinality thesis. *Journal of Hydrology*,
320(1-2), 18-36.

Calovi, D. S., Litchinko, A., Lecheval, V., Lopez, U., Perez Escudero, A., Chate, H.,
Sire, C., and Theraulaz, G. (2018). Disentangling and modeling interactions in fish
with burst-and-coast swimming. *PLoS Computational Biology*, 14(1), e1005933.

Epstein, J. M. (2023). Inverse generative social science: backward to the future.
*JASSS*, 26(2).

Grimm, V., Revilla, E., Berger, U., Jeltsch, F., Mooij, W. M., Railsback, S. F.,
Thulke, H.-H., Weiner, J., Wiegand, T., and DeAngelis, D. L. (2005).
Pattern-oriented modeling of agent-based complex systems. *Science*, 310(5750),
987-991.

Gunaratne, C., and Garibay, I. (2020). Evolutionary model discovery of causal factors
behind the socio-agricultural behavior of the Ancestral Pueblo. *PLOS ONE*.

Gunaratne, C., Rand, W., and Garibay, I. (2023). Generating mixed patterns of
residential segregation: an evolutionary approach. *JASSS*, 26(2).

Lei, L., Escobedo, R., Sire, C., and Theraulaz, G. (2020). Computational and robotic
modeling reveal parsimonious combinations of interactions between individuals in
schooling fish. *PLoS Computational Biology*, 16(3), e1007194.

Schelling, T. C. (1971). Dynamic models of segregation. *Journal of Mathematical
Sociology*, 1(2), 143-186.

Vovk, V., Gammerman, A., and Shafer, G. (2005). *Algorithmic Learning in a Random
World*. Springer.

---

## Figures

| File | Content |
|---|---|
| `fig1_ladder` | The information ladder at T=64. Two gains, then a flat top rung. |
| `fig6_equifinality` | **The central figure.** Set size vs observation length, and the set-size distribution at 90% confidence. |
| `fig2_window` | Accuracy vs observation length, all three rungs. |
| `fig3_conformal` | Coverage and set size vs alpha, at T=16 (tight) and T=64 (ceiling). |
| `fig4_confusion` | 11x11 confusion matrix; the high-k cluster. |
| `fig5_errors` | Where the errors go: right-k vs right-family. |

## TODO (not part of the paper)

- [ ] **Hazel `fish-long` array still running.** If it lands: (a) the transformer at
      T=128/256/512 confirms the GRU substitution at more than one window length,
      and (b) the raw-vs-invariant control completes. Neither is load-bearing --
      every claim in the paper rests on a finished 5-fold result.
- [ ] Convert to the submission format (no template specified by the CFP).
- [ ] Word count: **~4,500 inclusive**; limit 5,000. The ODD appendix is excluded
      by the CFP.
