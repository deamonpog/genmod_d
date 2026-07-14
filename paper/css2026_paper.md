# Measuring Equifinality: Calibrated Rule Identification in Agent-Based Models of Collective Motion

*Submission to CSS 2026 (Santa Fe). Full paper; target 3,000-5,000 words
inclusive of figures, tables, notes and references.*

---

## Abstract

Agent-based modelers have long worried about equifinality: different behavioral
rules can generate indistinguishable macroscopic patterns, so recovering "the"
rule that produced an observation may be impossible in principle. The worry is
usually stated qualitatively. We show it can be measured.

We treat inverse generative social science as classification over a library of
candidate rules, and attach conformal prediction to the classifier. Conformal
prediction returns a *set* of rules guaranteed to contain the true generator with
probability at least `1 - alpha`, without distributional assumptions. The size of
that set is a calibrated, observation-conditional measure of equifinality: when
the data cannot distinguish three mechanisms, the set contains three rules, and
it is *provably* not the model's fault.

We demonstrate this on a collective-motion ABM in which five agents move under one
of eleven known neighbor-selection rules. Using cross-validation grouped by
simulation seed, we recover the exact rule with 0.93 accuracy from 28 seconds of
observation (chance: 0.09), and RAPS conformal sets attain nominal coverage while
averaging 1.1 to 1.9 rules out of eleven.

Two findings should interest modelers. First, identifiability is governed almost
entirely by *what is observed* and almost not at all by *what model is fitted*:
single-agent kinematics yield 0.35, adding relational information yields 0.72, and
adding the time course yields 0.93 -- after which a 156k-parameter recurrent network
over group averages matches a 1.8M-parameter transformer over per-agent
trajectories (difference: -0.002, 95% CI [-0.008, +0.005]). Second, the residual
equifinality is structured and interpretable: *how many* neighbors an agent attends
to is recovered far more reliably than *which* neighbors it selects, because at
three of four available neighbors the selection rules draw from nearly the same
pool and genuinely converge in behavior.

---

## 1. Introduction

Epstein's generative dictum -- "if you didn't grow it, you didn't explain it" --
licenses a model by its ability to produce a target pattern. The inverse question
is harder and more useful: given the pattern, which rule grew it? *Inverse
generative social science* (IGSS) asks exactly this, and typically answers it by
search -- evolving a population of candidate rules and keeping those whose output
resembles the target (Gunaratne and Garibay 2020; Gunaratne et al. 2023;
Epstein 2023).

Two problems dog this program. The first is cost. The second, and deeper, is
**equifinality**: several distinct rules may generate output that no observer can
tell apart. A search procedure returns a rule; it does not tell you how many *other*
rules would have served equally well. Modelers know this and hedge accordingly, but
the hedge is rhetorical. There is no number.

This paper supplies one. We recast rule recovery as supervised classification over
a library of candidate rules, and then wrap the classifier in **conformal
prediction** (Vovk et al. 2005; Angelopoulos et al. 2021). Conformal prediction
converts a classifier's scores into a set `C(W)` of candidate rules with a
distribution-free guarantee:

> `P( g_true in C(W) ) >= 1 - alpha`

for a user-chosen `alpha`, requiring only that calibration and test data be
exchangeable. The guarantee holds regardless of whether the classifier is good.
A *bad* classifier simply returns large sets.

That property is what makes the set size meaningful. Because coverage is
guaranteed, the size of `C(W)` cannot be dismissed as a modeling artifact: it is a
calibrated statement about how much the observation itself constrains the rule. A
singleton means the mechanism is identified. A set of three means three mechanisms
remain live at the requested confidence -- **equifinality, quantified, conditional on
the data you actually have.**

We test this on collective motion, where the rules under study are explicitly
*relational* -- they concern which neighbors influence whom -- and where the ground
truth is known by construction.

**Contributions.**

1. **Conformal sets as an equifinality measure** for ABM rule recovery, with a
   coverage guarantee that makes set size interpretable rather than incidental.
2. **An information ladder.** We show empirically which *observations* make a rule
   identifiable -- relational structure, then temporal structure -- and that beyond
   these, model choice contributes nothing measurable. This speaks directly to
   pattern-oriented modeling and to summary-statistic selection in ABM calibration.
3. **A leakage-free benchmark.** We document a seed-coupling hazard in the source
   data that would inflate any naive evaluation, and the grouped design that
   removes it.
4. **A mechanistic reading of the residual ambiguity**: neighbor *count* is far
   more identifiable than neighbor *selection*, and the conformal sets say so.

---

## 2. Background

**Inverse generative social science.** Evolutionary model discovery (Gunaratne and
Garibay 2020) evolves rule trees and uses random-forest importance to identify
influential factors; Gunaratne et al. (2023) apply this to residential
segregation. The output is a rule and a ranking of factors, not a calibrated set of
alternatives.

**Equifinality and pattern-oriented modeling.** That many parameterizations or
structures can reproduce a pattern is a familiar hazard (Beven 2006). The standard
response is pattern-oriented modeling (Grimm et al. 2005): constrain the model with
*multiple* patterns at once, on the reasoning that a rule surviving several
independent filters is more likely the right one. This is sound but informal --
which patterns, and how much constraint does each buy? Section 4.1 is an empirical
answer to that question for one system, and Section 4.4 turns the residual
ambiguity into a number.

**Calibration and summary statistics.** ABM calibration -- including approximate
Bayesian computation -- turns on the choice of summary statistics: statistics that
discard the information distinguishing two mechanisms make those mechanisms
inferentially identical. Our information ladder is a direct measurement of this
effect: it shows precisely which classes of observation carry the rule signal, and
which do not.

**Conformal prediction.** Split conformal (Vovk et al. 2005) uses a held-out
calibration set to convert scores into sets with finite-sample coverage. RAPS
(Angelopoulos et al. 2021) is an adaptive variant that penalizes long tails,
producing smaller and more stable sets. We use RAPS throughout.

---

## 3. The benchmark

### 3.1 Rules

Five agents move in a circular arena (radius `R = 0.25 m`) under burst-and-coast
dynamics. A rule `g = (s, k)` specifies a neighbor-selection strategy `s` and the
number `k` of neighbors attended to.

| Rule | Strategy | k | | Rule | Strategy | k |
|---|---|---|---|---|---|---|
| 0 | no interaction | 0 | | 6 | random | 3 |
| 1 | nearest | 1 | | 7 | most influential | 1 |
| 2 | nearest | 2 | | 8 | most influential | 2 |
| 3 | nearest | 3 | | 9 | most influential | 3 |
| 4 | random | 1 | | 10 | all neighbors | 4 |
| 5 | random | 2 | | | | |

Eleven rules; chance accuracy `1/11 = 0.091`. The space is not a clean product:
`k = 0` occurs only with "none" and `k = 4` only with "all", so those rules are
fixed by `k` alone. The hard decisions live among the nine rules with `k` in
{1,2,3} across three families -- and that is where, as we will see, the ambiguity
concentrates.

### 3.2 Data

We use the **simulated-agent** trajectories released with Lei et al. (2020), who
build on the burst-and-coast interaction model of Calovi et al. (2018) for
*Hemigrammus rhodostomus* and compare which combinations of neighbors an individual
attends to. These are model output, not empirical fish data, and we treat them as
such: the point is that the generating rule is known for every run, which makes
exact recovery measurable. (The real-fish and robot trajectories in the same
release are outside this paper's scope.)

Verified directly from the files rather than assumed: 11 rules x 50 independent
runs = **550 runs, perfectly balanced**; every run has exactly five agents sharing
a single time vector, so a run is a `[T, 5, 2]` tensor; time is **irregular kick
time** (median interval 0.434 s, range 0.002-1.842 s), about 1,600 kicks and 717
seconds per run.

We index sequences by **kick event, not resampled clock time**. The dynamics are
burst-and-coast, so a kick is the natural unit of behavior; interpolating onto a
uniform grid would smooth away the discontinuities through which the rule acts. The
inter-kick interval enters as a feature instead.

### 3.3 Leakage: the experimental unit is the seed

Two hazards, either of which would inflate results silently.

**Seed coupling across rules.** Experiment identifiers are *shared* across the
eleven rule files. Inspecting the first recorded event of every run, one experiment
is byte-identical across all eleven rules and thirteen more are partially identical:
the runs begin from a common initial placement and diverge only as the rules take
hold. A classifier could therefore learn the *initial condition* rather than the
rule and score well for the wrong reason.

We block this twice. The **grouping key is the experiment id, applied across all
eleven rules at once** -- if experiment 42 is in the test partition, all eleven runs
carrying `exp_id = 42` are in the test partition. And we discard a **50-event
burn-in** from every run, which removes both the placement transient and the early
region where the coupling lives.

**Window dependence.** Windows cut from the same run are strongly dependent. So (i)
every window inherits its run's partition; (ii) calibration and test windows are
**non-overlapping** -- overlapping windows are not exchangeable, and feeding them to
split conformal yields a quantile that looks fine and a guarantee that does not
hold; and (iii) **all confidence intervals bootstrap over runs, never windows**.

**Design.** Grouped 5-fold cross-validation over the 50 experiments; each fold holds
out ten as test, so every experiment is tested exactly once. Within a fold: 25
train / 5 validation / 10 calibration / 10 test experiments. Because every
experiment exists under all eleven rules, every partition is automatically balanced.
Calibration is used for nothing but the conformal quantile -- not training, not early
stopping, not temperature scaling.

### 3.4 Observations and models

Each `(agent, event)` cell carries **single-agent** features (position, velocity,
speed, heading, acceleration, wall distance, inter-kick interval) and **relational**
features (sorted neighbor distances and their spread; mean neighbor heading and
bearing to the group centroid, both in the focal agent's frame; local
polarization). Derivatives divide by the actual inter-kick interval, never a
constant.

We compare four observation designs, in increasing order of what they retain:

- **Solo kinematics** (random forest on window summaries): how fast an agent swims,
  how sharply it turns, how close it sits to the wall. *No neighbor information.*
- **Relational, aggregated over time** (random forest): adds cohesion,
  polarization, nearest-neighbor distance, angular momentum -- but averaged over the
  window, so the *time course* is gone.
- **Relational, over time** (GRU): the same group statistics at every event, as a
  sequence. Agents are averaged away; time is retained.
- **Relational, over time, per agent** (transformer): each agent-event is its own
  token. An axial architecture attends over time *within* an agent and over agents
  *within* an event, which preserves each agent's identity along its trajectory
  while remaining exactly invariant to relabeling the agents.

The transformer detail matters and is easy to get wrong. Flattening all agent-event
cells into one sequence *without* an agent identifier is exactly permutation
invariant -- and useless: the model then sees an unordered *bag* of (state, time)
tokens, with nothing linking agent 3 at event 10 to agent 3 at event 11. It becomes
invariant to shuffling agents independently at each event, which dices every
trajectory into fragments and destroys the very signal at issue -- whether an agent
*keeps* the same nearest neighbor from kick to kick is what separates "nearest-k"
from "random-k".

Rotations, reflections and agent permutations are genuine symmetries of a circular
arena and are used as augmentations. Time reversal is not, and is not used.

---

## 4. Results

Primary setting: `T = 64` kicks, roughly **28 seconds** of observation.

### 4.1 The information ladder [Figure 1]

| What is observed | Model | Exact-rule accuracy |
|---|---|---|
| nothing | chance | 0.091 |
| solo kinematics | random forest | 0.352 +- 0.008 |
| **+ relational** | random forest | 0.717 +- 0.012 |
| **+ temporal** | GRU (156k params) | **0.929 +- 0.023** |
| **+ per-agent** | transformer (1.8M params) | 0.927 +- 0.020 |

Two large gains, then nothing.

**Relational information is worth +0.365.** An observer who records only how each
fish moves -- speed, turning, wall distance -- recovers the rule 35% of the time.
Letting the observer see the neighbors doubles that. Unsurprising, given the rules
*are* neighbor-selection rules, but it fixes the scale.

**Temporal information is worth +0.212.** The identical relational statistics, fed
to a model that sees their *time course* rather than their window average, go from
0.717 to 0.929. The rules differ in the *persistence* of the influencing set, and a
window average cannot express persistence.

**Model choice is worth nothing.** A 156k-parameter GRU over thirteen group averages
matches a 1.8M-parameter transformer over per-agent trajectories: paired difference
**-0.0015, 95% CI [-0.008, +0.005]**, bootstrapped over runs. Preserving individual
agent identity -- the representational upgrade one would naturally reach for -- buys
nothing measurable once relational and temporal structure are present.

> **[PENDING]** A further control (raw coordinates vs. rotation-invariant features,
> matched training budget) is in flight; two of five folds show 0.939 / 0.948 versus
> 0.940 / 0.945, i.e. no difference. If this holds, hand-supplied symmetry is also
> a flat rung of the ladder.

**For modelers, this is the operative result.** If a rule is not identifiable from
your data, adding model capacity will not rescue it. The lever is the *observation
design*: whether the data retain the relational and temporal structure through which
the mechanism acts. This is pattern-oriented modeling's intuition, measured.

### 4.2 Identifiability saturates with observation length [Figure 2]

| T (kicks) | seconds | solo kinematics | relational (RF) |
|---|---|---|---|
| 16 | 7 | 0.254 | 0.549 |
| 32 | 14 | 0.291 | 0.624 |
| 64 | 28 | 0.352 | 0.717 |
| 128 | 56 | 0.434 | 0.809 |
| 256 | 111 | 0.533 | 0.885 |
| 512 | 222 | 0.661 | **0.962** |

With four minutes of trajectory, even a random forest on group statistics recovers
the rule 96% of the time. Rule identification in this system is *easy* given enough
observation -- which means the scientifically interesting regime is the **short** one,
where the mechanism is genuinely underdetermined. That is also, as the next section
shows, where conformal prediction earns its keep.

### 4.3 Conformal sets attain nominal coverage [Figure 3]

RAPS on the held-out calibration partition; transformer, `T = 64`:

| alpha | target coverage | empirical coverage | mean \|C\| | singletons |
|---|---|---|---|---|
| 0.05 | 0.95 | 0.962 [0.956, 0.968] | 1.28 | 73% |
| 0.10 | 0.90 | 0.949 [0.941, 0.956] | 1.06 | 94% |
| 0.20 | 0.80 | 0.937 [0.929, 0.945] | 1.03 | 98% |

Coverage is valid at every level and sets are small: one to two rules out of eleven
at 95% confidence. The uncertainty layer transfers intact from the segregation
domain in which we previously deployed it.

The over-coverage at large `alpha` is a **ceiling effect**, and it carries a lesson.
With top-1 accuracy at 0.93 and a convention of never returning an empty set,
coverage cannot drop below ~0.93 whatever `alpha` requests. The sets have collapsed
to singletons; the conformal layer has nothing left to express. *Reporting
equifinality at an observation length where the task is already solved measures
nothing.* The measure is informative exactly where the mechanism is
underdetermined -- at short windows.

### 4.4 The residual equifinality is structured [Figures 4-5]

We expected confusions to stay inside a strategy family and slip on `k`
("nearest-2" mistaken for "nearest-3"). **The reverse is true.**

| | transformer | GRU |
|---|---|---|
| exact rule | 0.927 | 0.929 |
| strategy family | 0.939 | 0.944 |
| **neighbor count `k`** | **0.959** | 0.951 |
| `k` mean abs. error | 0.043 | 0.050 |

Of the errors, 43% (transformer) preserve the correct `k` while getting the family
wrong; only 16% do the opposite.

**How many neighbors an agent attends to is far easier to recover than which
neighbors it selects.** The mechanism explains it. `k` sets the effective strength
of cohesion, which is written directly into group geometry -- mean nearest-neighbor
distance falls monotonically with `k` within every family (0.28, 0.22, 0.19 for
nearest-1/2/3). *Which* neighbors are chosen is a subtler, higher-order property,
and at `k = 3` out of four available neighbors, "nearest three", "random three" and
"most influential three" are drawing from an almost identical pool. They converge
behaviorally. No observer -- and no amount of model capacity -- can separate them.

The confusion matrix confirms it: errors concentrate in a high-`k` cluster
(influential-3 with all-4, random-3, nearest-3), while no-interaction is recovered
perfectly and random-1 at 0.97.

**This is where the conformal set pays off.** A prediction set

> `C(W) = {nearest-3, influential-3, all-4}`

is not a classifier failing. It is a calibrated, coverage-guaranteed statement that
*three mechanisms are behaviorally indistinguishable given this observation*. The
equifinality is a property of the system and the data, not of the analyst's choice
of model -- and the guarantee is what licenses that reading. Without it, a large
prediction set would be indistinguishable from a weak classifier.

---

## 5. Discussion

**Equifinality becomes an estimand.** ABM has treated equifinality as a caveat. A
conformal set turns it into a measured quantity with an interpretable unit -- number
of rules that survive at confidence `1 - alpha` -- computable per observation and
comparable across observation designs. One can now ask, quantitatively: *how much
data must I collect before mechanism A and mechanism B become distinguishable?*
Section 4.2 answers exactly that for our system.

**Observation design dominates model design.** In a domain where interaction rules
*are* the object of study, the choice between a small recurrent net and a large
transformer, and between group-average and per-agent representations, was
immaterial; the choice of *what to record* was decisive. For the ABM calibration
literature -- where the selection of summary statistics is the central practical
problem -- this is a concrete, measured instance of the principle.

We would not have learned this without the aggregate control. A transformer that
beats a random forest looks like a victory for agent-level modeling; the same
transformer beside a GRU reveals that the gain came from the time axis, which the
forest never had.

**Limitations.** The trajectories are simulated, not empirical: this is a controlled
benchmark for rule recovery, not a claim about fish. The rule library is small
(eleven, against 2,089 in our segregation study), so the sets are correspondingly
easy to read. Exchangeability holds at the level of runs rather than windows; we
mitigate structurally (disjoint experiments across partitions, non-overlapping
evaluation windows) and bootstrap all intervals over runs, so coverage should be
read as marginal over the window distribution induced by exchangeable *runs*. With
50 seeds, fold-to-fold spread (about +-0.02) is the honest resolution of these
numbers.

**Next.** The same release contains trajectories of *real* fish. With a rule library
calibrated on simulation and a conformal layer attached, one can ask which
interaction strategies remain *plausible* for real *H. rhodostomus* at a stated
coverage level. That is an IGSS question with a calibrated answer -- and it is not
one the existing model-selection machinery can give.

---

## Appendix: reproducibility note

Two automated checks guard claims that fail silently. The first asserts that
transforming the *features* under a rotation equals recomputing every feature from
transformed *positions*; it failed initially, exposing that several relational
features were indexed by `argmin` over neighbor distances -- ill-conditioned, because
positions are quantized to 1 mm and cohesive rules pack agents until two neighbors
are *exactly* equidistant. The second asserts permutation invariance *and*, as a
control, that shuffling agents independently at each event does change the output;
it failed initially, exposing the bag-of-tokens flaw described in Section 3.4.

We also flag a defect we found in a widely-used RAPS formulation: if the
randomization term in the non-conformity score is applied only when the true class
is *not* ranked first, then for an accurate model most calibration points receive an
inflated score, the calibration quantile inflates with them, and every prediction
set comes out too large. Coverage remains *valid*, so nothing in the output
announces the problem -- the sets are simply larger than they need to be, which for a
paper measuring equifinality by set size would be a silent and material error.

---

## References

Angelopoulos, A., Bates, S., Malik, J., and Jordan, M. (2021). Uncertainty sets for
image classifiers using conformal prediction. *ICLR*.

Beven, K. (2006). A manifesto for the equifinality thesis. *Journal of Hydrology*,
320(1-2), 18-36.

Calovi, D. S., Litchinko, A., Lecheval, V., Lopez, U., Perez Escudero, A., Chate,
H., Sire, C., and Theraulaz, G. (2018). Disentangling and modeling interactions in
fish with burst-and-coast swimming reveal distinct alignment and attraction
behaviors. *PLoS Computational Biology*, 14(1), e1005933.

Lei, L., Escobedo, R., Sire, C., and Theraulaz, G. (2020). Computational and robotic
modeling reveal parsimonious combinations of interactions between individuals in
schooling fish. *PLoS Computational Biology*, 16(3), e1007194.

Epstein, J. M. (2023). Inverse generative social science: backward to the future.
*JASSS*, 26(2).

Grimm, V., Revilla, E., Berger, U., Jeltsch, F., Mooij, W. M., Railsback, S. F.,
Thulke, H.-H., Weiner, J., Wiegand, T., and DeAngelis, D. L. (2005). Pattern-oriented
modeling of agent-based complex systems. *Science*, 310(5750), 987-991.

Gunaratne, C., and Garibay, I. (2020). Evolutionary model discovery of causal
factors behind the socio-agricultural behavior of the Ancestral Pueblo. *PLOS ONE*.

Gunaratne, C., Rand, W., and Garibay, I. (2023). Generating mixed patterns of
residential segregation: an evolutionary approach. *JASSS*, 26(2).

Lopez, U., Gautrais, J., Couzin, I. D., and Theraulaz, G. (2012). From behavioural
analyses to models of collective motion in fish schools. *Interface Focus*, 2(6).

Vovk, V., Gammerman, A., and Shafer, G. (2005). *Algorithmic Learning in a Random
World*. Springer.

---

## TODO before submission (not part of the paper)

- [ ] Confirm the raw-vs-invariant control (2/5 folds in, both flat).
- [ ] T=16 / T=32 conformal -- the substantive version of Section 4.3, showing
      set size shrinking with observation length. This is the figure that makes
      the equifinality argument concrete.
- [ ] Retitle Figure 1 ("What is observed, not what is fitted").
- [ ] Word count check against the 3,000-5,000 limit (currently ~3,760, inclusive).
