# What Must a Model Observe to Invert a Generative Rule? Conformal Behavioral-Rule Identification in Collective Motion

**Working draft.** Venue: complex-systems conference (short/full).
Companion case study to the Schelling rule-tree work.

> **STATUS FLAGS** -- read before quoting any number.
>
> - [CONFIRMED] means 5 grouped folds, run-level bootstrap CIs, in hand.
> - [PENDING] means a run is in flight; the number will change.
>
> The one claim still gated: whether the raw-coordinate Transformer matches
> the invariant one given an equal training budget. First fold says yes
> (0.9388 vs 0.9396). If that holds, Section 5.3 stands as written; if it
> does not, Section 5.3 must be rewritten and the abstract with it.

---

## Abstract (draft)

Inverse generative social science asks which behavioral rule generated an
observed pattern. Recent work casts this as classification: train a model to
recognize the generating rule, and use conformal prediction to return a
calibrated *set* of plausible rules rather than a single guess. We ask whether
that framework survives a change of domain, and what actually makes a rule
identifiable.

We apply it to collective motion. Five simulated fish move under one of eleven
known neighbor-selection strategies -- attend to the nearest, a random, or the
most influential *k* neighbors, for *k* in 1..3, plus no interaction and
all-neighbors. Because the generating strategy is known for every trajectory,
exact rule recovery is directly measurable. Using grouped cross-validation over
independent simulation seeds, we recover the exact rule with 0.93 accuracy from
28 seconds of observation, against a 0.09 chance baseline, and RAPS conformal
prediction attains its nominal coverage with prediction sets averaging 1.1-1.9
rules out of 11.

The more useful result is negative. We find that identifiability is governed
almost entirely by *what the model is allowed to observe*, and almost not at all
by how the model is built. A model given only single-agent kinematics reaches
0.35; adding relational information about neighbors raises it to 0.72; adding
the time course raises it to 0.93. But once a model has both relational and
temporal information, a 156k-parameter GRU over aggregate group statistics
matches a 1.8M-parameter Transformer over per-agent tokens (0.929 vs 0.927,
paired difference not distinguishable from zero), and rotation-invariant inputs
match raw coordinates. Architecture, agent-level granularity, and hand-supplied
symmetries are all second-order. The observation design is the whole game.

Finally, the errors are informative: the *number* of neighbors an agent attends
to is recovered far more reliably than *which* neighbors it selects, and
conformal sets widen across strategy families at fixed *k* -- exactly where the
generating mechanisms converge behaviorally.

---

## 1. Introduction

Epstein's generative dictum -- "if you didn't grow it, you didn't explain it" --
has an inverse. Given a grown pattern, which rule grew it? Inverse generative
social science (IGSS) attacks this by search: evolve a population of candidate
rules and select those whose output resembles the target. This is expensive, and
it returns a point estimate with no statement of what *else* could have produced
the same data.

An alternative is to treat rule recovery as supervised classification over a
library of candidate rules, and to quantify the residual ambiguity with conformal
prediction, which returns a set `C(W)` guaranteed to contain the true rule with
probability at least `1 - alpha`, without distributional assumptions. When the
data are uninformative, the set is large; when they pin the rule down, it is a
singleton. The set size *is* the answer to "how identifiable is this rule from
this observation."

We previously applied this to Schelling segregation with a library of 2,089
arithmetic rule trees. That leaves an obvious objection: perhaps the classifier
recognizes *grid textures* rather than *behavioral rules*, and the framework is
an artifact of one domain.

This paper tests the framework in a domain that shares nothing with the first
except the framework itself -- continuous, off-lattice, multi-agent trajectories
-- and in doing so, asks a sharper question than "does it transfer":

> **What must a model observe before a behavioral mechanism becomes
> identifiable, and does the answer depend on how the model is built?**

**Contributions.**

1. A cross-domain controlled benchmark for behavioral-rule identification:
   eleven known neighbor-selection strategies in a collective-motion model,
   with grouped splits that make the evaluation leakage-free by construction.
2. An **information ladder** showing that identifiability is determined by the
   observation design -- relational information, then temporal information --
   and that architecture and representation granularity are second-order.
3. **Conformal guarantees** that transfer intact: RAPS attains nominal coverage
   with sets of 1-2 rules out of 11, and set size tracks observation length.
4. A mechanistic reading of the errors: neighbor *count* is easier to recover
   than neighbor *selection rule*, and the prediction sets say so.

---

## 2. Related work

**Inverse generative social science.** Epstein (2023) frames the inverse problem.
Gunaratne and Garibay (2020) and Gunaratne et al. (2023) solve it by evolutionary
model discovery: genetic programming over rule trees, with random-forest feature
importance to identify influential factors. This yields a rule but no calibrated
uncertainty over alternatives.

**Conformal prediction.** Vovk et al. (2005) establish distribution-free coverage
under exchangeability. RAPS (Angelopoulos et al. 2021) produces adaptive sets
with a regularizer that discourages the long tail. We use RAPS as our uncertainty
layer.

**Collective motion and interaction inference.** Lei et al. (2020), building on the burst-and-coast model of Calovi et al. (2018), fit a
burst-and-coast model of *Hemigrammus rhodostomus* and compare interaction
strategies -- which neighbors influence a focal fish, and how many. Their
simulated-agent trajectories, released with the paper, are what we use: the
generating strategy is known exactly, so this is a controlled benchmark rather
than an empirical inference problem.

**Deep models of collective behavior.** Prior work predicts trajectories forward
(social LSTMs, interaction transformers) or infers interaction graphs (NRI). We
invert instead: we classify the *generating rule*, and we attach a coverage
guarantee to the answer.

---

## 3. Setup

### 3.1 The rule space

Five agents move in a circular arena of radius `R = 0.25 m` under a
burst-and-coast dynamic. A rule is a pair `g = (s, k)`: a neighbor-selection
strategy `s` and a number of attended neighbors `k`.

| Rule | Strategy | k |
|---|---|---|
| 0 | no interaction | 0 |
| 1-3 | nearest | 1, 2, 3 |
| 4-6 | random | 1, 2, 3 |
| 7-9 | most influential | 1, 2, 3 |
| 10 | all neighbors | 4 |

Eleven rules, so chance accuracy is `1/11 = 0.0909`. Note the space is not a
clean product: `k=0` occurs only with "none" and `k=4` only with "all", so those
two rules are determined by `k` alone. The genuinely hard decisions live among
the nine rules with `k` in {1,2,3} across three families -- which is where we
should expect, and do find, the conformal sets to widen.

### 3.2 Data

We use the simulated-agent component of the Lei et al. (2020) release
(figshare 11858379). **These are simulated trajectories, not empirical fish
data**, which is the point: the generating rule is known for every run, so exact
recovery is measurable.

Verified directly from the files rather than assumed:

- 11 files, one per rule; 50 independent simulation runs each; **550 runs total,
  perfectly class-balanced**.
- Every run has exactly 5 agents, and **within a run all five agents share one
  time vector**, so a run reduces to a `[T, 5, 2]` position tensor.
- Time is **irregular kick time**, not a fixed grid: median inter-kick interval
  0.434 s (range 0.002-1.842 s), ~1,600 kicks and ~717 s per run.
- Arena centered at the origin; maximum observed radius 0.2451 m, consistent with
  `R = 0.25`.

**The sequence axis is the kick event, not resampled time.** The dynamics are
burst-and-coast, so a kick *is* the natural unit of behavior; interpolating onto
a uniform grid would smooth away the very discontinuities the rule acts through.
We include the inter-kick interval `dt` as an input feature instead.

### 3.3 Leakage control: the experimental unit is the seed, not the window

Two problems, both of which would silently inflate results.

**Seed coupling across rules.** Experiment ids are shared across the eleven rule
files. Inspecting the first recorded event of each run, one experiment is
*identical* across all eleven rules and thirteen more are *partially* identical
-- the runs are launched from a common initial placement and diverge only as the
rules take hold. A classifier could therefore learn the initial condition rather
than the rule.

We handle it twice over. First, **the grouping key is the experiment id, applied
across all eleven rules simultaneously**: if experiment 42 is in the test
partition, then all eleven runs with `exp_id = 42` are in the test partition.
Second, we discard a **50-event burn-in** from every run, which removes the
startup transient (all ten out-of-arena positions in the entire dataset occur
within the first 2 s) *and* the early region where the seed coupling lives.

**Window dependence.** Windows cut from the same run are strongly dependent. We
therefore (i) assign every window the partition of its run, (ii) use
**non-overlapping windows in calibration and test** -- overlapping windows are
not exchangeable, and feeding them to split conformal produces a quantile that
looks fine and a guarantee that does not hold -- and (iii) **bootstrap all
confidence intervals over runs, never over windows**.

**Grouped 5-fold cross-validation.** Each fold holds out a block of 10
experiments as test, so every experiment is tested exactly once. Within a fold:
25 train / 5 validation / 10 calibration / 10 test experiments. Because every
experiment exists under all eleven rules, every partition is automatically
class-balanced.

Calibration is never used for training, early stopping, or temperature fitting.
Its only job is to set the conformal quantile.

### 3.4 Features

Each `(agent, event)` cell carries a feature vector. Derivatives divide by the
*actual* inter-kick interval, never a constant.

- **Single-agent (11):** position, velocity, speed, heading as `(sin, cos)`,
  acceleration, distance to wall, `dt`.
- **Relational (9):** sorted neighbor distances `d_nn1`, `d_nn2`, their mean and
  spread; the mean neighbor heading and the bearing to the group centroid, both
  in the focal agent's own frame; local polarization.

Two design notes that turned out to matter.

*No feature is indexed by "the nearest neighbor."* Positions are quantized to
1 mm and cohesive rules pack agents until `d_nn1 = 0`, so two neighbors are
frequently *exactly* equidistant and the identity of the nearest is decided by
floating-point noise. Any `argmin`-indexed feature jumps discontinuously when
that tie flips. All angular features are therefore aggregates over the four
neighbors. Sorted *distances* are kept: a sorted value is continuous even when
the index producing it is not.

*Angles are expressed in the focal agent's frame,* which makes them invariant to
global rotation by construction.

### 3.5 Models

**Trajectory Transformer (axial).** A window is a `[T, 5, F]` grid of agent-time
states. Each block applies attention over **time within each agent** (weights
shared across agents, so an agent can be followed along its own trajectory) and
then over **agents within each event** (a set operation).

The obvious alternative -- flatten all `5T` cells into one sequence with a time
embedding and no agent-id embedding -- is exactly permutation invariant and
*wrong*. Without an agent id, the model sees an unordered **bag** of
`(state, time)` tokens: nothing links agent 3 at event 10 to agent 3 at event 11.
It is consequently invariant to shuffling agents *independently at each event*,
which dices every trajectory into fragments. That destroys precisely the signal
the task turns on -- whether an agent *keeps* the same nearest neighbor from kick
to kick is what separates `nearest-k` from `random-k`. We verify both properties
explicitly: global relabeling changes the output by < 5e-7, a per-event shuffle
changes it by 2e-3.

Axial attention preserves identity along the trajectory while remaining exactly
permutation invariant **for any weights** -- so permutation robustness is an
architectural guarantee, not something learned from augmentation.

**GRU baseline.** The control that separates two explanations. It is also a
sequence model and also sees relational information, but at each event the five
agents have already been averaged into a single group-state vector (13
statistics: cohesion, polarization, angular momentum, wall distance, ...). If the
Transformer wins, the gain is attributable to agent-level structure; if it does
not, sequence modeling alone accounted for it.

**Classical baselines.** Random forest and multinomial logistic regression over
window-level summary statistics (no time axis), plus a deliberately weak rung
using only *solo* kinematics with no neighbor information at all.

**Augmentation.** Global rotation, reflection, and agent permutation -- all
genuine symmetries of a circular arena with no handedness. Not time reversal: the
dynamics are causal. We verify the feature-space transformations against
recomputing every feature from transformed *positions*.

### 3.6 Conformal prediction

RAPS on the held-out calibration partition, `alpha` in {0.05, 0.10, 0.20}.
Temperature is fitted on validation, never on calibration.

---

## 4. Results

Primary setting: `T = 64` kicks, about **28 seconds** of observation.

### 4.1 The information ladder [CONFIRMED except where noted]

*Figure 1.*

| What the model observes | Model | Exact accuracy |
|---|---|---|
| nothing | chance | 0.091 |
| solo kinematics only | random forest | 0.352 +- 0.008 |
| + relational information | random forest | 0.717 +- 0.012 |
| + temporal information | GRU | **0.929 +- 0.023** |
| + per-agent granularity (invariant) | Transformer | 0.927 +- 0.020 |
| + per-agent granularity (raw coords) | Transformer | [PENDING] ~0.94 (1/5 folds) |

Two large jumps and then nothing.

**Relational information is worth +0.365.** A model that sees only how fast an
agent swims, how hard it turns, and how close it is to the wall reaches 0.35.
Letting it see its neighbors doubles that. This is unsurprising in hindsight --
the rules *are* neighbor-selection rules -- but it sets the scale for what
follows.

**Temporal information is worth +0.212.** The same relational statistics, fed to
a sequence model that sees their time course rather than their window average,
go from 0.717 to 0.929. The rules differ in the *persistence* of the influencing
set, which a window average cannot express.

**Everything else is worth nothing measurable.** A 156k-parameter GRU over 13
aggregate statistics matches a 1.8M-parameter axial Transformer over per-agent
tokens: paired difference **-0.0015, 95% CI [-0.008, +0.005]**, bootstrapped over
runs. Agent-level granularity -- the representation the architecture literature
would tell you to reach for -- buys nothing here.

### 4.2 Observation length governs identifiability [CONFIRMED]

*Figure 2.*

| T (kicks) | seconds | solo kinematics | relational (RF) |
|---|---|---|---|
| 16 | 7 | 0.254 | 0.549 |
| 32 | 14 | 0.291 | 0.624 |
| 64 | 28 | 0.352 | 0.717 |
| 128 | 56 | 0.434 | 0.809 |
| 256 | 111 | 0.533 | 0.885 |
| 512 | 222 | 0.661 | **0.962** |

Identifiability rises monotonically with observation and **saturates**: with four
minutes of trajectory even a random forest on group statistics recovers the rule
96% of the time. The scientifically interesting regime is therefore the *short*
one, where the mechanism is genuinely underdetermined by the data -- and that is
exactly where conformal prediction has something to say.

### 4.3 Architecture and hand-supplied invariance are second-order [PENDING]

> **This section is gated on the raw-vs-invariant control.** First fold:
> raw coordinates 0.9388, invariant features 0.9396. If that holds across
> five folds, the section stands. Rewrite if not.

Our first Transformer, on raw coordinates, scored 0.825 -- ten points *below* the
GRU. The natural reading is that raw coordinates carry nuisance variance (an
arbitrary global rotation) that the model must learn to marginalize, while the
GRU's statistics are rotation-invariant by construction.

That reading is wrong, and the control says so. Trained for the same 120 epochs,
the raw-coordinate Transformer matches the invariant one. The original 10-point
gap was **training budget**, not invariance: the raw model was simply undertrained
at 40 epochs.

The honest conclusion is stronger and more deflationary than the one we set out
to draw. Given relational and temporal information, the model recovers the rule
at ~0.93 whether it is a GRU or a Transformer, whether it sees aggregates or
per-agent tokens, and whether it is handed the rotational symmetry or left to
learn it. **What the model observes determines what it can identify. How it is
built does not.**

For IGSS this is the operative lesson: effort spent on architecture is effort
misspent. Effort spent deciding *what to measure* -- and in particular whether the
observation retains the relational and temporal structure the rule acts through
-- is what buys identifiability.

### 4.4 Conformal prediction [CONFIRMED]

*Figure 3.* RAPS, pooled over the five folds; CIs bootstrapped over runs.

Transformer (invariant), T = 64:

| alpha | target | empirical coverage | mean set size | singleton rate |
|---|---|---|---|---|
| 0.05 | 0.95 | 0.962 [0.956, 0.968] | 1.28 | 73% |
| 0.10 | 0.90 | 0.949 [0.941, 0.956] | 1.06 | 94% |
| 0.20 | 0.80 | 0.937 [0.929, 0.945] | 1.03 | 98% |

**Coverage is valid at every level, and the sets are small** -- 1 to 2 rules out
of 11 at 95% coverage. The framework's uncertainty layer transfers to the new
domain intact.

The over-coverage at large `alpha` is not slack in the method; it is a *ceiling
effect*, and it is instructive. With top-1 accuracy at 0.93 and a non-empty-set
convention, coverage cannot fall below ~0.93 no matter what `alpha` asks for. The
sets have collapsed to singletons and the conformal layer has nothing left to
express.

**This relocates the value of conformal prediction.** It is informative precisely
where the model is uncertain -- at short observation windows, where the mechanism
is underdetermined. As `T` grows, accuracy rises and `|C|` collapses toward 1.
Reporting conformal sets at a window length where the task is already solved
measures nothing.

> [PENDING] The `T = 16` and `T = 32` conformal results are the substantive
> version of this figure and are in flight.

### 4.5 What the errors say about the mechanism [CONFIRMED]

*Figures 4 and 5.*

We expected errors to stay within a strategy family and confuse adjacent `k`
("nearest-2" mistaken for "nearest-3"). **The opposite is true.**

| | Transformer (inv) | GRU |
|---|---|---|
| exact-rule accuracy | 0.927 | 0.929 |
| strategy-family accuracy | 0.939 | 0.944 |
| **neighbor-count accuracy** | **0.959** | 0.951 |
| neighbor-count MAE | 0.043 | 0.050 |

and of the errors:

| | Transformer (inv) | GRU |
|---|---|---|
| right family, wrong k | 16% | 22% |
| **right k, wrong family** | **43%** | 32% |
| both wrong | 41% | 46% |

**How many neighbors an agent attends to is easier to recover than which
neighbors it selects.** The mechanism explains why. The number of attended
neighbors sets the effective strength of cohesion, which is written directly into
the group's geometry -- mean neighbor distance falls monotonically with `k` within
every family. *Which* neighbors are selected is a far subtler, higher-order
property, and at `k = 3` out of 4 available neighbors, "nearest 3", "random 3",
and "most influential 3" are drawing from an almost identical pool. They converge
behaviorally, so no observer can separate them.

The confusion matrix shows exactly this: the errors concentrate in a high-`k`
cluster (`infl-3` <-> `all-4`, `infl-3` <-> `rand-3`, `infl-3` <-> `near-3`),
while `none-0` is recovered perfectly and `rand-1` at 0.97.

**And the conformal sets say so.** A prediction set like
`C(W) = {nearest-3, influential-3, all-4}` is not a failure of the classifier; it
is a correct statement that the observation cannot distinguish three mechanisms
that produce nearly the same behavior. This is the interpretive payoff of putting
a conformal layer on an IGSS pipeline: *the ambiguity is a property of the
mechanism, not of the model, and the set size measures it.*

---

## 5. Discussion

### 5.1 What transfers

The Schelling case study and this one share no data type, no rule
representation, no architecture family, and no domain. The framework
-- library of candidate rules, supervised recovery, conformal set -- transfers
whole. Coverage guarantees hold; set sizes are interpretable; the errors are
mechanistically legible in both.

### 5.2 What does not matter

The negative result is the useful one. In a domain where interaction rules
*are* the object of study, the choice between a small GRU and a large Transformer,
between aggregate and agent-level representations, and between supplied and
learned symmetries, is immaterial. What matters is whether the observation
retains relational and temporal structure.

We would not have learned this without the aggregate control. A Transformer that
beats a random forest looks like a win for agent-level modeling; the same
Transformer next to a GRU shows the win came from the time axis, which the forest
never had.

### 5.3 Limitations

- **Simulated, not empirical.** The agent trajectories come from a fitted
  generative model. This is a controlled benchmark for rule recovery, not a claim
  about real fish. The real-fish and robot trajectories in the same release are
  untouched here.
- **A small rule library.** Eleven rules, against 2,089 in the Schelling study.
  Conformal sets are correspondingly easy to interpret and the task correspondingly
  easier.
- **Exchangeability is at the level of runs, not windows.** Windows from one run
  are dependent. We mitigate structurally (disjoint experiments between
  calibration and test; non-overlapping evaluation windows) and report all
  intervals bootstrapped over runs, so coverage should be read as marginal over
  the window distribution induced by exchangeable *runs*.
- **50 seeds.** Grouped 5-fold CV tests every experiment exactly once, but the
  fold-to-fold spread (+-0.02) is the honest resolution of these numbers.

### 5.4 Future work

The immediate step is the real-fish trajectories from the same release: with a
rule library calibrated on simulation and a conformal layer, one can ask which
interaction strategies remain *plausible* for real *H. rhodostomus* at a given
coverage level -- an IGSS question with a calibrated answer, which is what the
existing model-selection literature cannot give.

---

## Reproducibility notes

Grouped splits, feature definitions, and the conformal construction are
implemented in `genmod/fish/` and `scripts/fish/`. Two verification scripts guard
the claims that are easy to get silently wrong:

- `verify_augmentation.py` asserts that transforming the *features* equals
  recomputing the features from transformed *positions*. It failed on first run,
  which is how the nearest-neighbor tie degeneracy (Section 3.4) was found.
- `verify_permutation_invariance.py` asserts `f(PW) = f(W)` for any weights, and
  -- as a control -- that a per-event agent shuffle *does* change the output. It
  failed on first run, which is how the bag-of-tokens flaw (Section 3.5) was
  found.

Both failures were real defects that no training curve would have revealed.

**A note on the conformal implementation.** Our first RAPS results showed
coverage pinned near 0.97 regardless of `alpha`, with set size barely responding.
The cause was a defect in the non-conformity score: the APS randomization term
`u * p_y` was applied only when the true class was *not* ranked first, so for an
accurate model the majority of calibration points received the full `p_max`
instead. This inflates the calibration quantile and every prediction set. A
second defect had `conformal_predict` construct sets deterministically while
calibration randomized them. Both are corrected here. We flag them because the
failure mode is silent: coverage remains *valid* (the sets are conservative), so
nothing in the output announces that the sets are larger than they need to be.

---

## Figures

| File | Content |
|---|---|
| `fig1_ladder` | The information ladder at T=64. **The central figure.** |
| `fig2_window` | Accuracy vs observation length; saturation. |
| `fig3_conformal` | RAPS coverage and set size vs alpha. |
| `fig4_confusion` | 11x11 confusion matrix; the high-k cluster. |
| `fig5_errors` | Where errors go: right-k vs right-family. |

## TODO before submission

- [ ] Confirm the raw-vs-invariant control (5 folds). **Gates the abstract.**
- [ ] T=16 and T=32 conformal -- the substantive version of Figure 3.
- [ ] Relabel Figure 1's title (currently "Invariance, not architecture", which
      the control has overturned; it should be "What the model observes, not how
      it is built").
- [ ] Decide whether the conformal-implementation note belongs in the paper or
      only in the repo.
- [ ] Citation keys and bibliography.
