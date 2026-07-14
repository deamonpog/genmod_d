# ODD Appendix: the generating model and the inversion protocol

*Supplementary to "Measuring Equifinality: Calibrated Rule Identification in
Agent-Based Models of Collective Motion." Per the CSS 2026 call, ODD
documentation is attached without counting toward the word limit.*

Two things need documenting, and they are not the same thing:

- **Part A** describes the agent-based model whose rules we are trying to
  recover. We did not build it; we restate it in ODD form so that a reader can
  judge what our classifier is being asked to invert.
- **Part B** documents *our* protocol -- the inversion pipeline -- to the same
  standard, because the reproducibility of the inverse claim depends on the
  data partitioning at least as much as on the model.

---

# Part A. The generating model (Lei et al. 2020)

## A.1 Purpose

To reproduce the collective motion of small groups of *Hemigrammus rhodostomus*
(rummy-nose tetra) and to determine which combinations of neighbor interactions
are sufficient to explain observed schooling. The model is a burst-and-coast
swimmer with pairwise wall and social interactions, fitted to experiment by
Calovi et al. (2018), and extended by Lei et al. (2020) to compare
*neighbor-selection strategies*.

For our purposes the model's status is that of a **known ground truth**: it is a
generator whose rule we hide and then attempt to recover.

## A.2 Entities, state variables, and scales

**Entities.** Five agents (fish) and one environment (a circular tank).

**Agent state.**

| Variable | Meaning | Unit |
|---|---|---|
| `(x, y)` | position | m |
| `theta` | heading | rad |
| `t_kick` | time of the agent's next kick | s |
| `l_kick` | length of the current kick | m |

**Environment.** A circular arena of radius `R = 0.25 m` for agents (the robot
arena in the same release is 0.42 m and is not used here). No obstacles.

**Temporal scale.** The model is **event-driven, not time-stepped.** An agent
advances by discrete *kicks*: a burst of propulsion followed by a passive coast
with exponentially decaying speed. Observations are emitted at kick events, so
the natural time index is the kick, not a clock tick. Empirically: median
inter-kick interval 0.434 s (range 0.002-1.842 s), about 1,600 kicks per run,
about 717 s of simulated time.

**Spatial scale.** Continuous and off-lattice. Body length ~0.03 m, so the arena
is roughly 16 body lengths across.

## A.3 Process overview and scheduling

At each of its kick events, an agent:

1. **Selects a subset of neighbors** to attend to, according to the rule in force
   (this is the object of our inference; see A.6).
2. **Computes a heading change** as the sum of
   - a *wall* term (repulsion, increasing as the wall is approached),
   - *social* terms from each selected neighbor -- an attraction/repulsion
     component depending on distance and a alignment component depending on
     relative heading,
   - a *stochastic* term.
3. **Draws a kick length and interval,** turns, and coasts.

Scheduling is **asynchronous**: agents kick at their own times. Neighbor states
are read at the moment of the kick.

## A.4 Design concepts

- **Emergence.** Schooling, milling and swarming phases emerge; none is coded.
- **Interaction.** Explicitly pairwise, and explicitly *selective* -- an agent
  responds only to the neighbors its rule admits. This selectivity is precisely
  what we invert.
- **Stochasticity.** Kick timing, kick length, and a heading noise term.
- **Sensing.** Agents sense neighbor position and heading without error.
- **Observation.** The released data record, at each kick event, the positions of
  all five agents. Headings and velocities are not recorded; we reconstruct them
  by finite differences over consecutive kicks (see B.3).

## A.5 Initialization

Agents are placed inside the arena and allowed to move; runs share initial
placements across rules (see B.2 -- this has a direct and dangerous consequence
for evaluation). Fifty independent runs (seeds) per rule.

## A.6 Submodels: the eleven rules

A rule is a pair `g = (s, k)`: a neighbor-selection strategy `s` and the number
`k` of neighbors attended to at each kick.

| Rule | `s` | `k` | The focal agent responds to ... |
|---|---|---|---|
| 0 | none | 0 | no one; wall + noise only |
| 1-3 | nearest | 1, 2, 3 | its `k` nearest neighbors |
| 4-6 | random | 1, 2, 3 | `k` neighbors drawn at random, re-drawn each kick |
| 7-9 | most influential | 1, 2, 3 | the `k` neighbors exerting the largest instantaneous influence |
| 10 | all | 4 | all four neighbors |

The families differ *only* in which neighbors enter the social sum -- the
interaction functions themselves are identical. This is what makes the benchmark
sharp, and also what makes the high-`k` rules genuinely hard to tell apart: with
`k = 3` of 4 available neighbors, "nearest three", "random three" and "most
influential three" draw from nearly the same pool on most kicks.

Note the asymmetry the rule table hides: `k = 0` occurs only under "none" and
`k = 4` only under "all", so those two rules are determined by `k` alone. Nine of
the eleven rules live in the `k` in {1,2,3} x {nearest, random, most influential}
grid, and that is where the residual equifinality concentrates (Section 4.4 of
the main paper).

## A.7 Data used

The simulated-agent files of the figshare release accompanying Lei et al. (2020)
(figshare 11858379): 11 rules x 50 runs x 5 agents. Whitespace-delimited, five
columns (`experiment_id`, `agent_id`, `t`, `x`, `y`). We use only the
**simulated-agent** files. The real-fish and robot files in the same release are
outside the scope of this paper, and the simulated trajectories are never
described as empirical data.

---

# Part B. The inversion protocol

## B.1 Purpose

Given a window of observed trajectory `W`, recover the generating rule
`g in G, |G| = 11`, and return a set `C(W)` that contains the true rule with
probability at least `1 - alpha`.

## B.2 Experimental unit and partitioning (the part that matters most)

**The experimental unit is the simulation seed, not the trajectory window.**

Two hazards, both of which inflate results silently if ignored.

**Hazard 1: seeds are coupled across rules.** Experiment identifiers are shared
across the eleven rule files, and the runs begin from a common initial
placement. Inspecting the first recorded event of every run: **one experiment is
identical across all eleven rules, and thirteen more are partially identical**.
A classifier could therefore learn the initial condition rather than the rule.

Mitigations (both applied):

- **Group by experiment id across all eleven rules simultaneously.** If
  experiment 42 is in the test partition, all eleven runs carrying
  `exp_id = 42` are in the test partition.
- **Discard a 50-event burn-in** from every run. This removes the placement
  transient (every out-of-arena position in the dataset -- ten in 4.4 million
  rows -- occurs within the first 2 s) *and* the early region in which the
  cross-rule coupling lives.

**Hazard 2: windows within a run are dependent.** Consecutive windows share most
of their events, and split conformal prediction assumes calibration and test
scores are exchangeable. Mitigations:

- Every window inherits the partition of its run.
- **Calibration and test windows are non-overlapping** (stride = window length).
  Training uses a dense stride, which is harmless because training makes no
  exchangeability assumption.
- **All confidence intervals bootstrap over runs, never over windows.**

**Design.** Grouped 5-fold cross-validation over the 50 experiments. Each fold
holds out a block of ten as test, so every experiment is tested exactly once
across the five folds. Within a fold:

| Partition | Experiments | Purpose |
|---|---|---|
| train | 25 | fit model parameters |
| validation | 5 | early stopping, temperature scaling |
| calibration | 10 | the conformal quantile, and nothing else |
| test | 10 | evaluation |

Because every experiment exists under all eleven rules, every partition is
automatically class-balanced. **The calibration partition is used for nothing but
the conformal quantile** -- not training, not model selection, not temperature.

## B.3 Observation model

The released data give positions only, so velocity, speed, heading and
acceleration are reconstructed by finite differences over consecutive kicks.
Because kick intervals are irregular, every derivative divides by the *actual*
interval `dt`, never a constant; `dt` also enters as a feature.

Per `(agent, event)` cell:

- **Single-agent (11):** normalized position, velocity, speed, heading as
  `(sin, cos)`, acceleration, distance to wall, `dt`.
- **Relational (9):** the sorted neighbor distances `d_nn1`, `d_nn2` and their
  mean and spread; the mean neighbor heading and the bearing to the group
  centroid, both rotated into the focal agent's own frame; local polarization.

**No feature is indexed by "the nearest neighbor."** Positions are quantized to
1 mm, and cohesive rules pack agents until `d_nn1 = 0`, so two neighbors are
frequently *exactly* equidistant and the `argmin` is decided by floating-point
noise. Any feature indexed by that `argmin` jumps discontinuously when the tie
flips. All angular features are therefore aggregates over the four neighbors.
Sorted *distances* are retained -- a sorted value is continuous even when the
index that produced it is not.

Feature scaling uses the median and inter-quartile range **fitted on the training
experiments of each fold only**, with clipping. Acceleration is a finite
difference over an interval that can be 2 ms, so its tails reach 570 sigma; a
mean/standard-deviation scaler would let those artifacts dictate the input scale.

## B.4 Observation windows

Windows are indexed by **kick event**, not resampled clock time: the dynamics are
burst-and-coast, so a kick is the natural unit of behavior, and interpolating
onto a uniform grid would smooth away the discontinuities through which the rule
acts.

Window lengths `T` in {16, 32, 64, 128, 256, 512} kicks, i.e. roughly 7 s to
222 s. `T = 64` (28 s) is the primary setting.

## B.5 Inference models

Four observation designs, in increasing order of what they retain (main paper,
Section 4.1). The two learned models:

- **GRU:** two layers, hidden 128 (~156k parameters), over a sequence of thirteen
  per-event group statistics. Agents are averaged away; time is retained.
- **Axial transformer:** `d_model = 128`, 8 heads, 4 axial blocks (~1.8M
  parameters). Each block attends over time *within* an agent (weights shared
  across agents) and then over agents *within* an event. No agent-identity
  embedding, so the model is **exactly permutation invariant for any weights**;
  the axial factorization preserves each agent's identity along its own
  trajectory, which a flat attention over all agent-event tokens would not.

Symmetries used as augmentation: global rotation, reflection, agent permutation.
**Not** time reversal -- the dynamics are causal, and a reversed trajectory does
not obey the same rule.

## B.6 Uncertainty quantification

RAPS (Angelopoulos et al. 2021) on the held-out calibration partition, at
`alpha` in {0.05, 0.10, 0.20}. Temperature scaling is fitted on validation, never
on calibration.

Reported: empirical marginal coverage, per-rule coverage, mean and median set
size, singleton rate, set-size distribution. All intervals bootstrap over runs.

## B.7 Verification

Two automated checks guard claims that would otherwise fail silently. Both failed
on first execution and both exposed real defects:

- **Symmetry check.** Asserts that transforming the *features* under a rotation
  equals recomputing every feature from transformed *positions*. Its failure
  exposed the nearest-neighbor tie degeneracy described in B.3.
- **Permutation check.** Asserts `f(PW) = f(W)` for any weights, *and*, as a
  control, that shuffling agents independently at each event *does* change the
  output. Its failure exposed that a flat transformer over agent-event tokens
  sees an unordered bag of `(state, time)` tokens and cannot follow an agent
  through time at all -- destroying exactly the neighbor-persistence signal that
  distinguishes "nearest-k" from "random-k".

Neither defect would have been visible in a training curve.

## B.8 Code and data

Data: figshare 11858379 (simulated-agent files only). Code: the pipeline is
implemented in `genmod/fish/` and `scripts/fish/`, with the grouped-split
assertions, the two verification scripts above, and the conformal construction
in `genmod/evaluation/conformal.py`.
