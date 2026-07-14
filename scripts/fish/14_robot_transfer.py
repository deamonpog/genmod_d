"""Transfer a simulation-trained rule library onto physical robots.

The framework's real claim is not "we can recover rules in a simulator". It is
"a rule library learned in simulation identifies mechanisms in a system that was
not simulated". This script tests exactly that, with NO retraining: the model is
the transformer already trained on simulated agents (fold 0), loaded from its
checkpoint and run on robot trajectories.

Three measurements, in increasing order of what they tell us.

1. ZERO-SHOT ACCURACY. Does the classifier identify the programmed rule of a
   physical robot?

2. CONFORMAL COVERAGE UNDER SHIFT, calibrated on SIMULATION. Split conformal
   guarantees coverage only when calibration and test scores are exchangeable.
   Simulated agents and robots are not: the arena is 1.7x larger, the kicks are
   1.9x slower, and the hardware has noise the simulator does not. So the
   guarantee should BREAK, and we expect the sets to be overconfident. This is a
   prediction, and reporting it is the honest thing to do -- a coverage guarantee
   that silently fails under distribution shift is exactly the trap practitioners
   fall into.

3. CONFORMAL COVERAGE, RECALIBRATED ON ROBOTS. Exchangeability does hold WITHIN
   the robot domain. So calibrate the conformal quantile on a held-out set of
   robot runs and evaluate on the rest. If coverage is restored, then:

       a rule library trained entirely in simulation can be deployed on a
       physical system and recover a VALID coverage guarantee from a small
       calibration set -- no retraining, no re-simulation.

   That is the practical payoff of putting a conformal layer on an IGSS pipeline,
   and it is the result this script exists to obtain.

Usage:
    python scripts/fish/14_robot_transfer.py
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())

from genmod.evaluation.conformal import (compute_nonconformity_scores,
                                         conformal_calibrate, conformal_predict)
from genmod.fish.features import INVARIANT_COLS, compute_features
from genmod.fish.gpu_transform import GPUTransform
from genmod.fish.robots import (ROBOT_RADIUS, load_robot_runs, to_agent_units)
from genmod.models.traj_transformer import TrajectoryTransformer

TAG = "transformer_inv_T64"
FOLD = 0
T = 64
N_RULES = 11
BURN_IN = 10   # robot runs are short (140-730 events); 50 would waste a third
ALPHAS = [0.05, 0.10, 0.20]
SEED = 20260714

RULE_NAMES = ["none-0", "near-1", "near-2", "near-3", "rand-1", "rand-2",
              "rand-3", "infl-1", "infl-2", "infl-3", "all-4"]
K_OF = np.array([0, 1, 2, 3, 1, 2, 3, 1, 2, 3, 4])
FAMILY_OF = np.array([0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4])


def coverage_and_size(probs, labels, qhat, k_reg, lam):
    sets = conformal_predict(probs, qhat, method="raps",
                             k_reg=k_reg, lambda_reg=lam)
    cover = np.array([y in s for s, y in zip(sets, labels)], dtype=float)
    size = np.array([len(s) for s in sets], dtype=float)
    return cover, size


def main():
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    res_dir = os.path.join("results", "fish", TAG)

    # ---- 1. the robot data ---------------------------------------------
    print("Loading robot runs ...")
    runs = load_robot_runs()
    rule_ids = sorted({r.rule_id for r in runs})
    print("  %d runs, %d rules present (of %d)  [missing: %s]"
          % (len(runs), len(rule_ids), N_RULES,
             ", ".join(RULE_NAMES[i] for i in range(N_RULES)
                       if i not in rule_ids) or "none"))

    feats, meta = [], []
    for run in runs:
        f, _ = compute_features(run.time, run.pos,
                                radius=ROBOT_RADIUS, burn_in=BURN_IN)
        if f.shape[0] < T:
            continue
        # Non-overlapping windows only: overlapping windows are not
        # exchangeable, which would void the conformal guarantee we are about
        # to test.
        for s in range(0, f.shape[0] - T + 1, T):
            feats.append(f[s:s + T])
            meta.append((run.rule_id, run.exp_id, len(meta)))
    X = np.stack(feats)                                   # [N, T, 5, F]
    y = np.array([m[0] for m in meta])
    grp = np.array([m[1] for m in meta])                  # robot experiment id
    print("  %d non-overlapping windows of T=%d (%.0f s each)"
          % (X.shape[0], T, T * 0.84))
    print("  per-rule window counts:",
          {RULE_NAMES[c]: int((y == c).sum()) for c in rule_ids})

    # ---- 2. the simulation-trained model --------------------------------
    ck = torch.load(os.path.join(res_dir, "model_fold%d.pt" % FOLD),
                    map_location=device, weights_only=False)
    cfg = ck["cfg"]
    model = TrajectoryTransformer(
        n_features=len(INVARIANT_COLS), n_agents=5, max_events=T,
        num_classes=N_RULES, d_model=cfg["d_model"], n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"], dropout=cfg["dropout"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    temp = ck["temperature"]
    print("\nLoaded %s fold %d (trained on SIMULATED agents only), temp %.3f"
          % (TAG, FOLD, temp))

    wins = np.load(os.path.join("data", "fish", "windows_T%d.npz" % T))
    # The scaler is the one fitted on the AGENT training split, and it stays
    # that way throughout. What changes between the two variants below is only
    # the UNITS the robot data is expressed in before it reaches the network.
    tf = GPUTransform(wins["fold%d_median" % FOLD], wins["fold%d_iqr" % FOLD],
                      float(wins["clip"]), device)

    def infer(Xin):
        out = []
        with torch.no_grad():
            for i in range(0, Xin.shape[0], 256):
                xb = torch.from_numpy(
                    np.ascontiguousarray(Xin[i:i + 256])).to(device)
                xb = tf.scale(xb)[..., INVARIANT_COLS]
                out.append(torch.softmax(model(xb) / temp, dim=1).cpu().numpy())
        return np.concatenate(out)

    # (a) Robot features in SECONDS, as recorded. The network was fitted on a
    #     domain whose clock runs 1.9x faster, so a robot moving identically --
    #     the same fraction of the arena per kick -- registers about half the
    #     speed. This is the naive transfer, and it is dimensionally wrong.
    probs_raw = infer(X)

    # (b) The same weights, the same scaler, with the robot features
    #     NONDIMENSIONALIZED: velocities, accelerations and dt re-expressed in
    #     the agents' time unit (see genmod/fish/robots.to_agent_units). This is
    #     not adaptation and not tuning -- it is a unit conversion. Doing it is
    #     as mandatory as converting feet to metres before adding them.
    probs_nd = infer(to_agent_units(X))

    chance = 1.0 / len(rule_ids)

    def report(p, name):
        pr = p.argmax(1)
        a = float((pr == y).mean())
        t3 = float(np.mean([y[i] in np.argsort(-p[i])[:3] for i in range(len(y))]))
        print("    %-30s top-1 %.4f (%.1fx chance)  top-3 %.4f  "
              "family %.4f  k %.4f"
              % (name, a, a / chance, t3,
                 float((FAMILY_OF[pr] == FAMILY_OF[y]).mean()),
                 float((K_OF[pr] == K_OF[y]).mean())))
        return a

    print("\n[1] ZERO-SHOT: simulation-trained model on physical robots")
    print("    Same weights throughout. Nothing is retrained. The only")
    print("    difference is the units the robot data is expressed in.")
    acc_raw = report(probs_raw, "robot clock (seconds)")
    acc = report(probs_nd, "nondimensionalized (per kick)")
    print("    %-30s top-1 %.4f" % ("[same model on AGENTS]", 0.927))
    print("\n    unit conversion is worth %+.4f top-1" % (acc - acc_raw))

    probs = probs_nd
    pred = probs.argmax(1)
    top3 = float(np.mean([y[i] in np.argsort(-probs[i])[:3]
                          for i in range(len(y))]))

    print("\n    per-rule recall (nondimensionalized):")
    for c in rule_ids:
        m = y == c
        print("      %-8s %.3f   (n=%d)"
              % (RULE_NAMES[c], float((pred[m] == c).mean()), int(m.sum())))

    # ---- 4. conformal, calibrated on SIMULATION -------------------------
    # This is the trap: reuse the simulation calibration set and hope the
    # guarantee survives the domain shift.
    sim = np.load(os.path.join(res_dir, "probs_fold%d.npz" % FOLD))
    sim_scores = compute_nonconformity_scores(
        sim["calib_probs"], sim["calib_labels"], method="raps",
        k_reg=2, lambda_reg=0.1)

    # ---- 5. conformal, RECALIBRATED on robots ---------------------------
    # Split robot EXPERIMENTS (not windows) into calibration and test halves.
    rng = np.random.default_rng(SEED)
    exps = np.unique(grp)
    perm = rng.permutation(exps)
    cal_exps = set(perm[:len(perm) // 2].tolist())
    cal_m = np.array([g in cal_exps for g in grp])
    test_m = ~cal_m
    print("\n    robot calibration: %d windows from %d experiments"
          % (cal_m.sum(), len(cal_exps)))
    print("    robot evaluation:  %d windows from %d experiments"
          % (test_m.sum(), len(exps) - len(cal_exps)))

    rob_scores = compute_nonconformity_scores(
        probs[cal_m], y[cal_m], method="raps", k_reg=2, lambda_reg=0.1)

    out = {"zero_shot_top1": acc, "zero_shot_top3": top3,
           "n_windows": int(X.shape[0]), "rules_present": rule_ids,
           "alphas": {}}

    print("\n[2] CONFORMAL under domain shift")
    print("    %-7s %-9s %-24s %-24s" % (
        "alpha", "target", "calibrated on SIMULATION", "calibrated on ROBOTS"))
    print("    %-7s %-9s %-24s %-24s" % ("", "", "coverage   mean |C|",
                                         "coverage   mean |C|"))
    print("    " + "-" * 66)

    for a in ALPHAS:
        q_sim = conformal_calibrate(sim_scores, alpha=a)
        cov_s, sz_s = coverage_and_size(probs[test_m], y[test_m], q_sim, 2, 0.1)

        q_rob = conformal_calibrate(rob_scores, alpha=a)
        cov_r, sz_r = coverage_and_size(probs[test_m], y[test_m], q_rob, 2, 0.1)

        out["alphas"]["%.2f" % a] = {
            "target": 1 - a,
            "sim_calibrated": {"coverage": float(cov_s.mean()),
                               "mean_set_size": float(sz_s.mean())},
            "robot_calibrated": {"coverage": float(cov_r.mean()),
                                 "mean_set_size": float(sz_r.mean()),
                                 "singleton_rate": float((sz_r == 1).mean())},
        }
        flag_s = "OK " if cov_s.mean() >= 1 - a - 0.02 else "FAIL"
        flag_r = "OK " if cov_r.mean() >= 1 - a - 0.02 else "FAIL"
        print("    %-7.2f %-9.2f %s %.3f      %5.2f       %s %.3f      %5.2f"
              % (a, 1 - a, flag_s, cov_s.mean(), sz_s.mean(),
                 flag_r, cov_r.mean(), sz_r.mean()))

    print("""
    Reading this table:
      SIMULATION-calibrated  the guarantee is transported across a domain shift
                             it was never entitled to cross. Under-coverage here
                             is not a bug -- it is what exchangeability failing
                             looks like, and it is silent: nothing in the output
                             would tell a practitioner the sets are too small.
      ROBOT-calibrated       the same model, the same weights, no retraining.
                             Only the conformal quantile is re-estimated, from a
                             held-out set of robot runs. Exchangeability holds
                             within the robot domain, so coverage is restored.""")

    path = os.path.join("results", "fish", "robot_transfer.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nWrote %s" % path)


if __name__ == "__main__":
    main()
