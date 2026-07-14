"""Generate the figures for the fish case study.

Reads only saved results (results/fish/*), so it never retrains anything and
can be re-run as new ablations land.

    fig1_ladder.pdf        the representation ladder at T=64
    fig2_window.pdf        accuracy vs observation length
    fig3_conformal.pdf     RAPS coverage and set size vs alpha
    fig4_confusion.pdf     confusion matrix of the invariant Transformer
    fig5_errors.pdf        where the errors go: family vs k

Colors are the Okabe-Ito categorical set, assigned in FIXED order and never
cycled: worst adjacent CVD separation is dE 17.9 (deuteranopia), well clear of
the 12 threshold. Three of the hues fall below 3:1 contrast against white, so
every mark is directly labeled -- the labels are the required relief, not
decoration.

Run from the repository root:
    python scripts/fish/13_make_figures.py
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.getcwd())

RES = os.path.join("results", "fish")
FIG = os.path.join("results", "figures", "fish")

# Okabe-Ito, in fixed assignment order. Validated: see module docstring.
BLUE = "#0072B2"
VERM = "#D55E00"
GREEN = "#009E73"
PINK = "#CC79A7"
ORANGE = "#E69F00"
SKY = "#56B4E9"
INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#e2e2e2"

RULE_NAMES = ["none-0", "near-1", "near-2", "near-3", "rand-1", "rand-2",
              "rand-3", "infl-1", "infl-2", "infl-3", "all-4"]
CHANCE = 1.0 / 11


def style(ax):
    """Recessive axes and grid: the data should be the only assertive thing."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, lw=0.8)


def load(tag, name):
    p = os.path.join(RES, tag, name)
    return json.load(open(p)) if os.path.exists(p) else None


def pooled_probs(tag):
    """Test probabilities and labels, pooled over the five folds."""
    d = os.path.join(RES, tag)
    folds = sorted(int(f[len("probs_fold"):-4]) for f in os.listdir(d)
                   if f.startswith("probs_fold"))
    P, Y = [], []
    for k in folds:
        z = np.load(os.path.join(d, "probs_fold%d.npz" % k))
        P.append(z["test_probs"])
        Y.append(z["test_labels"])
    return np.concatenate(P), np.concatenate(Y)


# ----------------------------------------------------------------------
def fig1_ladder(base):
    """The information ladder. This is the paper's central figure.

    transformer_T64 (raw coordinates, 40 epochs, 0.825) is deliberately NOT
    plotted. Its validation accuracy was still climbing when the epoch budget
    ran out -- early stopping never fired -- and a 120-epoch control on raw
    coordinates reaches the same range as the invariant model on the same
    folds. Plotting 0.825 beside the others would tell the reader that raw
    coordinates are a worse REPRESENTATION, which our own control contradicts.
    It is a training-budget artifact, not a rung of this ladder.
    """
    b = base["results"]["T64"]

    rows = [
        ("Chance", CHANCE, 0.0, MUTED),
        ("Solo kinematics\n(random forest)", b["rf_naive"]["acc"][0],
         b["rf_naive"]["acc"][1], SKY),
        ("+ relational\n(random forest)", b["rf"]["acc"][0],
         b["rf"]["acc"][1], BLUE),
    ]
    for tag, label, color in [
            ("gru_T64", "+ temporal\n(GRU, 156k params)", GREEN),
            ("transformer_inv_T64",
             "+ per-agent\n(Transformer, 1.8M params)", ORANGE)]:
        ev = load(tag, "evaluation.json")
        if ev:
            lo, hi = ev["top-1 accuracy"][1], ev["top-1 accuracy"][2]
            rows.append((label, ev["top-1 accuracy"][0], (hi - lo) / 2, color))

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    y = np.arange(len(rows))[::-1]
    for yi, (label, v, err, c) in zip(y, rows):
        ax.barh(yi, v, height=0.62, color=c, edgecolor="white", lw=2,
                zorder=3)
        if err:
            ax.errorbar(v, yi, xerr=err, color=INK, lw=1.2, capsize=3,
                        zorder=4)
        # Direct label: required relief for the low-contrast hues, and it
        # removes any need for the reader to track a value back to an axis.
        ax.text(v + 0.018, yi, "%.3f" % v, va="center", ha="left",
                fontsize=9.5, color=INK, zorder=5)

    ax.axvline(CHANCE, color=MUTED, ls=":", lw=1, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlim(0, 1.06)
    ax.set_xlabel("Exact-rule accuracy (11 classes)", fontsize=10, color=INK)
    ax.set_title("What the model observes, not how it is built\n"
                 "T = 64 kicks (28 s), 5 grouped folds",
                 fontsize=11, color=INK, loc="left", pad=12)
    style(ax)
    ax.grid(axis="y", lw=0)
    ax.grid(axis="x", color=GRID, lw=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_ladder.pdf"))
    fig.savefig(os.path.join(FIG, "fig1_ladder.png"), dpi=200)
    plt.close(fig)
    print("  fig1_ladder")


# ----------------------------------------------------------------------
def fig2_window(base):
    """Accuracy vs observation length. RQ3."""
    Ts = [16, 32, 64, 128, 256, 512]
    dt = 0.434  # median inter-kick interval, seconds

    series = [
        ("Solo kinematics (RF)", "rf_naive", SKY),
        ("Relational group stats (RF)", "rf", BLUE),
    ]

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    for label, key, color in series:
        v = [base["results"]["T%d" % T][key]["acc"][0] for T in Ts]
        e = [base["results"]["T%d" % T][key]["acc"][1] for T in Ts]
        ax.errorbar(Ts, v, yerr=e, color=color, lw=2, marker="o", ms=6,
                    capsize=3, label=label, zorder=3)
        ax.annotate(label, (Ts[-1], v[-1]), xytext=(6, 0),
                    textcoords="offset points", fontsize=8.5, color=color,
                    va="center")

    # The sequence model across every window length.
    xs, ys, es = [], [], []
    for T in Ts:
        ev = load("gru_T%d" % T, "evaluation.json")
        if ev:
            lo, hi = ev["top-1 accuracy"][1], ev["top-1 accuracy"][2]
            xs.append(T)
            ys.append(ev["top-1 accuracy"][0])
            es.append((hi - lo) / 2)
    if xs:
        ax.errorbar(xs, ys, yerr=es, color=GREEN, lw=2, marker="o", ms=6,
                    capsize=3, zorder=4, label="+ temporal (GRU)")
        ax.annotate("+ temporal (GRU)", (xs[-1], ys[-1]), xytext=(6, 0),
                    textcoords="offset points", fontsize=8.5, color=GREEN,
                    va="center")

    # The agent-level model, at the one window length where it was trained.
    ev = load("transformer_inv_T64", "evaluation.json")
    if ev:
        ax.plot([64], [ev["top-1 accuracy"][0]], marker="D", ms=9,
                color=ORANGE, zorder=5, ls="none",
                label="+ per-agent (Transformer)")

    ax.axhline(CHANCE, color=MUTED, ls=":", lw=1)
    ax.text(16, CHANCE + 0.012, "chance", fontsize=8, color=MUTED)
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ts)
    ax.set_xticklabels(["%d\n(%.0fs)" % (T, T * dt) for T in Ts], fontsize=8.5)
    ax.set_xlabel("Observation window: kick events (seconds)", fontsize=10,
                  color=INK)
    ax.set_ylabel("Exact-rule accuracy", fontsize=10, color=INK)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(14, 900)
    ax.set_title("Identifiability saturates with observation length\n"
                 "the informative regime is short windows",
                 fontsize=11, color=INK, loc="left", pad=12)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_window.pdf"))
    fig.savefig(os.path.join(FIG, "fig2_window.png"), dpi=200)
    plt.close(fig)
    print("  fig2_window")


# ----------------------------------------------------------------------
def fig3_conformal():
    """RAPS coverage and set size. RQ4."""
    # T=16 is where the model is genuinely uncertain and the guarantee is
    # actually tested; T=64 is where it has saturated. Showing both is the
    # point: the ceiling effect at T=64 is a property of an easy task, not
    # slack in the method.
    tags = [("gru_T16", "T=16 (7 s), acc 0.77", BLUE),
            ("gru_T64", "T=64 (28 s), acc 0.93", GREEN),
            ("transformer_inv_T64", "T=64, Transformer", ORANGE)]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))

    for tag, label, color in tags:
        c = load(tag, "conformal.json")
        if not c:
            continue
        alphas = sorted(float(a) for a in c["alphas"])
        cov = [c["alphas"]["%.2f" % a]["empirical_coverage"] for a in alphas]
        lo = [c["alphas"]["%.2f" % a]["coverage_ci95"][0] for a in alphas]
        hi = [c["alphas"]["%.2f" % a]["coverage_ci95"][1] for a in alphas]
        sz = [c["alphas"]["%.2f" % a]["mean_set_size"] for a in alphas]

        targ = [1 - a for a in alphas]
        axes[0].errorbar(targ, cov, yerr=[np.array(cov) - lo, np.array(hi) - cov],
                         color=color, lw=2, marker="o", ms=6, capsize=3,
                         label=label, zorder=3)
        axes[1].plot(targ, sz, color=color, lw=2, marker="o", ms=6,
                     label=label, zorder=3)

    lims = [0.78, 0.98]
    axes[0].plot(lims, lims, color=MUTED, ls="--", lw=1.2, zorder=2)
    axes[0].text(0.80, 0.815, "nominal", fontsize=8.5, color=MUTED, rotation=38)
    axes[0].set_xlabel("Target coverage $1-\\alpha$", fontsize=10, color=INK)
    axes[0].set_ylabel("Empirical coverage", fontsize=10, color=INK)
    axes[0].set_title("Coverage is valid at every level", fontsize=10.5,
                      color=INK, loc="left")
    axes[0].set_xlim(*lims)
    axes[0].set_ylim(0.78, 1.0)

    axes[1].axhline(1.0, color=MUTED, ls=":", lw=1)
    axes[1].text(0.955, 1.02, "singleton", fontsize=8.5, color=MUTED)
    axes[1].set_xlabel("Target coverage $1-\\alpha$", fontsize=10, color=INK)
    axes[1].set_ylabel("Mean prediction-set size", fontsize=10, color=INK)
    axes[1].set_title("Sets stay small: 1-2 rules out of 11", fontsize=10.5,
                      color=INK, loc="left")
    axes[1].set_xlim(*lims)
    axes[1].legend(frameon=False, fontsize=8.5, loc="upper left")

    for ax in axes:
        style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_conformal.pdf"))
    fig.savefig(os.path.join(FIG, "fig3_conformal.png"), dpi=200)
    plt.close(fig)
    print("  fig3_conformal")


# ----------------------------------------------------------------------
def fig4_confusion(tag="transformer_inv_T64"):
    """Confusion matrix. Sequential single hue, light to dark, per the rule
    that magnitude never gets a rainbow."""
    probs, labels = pooled_probs(tag)
    pred = probs.argmax(1)
    cm = np.zeros((11, 11))
    for t, p in zip(labels, pred):
        cm[t, p] += 1
    cm = cm / cm.sum(axis=1, keepdims=True)  # row-normalized: recall

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(11))
    ax.set_yticks(range(11))
    ax.set_xticklabels(RULE_NAMES, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticklabels(RULE_NAMES, fontsize=8.5)
    ax.set_xlabel("Predicted rule", fontsize=10, color=INK)
    ax.set_ylabel("True rule", fontsize=10, color=INK)

    for i in range(11):
        for j in range(11):
            if cm[i, j] >= 0.01:
                ax.text(j, i, "%.0f" % (100 * cm[i, j]), ha="center",
                        va="center", fontsize=7.5,
                        color="white" if cm[i, j] > 0.5 else INK)

    ax.set_title("Errors concentrate among the high-$k$ rules\n"
                 "%s, row-normalized (%%)" % tag.replace("_", " "),
                 fontsize=11, color=INK, loc="left", pad=12)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=MUTED, length=0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_confusion.pdf"))
    fig.savefig(os.path.join(FIG, "fig4_confusion.png"), dpi=200)
    plt.close(fig)
    print("  fig4_confusion")


# ----------------------------------------------------------------------
def fig5_errors():
    """Where the errors go. RQ2, and it overturns the stated hypothesis."""
    tags = [("transformer_inv_T64", "Transformer\n(agent tokens)"),
            ("gru_T64", "GRU\n(group stats)")]

    FAMILY_OF = np.array([0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4])
    K_OF = np.array([0, 1, 2, 3, 1, 2, 3, 1, 2, 3, 4])

    labels_, fam_, k_, both_ = [], [], [], []
    for tag, name in tags:
        if not os.path.isdir(os.path.join(RES, tag)):
            continue
        probs, y = pooled_probs(tag)
        p = probs.argmax(1)
        err = p != y
        n = err.sum()
        fam = ((FAMILY_OF[p] == FAMILY_OF[y]) & err).sum() / n
        kk = ((K_OF[p] == K_OF[y]) & err).sum() / n
        labels_.append(name)
        fam_.append(fam)
        k_.append(kk)
        both_.append(1 - fam - kk)

    x = np.arange(len(labels_))
    w = 0.26
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for i, (vals, lab, c) in enumerate([
            (fam_, "Right family, wrong $k$", BLUE),
            (k_, "Right $k$, wrong family", ORANGE),
            (both_, "Both wrong", MUTED)]):
        pos = x + (i - 1) * w
        ax.bar(pos, vals, width=w * 0.92, color=c, edgecolor="white", lw=2,
               label=lab, zorder=3)
        for xi, v in zip(pos, vals):
            ax.text(xi, v + 0.012, "%.0f%%" % (100 * v), ha="center",
                    fontsize=8.5, color=INK)

    ax.set_xticks(x)
    ax.set_xticklabels(labels_, fontsize=9)
    ax.set_ylabel("Share of errors", fontsize=10, color=INK)
    ax.set_ylim(0, 0.62)
    ax.set_title("How many neighbors is easier to recover than which neighbors\n"
                 "errors preserve $k$ far more often than the strategy family",
                 fontsize=11, color=INK, loc="left", pad=12)
    ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper center")
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig5_errors.pdf"))
    fig.savefig(os.path.join(FIG, "fig5_errors.png"), dpi=200)
    plt.close(fig)
    print("  fig5_errors")


def fig6_equifinality():
    """Equifinality decays as observation accumulates. THE central measurement.

    Panel A: mean prediction-set size vs observation length, one line per alpha.
    Panel B: the set-size distribution at alpha=0.10, which makes the abstract
    number concrete -- at 7 seconds, two thirds of observations cannot resolve
    the mechanism to a single rule.

    Set size, not accuracy, is the quantity of interest: it is a calibrated,
    coverage-guaranteed count of how many rules remain live given the data.
    """
    Ts = [16, 32, 64, 128, 256, 512]
    dt = 0.434
    alphas = [("0.05", "95% confidence", BLUE),
              ("0.10", "90% confidence", ORANGE),
              ("0.20", "80% confidence", GREEN)]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1))

    # ---- Panel A: mean |C| vs T -------------------------------------
    ax = axes[0]
    for a, label, color in alphas:
        xs, ys = [], []
        for T in Ts:
            c = load("gru_T%d" % T, "conformal.json")
            if c and a in c["alphas"]:
                xs.append(T)
                ys.append(c["alphas"][a]["mean_set_size"])
        if not xs:
            continue
        ax.plot(xs, ys, color=color, lw=2, marker="o", ms=6, zorder=3,
                label=label)
        # Direct-label the first point only. The three lines converge on the
        # right, so end-labels would collide; the legend carries identity there.
        ax.annotate("%.2f" % ys[0], (xs[0], ys[0]), xytext=(8, 4),
                    textcoords="offset points", fontsize=9, color=color,
                    ha="left")

    ax.axhline(1.0, color=MUTED, ls=":", lw=1, zorder=2)
    ax.text(500, 1.06, "singleton", fontsize=8.5, color=MUTED, ha="right")
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ts)
    ax.set_xticklabels(["%d\n(%.0fs)" % (T, T * dt) for T in Ts], fontsize=8.5)
    ax.set_xlim(14, 640)
    ax.set_ylim(0.9, 3.6)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_xlabel("Observation window: kick events (seconds)", fontsize=10,
                  color=INK)
    ax.set_ylabel("Mean prediction-set size  |C|", fontsize=10, color=INK)
    ax.set_title("Equifinality decays with observation", fontsize=11,
                 color=INK, loc="left", pad=10)
    style(ax)

    # ---- Panel B: set-size distribution at alpha = 0.10 ---------------
    ax = axes[1]
    buckets = [("1", BLUE), ("2", ORANGE), ("3", GREEN), (">=4", MUTED)]
    bottom = np.zeros(len(Ts))
    shares = {k: [] for k, _ in buckets}

    for T in Ts:
        c = load("gru_T%d" % T, "conformal.json")
        h = c["alphas"]["0.10"]["set_size_hist"] if c else {}
        tot = sum(h.values()) or 1
        for k, _ in buckets:
            if k == ">=4":
                v = sum(n for s, n in h.items() if int(s) >= 4)
            else:
                v = h.get(k, 0)
            shares[k].append(100.0 * v / tot)

    x = np.arange(len(Ts))
    for k, color in buckets:
        vals = np.array(shares[k])
        ax.bar(x, vals, bottom=bottom, width=0.68, color=color,
               edgecolor="white", lw=2, zorder=3, label="|C| = %s" % k)
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            if v >= 7:
                ax.text(xi, b + v / 2, "%.0f%%" % v, ha="center", va="center",
                        fontsize=8.5,
                        color="white" if color != MUTED else "white")
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(["%d\n(%.0fs)" % (T, T * dt) for T in Ts], fontsize=8.5)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Observation window: kick events (seconds)", fontsize=10,
                  color=INK)
    ax.set_ylabel("Share of observations (%)", fontsize=10, color=INK)
    ax.set_title("How many rules remain live, at 90% confidence",
                 fontsize=11, color=INK, loc="left", pad=10)
    ax.legend(frameon=False, fontsize=8.5, ncol=4, loc="lower center",
              bbox_to_anchor=(0.5, -0.30))
    style(ax)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig6_equifinality.pdf"))
    fig.savefig(os.path.join(FIG, "fig6_equifinality.png"), dpi=200)
    plt.close(fig)
    print("  fig6_equifinality")


def main():
    os.makedirs(FIG, exist_ok=True)
    plt.rcParams["font.family"] = "DejaVu Sans"
    base = json.load(open(os.path.join(RES, "baselines.json")))

    print("Writing figures to %s" % FIG)
    fig1_ladder(base)
    fig2_window(base)
    fig3_conformal()
    fig4_confusion()
    fig5_errors()
    fig6_equifinality()
    print("done")


if __name__ == "__main__":
    main()
