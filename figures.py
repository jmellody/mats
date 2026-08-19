"""Figures for the dose-response result.

    python figures.py

Writes four PNGs to figures/. No new compute -- reads dose_acts.npz.

  fig1_trajectories.png   the headline: both arms, mean +/- SE, per-unit faint
  fig2_slopes.png         slope distributions and the paired asymmetry
  fig3_saturation.png     step size by dose position, the strongest result
  fig4_spread.png         per-unit variation, honest about heterogeneity

Design notes: error bars are standard errors of the mean across units, which
is the right unit of analysis since slopes are estimated per unit. Paired
comparisons are drawn as connected points, not as separate bars, because the
design is within-unit and bars would misrepresent the test that was run.
"""

import json
import os
from collections import defaultdict
from math import sqrt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from acts import load_extracted
from probe import make_probe

LAYER = 27
C = 0.01
MAX_DOSE = 3
POS_C, NEG_C = "#2a6fb0", "#c0453a"
GREY = "#8a8a85"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.facecolor": "white",
})


def load():
    tr, trm = load_extracted("data/probe_train_acts.npz")
    labels = np.array([m["label"] for m in trm])
    probe = make_probe(C)
    probe.fit(tr[:, LAYER, :], labels)

    acts, meta = load_extracted("data/dose_acts.npz")
    margins = probe.decision_function(acts[:, LAYER, :])

    vals = defaultdict(lambda: defaultdict(dict))
    for row, v in zip(meta, margins):
        if row["dose"] == 0:
            vals[row["pair"]]["pos"][0] = v
            vals[row["pair"]]["neg"][0] = v
        else:
            vals[row["pair"]][row["arm"]][row["dose"]] = v

    doses = np.arange(MAX_DOSE + 1)
    curves = {"pos": [], "neg": []}
    slopes = {"pos": [], "neg": []}
    for _, arms in sorted(vals.items()):
        if not all(set(doses) <= set(arms[a]) for a in ("pos", "neg")):
            continue
        for a in ("pos", "neg"):
            y = np.array([arms[a][k] for k in doses])
            curves[a].append(y)
            slopes[a].append(np.polyfit(doses, y, 1)[0])
    return (doses,
            {a: np.stack(curves[a]) for a in curves},
            {a: np.array(slopes[a]) for a in slopes})


def sem(x, axis=0):
    return x.std(axis=axis, ddof=1) / sqrt(x.shape[axis])


def fig1(doses, curves, out):
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    n = curves["pos"].shape[0]
    for arm, col, lab in [("pos", POS_C, "Correct statements"),
                          ("neg", NEG_C, "Incorrect statements")]:
        Y = curves[arm]
        for row in Y:
            ax.plot(doses, row, color=col, alpha=0.05, lw=0.7, zorder=1)
        m, e = Y.mean(0), sem(Y)
        ax.fill_between(doses, m - e, m + e, color=col, alpha=0.2, zorder=2)
        ax.plot(doses, m, color=col, lw=2.4, marker="o", ms=6,
                label=lab, zorder=3)

    ax.set_xticks(doses)
    ax.set_xlabel("Number of signals in conversation")
    ax.set_ylabel("Probe margin  (higher = model reads user as expert)")
    ax.set_title("User-expertise representation by evidence dose",
                 fontsize=11.5, pad=10)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.annotate(f"n = {n} units\nshaded = ±1 SEM", xy=(0.98, 0.04),
                xycoords="axes fraction", ha="right", va="bottom",
                fontsize=8, color=GREY)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig2(slopes, out):
    bp, bn = slopes["pos"], slopes["neg"]
    n = len(bp)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.0))

    ax = axes[0]
    bins = np.linspace(min(bp.min(), bn.min()), max(bp.max(), bn.max()), 26)
    ax.hist(bp, bins=bins, color=POS_C, alpha=0.55, label="Correct")
    ax.hist(bn, bins=bins, color=NEG_C, alpha=0.55, label="Incorrect")
    ax.axvline(0, color="black", lw=1, ls="--", alpha=0.6)
    ax.axvline(bp.mean(), color=POS_C, lw=2)
    ax.axvline(bn.mean(), color=NEG_C, lw=2)
    ax.set_xlabel("Slope (margin change per signal)")
    ax.set_ylabel("Units")
    ax.set_title("Slopes move in opposite directions", fontsize=10.5)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    asym = np.abs(bp) - np.abs(bn)
    rng = np.random.default_rng(0)
    x = 1 + rng.normal(0, 0.055, n)
    ax.scatter(x, asym, s=16, color=GREY, alpha=0.55, zorder=2)
    m = asym.mean()
    e = 1.96 * asym.std(ddof=1) / sqrt(n)
    ax.errorbar([1], [m], yerr=[e], fmt="o", color="black", ms=8,
                capsize=6, lw=2, zorder=3)
    ax.axhline(0, color="black", lw=1, ls="--", alpha=0.6)
    ax.set_xlim(0.75, 1.25)
    ax.set_xticks([])
    ax.set_ylabel("|positive slope| − |negative slope|")
    ax.set_title("Asymmetry: null", fontsize=10.5)
    ax.annotate(f"mean {m:+.3f}\n95% CI [{m-e:+.3f}, {m+e:+.3f}]\n"
                f"{int((asym > 0).sum())}/{n} units positive",
                xy=(0.97, 0.97), xycoords="axes fraction",
                ha="right", va="top", fontsize=8.5, color="black")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig3(curves, out):
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    width = 0.36
    pos_x = np.arange(MAX_DOSE)
    for k, (arm, col, lab) in enumerate([
            ("pos", POS_C, "Correct"), ("neg", NEG_C, "Incorrect")]):
        Y = curves[arm]
        steps = np.diff(Y, axis=1)
        m, e = steps.mean(0), sem(steps)
        ax.bar(pos_x + (k - 0.5) * width, m, width, yerr=e, capsize=4,
               color=col, alpha=0.85, label=lab,
               error_kw={"lw": 1.2, "ecolor": "#444"})
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(pos_x)
    ax.set_xticklabels([f"{i+1}{'st' if i==0 else 'nd' if i==1 else 'rd'}"
                        for i in pos_x])
    ax.set_xlabel("Which signal in the sequence")
    ax.set_ylabel("Change in probe margin")
    ax.set_title("The first signal does nearly all the work",
                 fontsize=11.5, pad=10)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig4(slopes, out):
    bp, bn = slopes["pos"], slopes["neg"]
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    lim = max(np.abs(np.r_[bp, bn])) * 1.08
    ax.axhline(0, color=GREY, lw=0.8)
    ax.axvline(0, color=GREY, lw=0.8)
    ax.plot([-lim, lim], [lim, -lim], color="black", lw=0.9, ls="--",
            alpha=0.5, zorder=1)
    ax.scatter(bp, bn, s=22, color="#4a4a48", alpha=0.6, zorder=2)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Positive-arm slope")
    ax.set_ylabel("Negative-arm slope")
    ax.set_title("Per-unit slopes\n(dashed = perfectly mirrored arms)",
                 fontsize=10.5)
    q = int(((bp > 0) & (bn < 0)).sum())
    ax.annotate(f"{q}/{len(bp)} units in the\nexpected quadrant",
                xy=(0.03, 0.03), xycoords="axes fraction",
                fontsize=8.5, color=GREY)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    doses, curves, slopes = load()
    fig1(doses, curves, "figures/fig1_trajectories.png")
    fig2(slopes, "figures/fig2_slopes.png")
    fig3(curves, "figures/fig3_saturation.png")
    fig4(slopes, "figures/fig4_spread.png")
    print(f"n = {curves['pos'].shape[0]} units")
    for f in sorted(os.listdir("figures")):
        print(f"  figures/{f}")
