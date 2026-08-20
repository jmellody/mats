"""Figures.

    python figures.py

Writes PNGs to figures/<MODEL_TAG>/. No new compute -- reads saved activations.

  fig1_trajectories.png   dose: both arms, mean +/- SEM, per-unit faint lines
  fig2_slopes.png         dose: slope distributions and the paired asymmetry
  fig3_saturation.png     dose: step size by position
  fig4_spread.png         dose: per-unit slope scatter
  fig5_retention.png      contradiction: the four conditions

Error bars are SEM across units, which is the right unit of analysis since
slopes are estimated per unit. Paired comparisons are drawn as points with a CI
rather than as bars, because the design is within-unit and bars would
misrepresent the test that was run.

The contradiction figure is skipped if contra_acts.npz is absent, so this runs
after the dose stage alone.
"""

import os
from collections import defaultdict
from math import sqrt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
from acts import load as load_acts
from probe import make_probe

MAX_DOSE = 3
POS_C, NEG_C = "#2a6fb0", "#c0453a"
GREY = "#8a8a85"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.facecolor": "white",
})


def fig_dir():
    d = os.path.join("figures", config.MODEL_TAG)
    os.makedirs(d, exist_ok=True)
    return d


def fit_probe():
    tr, trm = load_acts(config.data_path("probe_train_acts.npz"))
    p = make_probe(config.C)
    p.fit(tr[:, config.LAYER, :], np.array([m["label"] for m in trm]))
    return p


def sem(x, axis=0):
    return x.std(axis=axis, ddof=1) / sqrt(x.shape[axis])


def load_dose(probe):
    a, meta = load_acts(config.data_path("dose_acts.npz"))
    margins = probe.decision_function(a[:, config.LAYER, :])

    vals = defaultdict(lambda: defaultdict(dict))
    for row, v in zip(meta, margins):
        if row["dose"] == 0:
            vals[row["pair"]]["pos"][0] = v
            vals[row["pair"]]["neg"][0] = v
        else:
            vals[row["pair"]][row["arm"]][row["dose"]] = v

    doses = np.arange(MAX_DOSE + 1)
    curves, slopes = {"pos": [], "neg": []}, {"pos": [], "neg": []}
    for _, arms in sorted(vals.items()):
        if not all(set(doses) <= set(arms[k]) for k in ("pos", "neg")):
            continue
        for k in ("pos", "neg"):
            y = np.array([arms[k][j] for j in doses])
            curves[k].append(y)
            slopes[k].append(np.polyfit(doses, y, 1)[0])
    return (doses,
            {k: np.stack(v) for k, v in curves.items()},
            {k: np.array(v) for k, v in slopes.items()})


def load_contra(probe):
    a, meta = load_acts(config.data_path("contra_acts.npz"))
    margins = probe.decision_function(a[:, config.LAYER, :])
    vals = defaultdict(dict)
    for row, v in zip(meta, margins):
        vals[row["pair"]][row["condition"]] = v
    conds = ["ppp", "ppn", "nnp", "nnn"]
    keep = sorted(p for p, c in vals.items() if set(conds) <= c.keys())
    return {k: np.array([vals[p][k] for p in keep]) for k in conds}, len(keep)


def stamp(ax):
    ax.annotate(f"{config.MODEL_TAG}  L{config.LAYER}", xy=(0.995, 1.02),
                xycoords="axes fraction", ha="right", va="bottom",
                fontsize=7, color=GREY)


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
        ax.plot(doses, m, color=col, lw=2.4, marker="o", ms=6, label=lab,
                zorder=3)
    ax.set_xticks(doses)
    ax.set_xlabel("Number of signals in conversation")
    ax.set_ylabel("Probe margin  (higher = model reads user as expert)")
    ax.set_title("User-expertise representation by evidence dose",
                 fontsize=11.5, pad=10)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.annotate(f"n = {n} units\nshaded = ±1 SEM", xy=(0.98, 0.04),
                xycoords="axes fraction", ha="right", va="bottom",
                fontsize=8, color=GREY)
    stamp(ax)
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
    stamp(ax)

    ax = axes[1]
    asym = np.abs(bp) - np.abs(bn)
    rng = np.random.default_rng(0)
    ax.scatter(1 + rng.normal(0, 0.055, n), asym, s=16, color=GREY,
               alpha=0.55, zorder=2)
    m = asym.mean()
    e = 1.96 * asym.std(ddof=1) / sqrt(n)
    ax.errorbar([1], [m], yerr=[e], fmt="o", color="black", ms=8, capsize=6,
                lw=2, zorder=3)
    ax.axhline(0, color="black", lw=1, ls="--", alpha=0.6)
    ax.set_xlim(0.75, 1.25)
    ax.set_xticks([])
    ax.set_ylabel("|positive slope| − |negative slope|")
    sig = "null" if abs(m) < e else "significant"
    ax.set_title(f"Valence asymmetry: {sig}", fontsize=10.5)
    ax.annotate(f"mean {m:+.3f}\n95% CI [{m-e:+.3f}, {m+e:+.3f}]\n"
                f"{int((asym > 0).sum())}/{n} units positive",
                xy=(0.97, 0.97), xycoords="axes fraction", ha="right",
                va="top", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig3(curves, out):
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    width = 0.36
    xs = np.arange(MAX_DOSE)
    for k, (arm, col, lab) in enumerate([("pos", POS_C, "Correct"),
                                         ("neg", NEG_C, "Incorrect")]):
        steps = np.diff(curves[arm], axis=1)
        ax.bar(xs + (k - 0.5) * width, steps.mean(0), width,
               yerr=sem(steps), capsize=4, color=col, alpha=0.85, label=lab,
               error_kw={"lw": 1.2, "ecolor": "#444"})
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(["1st", "2nd", "3rd"][:MAX_DOSE])
    ax.set_xlabel("Which signal in the sequence")
    ax.set_ylabel("Change in probe margin")
    ax.set_title("The first signal does nearly all the work",
                 fontsize=11.5, pad=10)
    ax.legend(frameon=False, fontsize=9)
    stamp(ax)
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
                xy=(0.03, 0.03), xycoords="axes fraction", fontsize=8.5,
                color=GREY)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig5(c, n, out):
    """Contradiction: four conditions, with the two retention contrasts."""
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2),
                             gridspec_kw={"width_ratios": [1.25, 1]})

    ax = axes[0]
    order = ["ppp", "ppn", "nnp", "nnn"]
    labels = ["+ + +", "+ + −", "− − +", "− − −"]
    cols = [POS_C, "#7a94b8", "#c08a86", NEG_C]
    means = [c[k].mean() for k in order]
    errs = [sem(c[k]) for k in order]
    ax.bar(range(4), means, 0.62, yerr=errs, capsize=4, color=cols,
           alpha=0.9, error_kw={"lw": 1.2, "ecolor": "#444"})
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Signal sequence")
    ax.set_ylabel("Probe margin")
    ax.set_title("Conditions ending on the same valence differ by history",
                 fontsize=10.5, pad=10)
    ax.annotate(f"n = {n} units", xy=(0.02, 0.04), xycoords="axes fraction",
                ha="left", va="bottom", fontsize=8, color=GREY)

    ax = axes[1]
    contrasts = [
        ("ppn − nnn\n(both end −)", c["ppn"] - c["nnn"]),
        ("ppp − nnp\n(both end +)", c["ppp"] - c["nnp"]),
    ]
    rng = np.random.default_rng(0)
    for i, (lab, d) in enumerate(contrasts):
        ax.scatter(i + rng.normal(0, 0.05, len(d)), d, s=14, color=GREY,
                   alpha=0.5, zorder=2)
        m = d.mean()
        e = 1.96 * d.std(ddof=1) / sqrt(len(d))
        ax.errorbar([i], [m], yerr=[e], fmt="o", color="black", ms=8,
                    capsize=6, lw=2, zorder=3)
    ax.axhline(0, color="black", lw=1, ls="--", alpha=0.6)
    ax.set_xticks(range(len(contrasts)))
    ax.set_xticklabels([lab for lab, _ in contrasts], fontsize=9)
    ax.set_ylabel("Retention (margin difference)")
    ax.set_title("Earlier evidence survives contradiction", fontsize=10.5,
                 pad=10)
    stamp(ax)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


if __name__ == "__main__":
    d = fig_dir()
    probe = fit_probe()
    print(f"model {config.MODEL}   layer {config.LAYER}")

    doses, curves, slopes = load_dose(probe)
    fig1(doses, curves, f"{d}/fig1_trajectories.png")
    fig2(slopes, f"{d}/fig2_slopes.png")
    fig3(curves, f"{d}/fig3_saturation.png")
    fig4(slopes, f"{d}/fig4_spread.png")
    print(f"dose: n = {curves['pos'].shape[0]} units")

    try:
        c, n = load_contra(probe)
        fig5(c, n, f"{d}/fig5_retention.png")
        print(f"contradiction: n = {n} units")
    except FileNotFoundError:
        print("contradiction: no contra_acts.npz, skipping fig5")

    for f in sorted(os.listdir(d)):
        print(f"  {d}/{f}")