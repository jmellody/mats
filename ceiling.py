"""Is the positive arm hitting a ceiling?

    PROBE_LAYER=26 python ceiling.py

No new compute -- reads saved activations.

THE PROBLEM

On Qwen3.5-4B the dose-0 (filler-only) condition sits at margin +2.86, and the
positive arm barely moves (+0.13, p=0.078) while the negative arm moves hugely
(-1.35, p<0.0001). That asymmetry is the opposite of what Qwen2.5-3B showed.

Two explanations, and they are very different findings:

  CEILING   filler already reads as near-maximally expert, so positive signals
            have no headroom while negative ones have the whole range. The
            asymmetry is an artifact of where the baseline happens to land.

  REAL      there is plenty of headroom above +2.86 and the positive arm simply
            does not use it. Negative evidence genuinely dominates on this
            model.

This distinguishes them by comparing the experiment's margins against the range
the probe actually produces on its own training data, and by checking whether
per-unit positive slopes shrink as their starting point rises.
"""

from collections import defaultdict

import numpy as np

import config
from acts import load
from probe import make_probe

MAX_DOSE = 3


def main():
    tr, trm = load(config.data_path("probe_train_acts.npz"))
    labels = np.array([r["label"] for r in trm])
    probe = make_probe(config.C)
    probe.fit(tr[:, config.LAYER, :], labels)
    train = probe.decision_function(tr[:, config.LAYER, :])

    print(f"model {config.MODEL}   layer {config.LAYER}\n")
    print("=" * 66)
    print("1. WHERE DOES THE PROBE'S OWN SCALE TOP OUT?")
    print("=" * 66)
    pos_tr, neg_tr = train[labels == 1], train[labels == 0]
    print(f"  training margins   min {train.min():+.2f}   "
          f"max {train.max():+.2f}")
    print(f"  correct-half       mean {pos_tr.mean():+.2f}   "
          f"95th {np.percentile(pos_tr, 95):+.2f}   max {pos_tr.max():+.2f}")
    print(f"  incorrect-half     mean {neg_tr.mean():+.2f}   "
          f"5th  {np.percentile(neg_tr, 5):+.2f}   min {neg_tr.min():+.2f}")

    a, meta = load(config.data_path("dose_acts.npz"))
    m = probe.decision_function(a[:, config.LAYER, :])

    vals = defaultdict(lambda: defaultdict(dict))
    for row, v in zip(meta, m):
        if row["dose"] == 0:
            vals[row["pair"]]["pos"][0] = v
            vals[row["pair"]]["neg"][0] = v
        else:
            vals[row["pair"]][row["arm"]][row["dose"]] = v

    doses = np.arange(MAX_DOSE + 1)
    curves = {"pos": [], "neg": []}
    for _, arms in sorted(vals.items()):
        if not all(set(doses) <= set(arms[k]) for k in ("pos", "neg")):
            continue
        for k in ("pos", "neg"):
            curves[k].append(np.array([arms[k][j] for j in doses]))
    P, N = np.stack(curves["pos"]), np.stack(curves["neg"])
    base = P[:, 0]

    print("\n" + "=" * 66)
    print("2. HOW MUCH HEADROOM DID THE POSITIVE ARM HAVE?")
    print("=" * 66)
    top = np.percentile(pos_tr, 95)
    head = top - base.mean()
    print(f"  dose-0 baseline          {base.mean():+.2f}")
    print(f"  95th pct of correct-half {top:+.2f}")
    print(f"  headroom above baseline  {head:+.2f}")
    print(f"  observed pos movement    {(P[:, 3] - P[:, 0]).mean():+.2f}"
          f"  over 3 signals")
    frac = np.mean(base > np.percentile(pos_tr, 90))
    print(f"  {frac:.0%} of units start above the 90th percentile of the")
    print(f"  probe's own correct-half distribution")

    if head < 1.0:
        print("\n  CEILING. The baseline already sits at the top of the range")
        print("  the probe produces on genuinely-expert text. The positive")
        print("  arm had nowhere to go, so the asymmetry is an artifact of")
        print("  where filler lands, not a fact about the model.")
    elif head > 3.0:
        print("\n  NO CEILING. There is substantial headroom the positive arm")
        print("  did not use. Negative evidence genuinely dominates here.")
    else:
        print("\n  AMBIGUOUS headroom. Lean on section 3.")

    print("\n" + "=" * 66)
    print("3. DO HIGH-STARTING UNITS MOVE LESS? (the ceiling signature)")
    print("=" * 66)
    print("  Under a ceiling, units starting higher should show smaller")
    print("  positive slopes. Under a real effect, starting point should")
    print("  not matter.")
    for k, Y, lab in [("pos", P, "positive"), ("neg", N, "negative")]:
        slope = np.array([np.polyfit(doses, y, 1)[0] for y in Y])
        r = np.corrcoef(base, slope)[0, 1]
        n = len(base)
        t = r * np.sqrt((n - 2) / max(1 - r ** 2, 1e-9))
        print(f"    {lab:9} corr(baseline, slope) = {r:+.3f}  t = {t:+.2f}")
    print("\n  Strongly negative for the positive arm = ceiling.")
    print("  Near zero = the positive arm is genuinely flat.")

    print("\n" + "=" * 66)
    print("4. SPLIT BY BASELINE")
    print("=" * 66)
    med = np.median(base)
    lo, hi = base <= med, base > med
    print(f"  {'group':16} {'baseline':>9} {'pos move':>10} {'neg move':>10}")
    for name, mask in [("low baseline", lo), ("high baseline", hi)]:
        print(f"  {name:16} {base[mask].mean():>+9.2f} "
              f"{(P[mask, 3] - P[mask, 0]).mean():>+10.2f} "
              f"{(N[mask, 3] - N[mask, 0]).mean():>+10.2f}")
    lo_move = (P[lo, 3] - P[lo, 0]).mean()
    hi_move = (P[hi, 3] - P[hi, 0]).mean()
    print(f"\n  positive movement in low-baseline units:  {lo_move:+.2f}")
    print(f"  positive movement in high-baseline units: {hi_move:+.2f}")
    if lo_move > 0.3 and lo_move > hi_move * 2:
        print("\n  The positive arm DOES move where there is room. The flat")
        print("  overall slope is a ceiling, not an absence of effect.")
    elif abs(lo_move) < 0.3:
        print("\n  The positive arm is flat even where there is room.")
        print("  Not a ceiling -- positive evidence genuinely does little.")

    print("\n" + "=" * 66)
    print("5. WHAT TO DO ABOUT IT")
    print("=" * 66)
    print("  If this is a ceiling, the fix is a lower-starting baseline:")
    print("  filler that reads as less expert, so the positive arm has")
    print("  somewhere to go. That is a stimulus change, not an analysis")
    print("  one -- and it means the 3B/4B difference may be about where")
    print("  each model places filler rather than about how they update.")
    print("  Report the comparison with that caveat either way.")


if __name__ == "__main__":
    main()
