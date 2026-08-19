"""Is the RQ1 asymmetry driven by one or two stimuli?

    python robustness.py

No new compute -- reads data/rq1_acts.npz.

At n=30 a single extreme pair can carry a p<0.01 result on its own. Three
checks:

  1. Leave-one-out: recompute the test 30 times, dropping one pair each time.
     If any single drop kills significance, the effect is that pair.
  2. Rank-based test: sign test on the paired differences, which ignores
     magnitude entirely. If the sign test agrees, the effect is not an
     artifact of a few large values.
  3. Trimmed mean: drop the top and bottom 10% and recompute.

Report all three in the writeup whatever they say. An effect that survives is
much more credible; one that does not, you needed to know.
"""

from collections import defaultdict
from math import comb, erf, sqrt

import numpy as np

from extract import load_extracted
from probe import make_probe

LAYER = 27
C = 0.01


def paired_t(x):
    n = len(x)
    if n < 2 or x.std(ddof=1) == 0:
        return 0.0, 1.0
    t = x.mean() / (x.std(ddof=1) / np.sqrt(n))
    p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
    return t, p


def sign_test(x):
    """Exact two-sided binomial test on the signs. Magnitude-free."""
    pos = int((x > 0).sum())
    n = int((x != 0).sum())
    if n == 0:
        return 0, 1.0
    k = max(pos, n - pos)
    tail = sum(comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    return pos, min(1.0, 2 * tail)


def trimmed_mean(x, frac=0.1):
    k = int(len(x) * frac)
    s = np.sort(x)
    return s[k:len(s) - k].mean() if k else x.mean()


def load_effects():
    acts, meta = load_extracted("data/probe_train_acts.npz")
    labels = np.array([m["label"] for m in meta])
    probe = make_probe(C)
    probe.fit(acts[:, LAYER, :], labels)

    ea, em = load_extracted("data/rq1_acts.npz")
    margins = probe.decision_function(ea[:, LAYER, :])

    by_pair = defaultdict(dict)
    concepts = {}
    for row, v in zip(em, margins):
        by_pair[row["pair"]][row["condition"]] = v
        concepts[row["pair"]] = row["concept"]

    pairs = sorted(p for p, d in by_pair.items()
                   if {"baseline", "pos", "neg"} <= d.keys())
    up = np.array([by_pair[p]["pos"] - by_pair[p]["baseline"] for p in pairs])
    down = np.array([by_pair[p]["neg"] - by_pair[p]["baseline"] for p in pairs])
    return pairs, up, down, concepts


def main():
    pairs, up, down, concepts = load_effects()
    asym = np.abs(up) - np.abs(down)
    n = len(asym)

    t, p = paired_t(asym)
    print(f"FULL SAMPLE (n={n})")
    print(f"  asymmetry {asym.mean():+.3f}  t = {t:+.2f}  p = {p:.4f}\n")

    print("1. LEAVE-ONE-OUT")
    worst_p, worst_i = 0.0, None
    for i in range(n):
        sub = np.delete(asym, i)
        _, pi = paired_t(sub)
        if pi > worst_p:
            worst_p, worst_i = pi, i
    print(f"  worst p after dropping one pair: {worst_p:.4f} "
          f"(pair {pairs[worst_i]}, {concepts[pairs[worst_i]]})")
    if worst_p > 0.05:
        print("  ^ significance depends on a single stimulus. Not an effect.")
    else:
        print("  ^ survives every single-pair deletion")

    order = np.argsort(-np.abs(asym))
    print("\n  largest 5 contributions:")
    for i in order[:5]:
        print(f"    {asym[i]:+7.3f}  up {up[i]:+6.3f}  down {down[i]:+6.3f}  "
              f"{concepts[pairs[i]]}")

    print("\n2. SIGN TEST (magnitude-free)")
    pos, sp = sign_test(asym)
    print(f"  {pos}/{n} pairs positive, exact p = {sp:.4f}")
    if sp < 0.05:
        print("  ^ agrees with the t-test; effect is not carried by outliers")
    else:
        print("  ^ disagrees with the t-test; a few large values are driving it")

    print("\n3. TRIMMED MEAN (10% each tail)")
    tm = trimmed_mean(asym)
    print(f"  full mean {asym.mean():+.3f}   trimmed {tm:+.3f}   "
          f"median {np.median(asym):+.3f}")
    if abs(tm) < abs(asym.mean()) * 0.5:
        print("  ^ trimming halves the effect; distribution is skewed")

    print("\n4. DIRECTION CHECK -- what is actually happening")
    print(f"  pos moves probe {up.mean():+.3f}  "
          f"({(up > 0).sum()}/{n} pairs upward)")
    print(f"  neg moves probe {down.mean():+.3f}  "
          f"({(down > 0).sum()}/{n} pairs upward)")
    if down.mean() > 0:
        print("\n  Both signals move the probe UP. This is not an asymmetry")
        print("  between an upgrade and a downgrade -- it is two upgrades of")
        print("  different size. The baseline lacks technical content, so the")
        print("  dominant contrast is technicality, not correctness.")
        print("  Run neutral_baseline.py before interpreting further.")


if __name__ == "__main__":
    main()
