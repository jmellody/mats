"""Proper statistics for the RQ1 result.

    python final_stats.py

No new compute. Reads the saved activations.

WHAT THIS ADDS OVER neutral_baseline.py

1. DIRECT PAIRED TEST of pos vs neg. Each pair contributes both conditions on
   the same concept with the same wrapper, so subtracting within a pair cancels
   all pair-level variation. Neutral's own noise drops out entirely. This is
   the cleanest statement of the asymmetry.

2. CONFIDENCE INTERVALS on everything. "p = 0.88" is not a finding. The CI
   tells you what size of effect you have actually ruled out.

3. EQUIVALENCE TEST (TOST) on the null. A non-significant t-test is not
   evidence of no effect. TOST asks the right question: can you reject the
   presence of an effect larger than some bound? Only that licenses a claim
   like "negative evidence does not register".

4. BOOTSTRAP CIs, which do not assume normality. With n=30 and a skewed
   effect distribution the t-interval may be optimistic.
"""

from collections import defaultdict
from math import erf, sqrt

import numpy as np

from extract import load_extracted
from probe import make_probe

LAYER = 27
C = 0.01
BOOT = 20000


def norm_p(t):
    return 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))


def paired(x, label, bound=None):
    """Paired t-test on within-pair differences, with CIs."""
    n = len(x)
    m, sd = x.mean(), x.std(ddof=1)
    sem = sd / sqrt(n)
    t = m / sem if sem else 0.0
    p = norm_p(t)
    ci = 1.96 * sem

    rng = np.random.default_rng(0)
    boot = np.array([rng.choice(x, n, replace=True).mean() for _ in range(BOOT)])
    blo, bhi = np.percentile(boot, [2.5, 97.5])

    d = m / sd if sd else 0.0
    print(f"\n{label}")
    print(f"  mean {m:+.3f}   sd {sd:.3f}   n {n}")
    print(f"  t({n-1}) = {t:+.2f}   p = {p:.4f}   Cohen's d = {d:+.2f}")
    print(f"  95% CI      [{m-ci:+.3f}, {m+ci:+.3f}]")
    print(f"  bootstrap   [{blo:+.3f}, {bhi:+.3f}]")
    print(f"  {(x > 0).sum()}/{n} pairs positive")

    if bound is not None:
        # TOST: two one-sided tests against +/- bound
        t_lo = (m + bound) / sem
        t_hi = (m - bound) / sem
        p_tost = max(norm_p(t_lo) / 2 if t_lo > 0 else 1.0,
                     norm_p(t_hi) / 2 if t_hi < 0 else 1.0)
        verdict = ("equivalent" if p_tost < 0.05
                   else "cannot rule out an effect this large")
        print(f"  TOST vs +/-{bound}: p = {p_tost:.4f} -> {verdict}")
    return m, sem


def main():
    acts, meta = load_extracted("data/probe_train_acts.npz")
    labels = np.array([m["label"] for m in meta])
    probe = make_probe(C)
    probe.fit(acts[:, LAYER, :], labels)
    train_margins = probe.decision_function(acts[:, LAYER, :])
    train_sd = train_margins.std()

    vals, concepts = defaultdict(dict), {}
    for f in ["data/rq1_acts.npz", "data/neutral_acts.npz"]:
        a, m = load_extracted(f)
        for row, v in zip(m, probe.decision_function(a[:, LAYER, :])):
            vals[row["pair"]][row["condition"]] = v
            concepts[row["pair"]] = row["concept"]

    need = {"baseline", "pos", "neg", "neutral"}
    pairs = sorted(p for p, d in vals.items() if need <= d.keys())
    g = lambda c: np.array([vals[p][c] for p in pairs])
    filler, neutral, pos, neg = g("baseline"), g("neutral"), g("pos"), g("neg")
    n = len(pairs)

    print("=" * 62)
    print(f"CONDITION MEANS (probe margin, layer {LAYER}, C={C})")
    print("=" * 62)
    print(f"{'condition':12} {'mean':>8} {'sd':>7}   scale reference:")
    print(f"{'filler':12} {filler.mean():>+8.3f} {filler.std():>7.3f}   "
          f"training margin sd = {train_sd:.2f}")
    print(f"{'neutral':12} {neutral.mean():>+8.3f} {neutral.std():>7.3f}")
    print(f"{'neg':12} {neg.mean():>+8.3f} {neg.std():>7.3f}")
    print(f"{'pos':12} {pos.mean():>+8.3f} {pos.std():>7.3f}")

    # a defensible equivalence bound: a quarter of the training-margin sd,
    # i.e. an effect too small to matter on the probe's own scale
    bound = round(0.25 * train_sd, 2)

    print("\n" + "=" * 62)
    print("1. THE DIRECT PAIRED TEST -- pos vs neg")
    print("=" * 62)
    print("   Within-pair difference. Same concept, same wrapper, so all")
    print("   pair-level variation cancels. This is the asymmetry claim.")
    paired(pos - neg, "pos - neg")

    print("\n" + "=" * 62)
    print("2. EACH SIGNAL AGAINST THE MATCHED NEUTRAL")
    print("=" * 62)
    paired(pos - neutral, "pos - neutral")
    paired(neg - neutral, "neg - neutral", bound=bound)

    print("\n" + "=" * 62)
    print("3. IS THE ASYMMETRY ITSELF SIGNIFICANT?")
    print("=" * 62)
    print("   |up| vs |down|: does one direction move the probe further?")
    paired(np.abs(pos - neutral) - np.abs(neg - neutral), "|up| - |down|")

    print("\n" + "=" * 62)
    print("4. TECHNICALITY CONFOUND, QUANTIFIED")
    print("=" * 62)
    print("   How much of the original result was technical content rather")
    print("   than correctness?")
    paired(neutral - filler, "neutral - filler")

    print("\n" + "=" * 62)
    print("HOW TO WRITE THIS UP")
    print("=" * 62)
    up, down = (pos - neutral).mean(), (neg - neutral).mean()
    ci_d = 1.96 * (neg - neutral).std(ddof=1) / sqrt(n)
    print(f"  Correct statements shift the representation toward 'expert'")
    print(f"  ({up:+.3f}). Misconceptions produce no detectable shift")
    print(f"  ({down:+.3f}, 95% CI [{down-ci_d:+.3f}, {down+ci_d:+.3f}]).")
    print()
    print("  State the CI, not just the p-value. You have ruled out a")
    print(f"  downward shift larger than about {abs(down - ci_d):.2f} units,")
    print("  not established that the effect is exactly zero.")
    print()
    print("  Name these limitations:")
    print("   - the probe never saw neutral statements in training, so the")
    print("     neutral anchor is an extrapolation")
    print("   - pos > neg is partly guaranteed by how the probe was trained;")
    print("     the informative part is that neutral sits near neg, not")
    print("     midway between")
    print("   - single model, single domain, n=30 concepts")

    print("\n" + "=" * 62)
    print("PER-CONCEPT SPREAD")
    print("=" * 62)
    eff = sorted(zip(pos - neutral, [concepts[p] for p in pairs]))
    for v, c in eff[:3]:
        print(f"  {v:+7.3f}  {c}")
    print("  ...")
    for v, c in eff[-3:]:
        print(f"  {v:+7.3f}  {c}")


if __name__ == "__main__":
    main()
