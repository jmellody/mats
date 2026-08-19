"""Is the correctness effect real, or residual technical density?

    python orthogonalize.py

No new compute. Reads the saved activations.

THE QUESTION

The probe was trained only to separate correct from incorrect statements. It was
never shown technicality. Yet filler sits at -1.02 and technicality-matched
neutral at -0.12, a gap of +0.90 (d=0.95) -- larger than the correctness effect
itself (+0.44). So the learned direction encodes technical density as well as
correctness, most likely because the correct statements are on average slightly
denser than the incorrect ones despite the length and vocabulary matching.

THE TEST

Technicality has an estimable direction in activation space:

    v_tech = mean(neutral activations) - mean(filler activations)

Both conditions are non-committal about the user's understanding and differ
almost only in register, so this isolates the nuisance axis. Project it out of
the probe direction, re-score, and see what survives.

  - pos-neg survives  -> the correctness effect is not density. Real result.
  - pos-neg collapses -> what looked like correctness WAS residual density.
                         Also a real result, and a more interesting one about
                         what expertise probes capture.

Either way this is the check a reviewer asks for, so run it before writing.
"""

from collections import defaultdict
from math import erf, sqrt

import numpy as np

from extract import load_extracted
from probe import make_probe, probe_direction

LAYER = 27
C = 0.01
BOOT = 20000


def norm_p(t):
    return 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))


def report(x, label):
    n = len(x)
    m, sd = x.mean(), x.std(ddof=1)
    sem = sd / sqrt(n)
    t = m / sem if sem else 0.0
    rng = np.random.default_rng(0)
    boot = np.array([rng.choice(x, n, replace=True).mean() for _ in range(BOOT)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  {label:20} {m:+7.3f}  t = {t:+5.2f}  p = {norm_p(t):.4f}  "
          f"d = {m/sd if sd else 0:+.2f}  boot [{lo:+.3f}, {hi:+.3f}]")
    return m, (lo, hi)


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n else v


def main():
    # --- probe direction from training data
    tr_acts, tr_meta = load_extracted("data/probe_train_acts.npz")
    labels = np.array([m["label"] for m in tr_meta])
    probe = make_probe(C)
    probe.fit(tr_acts[:, LAYER, :], labels)
    w = probe_direction(probe)

    # --- experiment activations, keyed by pair and condition
    acts_by = defaultdict(dict)
    concepts = {}
    for f in ["data/rq1_acts.npz", "data/neutral_acts.npz"]:
        a, m = load_extracted(f)
        for row, vec in zip(m, a[:, LAYER, :]):
            acts_by[row["pair"]][row["condition"]] = vec
            concepts[row["pair"]] = row["concept"]

    need = {"baseline", "pos", "neg", "neutral"}
    pairs = sorted(p for p, d in acts_by.items() if need <= d.keys())
    n = len(pairs)
    stack = lambda c: np.stack([acts_by[p][c] for p in pairs])
    A = {c: stack(c) for c in ["baseline", "neutral", "pos", "neg"]}

    # --- the technicality direction
    v_tech = unit(A["neutral"].mean(0) - A["baseline"].mean(0))
    cos = float(np.dot(unit(w), v_tech))

    print("=" * 62)
    print("ALIGNMENT BETWEEN THE TWO DIRECTIONS")
    print("=" * 62)
    print(f"  cosine(probe, technicality) = {cos:+.3f}")
    print(f"  shared variance             = {cos**2:.1%}")
    if abs(cos) > 0.3:
        print("  ^ substantially entangled; orthogonalisation matters here")
    else:
        print("  ^ only weakly entangled; the probe is largely not density")

    # --- orthogonalised probe direction
    w_orth = unit(unit(w) - cos * v_tech)

    def score(vecs, direction):
        return vecs @ direction

    print("\n" + "=" * 62)
    print("BEFORE ORTHOGONALISATION")
    print("=" * 62)
    s = {c: score(A[c], unit(w)) for c in A}
    report(s["neutral"] - s["baseline"], "neutral - filler")
    report(s["pos"] - s["neg"], "pos - neg")
    pre_pos, _ = report(s["pos"] - s["neutral"], "pos - neutral")
    report(s["neg"] - s["neutral"], "neg - neutral")

    print("\n" + "=" * 62)
    print("AFTER PROJECTING OUT TECHNICALITY")
    print("=" * 62)
    o = {c: score(A[c], w_orth) for c in A}
    report(o["neutral"] - o["baseline"], "neutral - filler")
    post_pn, pn_ci = report(o["pos"] - o["neg"], "pos - neg")
    post_pos, _ = report(o["pos"] - o["neutral"], "pos - neutral")
    report(o["neg"] - o["neutral"], "neg - neutral")

    print("\n" + "=" * 62)
    print("VERDICT")
    print("=" * 62)
    pre_pn = (s["pos"] - s["neg"]).mean()
    retained = post_pn / pre_pn if pre_pn else 0.0
    print(f"  pos-neg before {pre_pn:+.3f}, after {post_pn:+.3f} "
          f"({retained:.0%} retained)")
    survives = pn_ci[0] > 0 or pn_ci[1] < 0

    if survives and retained > 0.5:
        print("\n  The correctness effect survives orthogonalisation. The")
        print("  representation distinguishes correct from incorrect use")
        print("  independently of technical register. This is the stronger")
        print("  version of your result -- state it this way.")
    elif survives:
        print("\n  Survives but much reduced. Most of the apparent correctness")
        print("  effect was technical density; a smaller genuine component")
        print("  remains. Report both numbers.")
    else:
        print("\n  The correctness effect does NOT survive. What looked like")
        print("  sensitivity to whether the user was right was residual")
        print("  technical density. This is the more interesting finding:")
        print("  a probe trained to detect correct vs incorrect usage")
        print("  generalises as a register detector, not a correctness one.")
        print("  That is a construct-validity result about a method in")
        print("  active use -- lead the writeup with it.")

    # --- also check the probe's own accuracy after orthogonalisation
    print("\n" + "=" * 62)
    print("SANITY: TRAINING ACCURACY AFTER ORTHOGONALISATION")
    print("=" * 62)
    for name, d in [("original", unit(w)), ("orthogonalised", w_orth)]:
        m = tr_acts[:, LAYER, :] @ d
        thr = np.median(m)
        acc = max(((m > thr) == (labels == 1)).mean(),
                  ((m > thr) == (labels == 0)).mean())
        pos_m, neg_m = m[labels == 1], m[labels == 0]
        dd = (pos_m.mean() - neg_m.mean()) / sqrt(
            (pos_m.std() ** 2 + neg_m.std() ** 2) / 2)
        print(f"  {name:16} median-split acc {acc:.3f}   Cohen's d {dd:+.2f}")
    print("\n  In-sample, not cross-validated -- for direction comparison only.")

    print("\n" + "=" * 62)
    print("LIMITATION TO STATE")
    print("=" * 62)
    print("  v_tech is estimated from 30 neutral and 30 filler statements, so")
    print("  it is a noisy estimate of the technicality axis. Projecting out a")
    print("  noisy direction removes some real signal too, which biases the")
    print("  post-orthogonalisation effect downward. Treat the 'after' number")
    print("  as a conservative lower bound, not a point estimate.")


if __name__ == "__main__":
    main()
