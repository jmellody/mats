"""Dose-response: how far does the probe move per additional signal?

    python dose.py build
    python dose.py extract
    python dose.py analyse

WHY THIS DESIGN EXISTS

The single-signal design needed a "neutral" reference condition, and there is
no principled one. Every statement you can put in that slot carries some
information about the user, so you are never measuring against zero -- you are
measuring against another point whose position you cannot justify. That showed
up concretely: orthogonalising the probe direction reversed which side of
neutral the pos and neg conditions fell on, while leaving pos-neg untouched.

Here, asymmetry is the difference between two SLOPES. Each arm supplies its own
baseline, so the intercept -- the thing that was unstable -- cancels out.

    positive arm:  0, 1, 2, 3 correct statements     -> slope b_pos
    negative arm:  0, 1, 2, 3 incorrect statements   -> slope b_neg

    asymmetry = |b_pos| - |b_neg|, tested pairwise across units

The dose-0 condition is shared by both arms, so it anchors nothing -- it only
pins the common intercept, which drops out of every slope.

BONUS: linearity. If the probe moves less for the third signal than the first,
the representation saturates -- a dynamics finding the single-signal design
could not see at all.

Filler occupies unused signal slots so every conversation has the same number
of turns and the same technical density. Only the count of real signals varies.
"""

import json
import random
import sys
from collections import defaultdict
from math import erf, sqrt

import numpy as np

import config
from acts import extract as extract_acts
from acts import load as load_acts
from acts import load_model
from probe import make_probe

MAX_DOSE = 3
BOOT = 10000

STIM_FILE = "data/dose.jsonl"
ACTS_FILE = "dose_acts.npz"          # resolved under data/<MODEL_TAG>/

# Neutral procedural filler for unused slots. Technical register, so dose does
# not confound with total technical content -- every conversation carries three
# technical-register statements; only their correctness status varies.
FILLERS = [
    "I ran the models last night and exported the output tables into a spreadsheet.",
    "The data is in long format with one row per observation and the groups coded as factors.",
    "I kept the raw and cleaned files separate, and the whole script runs end to end.",
    "The estimates and their intervals are saved in one table with the cell counts.",
    "I used the package defaults for fitting and have not touched the convergence settings.",
    "All the variables are still on their original scales and the summaries are printed out.",
]


def conv(*msgs):
    return [{"role": r, "content": c} for r, c in msgs]


def build(path="data/pairs.json", out=STIM_FILE, seed=0):
    """Each unit x arm x dose.

    A "unit" is MAX_DOSE distinct minimal pairs. Dose k uses the first k of
    them, so the positive and negative arms are matched pair-by-pair at every
    dose level.

    Repeating one sentence would confound dose with repetition -- three copies
    of the same statement is one piece of evidence plus a user who repeats
    themselves, and the model may represent that instead.

    Signals occupy the LAST slots so the most recent turns always carry the
    manipulation, holding recency constant across doses.
    """
    d = json.load(open(path))
    curated = [p for p in d["pairs"] if p.get("keep", True)]
    pool = curated + [p for p in d["pairs"] if not p.get("keep", True)]
    wrappers = d["wrappers"]
    rng = random.Random(seed)
    rng.shuffle(pool)

    n_units = len(pool) // MAX_DOSE
    if n_units < 10:
        print(f"WARNING: only {n_units} units. Generate more pairs.")
    units = [pool[i * MAX_DOSE:(i + 1) * MAX_DOSE] for i in range(n_units)]

    rows = []
    for i, unit in enumerate(units):
        w = wrappers[i % len(wrappers)]
        fill = [FILLERS[(i + k) % len(FILLERS)] for k in range(MAX_DOSE)]
        for arm, key in [("pos", "correct"), ("neg", "incorrect")]:
            for dose in range(MAX_DOSE + 1):
                if arm == "neg" and dose == 0:
                    continue  # dose 0 is shared; emit once
                signals = [unit[k][key] for k in range(dose)]
                slots = fill[:MAX_DOSE - dose] + signals
                turns = [("user", w), ("assistant", config.ACK)]
                for s in slots:
                    turns += [("user", s), ("assistant", config.ACK)]
                turns.append(("user", config.FOLLOWUP))
                rows.append({
                    "id": f"{arm}_{dose}_{i}",
                    "arm": arm if dose else "both", "dose": dose,
                    "pair": i,
                    "concept": "|".join(p["concept"]
                                        for p in unit[:max(dose, 1)]),
                    "turns": conv(*turns),
                })

    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    lens = {len(r["turns"]) for r in rows}
    print(f"{n_units} units x {2 * MAX_DOSE + 1} conditions = "
          f"{len(rows)} conversations -> {out}")
    print(f"turn counts: {lens}  (must be a single value)")


def extract():
    model, tok = load_model()
    extract_acts(STIM_FILE, config.data_path(ACTS_FILE),
                 model=model, tok=tok, per_turn=False)


def norm_p(t):
    return 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))


def report(x, label):
    n = len(x)
    m, sd = x.mean(), x.std(ddof=1)
    sem = sd / sqrt(n) if n > 1 else 0
    t = m / sem if sem else 0.0
    ci = 1.96 * sem
    rng = np.random.default_rng(0)
    boot = np.percentile(
        [rng.choice(x, n, replace=True).mean() for _ in range(BOOT)],
        [2.5, 97.5])
    print(f"  {label:20} {m:+.4f}  t = {t:+5.2f}  p = {norm_p(t):.4f}  "
          f"CI [{m-ci:+.4f}, {m+ci:+.4f}]  "
          f"boot [{boot[0]:+.4f}, {boot[1]:+.4f}]")
    return m


def main():
    tr, trm = load_acts(config.data_path("probe_train_acts.npz"))
    probe = make_probe(config.C)
    probe.fit(tr[:, config.LAYER, :], np.array([m["label"] for m in trm]))

    a, meta = load_acts(config.data_path(ACTS_FILE))
    margins = probe.decision_function(a[:, config.LAYER, :])
    print(f"model {config.MODEL}   layer {config.LAYER}   C {config.C}")

    vals = defaultdict(lambda: defaultdict(dict))
    concepts = {}
    for row, v in zip(meta, margins):
        concepts[row["pair"]] = row["concept"]
        if row["dose"] == 0:
            vals[row["pair"]]["pos"][0] = v
            vals[row["pair"]]["neg"][0] = v
        else:
            vals[row["pair"]][row["arm"]][row["dose"]] = v

    doses = np.arange(MAX_DOSE + 1)
    slopes = {"pos": [], "neg": []}
    curves = {"pos": [], "neg": []}
    keep = []
    for pair, arms in sorted(vals.items()):
        if not all(set(doses) <= set(arms[k]) for k in ("pos", "neg")):
            continue
        keep.append(pair)
        for k in ("pos", "neg"):
            y = np.array([arms[k][j] for j in doses])
            curves[k].append(y)
            slopes[k].append(np.polyfit(doses, y, 1)[0])

    n = len(keep)
    bp, bn = np.array(slopes["pos"]), np.array(slopes["neg"])
    print(f"n = {n} units, doses 0-{MAX_DOSE}\n")

    print("=" * 66)
    print("MEAN TRAJECTORY (probe margin by dose)")
    print("=" * 66)
    print(f"  {'dose':>5} " + " ".join(f"{d:>9}" for d in doses))
    for k in ("pos", "neg"):
        m = np.stack(curves[k]).mean(0)
        print(f"  {k:>5} " + " ".join(f"{v:>+9.3f}" for v in m))

    print("\n" + "=" * 66)
    print("1. SLOPES -- movement per additional signal")
    print("=" * 66)
    report(bp, "positive slope")
    report(bn, "negative slope")

    print("\n" + "=" * 66)
    print("2. ASYMMETRY TEST -- no neutral anchor involved")
    print("=" * 66)
    print("  Paired across units. Intercepts cancel, so this does not")
    print("  depend on any reference condition.")
    report(np.abs(bp) - np.abs(bn), "|b_pos| - |b_neg|")
    report(bp + bn, "b_pos + b_neg")
    print("  (near zero = mirror-image arms; positive = upward drift")
    print("   regardless of valence, i.e. a technicality effect)")

    print("\n" + "=" * 66)
    print("3. LINEARITY -- does the representation saturate?")
    print("=" * 66)
    for k in ("pos", "neg"):
        Y = np.stack(curves[k])
        first = Y[:, 1] - Y[:, 0]
        last = Y[:, MAX_DOSE] - Y[:, MAX_DOSE - 1]
        print(f"\n  {k} arm:")
        report(first, "  1st signal step")
        report(last, f"  {MAX_DOSE}rd signal step")
        report(first - last, "  step shrinkage")
    print("\n  positive shrinkage = later signals matter less (saturating)")

    print("\n" + "=" * 66)
    print("4. ROBUSTNESS")
    print("=" * 66)
    asym = np.abs(bp) - np.abs(bn)
    worst = max(norm_p(np.delete(asym, i).mean() /
                       (np.delete(asym, i).std(ddof=1) / sqrt(n - 1)))
                for i in range(n))
    print(f"  worst p after dropping any one unit: {worst:.4f}")
    print(f"  {int((asym > 0).sum())}/{n} units show |b_pos| > |b_neg|")
    print(f"  slope sd: pos {bp.std():.3f}, neg {bn.std():.3f}")

    ext = sorted(zip(asym, [concepts[p] for p in keep]))
    print("\n  most negative asymmetry:")
    for v, c in ext[:3]:
        print(f"    {v:+.3f}  {c[:60]}")
    print("  most positive:")
    for v, c in ext[-3:]:
        print(f"    {v:+.3f}  {c[:60]}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "analyse"
    {"build": build, "extract": extract, "analyse": main}[cmd]()