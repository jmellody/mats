"""Dose-response: how far does the probe move per additional signal?

    python dose.py build
    python dose.py extract      # ~240 conversations, ~45 min
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

    asymmetry = |b_pos| - |b_neg|, tested pairwise across concepts

The dose-0 condition appears in both arms and is the same conversation, so it
anchors nothing -- it only pins the shared intercept, which then drops out of
every slope.

BONUS: linearity. If the probe moves less for the third signal than the first,
the representation saturates, which is a dynamics finding the single-signal
design could not see at all.

Filler occupies unused signal slots so that every conversation has the same
number of turns and the same length. Only the count of real signals varies.
"""

import json
import sys
from collections import defaultdict
from math import erf, sqrt

import numpy as np

from acts import extract_dataset, load_extracted, load_model
from probe import make_probe

MODEL = "Qwen/Qwen2.5-3B-Instruct"
LAYER = 27
C = 0.01
MAX_DOSE = 3
BOOT = 10000

ACK = "Got it. Anything else about the setup I should know?"
FOLLOWUP = "So how should I decide what to actually report?"

# Neutral procedural filler for unused slots. Technical register so that dose
# does not confound with total technical content -- every conversation has
# three technical-register statements, only their correctness status varies.
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


def build(path="data/pairs.json", out="data/dose.jsonl", seed=0):
    """Each unit x arm x dose.

    A "unit" is a set of MAX_DOSE distinct minimal pairs. Dose k uses the first
    k of them, so the positive and negative arms are matched pair-by-pair at
    every dose level.

    Repeating one sentence would have confounded dose with repetition -- three
    copies of the same statement is not three pieces of evidence, it is one
    piece plus a user who repeats themselves, and the model may represent that
    instead. Distinct statements avoid it.

    Signals occupy the LAST slots so the most recent turns always carry the
    manipulation, holding recency constant across doses.
    """
    import random
    d = json.load(open(path))
    curated = [p for p in d["pairs"] if p.get("keep", True)]
    pool = curated + [p for p in d["pairs"] if not p.get("keep", True)]
    wrappers = d["wrappers"]
    rng = random.Random(seed)

    n_units = len(pool) // MAX_DOSE
    if n_units < 10:
        print(f"WARNING: only {n_units} units. Generate more pairs.")
    rng.shuffle(pool)
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
                turns = [("user", w), ("assistant", ACK)]
                for s in slots:
                    turns += [("user", s), ("assistant", ACK)]
                turns.append(("user", FOLLOWUP))
                rows.append({
                    "id": f"{arm}_{dose}_{i}",
                    "arm": arm if dose else "both", "dose": dose,
                    "pair": i,
                    "concept": "|".join(p["concept"] for p in unit[:max(dose, 1)]),
                    "turns": conv(*turns),
                })

    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    lens = [len(r["turns"]) for r in rows]
    print(f"{n_units} units x {2 * MAX_DOSE + 1} conditions = "
          f"{len(rows)} conversations -> {out}")
    print(f"turn counts: {set(lens)}  (must be a single value)")
    print(f"est. extraction: {len(rows) * 11 / 60:.0f} min on CPU")


def extract():
    model, tok = load_model(MODEL)
    extract_dataset("data/dose.jsonl", "data/dose_acts.npz",
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
        [rng.choice(x, n, replace=True).mean() for _ in range(BOOT)], [2.5, 97.5])
    print(f"  {label:20} {m:+.4f}  t = {t:+5.2f}  p = {norm_p(t):.4f}  "
          f"CI [{m-ci:+.4f}, {m+ci:+.4f}]  boot [{boot[0]:+.4f}, {boot[1]:+.4f}]")
    return m


def main():
    tr, trm = load_extracted("data/probe_train_acts.npz")
    labels = np.array([m["label"] for m in trm])
    probe = make_probe(C)
    probe.fit(tr[:, LAYER, :], labels)

    acts, meta = load_extracted("data/dose_acts.npz")
    margins = probe.decision_function(acts[:, LAYER, :])

    # vals[pair][arm][dose] = margin
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
    slopes, curves = {"pos": [], "neg": []}, {"pos": [], "neg": []}
    keep = []
    for pair, arms in sorted(vals.items()):
        if not all(set(doses) <= set(arms[a]) for a in ("pos", "neg")):
            continue
        keep.append(pair)
        for a in ("pos", "neg"):
            y = np.array([arms[a][k] for k in doses])
            curves[a].append(y)
            slopes[a].append(np.polyfit(doses, y, 1)[0])

    n = len(keep)
    bp = np.array(slopes["pos"])
    bn = np.array(slopes["neg"])
    print(f"n = {n} concepts, doses 0-{MAX_DOSE}\n")

    print("=" * 66)
    print("MEAN TRAJECTORY (probe margin by dose)")
    print("=" * 66)
    print(f"  {'dose':>5} " + " ".join(f"{d:>9}" for d in doses))
    for a in ("pos", "neg"):
        m = np.stack(curves[a]).mean(0)
        print(f"  {a:>5} " + " ".join(f"{v:>+9.3f}" for v in m))

    print("\n" + "=" * 66)
    print("1. SLOPES -- movement per additional signal")
    print("=" * 66)
    report(bp, "positive slope")
    report(bn, "negative slope")

    print("\n" + "=" * 66)
    print("2. THE ASYMMETRY TEST -- no neutral anchor involved")
    print("=" * 66)
    print("  Paired across concepts. Intercepts cancel, so this does not")
    print("  depend on any reference condition.")
    report(np.abs(bp) - np.abs(bn), "|b_pos| - |b_neg|")
    report(bp + bn, "b_pos + b_neg")
    print("  (b_pos + b_neg near zero means the arms are mirror images;")
    print("   a positive sum means the pair drifts upward regardless of sign)")

    print("\n" + "=" * 66)
    print("3. LINEARITY -- does the representation saturate?")
    print("=" * 66)
    for a in ("pos", "neg"):
        Y = np.stack(curves[a])
        first = Y[:, 1] - Y[:, 0]
        last = Y[:, MAX_DOSE] - Y[:, MAX_DOSE - 1]
        print(f"\n  {a} arm:")
        report(first, "  1st signal step")
        report(last, f"  {MAX_DOSE}th signal step")
        report(first - last, "  step shrinkage")
        print("    positive shrinkage = later signals matter less "
              "(saturating)")

    print("\n" + "=" * 66)
    print("4. ROBUSTNESS")
    print("=" * 66)
    asym = np.abs(bp) - np.abs(bn)
    worst = max(norm_p(np.delete(asym, i).mean() /
                       (np.delete(asym, i).std(ddof=1) / sqrt(n - 1)))
                for i in range(n))
    print(f"  worst p after dropping any one concept: {worst:.4f}")
    pos_count = int((asym > 0).sum())
    print(f"  {pos_count}/{n} concepts show |b_pos| > |b_neg|")
    print(f"  slope sd: pos {bp.std():.3f}, neg {bn.std():.3f}")

    ext = sorted(zip(asym, [concepts[p] for p in keep]))
    print("\n  most negative asymmetry:")
    for v, c in ext[:3]:
        print(f"    {v:+.3f}  {c}")
    print("  most positive:")
    for v, c in ext[-3:]:
        print(f"    {v:+.3f}  {c}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "analyse"
    {"build": build, "extract": extract, "analyse": main}[cmd]()
