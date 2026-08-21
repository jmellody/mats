"""RQ2: does earlier evidence survive a contradiction?

    python contradiction.py build
    python contradiction.py extract
    python contradiction.py analyse

THE QUESTION

The dose experiment showed the first signal dominates, but every conversation
there was single-valence, so that was only established within a run of agreeing
evidence. Stickiness -- an established representation resisting revision --
needs mixed valence.

THE DESIGN

Four conditions, each three signals long, all ending on the same follow-up turn:

    ppp   + + +     pure positive
    ppn   + + -     positive established, then contradicted
    nnn   - - -     pure negative
    nnp   - - +     negative established, then contradicted

THE COMPARISON, and why it needs no steps

ppn and nnn both END on a negative statement, in the same slot, with the same
number of turns. They differ only in what came before, so compare their final
margins directly:

    ppn > nnn   ->  earlier positive evidence still shows through; the
                    representation retains its history despite contradiction
    ppn = nnn   ->  the most recent signal overwrites everything

nnp vs ppp is the mirror. Averaging the two guards against a result specific to
one direction.

Every conversation is read at the same position -- the shared follow-up turn,
exactly as in probe training -- so nothing is out of distribution. That was the
flaw in measuring intermediate steps: a prefix ending on a signal statement is
a different kind of input from anything the probe was fitted on.

Signals come from distinct minimal pairs, matched across conditions by source
pair, so the k-th slot concerns the same concept in every condition.
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

N_SIG = 3
BOOT = 10000

STIM_FILE = "data/contra.jsonl"
ACTS_FILE = "contra_acts.npz"        # resolved under data/<MODEL_TAG>/

CONDITIONS = {
    "ppp": ("correct", "correct", "correct"),
    "ppn": ("correct", "correct", "incorrect"),
    "nnn": ("incorrect", "incorrect", "incorrect"),
    "nnp": ("incorrect", "incorrect", "correct"),
}


def conv(*msgs):
    return [{"role": r, "content": c} for r, c in msgs]


def build(path="data/pairs.json", out=STIM_FILE, seed=1):
    d = json.load(open(path))
    pool = ([p for p in d["pairs"] if p.get("keep", True)]
            + [p for p in d["pairs"] if not p.get("keep", True)])
    wrappers = d["wrappers"]
    rng = random.Random(seed)
    rng.shuffle(pool)

    n_units = len(pool) // N_SIG
    units = [pool[i * N_SIG:(i + 1) * N_SIG] for i in range(n_units)]

    rows = []
    for i, unit in enumerate(units):
        w = wrappers[i % len(wrappers)]
        for cond, valences in CONDITIONS.items():
            turns = [("user", w), ("assistant", config.ACK)]
            for k, val in enumerate(valences):
                turns += [("user", unit[k][val]), ("assistant", config.ACK)]
            turns.append(("user", config.FOLLOWUP))
            rows.append({
                "id": f"{cond}_{i}", "condition": cond, "pair": i,
                "concepts": [p["concept"] for p in unit],
                "turns": conv(*turns),
            })

    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{n_units} units x 4 conditions = {len(rows)} -> {out}")
    print(f"turn counts: {set(len(r['turns']) for r in rows)} "
          f"(must be one value)")


def extract():
    model, tok = load_model()
    extract_acts(STIM_FILE, config.data_path(ACTS_FILE),
                 model=model, tok=tok, per_turn=False)


def norm_p(t):
    return 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))


def report(x, label):
    n = len(x)
    m, sd = x.mean(), x.std(ddof=1)
    se = sd / sqrt(n) if n > 1 else 0
    t = m / se if se else 0.0
    ci = 1.96 * se
    rng = np.random.default_rng(0)
    b = np.percentile([rng.choice(x, n, replace=True).mean()
                       for _ in range(BOOT)], [2.5, 97.5])
    print(f"  {label:24} {m:+.4f}  t = {t:+5.2f}  p = {norm_p(t):.4f}  "
          f"CI [{m-ci:+.4f}, {m+ci:+.4f}]  boot [{b[0]:+.4f}, {b[1]:+.4f}]")
    return m, se


def main():
    tr, trm = load_acts(config.data_path("probe_train_acts.npz"))
    probe = make_probe(config.C)
    probe.fit(tr[:, config.LAYER, :], np.array([m["label"] for m in trm]))

    a, meta = load_acts(config.data_path(ACTS_FILE))
    margins = probe.decision_function(a[:, config.LAYER, :])
    print(f"model {config.MODEL}   layer {config.LAYER}   C {config.C}")

    vals = defaultdict(dict)
    for row, v in zip(meta, margins):
        vals[row["pair"]][row["condition"]] = v

    keep = sorted(p for p, c in vals.items() if set(CONDITIONS) <= c.keys())
    n = len(keep)
    g = lambda k: np.array([vals[p][k] for p in keep])
    ppp, ppn, nnn, nnp = g("ppp"), g("ppn"), g("nnn"), g("nnp")

    print(f"n = {n} units\n")
    print("=" * 72)
    print("FINAL MARGINS (all read at the same follow-up turn)")
    print("=" * 72)
    for name, arr in [("ppp  + + +", ppp), ("ppn  + + -", ppn),
                      ("nnp  - - +", nnp), ("nnn  - - -", nnn)]:
        print(f"  {name:12} {arr.mean():+.3f}   sd {arr.std():.3f}")

    print("\n" + "=" * 72)
    print("1. RETENTION TEST -- both conditions end on a negative signal")
    print("=" * 72)
    print("  ppn and nnn match in structure and in the final statement's")
    print("  valence. Only the history differs.")
    report(ppn - nnn, "ppn - nnn")
    print("  > 0 means earlier positive evidence still shows through")

    print("\n" + "=" * 72)
    print("2. MIRROR TEST -- both end on a positive signal")
    print("=" * 72)
    report(ppp - nnp, "ppp - nnp")
    print("  > 0 means earlier positive evidence shows through here too")

    print("\n" + "=" * 72)
    print("3. COMBINED RETENTION")
    print("=" * 72)
    comb = ((ppn - nnn) + (ppp - nnp)) / 2
    m, se = report(comb, "mean retention")

    print("\n" + "=" * 72)
    print("4. SCALE REFERENCE -- retention vs the signal itself")
    print("=" * 72)
    recency = ((ppp - ppn) + (nnp - nnn)) / 2
    mr, _ = report(recency, "final-signal effect")
    print("  effect of flipping only the LAST signal, history held fixed")
    if mr:
        print(f"\n  retention / recency = {m / mr:.2f}")
        print("  ~1.0  history and the most recent signal matter equally")
        print("  <0.5  dominated by the most recent statement")
        print("  >1.5  dominated by what it saw first")
        print("\n  CAVEAT: retention sums TWO earlier signals against ONE")
        print("  recent one, so part of any ratio above 1 is accumulation")
        print("  rather than primacy. A count-matched (+ - vs - +) design")
        print("  is needed to separate them.")

    print("\n" + "=" * 72)
    print("5. ORDER CHECK -- does + + - land where - - + does?")
    print("=" * 72)
    print("  Note these differ in composition as well as order, so this is")
    print("  suggestive rather than a clean order test.")
    report(ppn - nnp, "ppn - nnp")

    print("\n" + "=" * 72)
    print("HOW TO READ THIS")
    print("=" * 72)
    p = norm_p(m / se) if se else 1.0
    if p < 0.05 and m > 0:
        print("  Earlier evidence survives contradiction. With the dose")
        print("  result, the picture is: the representation is set early")
        print("  and later evidence -- including contradicting evidence --")
        print("  shifts it only partially.")
    elif p < 0.05 and m < 0:
        print("  Unexpected direction: later evidence dominates AND")
        print("  overshoots. Investigate before writing up.")
    else:
        print("  No detectable retention. The most recent signal largely")
        print("  determines the representation regardless of history.")
        print("  With the saturation result, that suggests first-signal")
        print("  dominance is about position in the sequence, not the")
        print("  model defending an established view. Report the CI.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "analyse"
    {"build": build, "extract": extract, "analyse": main}[cmd]()