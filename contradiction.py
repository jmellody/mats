"""RQ2: does earlier evidence survive a contradiction?

    python contradiction.py build
    python contradiction.py extract      # 160 conversations, ~30 min
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
number of turns. They differ only in what came before. So compare their final
margins directly:

    ppn > nnn   ->  the earlier positive evidence still shows through. The
                    representation retains its history despite contradiction.
    ppn = nnn   ->  the most recent signal overwrites everything. No retention.

nnp vs ppp is the mirror. Averaging the two guards against a result that is
specific to one direction.

This reads every conversation at the same position -- the shared follow-up turn,
exactly as in probe training -- so nothing is out of distribution. That was the
flaw in measuring intermediate steps: a prefix ending on a signal statement is a
different kind of input from anything the probe was fitted on.

Signals are drawn from distinct minimal pairs, matched across conditions by
source pair, so the k-th slot concerns the same concept in every condition.
"""

import json
import random
import sys
from collections import defaultdict
from math import erf, sqrt

import numpy as np

from extract import load_extracted
from probe import make_probe

import os, torch
torch.set_num_threads(os.cpu_count())

LAYER = 27
C = 0.01
N_SIG = 3
BOOT = 10000

ACK = "Got it. Anything else about the setup I should know?"
FOLLOWUP = "So how should I decide what to actually report?"

CONDITIONS = {
    "ppp": ("correct", "correct", "correct"),
    "ppn": ("correct", "correct", "incorrect"),
    "nnn": ("incorrect", "incorrect", "incorrect"),
    "nnp": ("incorrect", "incorrect", "correct"),
}


def conv(*msgs):
    return [{"role": r, "content": c} for r, c in msgs]


def build(path="data/pairs.json", out="data/contra.jsonl", seed=1):
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
            turns = [("user", w), ("assistant", ACK)]
            for k, val in enumerate(valences):
                turns += [("user", unit[k][val]), ("assistant", ACK)]
            turns.append(("user", FOLLOWUP))
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
    print(f"est. {len(rows) * 11 / 60:.0f} min on CPU")


def extract():
    import os
    import torch
    torch.set_num_threads(os.cpu_count() or 4)
    from extract import extract_dataset, load_model
    model, tok = load_model("Qwen/Qwen2.5-3B-Instruct")
    extract_dataset("data/contra.jsonl", "data/contra_acts.npz",
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
    tr, trm = load_extracted("data/probe_train_acts.npz")
    probe = make_probe(C)
    probe.fit(tr[:, LAYER, :], np.array([m["label"] for m in trm]))

    acts, meta = load_extracted("data/contra_acts.npz")
    margins = probe.decision_function(acts[:, LAYER, :])

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
    print("  ppn and nnn are identical in structure and in the final")
    print("  statement's valence. Only the history differs.")
    dn, _ = report(ppn - nnn, "ppn - nnn")
    print("  > 0 means earlier positive evidence still shows through")

    print("\n" + "=" * 72)
    print("2. MIRROR TEST -- both end on a positive signal")
    print("=" * 72)
    dp, _ = report(ppp - nnp, "ppp - nnp")
    print("  > 0 means earlier positive evidence still shows through here too")

    print("\n" + "=" * 72)
    print("3. COMBINED RETENTION")
    print("=" * 72)
    comb = ((ppn - nnn) + (ppp - nnp)) / 2
    m, se = report(comb, "mean retention")

    print("\n" + "=" * 72)
    print("4. SCALE REFERENCE -- how big is retention vs the signal itself?")
    print("=" * 72)
    recency = ((ppp - ppn) + (nnp - nnn)) / 2
    mr, _ = report(recency, "final-signal effect")
    print("  effect of flipping only the LAST signal, holding history fixed")
    if mr != 0:
        print(f"\n  retention / recency = {m / mr:.2f}")
        print("  ~1.0  history and the most recent signal matter equally")
        print("  <0.5  the model is dominated by the most recent statement")
        print("  >1.5  the model is dominated by what it saw first")

    print("\n" + "=" * 72)
    print("5. ORDER CHECK -- does + + - land where - - + does?")
    print("=" * 72)
    print("  Both contain two of one valence and one of the other.")
    print("  If order did not matter these would be equal.")
    report(ppn - nnp, "ppn - nnp")

    print("\n" + "=" * 72)
    print("HOW TO READ THIS")
    print("=" * 72)
    p = norm_p(m / se) if se else 1.0
    if p < 0.05 and m > 0:
        print("  Earlier evidence survives contradiction. Combined with the")
        print("  dose result, the picture is: the representation is set")
        print("  early and later evidence -- including contradicting")
        print("  evidence -- shifts it only partially.")
    elif p < 0.05 and m < 0:
        print("  Unexpected direction. Later evidence dominates AND")
        print("  overshoots. Worth investigating before writing up.")
    else:
        print("  No detectable retention. The most recent signal largely")
        print("  determines the representation regardless of history.")
        print("  Combined with the saturation result, that suggests the")
        print("  first-signal dominance in the dose experiment was about")
        print("  position in the sequence, not about the model defending")
        print("  an established view. Report the CI, not just the null.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "analyse"
    {"build": build, "extract": extract, "analyse": main}[cmd]()
