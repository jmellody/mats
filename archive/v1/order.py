"""Order and separation: two things the contradiction design confounded.

    python order.py build
    python order.py extract
    python order.py analyse

TWO QUESTIONS

1. ORDER, COUNT-MATCHED. The retention result compared TWO earlier signals
   against ONE recent one, so part of the 1.55 ratio is accumulation rather
   than primacy. Holding count constant separates them:

       pn   + then -        same composition, opposite order
       np   - then +

   If pn > np, order matters independently of how much evidence there is.
   That is the clean primacy claim. pp and nn are same-count controls.

2. SEPARATION. In the contradiction design the contradiction always arrived
   immediately after the establishing evidence. Does earlier evidence still
   survive when the contradiction comes several turns later?

       pn_gap   +  filler  filler  -
       np_gap   -  filler  filler  +

   Compared against adjacent pn / np. If the gap versions show weaker order
   effects, retention is about recent context rather than history.

A NOTE ON FILLER, informed by the ceiling result

The dose experiment used technical-register filler, which on Qwen3.5-4B read as
near-maximally expert and pushed 64% of units against the probe's ceiling. Here
the filler is deliberately plain, so the baseline starts low and both directions
have headroom.

That reintroduces a technicality confound in absolute terms -- but every
comparison here is between conditions with IDENTICAL composition and identical
filler count, differing only in order. The confound is constant across the
conditions being compared, so it cancels. Do not compare absolute margins
across this experiment and the dose one; the baselines are not the same.
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

N_SLOTS = 4
BOOT = 10000

STIM_FILE = "data/order.jsonl"
ACTS_FILE = "order_acts.npz"

# Plain, non-technical filler. Keeps the baseline off the ceiling.
FILLERS = [
    "I put the whole thing together over the weekend and it's all in one place now.",
    "It took a while to collect but I finally have everything I need in the file.",
    "I've been meaning to look at this properly for a couple of weeks now.",
    "There's quite a lot of it, more than I expected when I started.",
    "I tidied it up a bit yesterday so it's easier to work with now.",
    "It's all in one folder along with the notes I made at the time.",
]

# slot pattern per condition: "p" correct, "n" incorrect, "f" filler
CONDITIONS = {
    "pn":     "ffpn",   # + then -, adjacent
    "np":     "ffnp",   # - then +, adjacent
    "pp":     "ffpp",   # same-count control
    "nn":     "ffnn",   # same-count control
    "pn_gap": "pffn",   # + then -, separated
    "np_gap": "nffp",   # - then +, separated
}


def conv(*msgs):
    return [{"role": r, "content": c} for r, c in msgs]


def build(path="data/pairs.json", out=STIM_FILE, seed=2):
    d = json.load(open(path))
    pool = ([p for p in d["pairs"] if p.get("keep", True)]
            + [p for p in d["pairs"] if not p.get("keep", True)])
    wrappers = d["wrappers"]
    rng = random.Random(seed)
    rng.shuffle(pool)

    # two distinct pairs per unit: one supplies the positive-slot statement,
    # the other the negative-slot statement, so pn and np draw on the same two
    # concepts in both orders
    n_units = len(pool) // 2
    units = [pool[2 * i:2 * i + 2] for i in range(n_units)]

    rows = []
    for i, (a, b) in enumerate(units):
        w = wrappers[i % len(wrappers)]
        fill = [FILLERS[(i + k) % len(FILLERS)] for k in range(N_SLOTS)]
        for cond, pattern in CONDITIONS.items():
            turns = [("user", w), ("assistant", config.ACK)]
            fi = 0
            seen_sig = 0
            for ch in pattern:
                if ch == "f":
                    s = fill[fi]
                    fi += 1
                else:
                    src = a if seen_sig == 0 else b
                    s = src["correct" if ch == "p" else "incorrect"]
                    seen_sig += 1
                turns += [("user", s), ("assistant", config.ACK)]
            turns.append(("user", config.FOLLOWUP))
            rows.append({
                "id": f"{cond}_{i}", "condition": cond, "pair": i,
                "pattern": pattern,
                "concepts": [a["concept"], b["concept"]],
                "turns": conv(*turns),
            })

    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{n_units} units x {len(CONDITIONS)} conditions = {len(rows)} "
          f"-> {out}")
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
    print(f"  {label:26} {m:+.4f}  t = {t:+5.2f}  p = {norm_p(t):.4f}  "
          f"CI [{m-ci:+.4f}, {m+ci:+.4f}]  boot [{b[0]:+.4f}, {b[1]:+.4f}]")
    return m, se


def main():
    tr, trm = load_acts(config.data_path("probe_train_acts.npz"))
    labels = np.array([m["label"] for m in trm])
    probe = make_probe(config.C)
    probe.fit(tr[:, config.LAYER, :], labels)
    train = probe.decision_function(tr[:, config.LAYER, :])

    a, meta = load_acts(config.data_path(ACTS_FILE))
    margins = probe.decision_function(a[:, config.LAYER, :])
    print(f"model {config.MODEL}   layer {config.LAYER}   C {config.C}")

    vals = defaultdict(dict)
    for row, v in zip(meta, margins):
        vals[row["pair"]][row["condition"]] = v
    keep = sorted(p for p, c in vals.items() if set(CONDITIONS) <= c.keys())
    n = len(keep)
    g = lambda k: np.array([vals[p][k] for p in keep])
    pn, np_, pp, nn = g("pn"), g("np"), g("pp"), g("nn")
    pn_gap, np_gap = g("pn_gap"), g("np_gap")

    print(f"n = {n} units\n")
    print("=" * 72)
    print("HEADROOM CHECK (the dose experiment failed this)")
    print("=" * 72)
    top = np.percentile(train[labels == 1], 95)
    bot = np.percentile(train[labels == 0], 5)
    mid = np.concatenate([pn, np_]).mean()
    print(f"  probe range on training text: [{bot:+.2f}, {top:+.2f}]")
    print(f"  mixed-condition mean:          {mid:+.2f}")
    print(f"  headroom above {top - mid:+.2f}   below {mid - bot:+.2f}")
    if min(top - mid, mid - bot) < 1.0:
        print("  WARNING: little room in one direction. Interpret with care.")
    else:
        print("  Both directions have room.")

    print("\n" + "=" * 72)
    print("CONDITION MEANS")
    print("=" * 72)
    for name, arr in [("pp   + +", pp), ("pn   + -", pn), ("np   - +", np_),
                      ("nn   - -", nn), ("pn_gap  + . . -", pn_gap),
                      ("np_gap  - . . +", np_gap)]:
        print(f"  {name:18} {arr.mean():+.3f}   sd {arr.std():.3f}")

    print("\n" + "=" * 72)
    print("1. ORDER, COUNT-MATCHED -- the clean primacy test")
    print("=" * 72)
    print("  pn and np contain the same two statements in opposite order.")
    print("  Composition, length, and filler count are identical.")
    m, se = report(pn - np_, "pn - np")
    print("  > 0 = the FIRST signal has more influence (primacy)")
    print("  < 0 = the LAST signal has more influence (recency)")

    print("\n" + "=" * 72)
    print("2. HOW BIG IS ORDER RELATIVE TO COMPOSITION?")
    print("=" * 72)
    comp, _ = report(pp - nn, "pp - nn (composition)")
    if comp:
        print(f"\n  order / composition = {m / comp:.3f}")
        print("  the share of the expertise signal carried by order alone")

    print("\n" + "=" * 72)
    print("3. SEPARATION -- does a gap weaken the order effect?")
    print("=" * 72)
    gap_eff, _ = report(pn_gap - np_gap, "pn_gap - np_gap")
    report((pn - np_) - (pn_gap - np_gap), "adjacent - separated")
    print("  > 0 = separation weakens order, so the effect is about recent")
    print("        context rather than conversation history")
    print("  ~ 0 = order survives separation, i.e. genuine history retention")

    print("\n" + "=" * 72)
    print("4. CONTROLS")
    print("=" * 72)
    report(pn - pp, "pn - pp")
    report(np_ - nn, "np - nn")
    print("  both should be non-zero; if not, the second slot did nothing")

    print("\n" + "=" * 72)
    print("HOW TO READ THIS")
    print("=" * 72)
    p = norm_p(m / se) if se else 1.0
    if p < 0.05 and m > 0:
        print("  Primacy, count-matched. The first signal carries more weight")
        print("  than the second even when the amount of evidence is held")
        print("  fixed. This closes the accumulation caveat on the retention")
        print("  result.")
    elif p < 0.05 and m < 0:
        print("  Recency, count-matched. The most recent signal dominates.")
        print("  The retention result was accumulation, not primacy -- two")
        print("  signals beat one regardless of position. Say so plainly.")
    else:
        print("  No order effect once count is held constant. The retention")
        print("  result reflects amount of evidence, not its position.")
        print("  Report the CI; this is an informative null.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "analyse"
    {"build": build, "extract": extract, "analyse": main}[cmd]()
