"""Stated vs demonstrated expertise: which wins when they conflict?

    export PROBE_LAYER=13
    python conflict.py build
    python conflict.py extract
    python conflict.py analyse

THE QUESTION

Every experiment so far used cues that agree with each other. Real users say
one thing and do another all the time -- and Reading Between the Prompts
(arXiv 2505.16467) found that inferred demographic attributes persist even when
the user explicitly identifies with a different group. This is the expertise
version of that test.

2x2 FACTORIAL

                        demonstrated HIGH        demonstrated LOW
    stated HIGH      "biostatistician" + Stan   "biostatistician" + Excel
    stated LOW       "nurse" + Stan             "nurse" + Excel

The off-diagonal cells are the conflicts. Two main effects fall out:

    stated effect       = (HH + HL)/2 - (LH + LL)/2
    demonstrated effect = (HH + LH)/2 - (HL + LL)/2

Their RATIO is the finding. Ratio near 1 means the model weighs a stated
identity as heavily as demonstrated behaviour. Ratio well below 1 means it
discounts what people say about themselves in favour of what they do.

The interaction term matters too: if it is large, conflict is not simply the
sum of two signals -- the model does something specific when the cues disagree,
which would be the more interesting result.

WHY THE CUE TYPES ARE role AND tooling

Both were in the probe's training set, so neither is out of distribution. But
the probe never saw them COMBINED, and never saw them disagree. A conversation
where a stated biostatistician works in Excel is genuinely novel to it.

ORDER IS COUNTERBALANCED. Half the conversations put the role first, half put
the tooling first, so the small recency effect found in order() cannot masquerade
as a difference between stated and demonstrated cues.
"""

import json
import os
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

STIM = "data/conflict.jsonl"
ACTS = "conflict_acts.npz"
BOOT = 10000

FILLERS = [
    "I put the whole thing together over the weekend and it's all in one place now.",
    "It took a while to collect but I finally have everything I need in the file.",
    "I've been meaning to look at this properly for a couple of weeks now.",
    "There's quite a lot of it, more than I expected when I started.",
]


def conv(*m):
    return [{"role": r, "content": c} for r, c in m]


def build(path="data/identity.json", out=STIM, seed=11):
    rows = json.load(open(path))
    wr = json.load(open("data/pairs.json"))["wrappers"]
    rng = random.Random(seed)

    pool = {}
    for cue in ("role", "tooling"):
        for lab in (1, 0):
            v = [r["text"] for r in rows
                 if r["cue"] == cue and r["label"] == lab]
            rng.shuffle(v)
            pool[(cue, lab)] = v

    n = min(len(v) for v in pool.values())
    print(f"{n} usable items per cell")

    out_rows = []
    for i in range(n):
        w = wr[i % len(wr)]
        fill = FILLERS[i % len(FILLERS)]
        role_first = (i % 2 == 0)   # counterbalance order
        for sl in (1, 0):
            for dl in (1, 0):
                r_txt = pool[("role", sl)][i]
                t_txt = pool[("tooling", dl)][i]
                a, b = (r_txt, t_txt) if role_first else (t_txt, r_txt)
                cond = f"s{'H' if sl else 'L'}_d{'H' if dl else 'L'}"
                out_rows.append({
                    "id": f"{cond}_{i}", "condition": cond,
                    "stated": sl, "demonstrated": dl, "pair": i,
                    "role_first": role_first,
                    "turns": conv(("user", w), ("assistant", config.ACK),
                                  ("user", fill), ("assistant", config.ACK),
                                  ("user", a), ("assistant", config.ACK),
                                  ("user", b), ("assistant", config.ACK),
                                  ("user", config.FOLLOWUP)),
                })

    with open(out, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(out_rows)} conversations -> {out}")
    print(f"turn counts: {set(len(r['turns']) for r in out_rows)}")
    print(f"role-first: {sum(r['role_first'] for r in out_rows)}/"
          f"{len(out_rows)}")


def extract():
    model, tok = load_model()
    extract_acts(STIM, config.data_path(ACTS), model=model, tok=tok,
                 per_turn=False)


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
    print(f"  {label:28} {m:+.4f}  t = {t:+6.2f}  p = {norm_p(t):.4f}  "
          f"CI [{m-ci:+.4f}, {m+ci:+.4f}]  boot [{b[0]:+.4f}, {b[1]:+.4f}]")
    return m


def main():
    a, m = load_acts(config.data_path(os.environ.get("PROBE_TRAIN","ag_train_acts.npz")))
    y = np.array([r["label"] for r in m])
    probe = make_probe(config.C)
    probe.fit(a[:, config.LAYER, :], y)
    ts = probe.decision_function(a[:, config.LAYER, :])

    ea, meta = load_acts(config.data_path(ACTS))
    s = probe.decision_function(ea[:, config.LAYER, :])
    print(f"model {config.MODEL}   layer {config.LAYER}   v2 identity probe\n")

    v = defaultdict(dict)
    first = {}
    for r, x in zip(meta, s):
        v[r["pair"]][r["condition"]] = x
        first[r["pair"]] = r["role_first"]
    need = {"sH_dH", "sH_dL", "sL_dH", "sL_dL"}
    keep = sorted(p for p, c in v.items() if need <= c.keys())
    g = lambda k: np.array([v[p][k] for p in keep])
    HH, HL, LH, LL = g("sH_dH"), g("sH_dL"), g("sL_dH"), g("sL_dL")
    n = len(keep)

    print("=" * 72)
    print("REFERENCE")
    print("=" * 72)
    print(f"  training high cues {ts[y == 1].mean():+.2f}   "
          f"low cues {ts[y == 0].mean():+.2f}")

    print("\n" + "=" * 72)
    print(f"THE 2x2  (n = {n} units)")
    print("=" * 72)
    print(f"  {'':22} {'demo HIGH':>12} {'demo LOW':>12}")
    print(f"  {'stated HIGH':22} {HH.mean():>+12.3f} {HL.mean():>+12.3f}")
    print(f"  {'stated LOW':22} {LH.mean():>+12.3f} {LL.mean():>+12.3f}")
    print("\n  off-diagonal cells are the conflicts")

    print("\n" + "=" * 72)
    print("MAIN EFFECTS")
    print("=" * 72)
    stated = (HH + HL) / 2 - (LH + LL) / 2
    demo = (HH + LH) / 2 - (HL + LL) / 2
    ms = report(stated, "stated (what they claim)")
    md = report(demo, "demonstrated (what they do)")

    print("\n" + "=" * 72)
    print("WHICH WINS?")
    print("=" * 72)
    if md:
        print(f"  stated / demonstrated = {ms / md:.2f}")
        print("  ~1.0  weighed equally")
        print("  <0.5  the model discounts what users claim about themselves")
        print("  >2.0  a stated identity overrides demonstrated behaviour")
    report(stated - demo, "stated - demonstrated")

    print("\n" + "=" * 72)
    print("INTERACTION -- is conflict more than the sum of two cues?")
    print("=" * 72)
    inter = (HH - HL) - (LH - LL)
    report(inter, "interaction")
    print("  ~0 means the cues combine additively; the model simply adds")
    print("  evidence. A large term means it does something specific when")
    print("  the cues disagree, which would be the more interesting result.")

    print("\n" + "=" * 72)
    print("ORDER CONTROL")
    print("=" * 72)
    rf = np.array([first[p] for p in keep])
    if rf.sum() and (~rf).sum():
        print(f"  role-first units:    stated {stated[rf].mean():+.3f}  "
              f"demo {demo[rf].mean():+.3f}")
        print(f"  tooling-first units: stated {stated[~rf].mean():+.3f}  "
              f"demo {demo[~rf].mean():+.3f}")
        print("  If the stated/demonstrated gap flips with order, it is the")
        print("  recency effect from order(), not a difference between cue")
        print("  kinds. Counterbalancing means the main effects above")
        print("  already average over it.")

    print("\n" + "=" * 72)
    print("HOW TO READ THIS")
    print("=" * 72)
    print("  Reading Between the Prompts (2505.16467) found inferred")
    print("  demographic attributes persisting even when a user explicitly")
    print("  identifies otherwise. If the stated effect here is small")
    print("  relative to the demonstrated one, that is the expertise")
    print("  analogue: the model weighs how someone works more heavily than")
    print("  what they say they are.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "analyse"
    {"build": build, "extract": extract, "analyse": main}[cmd]()
