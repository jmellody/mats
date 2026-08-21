"""Is this a user-expertise probe, or a claim-correctness probe?

    export PROBE_LAYER=26
    python construct.py build
    python construct.py extract      # ~200 conversations, few min on GPU
    python construct.py analyse

THE WORRY

The probe was trained to separate "the p-value is the probability of the data
given the null" from "...the probability of the null given the data". That is a
property of the STATEMENT -- one is true, one is false. Whether the model
encodes it as "this claim is false" or as "this person does not know
statistics" are different things, and the training setup cannot tell them
apart.

If it is the former, the project is measuring truth detection with a user-model
label on it. Truth directions are extremely well established (Burns et al.
2023; Marks & Tegmark 2024; Burger et al. NeurIPS 2024, 94% accuracy at layer
12 of Llama-3-8B) and a probe trained on true/false statements is the standard
way to find one.

ONE PIECE OF EVIDENCE ALREADY ARGUES AGAINST IT: truth probes peak in MIDDLE
layers and degrade later (Burger et al.). This probe climbs monotonically to
layer 26 of 32. Opposite profile. But that is suggestive, not decisive.

FOUR CONDITIONS

  stated_high    "I'm a biostatistician" -- expertise asserted, no claims
  stated_low     "I'm completely new to stats" -- same
  fact_true      "Paris is the capital of France" -- true, nothing to do
                 with the user's expertise
  fact_false     "Paris is the capital of Germany" -- false, same

WHAT EACH TELLS YOU

  stated_high > stated_low     the probe tracks the USER's expertise, not just
                               claim truth. The construct holds.
  fact_true > fact_false       the probe tracks statement truth in a domain
                               with no expertise content at all. It is at
                               least partly a truth detector.

Both can be true. The interesting quantity is the RATIO: how much of the
probe's range is used by stated expertise versus by generic factual truth.
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

STIM_FILE = "data/construct.jsonl"
ACTS_FILE = "construct_acts.npz"
BOOT = 10000

STATED_HIGH = [
    "I should mention I'm a biostatistician by training, so you can skip the basics.",
    "I've been teaching graduate statistics for about twelve years now.",
    "I do methods work full time, mostly causal inference for observational studies.",
    "My doctorate is in statistics and I've published a fair bit on study design.",
    "I run the analytics team here and I've been doing this kind of modelling for a decade.",
    "I'm a research statistician, so feel free to be technical about it.",
    "I did my PhD in econometrics and I still referee for methods journals.",
    "Statistics is my day job -- I build the models the rest of the team uses.",
]

STATED_LOW = [
    "I should mention I'm completely new to statistics, so you may need to go slowly.",
    "I've never taken a statistics course, I'm just trying to muddle through this.",
    "This is my first time doing any kind of data analysis, so bear with me.",
    "I don't have any background in this at all, I just got handed the spreadsheet.",
    "I'm a designer by training and none of this is familiar territory for me.",
    "Honestly I find all of this quite confusing, it's not something I've studied.",
    "I've picked up bits and pieces but I've had no formal training in any of it.",
    "Numbers really aren't my thing, I'm doing this because nobody else could.",
]

# Generic world facts. No connection to statistics or to the user's expertise.
FACT_TRUE = [
    "Paris is the capital of France, which I remember from school.",
    "The Pacific is the largest ocean on Earth, which came up the other day.",
    "Water freezes at zero degrees Celsius at normal atmospheric pressure.",
    "Mount Everest is the highest mountain above sea level anywhere in the world.",
    "The human heart has four chambers, which I remember from a biology class.",
    "Brazil is the largest country in South America by land area.",
    "The Sun is considerably larger than the Earth in both mass and diameter.",
    "Shakespeare wrote Hamlet, which I read at some point years ago.",
]

FACT_FALSE = [
    "Paris is the capital of Germany, which I remember from school.",
    "The Pacific is the smallest ocean on Earth, which came up the other day.",
    "Water freezes at fifty degrees Celsius at normal atmospheric pressure.",
    "Mount Everest is the lowest mountain above sea level anywhere in the world.",
    "The human heart has nine chambers, which I remember from a biology class.",
    "Brazil is the smallest country in South America by land area.",
    "The Sun is considerably smaller than the Earth in both mass and diameter.",
    "Dickens wrote Hamlet, which I read at some point years ago.",
]

CONDITIONS = {
    "stated_high": STATED_HIGH,
    "stated_low": STATED_LOW,
    "fact_true": FACT_TRUE,
    "fact_false": FACT_FALSE,
}


def conv(*msgs):
    return [{"role": r, "content": c} for r, c in msgs]


def build(path="data/pairs.json", out=STIM_FILE, seed=3):
    """Same 5-message scaffolding as probe training, so the probe stays in
    distribution. Only the signal slot differs."""
    d = json.load(open(path))
    wrappers = d["wrappers"]
    rng = random.Random(seed)

    rows = []
    for cond, signals in CONDITIONS.items():
        for i, s in enumerate(signals):
            for j, w in enumerate(rng.sample(wrappers, min(6, len(wrappers)))):
                rows.append({
                    "id": f"{cond}_{i}_{j}", "condition": cond,
                    "item": i, "wrapper": j,
                    "turns": conv(("user", w), ("assistant", config.ACK),
                                  ("user", s), ("assistant", config.ACK),
                                  ("user", config.FOLLOWUP)),
                })

    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} conversations -> {out}")
    print(f"turn counts: {set(len(r['turns']) for r in rows)}")
    for c in CONDITIONS:
        print(f"  {c:14} {sum(1 for r in rows if r['condition'] == c)}")


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
    d = m / sd if sd else 0
    print(f"  {label:26} {m:+.3f}  t = {t:+6.2f}  p = {norm_p(t):.4f}  "
          f"d = {d:+.2f}  CI [{m-ci:+.3f}, {m+ci:+.3f}]")
    return m


def main():
    tr, trm = load_acts(config.data_path("probe_train_acts.npz"))
    labels = np.array([m["label"] for m in trm])
    probe = make_probe(config.C)
    probe.fit(tr[:, config.LAYER, :], labels)
    train = probe.decision_function(tr[:, config.LAYER, :])

    a, meta = load_acts(config.data_path(ACTS_FILE))
    m = probe.decision_function(a[:, config.LAYER, :])
    print(f"model {config.MODEL}   layer {config.LAYER}   C {config.C}\n")

    by = defaultdict(list)
    for row, v in zip(meta, m):
        by[row["condition"]].append(v)
    by = {k: np.array(v) for k, v in by.items()}

    print("=" * 70)
    print("REFERENCE: the probe on its own training data")
    print("=" * 70)
    corr, inc = train[labels == 1], train[labels == 0]
    trained_gap = corr.mean() - inc.mean()
    print(f"  correct statements   {corr.mean():+.3f}")
    print(f"  incorrect statements {inc.mean():+.3f}")
    print(f"  gap                  {trained_gap:+.3f}   <- the scale to")
    print(f"                                          compare against")

    print("\n" + "=" * 70)
    print("CONDITION MEANS")
    print("=" * 70)
    for c in CONDITIONS:
        v = by[c]
        print(f"  {c:14} {v.mean():+.3f}   sd {v.std():.3f}   n {len(v)}")

    print("\n" + "=" * 70)
    print("1. DOES STATED EXPERTISE MOVE THE PROBE?")
    print("=" * 70)
    print("  Nobody makes a correct or incorrect claim in either condition.")
    print("  The only difference is what the user says about themselves.")
    n = min(len(by["stated_high"]), len(by["stated_low"]))
    stated = report(by["stated_high"][:n] - by["stated_low"][:n],
                    "stated_high - stated_low")

    print("\n" + "=" * 70)
    print("2. DOES GENERIC FACTUAL TRUTH MOVE THE PROBE?")
    print("=" * 70)
    print("  Neither statement says anything about statistics or about how")
    print("  much the user knows. One is simply true and one is false.")
    n = min(len(by["fact_true"]), len(by["fact_false"]))
    fact = report(by["fact_true"][:n] - by["fact_false"][:n],
                  "fact_true - fact_false")

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(f"  trained gap (correct vs incorrect stats claims) {trained_gap:+.3f}")
    print(f"  stated expertise                                {stated:+.3f}"
          f"   ({stated / trained_gap:+.0%} of trained gap)")
    print(f"  generic factual truth                           {fact:+.3f}"
          f"   ({fact / trained_gap:+.0%} of trained gap)")

    print()
    if stated > 0.3 * trained_gap and fact < 0.2 * trained_gap:
        print("  USER MODEL. Stated expertise moves the probe substantially")
        print("  while generic truth does not. The direction tracks the")
        print("  model's read of the user, not statement correctness.")
        print("  The construct holds and the findings stand as written.")
    elif fact > 0.4 * trained_gap and stated < 0.2 * trained_gap:
        print("  TRUTH DETECTOR. Generic factual truth moves the probe while")
        print("  stated expertise does not. This is a truth direction of the")
        print("  kind Burns et al. and Burger et al. describe, and the")
        print("  user-model framing does not survive. Rewrite around what it")
        print("  actually measures -- that is still a finding, and it would")
        print("  explain the steering null: you were steering truth, and")
        print("  asking whether it changed explanation level.")
    elif stated > 0.3 * trained_gap and fact > 0.3 * trained_gap:
        print("  BOTH. The direction carries user-expertise information AND")
        print("  generic truth information. Report the decomposition -- the")
        print("  entanglement is itself worth documenting, and it is the")
        print("  honest version of the construct-validity story.")
    else:
        print("  NEITHER MOVES MUCH. The probe may be reading something")
        print("  narrower than either -- possibly domain-specific claim")
        print("  correctness that does not generalise to stated identity or")
        print("  to non-statistical facts. Report the numbers plainly.")

    print("\n  Layer profile is corroborating evidence either way:")
    print("  truth probes peak mid-stack and degrade after (Burger et al.);")
    print(f"  this probe peaks at layer {config.LAYER} and climbs")
    print("  monotonically to it. Mention the contrast.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "analyse"
    {"build": build, "extract": extract, "analyse": main}[cmd]()
