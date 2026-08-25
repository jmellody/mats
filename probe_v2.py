"""Train the v2 probe on identity cues, and check it is a user model this time.

    export PROBE_LAYER=26      # will be re-swept; this is only a fallback
    python probe_v2.py sweep      # layer sweep, grouped by item
    python probe_v2.py validate   # the tests v1 failed

VALIDATION IS NOT OPTIONAL HERE. v1 hit 0.737 held-out accuracy and turned out
to be a truth detector. Accuracy alone proves nothing about what a probe has
learned. Three checks:

  1. CUE-TYPE GENERALISATION. Train on role/context/request, test on tooling --
     a cue type the probe has never seen, sharing almost no vocabulary with the
     others. Transfer means it found something about the user. Collapse to
     chance means it found phrases.

  2. TRUTH CONTAMINATION. Score "Paris is the capital of France" against
     "...of Germany". v1 reproduced 98% of its trained gap on this. v2 should
     be near zero -- there are no truth claims anywhere in its training data,
     so any large gap means the direction is entangled with truth regardless.

  3. PER-CUE BREAKDOWN. If accuracy is carried by one cue type and near chance
     on the others, the probe is narrower than the label suggests. Report it.
"""

import json
import sys
from collections import defaultdict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold

import config
from acts import load
from probe import make_probe

TRAIN = "id_train_acts.npz"
HELD = "id_heldout_acts.npz"


def sweep():
    a, m = load(config.data_path(TRAIN))
    y = np.array([r["label"] for r in m])
    g = np.array([r["group"] for r in m])
    print(f"{a.shape[0]} samples, {len(set(g))} items, {a.shape[1]} layers")
    print(f"cue types in training: {sorted(set(r['cue'] for r in m))}\n")

    res = []
    for L in range(a.shape[1]):
        X = a[:, L, :]
        accs, f1s = [], []
        for tr, te in GroupKFold(n_splits=5).split(X, y, g):
            p = make_probe(config.C)
            p.fit(X[tr], y[tr])
            pred = p.predict(X[te])
            accs.append(accuracy_score(y[te], pred))
            f1s.append(f1_score(y[te], pred, average="macro"))
        res.append({"layer": L, "acc": float(np.mean(accs)),
                    "sd": float(np.std(accs)), "f1": float(np.mean(f1s))})

    print(f"  {'layer':>5} {'acc':>7} {'sd':>6} {'f1':>7}")
    for r in res:
        bar = "#" * int(max(0, r["acc"] - 0.5) * 100)
        print(f"  {r['layer']:>5} {r['acc']:>7.3f} {r['sd']:>6.3f} "
              f"{r['f1']:>7.3f}  {bar}")

    usable = res[:int(len(res) * 0.85)]
    b = max(usable, key=lambda r: r["acc"])
    depth = b["layer"] / (a.shape[1] - 1)
    print(f"\n  best usable layer {b['layer']} (acc {b['acc']:.3f}, "
          f"{depth:.0%} depth)")
    json.dump({"layer": b["layer"], "acc": b["acc"], "sweep": res},
              open(config.data_path("chosen_layer_v2.json"), "w"), indent=2)
    print("\n  Compare the profile with v1. Truth probes peak mid-stack and")
    print("  degrade (Burger et al.); a user-attribute probe need not.")
    print(f"\n  Then: PROBE_LAYER={b['layer']} python probe_v2.py validate")


def validate():
    a, m = load(config.data_path(TRAIN))
    y = np.array([r["label"] for r in m])
    probe = make_probe(config.C)
    probe.fit(a[:, config.LAYER, :], y)
    tr_margin = probe.decision_function(a[:, config.LAYER, :])
    gap = tr_margin[y == 1].mean() - tr_margin[y == 0].mean()

    print(f"model {config.MODEL}   layer {config.LAYER}   C {config.C}")
    print(f"trained gap (high vs low identity cues): {gap:+.3f}\n")

    print("=" * 70)
    print("1. CUE-TYPE GENERALISATION  -- the test v1 never had")
    print("=" * 70)
    try:
        ah, mh = load(config.data_path(HELD))
        yh = np.array([r["label"] for r in mh])
        cue = mh[0]["cue"]
        s = probe.decision_function(ah[:, config.LAYER, :])
        acc = ((s > 0) == (yh == 1)).mean()
        hi, lo = s[yh == 1], s[yh == 0]
        d = (hi.mean() - lo.mean()) / np.sqrt(
            (hi.std() ** 2 + lo.std() ** 2) / 2)
        print(f"  held-out cue type: {cue}   n = {len(yh)}")
        print(f"  accuracy {acc:.3f}   Cohen's d {d:+.2f}")
        print(f"  gap {hi.mean() - lo.mean():+.3f} "
              f"({(hi.mean() - lo.mean()) / gap:+.0%} of trained gap)")
        if acc > 0.70:
            print("\n  TRANSFERS. The probe recognises high vs low expertise")
            print("  in a cue type it never saw, sharing little vocabulary")
            print("  with training. This is a user representation.")
        elif acc > 0.58:
            print("\n  PARTIAL TRANSFER. Some generalisation, but weaker than")
            print("  within-training accuracy. Report both numbers.")
        else:
            print("\n  DOES NOT TRANSFER. The probe learned the vocabulary of")
            print("  the training cue types, not a user attribute. Same")
            print("  failure as v1 in a different guise -- do not build")
            print("  dynamics experiments on this direction.")
    except FileNotFoundError:
        print("  no held-out activations; extract data/id_heldout.jsonl first")

    print("\n" + "=" * 70)
    print("2. TRUTH CONTAMINATION  -- what killed v1")
    print("=" * 70)
    try:
        ac, mc = load(config.data_path("construct_acts.npz"))
        s = probe.decision_function(ac[:, config.LAYER, :])
        by = defaultdict(list)
        for r, v in zip(mc, s):
            by[r["condition"]].append(v)
        by = {k: np.array(v) for k, v in by.items()}
        ft = by["fact_true"].mean() - by["fact_false"].mean()
        se = by["stated_high"].mean() - by["stated_low"].mean()
        print(f"  generic factual truth  {ft:+.3f}  "
              f"({ft / gap:+.0%} of trained gap)")
        print(f"  stated expertise       {se:+.3f}  "
              f"({se / gap:+.0%} of trained gap)")
        print(f"\n  v1 for comparison: truth +98%, stated expertise -7%")
        if abs(ft) < 0.3 * abs(gap) and se > 0.4 * gap:
            print("\n  CLEAN. Tracks stated expertise, largely ignores")
            print("  generic truth. The construct holds this time.")
        elif abs(ft) > 0.5 * abs(gap):
            print("\n  STILL TRUTH-CONTAMINATED. Check the stimuli for")
            print("  sentences that make claims about how statistics works")
            print("  (stimuli_v2.py inspect) and remove them.")
        else:
            print("\n  Mixed. Report the decomposition rather than a verdict.")
    except FileNotFoundError:
        print("  no construct_acts.npz; run construct.py extract first")

    print("\n" + "=" * 70)
    print("3. PER-CUE BREAKDOWN")
    print("=" * 70)
    by = defaultdict(lambda: {"s": [], "y": []})
    for r, v in zip(m, tr_margin):
        by[r["cue"]]["s"].append(v)
        by[r["cue"]]["y"].append(r["label"])
    print(f"  {'cue':10} {'in-sample acc':>14} {'gap':>8}")
    for c, d in sorted(by.items()):
        s, yy = np.array(d["s"]), np.array(d["y"])
        acc = ((s > 0) == (yy == 1)).mean()
        g = s[yy == 1].mean() - s[yy == 0].mean()
        print(f"  {c:10} {acc:>14.3f} {g:>8.3f}")
    print("\n  In-sample, so inflated -- for comparing cue types only.")
    print("  If one cue carries everything, the probe is narrower than")
    print("  the label suggests and that belongs in limitations.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "validate"
    {"sweep": sweep, "validate": validate}[cmd]()
