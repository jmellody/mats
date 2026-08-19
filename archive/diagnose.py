"""Diagnose the probe before trusting any experiment numbers.

    python diagnose.py 27

Three checks, in the order that matters:

1. VALIDITY. Does the probe separate correct from incorrect statements at all,
   on held-out pairs? If not, nothing downstream means anything -- the probe is
   reading technical density, not expertise. This is the check that the layer
   sweep accuracy only weakly implies.

2. SATURATION. predict_proba on 2048-dim logistic regression is wildly
   overconfident and collapses to 0/1, destroying the graded signal your
   dynamics question depends on. decision_function (signed distance from the
   hyperplane) keeps it. Everything downstream should use margins, not
   probabilities.

3. BASELINE DRIFT. The baseline condition has no expertise signal in it. If the
   probe moves across turns there, it is tracking turn position or filler text,
   and every condition difference is contaminated by that.
"""

import sys
from collections import defaultdict

import numpy as np
from sklearn.model_selection import GroupKFold

from extract import load_extracted
from probe import make_probe


def margins_cv(acts, labels, groups, layer, C=0.01):
    """Held-out decision-function values. Lower C = stronger regularisation,
    which is what stops the saturation."""
    labels, groups = np.asarray(labels), np.asarray(groups)
    X = acts[:, layer, :]
    out = np.zeros(len(labels))
    gkf = GroupKFold(n_splits=min(5, len(set(groups))))
    for tr, te in gkf.split(X, labels, groups):
        p = make_probe(C)
        p.fit(X[tr], labels[tr])
        out[te] = p.decision_function(X[te])
    return out


def check_validity(acts, meta, layer):
    labels = np.array([m["label"] for m in meta])
    groups = [m.get("group", m["id"]) for m in meta]

    print("=" * 60)
    print("1. VALIDITY -- can the probe tell correct from incorrect?")
    print("=" * 60)
    for C in [1.0, 0.1, 0.01, 0.001]:
        m = margins_cv(acts, labels, groups, layer, C)
        pos, neg = m[labels == 1], m[labels == 0]
        d = (pos.mean() - neg.mean()) / np.sqrt(
            (pos.std() ** 2 + neg.std() ** 2) / 2)
        acc = ((m > 0) == (labels == 1)).mean()
        sat = np.mean(np.abs(m) > 5)
        print(f"  C={C:<6} acc {acc:.3f}  Cohen's d {d:+.2f}  "
              f"margin range [{m.min():+.1f}, {m.max():+.1f}]  "
              f"saturated {sat:.0%}")
    print("\n  Cohen's d under ~0.5 means the probe barely separates the")
    print("  classes. Any experiment built on it will measure noise.")


def check_experiment(train_acts, train_meta, layer, C=0.01):
    print("\n" + "=" * 60)
    print("2. EXPERIMENT -- margins, not probabilities")
    print("=" * 60)
    labels = np.array([m["label"] for m in train_meta])
    probe = make_probe(C)
    probe.fit(train_acts[:, layer, :], labels)

    acts, meta = load_extracted("data/experiment_acts.npz")
    m = probe.decision_function(acts[:, layer, :])

    traj = defaultdict(lambda: defaultdict(list))
    for row, v in zip(meta, m):
        traj[row["condition"]][row["n_turns"]].append(v)

    turns = sorted({row["n_turns"] for row in meta})
    print(f"\n{'condition':<14} " + " ".join(f"t{t:<8}" for t in turns))
    means = {}
    for cond in ["baseline", "pos_only", "neg_only",
                 "pos_then_neg", "neg_then_pos"]:
        if cond not in traj:
            continue
        row = [np.mean(traj[cond][t]) for t in turns]
        means[cond] = row
        print(f"{cond:<14} " + " ".join(f"{v:<+9.3f}" for v in row))

    print("\n" + "=" * 60)
    print("3. BASELINE DRIFT -- the negative control")
    print("=" * 60)
    if "baseline" in means:
        b = means["baseline"]
        drift = max(b) - min(b)
        print(f"  baseline range across turns: {drift:.3f}")
        if "pos_only" in means:
            effect = abs(means["pos_only"][-1] - b[-1])
            print(f"  pos_only effect at final turn: {effect:.3f}")
            if drift > effect:
                print("\n  Baseline drift exceeds the effect. The probe is")
                print("  tracking turn position or filler, not the user.")
                print("  Fix this before interpreting any condition.")

    print("\n  Signal-recency check (t3 vs t5):")
    if "pos_then_neg" in means and "neg_then_pos" in means:
        early = means["pos_then_neg"][1] - means["neg_then_pos"][1]
        print(f"    signal 2 turns back: pos-neg separation {early:+.3f}")
    if "pos_only" in means and "neg_only" in means:
        late = means["pos_only"][2] - means["neg_only"][2]
        print(f"    signal 1 turn back:  pos-neg separation {late:+.3f}")
    print("    if the second is much larger, the representation is")
    print("    recency-driven and your dynamics question changes shape")


if __name__ == "__main__":
    layer = int(sys.argv[1]) if len(sys.argv) > 1 else 27
    acts, meta = load_extracted("data/probe_train_acts.npz")
    print(f"layer {layer}, {acts.shape[0]} training samples\n")
    check_validity(acts, meta, layer)
    try:
        check_experiment(acts, meta, layer)
    except FileNotFoundError:
        print("\n(no experiment_acts.npz yet)")
