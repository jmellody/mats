"""Re-run the layer sweep with pair-grouped cross-validation.

    python resweep.py

Why this exists: minimal pairs are near-identical text with opposite labels.
If the two halves of a pair land on opposite sides of a CV split, the probe
learns the wording from one half and confidently predicts the wrong label on
the other. That produces reliably BELOW-chance accuracy, which is what a random
StratifiedKFold gives you here.

GroupKFold keeps both halves of a pair together. Whatever accuracy survives is
the probe generalising to concepts and phrasings it has not seen, which is the
only number worth trusting.

No re-extraction: this reads the saved npz.
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold

from extract import load_extracted
from probe import make_probe


def sweep_grouped(acts, labels, groups, n_splits=5):
    labels, groups = np.asarray(labels), np.asarray(groups)
    results = []
    for layer in range(acts.shape[1]):
        X = acts[:, layer, :]
        gkf = GroupKFold(n_splits=n_splits)
        accs, f1s = [], []
        for tr, te in gkf.split(X, labels, groups):
            p = make_probe(0.01)
            p.fit(X[tr], labels[tr])
            pred = p.predict(X[te])
            accs.append(accuracy_score(labels[te], pred))
            f1s.append(f1_score(labels[te], pred, average="macro"))
        results.append({"layer": layer, "acc": float(np.mean(accs)),
                        "acc_std": float(np.std(accs)),
                        "f1": float(np.mean(f1s))})
    return results


if __name__ == "__main__":
    acts, meta = load_extracted("data/probe_train_acts.npz")
    labels = [m["label"] for m in meta]
    groups = [m.get("group", m["id"]) for m in meta]

    n_groups = len(set(groups))
    print(f"{acts.shape[0]} samples, {n_groups} groups (stimulus pairs)")
    if n_groups < 10:
        print("too few groups for 5-fold -- results will be unstable")

    results = sweep_grouped(acts, labels, groups,
                            n_splits=min(5, n_groups))

    print(f"\n{'layer':>5}  {'acc':>6}  {'sd':>5}  {'f1':>6}")
    for r in results:
        bar = "#" * int(max(0, r["acc"] - 0.5) * 100)
        print(f"{r['layer']:>5}  {r['acc']:>6.3f}  {r['acc_std']:>5.3f}  "
              f"{r['f1']:>6.3f}  {bar}")

    usable = results[:int(len(results) * 0.85)]
    b = max(usable, key=lambda r: r["acc"])
    print(f"\nbest usable layer: {b['layer']} "
          f"(acc {b['acc']:.3f}, f1 {b['f1']:.3f})")

    mid = [r["acc"] for r in results[8:28]]
    if max(mid) < 0.6:
        print("\nMiddle layers near chance. The probe is not finding a")
        print("generalisable expertise direction. Likely causes, in order:")
        print("  1. signals too subtle -- try blatant errors as a pilot")
        print("  2. probing the wrong token position (try mean over the")
        print("     signal turn rather than the final token)")
        print("  3. 3B model may not represent this attribute cleanly")
    if min(r["acc"] for r in results) < 0.45:
        print("\nStill below chance somewhere -- grouping did not fully fix it.")
        print("Check that meta carries a 'group' field per stimulus pair.")
