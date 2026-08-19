"""Train per-layer linear probes for user expertise and pick a working layer."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_probe(C=1.0):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, max_iter=2000, class_weight="balanced"),
    )


def layer_sweep(acts, labels, n_splits=5, seed=0, groups=None):
    """acts: [n_samples, n_layers, d_model]; labels: [n_samples]

    groups: optional array of conversation ids. If your stimuli were built from
    templates, samples from the same template must not straddle the split --
    otherwise you are measuring memorisation of the template, not expertise.
    """
    labels = np.asarray(labels)
    n_layers = acts.shape[1]
    results = []

    for layer in range(n_layers):
        X = acts[:, layer, :]
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        accs, f1s = [], []
        for tr, te in skf.split(X, labels):
            if groups is not None:
                overlap = set(np.asarray(groups)[tr]) & set(np.asarray(groups)[te])
                if overlap:
                    print(f"  WARNING layer {layer}: {len(overlap)} groups straddle split")
            p = make_probe()
            p.fit(X[tr], labels[tr])
            pred = p.predict(X[te])
            accs.append(accuracy_score(labels[te], pred))
            f1s.append(f1_score(labels[te], pred, average="macro"))
        results.append(
            {
                "layer": layer,
                "acc": float(np.mean(accs)),
                "acc_std": float(np.std(accs)),
                "f1": float(np.mean(f1s)),
            }
        )
    return results


def best_layer(results, exclude_last_frac=0.15):
    """Highest accuracy, ignoring the final layers.

    The last few layers drift toward next-token prediction and their probes
    tend to be less useful for steering.
    """
    n = len(results)
    cutoff = int(n * (1 - exclude_last_frac))
    usable = results[:cutoff]
    return max(usable, key=lambda r: r["acc"])


def fit_final_probe(acts, labels, layer, C=1.0):
    p = make_probe(C)
    p.fit(acts[:, layer, :], np.asarray(labels))
    return p


def score(probe, acts, layer):
    """P(expert) for each sample, at the chosen layer."""
    return probe.predict_proba(acts[:, layer, :])[:, 1]


def probe_direction(probe):
    """Unit-norm direction in activation space. Use this for steering later.

    Note: the direction that reads best is often not the direction that steers
    best. Verify causally before claiming anything about it.
    """
    scaler = probe.named_steps["standardscaler"]
    lr = probe.named_steps["logisticregression"]
    w = lr.coef_[0] / scaler.scale_
    return w / np.linalg.norm(w)


def print_sweep(results):
    print(f"{'layer':>5}  {'acc':>6}  {'sd':>5}  {'f1':>6}")
    for r in results:
        print(f"{r['layer']:>5}  {r['acc']:>6.3f}  {r['acc_std']:>5.3f}  {r['f1']:>6.3f}")
    b = best_layer(results)
    print(f"\nbest usable layer: {b['layer']} (acc {b['acc']:.3f}, f1 {b['f1']:.3f})")
