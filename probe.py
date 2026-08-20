"""Linear probes for user expertise.

THREE DEFAULTS HERE ARE DELIBERATE, and each one fixes a bug that produced a
wrong-looking-but-plausible result earlier in this project.

1. GROUPED CROSS-VALIDATION. Minimal pairs are near-identical text with
   opposite labels. Under a random split, one half of a pair trains and the
   other tests, so the probe learns the wording and confidently predicts the
   wrong label -- producing reliably BELOW-chance accuracy. The old code only
   printed a warning when this happened. Now GroupKFold is used whenever groups
   are supplied, and a warning fires if they are not.

2. C DEFAULTS TO config.C (0.01), NOT 1.0. At C=1.0 the decision function
   saturates: half the margins exceed |5| and predict_proba collapses to 0/1,
   destroying the graded signal the dynamics experiments depend on.

3. score() RETURNS MARGINS, NOT PROBABILITIES. predict_proba on 2048
   dimensions is wildly overconfident. decision_function is the signed distance
   from the hyperplane and stays graded.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config


def make_probe(C=None):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=config.C if C is None else C,
                           max_iter=2000, class_weight="balanced"),
    )


def layer_sweep(acts, labels, groups=None, n_splits=5, seed=0, C=None):
    """acts: [n_samples, n_layers, d_model]

    groups: one id per sample identifying its stimulus pair. Both halves of a
    pair are then kept on the same side of every split. Strongly recommended --
    without it the accuracies are not interpretable.
    """
    labels = np.asarray(labels)
    if groups is None:
        print("  WARNING: no groups given. Minimal pairs will straddle CV")
        print("  splits and accuracy may fall below chance. Pass groups.")
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                   random_state=seed)
        split = lambda X: splitter.split(X, labels)
    else:
        groups = np.asarray(groups)
        k = min(n_splits, len(set(groups)))
        splitter = GroupKFold(n_splits=k)
        split = lambda X: splitter.split(X, labels, groups)

    results = []
    for layer in range(acts.shape[1]):
        X = acts[:, layer, :]
        accs, f1s = [], []
        for tr, te in split(X):
            p = make_probe(C)
            p.fit(X[tr], labels[tr])
            pred = p.predict(X[te])
            accs.append(accuracy_score(labels[te], pred))
            f1s.append(f1_score(labels[te], pred, average="macro"))
        results.append({"layer": layer, "acc": float(np.mean(accs)),
                        "acc_std": float(np.std(accs)),
                        "f1": float(np.mean(f1s))})
    return results


def best_layer(results, exclude_last_frac=0.15):
    """Highest accuracy, ignoring the final layers.

    Late layers drift toward next-token prediction; their probes read well but
    tend to steer poorly.
    """
    cutoff = int(len(results) * (1 - exclude_last_frac))
    return max(results[:cutoff], key=lambda r: r["acc"])


def fit_final_probe(acts, labels, layer=None, C=None):
    p = make_probe(C)
    p.fit(acts[:, config.LAYER if layer is None else layer, :],
          np.asarray(labels))
    return p


def score(probe, acts, layer=None):
    """Signed distance from the hyperplane. Higher = read as more expert.

    Margins, not probabilities -- see the module docstring.
    """
    return probe.decision_function(
        acts[:, config.LAYER if layer is None else layer, :])


def probe_direction(probe):
    """Unit-norm reading direction in activation space.

    NOTE FOR STEERING: the direction that reads best is often not the direction
    that steers best -- TalkTuner had to train separate control probes for this
    reason. Verify causally before claiming a direction is used by the model.
    Compare against mean_diff_direction below.
    """
    scaler = probe.named_steps["standardscaler"]
    lr = probe.named_steps["logisticregression"]
    w = lr.coef_[0] / scaler.scale_
    return w / np.linalg.norm(w)


def mean_diff_direction(acts, labels, layer=None):
    """Difference-of-means direction: mean(positive) - mean(negative).

    Often steers better than the logistic direction, which is optimised to
    discriminate rather than to move the representation. Worth trying both.
    """
    layer = config.LAYER if layer is None else layer
    labels = np.asarray(labels)
    X = acts[:, layer, :]
    v = X[labels == 1].mean(0) - X[labels == 0].mean(0)
    return v / np.linalg.norm(v)


def margin_scale(probe, acts, layer=None):
    """SD of training margins. Use this to express steering strength in units
    the probe itself defines, rather than in arbitrary activation norms."""
    return float(score(probe, acts, layer).std())


def print_sweep(results, n_layers=None):
    print(f"  {'layer':>5} {'acc':>7} {'sd':>6} {'f1':>7}")
    for r in results:
        bar = "#" * int(max(0, r["acc"] - 0.5) * 100)
        print(f"  {r['layer']:>5} {r['acc']:>7.3f} {r['acc_std']:>6.3f} "
              f"{r['f1']:>7.3f}  {bar}")
    b = best_layer(results)
    depth = b["layer"] / max(len(results) - 1, 1)
    print(f"\n  best usable layer: {b['layer']} "
          f"(acc {b['acc']:.3f}, f1 {b['f1']:.3f}, {depth:.0%} depth)")

    if min(r["acc"] for r in results) < 0.45:
        print("\n  WARNING: below-chance accuracy. Almost always means minimal")
        print("  pairs are straddling CV splits -- check that groups is passed.")
    if max(r["acc"] for r in results[:6]) >= b["acc"] - 0.02:
        print("\n  WARNING: early layers match the peak. The probe may be")
        print("  reading vocabulary rather than expertise. Check that correct")
        print("  and incorrect statements share their technical terms.")
    return b