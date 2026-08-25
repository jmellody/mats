import numpy as np
from sklearn.metrics import roc_auc_score


def load(path):
    d = np.load(path, allow_pickle=True)
    return d["acts"], d["topic"], d["label"], d["idx"]


def split(idx, frac=0.5, seed=0):
    rng = np.random.default_rng(seed)
    u = np.unique(idx)
    tr = set(rng.choice(u, int(len(u) * frac), replace=False).tolist())
    return np.array([i in tr for i in idx])


def direction(X_exp, X_nov):
    d = X_exp.mean(0) - X_nov.mean(0)
    return d / np.linalg.norm(d)


def fit_topic(acts, topic, label, idx, t, layer, seed=0):
    """Returns (direction, auc, expert_mean, novice_mean) fit on train half, auc on test half."""
    m = topic == t
    tr = split(idx[m], seed=seed)
    X, y = acts[m][:, layer, :], label[m]
    e, n = y == "expert", y == "novice"

    d = direction(X[e & tr], X[n & tr])
    te = ~tr & (e | n)
    auc = roc_auc_score((y[te] == "expert").astype(int), X[te] @ d)
    return d, auc, X[e & tr] @ d, X[n & tr] @ d


def layer_sweep(acts, topic, label, idx, topics, seed=0):
    return np.array([
        [fit_topic(acts, topic, label, idx, t, L, seed)[1] for t in topics]
        for L in range(acts.shape[1])
    ])


def calibrate(acts, topic, label, idx, t, layer, seed=0):
    """Position of each neutral prompt between novice (0) and expert (1) anchors."""
    d, auc, se, sn = fit_topic(acts, topic, label, idx, t, layer, seed)
    m = (topic == t) & (label == "neutral")
    s = acts[m][:, layer, :] @ d
    lo, hi = sn.mean(), se.mean()
    return (s - lo) / (hi - lo), auc, (sn - lo) / (hi - lo), (se - lo) / (hi - lo)


def bootstrap_ci(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    b = rng.choice(x, (n, len(x)), replace=True).mean(1)
    return np.percentile(b, [2.5, 97.5])
