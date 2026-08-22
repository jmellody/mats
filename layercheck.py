import json, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

TAG = os.environ.get("PROBE_TAG", "qwen35-4b")
TRAIN = os.environ.get("PROBE_TRAIN", "ag_train_acts.npz")
HELD = os.environ.get("PROBE_HELD", "ag_heldout_acts.npz")

def load(name):
    d = np.load(f"data/{TAG}/{name}", allow_pickle=False)
    return d["acts"], json.loads(str(d["meta"]))

a, m = load(TRAIN)
y = np.array([r["label"] for r in m])
ah, mh = load(HELD)
yh = np.array([r["label"] for r in mh])

print("train: %s   held-out cue: %s   n=%d" % (TRAIN, mh[0]["cue"], len(yh)))
print("  %-6s %12s %12s" % ("layer", "in-sample", "held-out"))
for L in range(33):
    p = make_pipeline(StandardScaler(),
                      LogisticRegression(C=0.01, max_iter=2000,
                                         class_weight="balanced"))
    p.fit(a[:, L, :], y)
    print("  L%-5d %12.3f %12.3f" % (
        L, roc_auc_score(y, p.decision_function(a[:, L, :])),
        roc_auc_score(yh, p.decision_function(ah[:, L, :]))))
