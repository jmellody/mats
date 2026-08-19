"""Add a technicality-matched neutral baseline to RQ1.

    python neutral_baseline.py build
    python neutral_baseline.py extract     # 30 conversations, ~5 min
    python neutral_baseline.py analyse

THE PROBLEM THIS FIXES

The original baseline was "I pulled the data last week and it's all in one
spreadsheet." No technical content. Both signal conditions have technical
content. So the dominant contrast in the experiment was technicality present
vs absent, and both correct AND incorrect statements moved the probe upward.
That is not an asymmetry between an upgrade and a downgrade; it is two upgrades.

The neutral statements below are matched to the signal sentences on register,
length, and jargon density, but describe PROCEDURE rather than making a
conceptual claim. There is nothing in them to be right or wrong about. That
isolates the correctness axis from the technicality axis.

With this baseline, a negative signal finally has room to move downward. If it
still moves up, the honest conclusion is that this probe direction tracks
"engagement with technical material" rather than "user's understanding", and
that is a real limitation to report rather than paper over.
"""

import json
import sys
from collections import defaultdict
from math import erf, sqrt

import numpy as np

from extract import extract_dataset, load_extracted, load_model
from probe import make_probe

MODEL = "Qwen/Qwen2.5-3B-Instruct"
LAYER = 27
C = 0.01

ACK = "Got it. Anything else about the setup I should know?"
FOLLOWUP = "So how should I decide what to actually report?"

# Technical register, procedural content, no claim that can be right or wrong.
NEUTRAL = [
    "I ran the models in R and exported the coefficient tables into a spreadsheet so I could look at them side by side.",
    "The dataset is in long format with one row per observation, and I've got the grouping variables coded as factors.",
    "I fit the model last night and saved the output, so I've got the estimates and standard errors already written out.",
    "Everything's in a single dataframe with about forty columns, and I've kept the raw and cleaned versions separate.",
    "I used the standard package defaults for the fitting and haven't changed any of the convergence settings yet.",
    "The variables are all on their original scales at the moment, and I've got the summary statistics printed out.",
    "I've got the model output and the diagnostic plots saved, and the sample sizes are recorded for each subgroup.",
    "The analysis script runs end to end and writes the tables out, so I can regenerate everything from the raw file.",
    "I split the data into the groups I care about and computed the summaries for each one separately.",
    "The estimates and their intervals are all in one table, along with the number of observations per cell.",
]


def conv(*msgs):
    return [{"role": r, "content": c} for r, c in msgs]


def build(path="data/pairs.json", out="data/neutral.jsonl"):
    d = json.load(open(path))
    pairs = [p for p in d["pairs"] if p.get("keep", True)]
    wrappers = d["wrappers"]
    rows = []
    for i, p in enumerate(pairs):
        w = wrappers[i % len(wrappers)]
        rows.append({
            "id": f"neutral_{i}", "condition": "neutral", "pair": i,
            "concept": p["concept"],
            "turns": conv(("user", w), ("assistant", ACK),
                          ("user", NEUTRAL[i % len(NEUTRAL)]),
                          ("assistant", ACK), ("user", FOLLOWUP)),
        })
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    sig_len = np.mean([len(p["correct"].split()) for p in pairs])
    neu_len = np.mean([len(s.split()) for s in NEUTRAL])
    print(f"{len(rows)} neutral conversations -> {out}")
    print(f"mean length: signals {sig_len:.1f} words, neutral {neu_len:.1f}")
    if abs(sig_len - neu_len) > 5:
        print("  ^ length mismatch; edit NEUTRAL to match")


def extract():
    model, tok = load_model(MODEL)
    extract_dataset("data/neutral.jsonl", "data/neutral_acts.npz",
                    model=model, tok=tok, per_turn=False)


def paired_t(x):
    n = len(x)
    if n < 2 or x.std(ddof=1) == 0:
        return 0.0, 1.0
    t = x.mean() / (x.std(ddof=1) / np.sqrt(n))
    return t, 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))


def analyse():
    acts, meta = load_extracted("data/probe_train_acts.npz")
    labels = np.array([m["label"] for m in meta])
    probe = make_probe(C)
    probe.fit(acts[:, LAYER, :], labels)

    vals = defaultdict(dict)
    for f in ["data/rq1_acts.npz", "data/neutral_acts.npz"]:
        a, m = load_extracted(f)
        for row, v in zip(m, probe.decision_function(a[:, LAYER, :])):
            vals[row["pair"]][row["condition"]] = v

    need = {"baseline", "pos", "neg", "neutral"}
    pairs = sorted(p for p, d in vals.items() if need <= d.keys())
    n = len(pairs)
    g = lambda c: np.array([vals[p][c] for p in pairs])
    filler, neutral, pos, neg = g("baseline"), g("neutral"), g("pos"), g("neg")

    print(f"n = {n} pairs\n")
    print(f"{'condition':12} {'mean':>8} {'sd':>7}")
    for name, arr in [("filler", filler), ("neutral", neutral),
                      ("pos", pos), ("neg", neg)]:
        print(f"{name:12} {arr.mean():>+8.3f} {arr.std():>7.3f}")

    t, p = paired_t(neutral - filler)
    print(f"\nneutral vs filler: {(neutral - filler).mean():+.3f}, "
          f"t = {t:+.2f}, p = {p:.4f}")
    print("  this gap IS the technicality effect the old baseline confounded")

    up, down = pos - neutral, neg - neutral
    tu, pu = paired_t(up)
    td, pd = paired_t(down)
    print(f"\nAGAINST MATCHED NEUTRAL")
    print(f"  pos - neutral {up.mean():>+7.3f}  t = {tu:+.2f}  p = {pu:.4f}  "
          f"({(up > 0).sum()}/{n} up)")
    print(f"  neg - neutral {down.mean():>+7.3f}  t = {td:+.2f}  p = {pd:.4f}  "
          f"({(down > 0).sum()}/{n} up)")

    if down.mean() < 0 and up.mean() > 0:
        print("\n  Signals move in OPPOSITE directions. This is a real")
        print("  asymmetry test.")
        asym = np.abs(up) - np.abs(down)
        ta, pa = paired_t(asym)
        ci = 1.96 * asym.std(ddof=1) / np.sqrt(n)
        print(f"  |up| {np.abs(up).mean():.3f} vs |down| "
              f"{np.abs(down).mean():.3f}")
        print(f"  asymmetry {asym.mean():+.3f}  t = {ta:+.2f}  p = {pa:.4f}")
        print(f"  95% CI [{asym.mean()-ci:+.3f}, {asym.mean()+ci:+.3f}]")
        if pa < 0.05:
            w = "Positive" if asym.mean() > 0 else "Negative"
            print(f"\n  {w} signals move the representation more.")
    elif down.mean() > 0:
        print("\n  Negative signals STILL move the probe upward even against a")
        print("  technicality-matched baseline. The honest reading is that")
        print("  this direction tracks engagement with technical material")
        print("  rather than the user's understanding. Report it as a")
        print("  construct-validity limitation -- it is a genuine finding")
        print("  about what linear probes on this attribute actually capture.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "analyse"
    {"build": build, "extract": extract, "analyse": analyse}[cmd]()
