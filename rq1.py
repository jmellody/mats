"""RQ1: are positive and negative signals symmetric in their effect?

    python rq1.py build     # write matched 5-message conversations
    python rq1.py extract   # forward passes (the slow part, ~15 min)
    python rq1.py analyse   # paired stats, no model needed

WHY THIS DESIGN

The pilot failed because experiment conversations were 7 messages while probe
training was 5, so the probe was scoring inputs far outside its training
distribution -- baseline drifted by 9.2 while the actual effect was 1.06.

Here every conversation is the same 5-message skeleton:

    wrapper, ACK, <signal slot>, ACK, FOLLOWUP

with the slot holding filler, the correct statement, or the incorrect one.
Structurally identical to probe training, so the probe is in distribution. The
only thing that varies is the signal. That is the whole experiment.

Because each pair supplies all three conditions with the same wrapper, every
pair is its own control and the comparison is paired -- far more powerful at
n=30 than comparing group means.

This design cannot answer RQ2 (establish vs update), which needs two signal
slots and therefore a separately trained 7-message probe. Do RQ1 first; if the
asymmetry is not there, RQ2 is not worth the extra hour.
"""

import json
import sys
from collections import defaultdict

import numpy as np

from extract import extract_dataset, load_extracted, load_model
from probe import make_probe

MODEL = "Qwen/Qwen2.5-3B-Instruct"
LAYER = 27
C = 0.01  # from diagnose.py: d=0.83, no saturation

ACK = "Got it. Anything else about the setup I should know?"
FOLLOWUP = "So how should I decide what to actually report?"
FILLER = "I pulled the data last week and it's all in one spreadsheet."


def conv(*msgs):
    return [{"role": r, "content": c} for r, c in msgs]


def build(path="data/pairs.json", out="data/rq1.jsonl"):
    d = json.load(open(path))
    pairs = [p for p in d["pairs"] if p.get("keep", True)]
    wrappers = d["wrappers"]
    rows = []
    for i, p in enumerate(pairs):
        w = wrappers[i % len(wrappers)]
        for cond, signal in [("baseline", FILLER),
                             ("pos", p["correct"]),
                             ("neg", p["incorrect"])]:
            rows.append({
                "id": f"{cond}_{i}", "condition": cond, "pair": i,
                "concept": p["concept"],
                "turns": conv(("user", w), ("assistant", ACK),
                              ("user", signal), ("assistant", ACK),
                              ("user", FOLLOWUP)),
            })
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(pairs)} pairs x 3 conditions = {len(rows)} -> {out}")
    print("Every conversation is 5 messages, matching probe training.")


def extract(inp="data/rq1.jsonl", out="data/rq1_acts.npz"):
    model, tok = load_model(MODEL)
    extract_dataset(inp, out, model=model, tok=tok, per_turn=False)


def fit_probe():
    acts, meta = load_extracted("data/probe_train_acts.npz")
    labels = np.array([m["label"] for m in meta])
    p = make_probe(C)
    p.fit(acts[:, LAYER, :], labels)
    lo, hi = np.percentile(p.decision_function(acts[:, LAYER, :]), [1, 99])
    return p, (lo, hi)


def analyse():
    probe, train_range = fit_probe()
    acts, meta = load_extracted("data/rq1_acts.npz")
    m = probe.decision_function(acts[:, LAYER, :])

    by_pair = defaultdict(dict)
    for row, v in zip(meta, m):
        by_pair[row["pair"]][row["condition"]] = v

    complete = [p for p in by_pair.values()
                if {"baseline", "pos", "neg"} <= p.keys()]
    n = len(complete)
    base = np.array([p["baseline"] for p in complete])
    pos = np.array([p["pos"] for p in complete])
    neg = np.array([p["neg"] for p in complete])

    print(f"n = {n} pairs\n")

    lo, hi = train_range
    frac_out = np.mean((m < lo) | (m > hi))
    print(f"training margin range (1-99%): [{lo:+.2f}, {hi:+.2f}]")
    print(f"experiment range: [{m.min():+.2f}, {m.max():+.2f}]")
    print(f"{frac_out:.0%} of experiment margins outside training range")
    if frac_out > 0.2:
        print("  ^ still out of distribution -- results unreliable\n")
    else:
        print("  ^ in distribution, good\n")

    up = pos - base
    down = neg - base

    print(f"{'':16} {'mean':>8} {'sd':>7} {'sem':>7}")
    print(f"{'baseline':16} {base.mean():>+8.3f} {base.std():>7.3f}")
    print(f"{'pos - baseline':16} {up.mean():>+8.3f} {up.std():>7.3f} "
          f"{up.std()/np.sqrt(n):>7.3f}")
    print(f"{'neg - baseline':16} {down.mean():>+8.3f} {down.std():>7.3f} "
          f"{down.std()/np.sqrt(n):>7.3f}")

    # paired test on |up| vs |down| -- the asymmetry question
    asym = np.abs(up) - np.abs(down)
    t, p_val = paired_t(asym)
    print(f"\nRQ1  |up| = {np.abs(up).mean():.3f}   "
          f"|down| = {np.abs(down).mean():.3f}")
    print(f"     asymmetry = {asym.mean():+.3f}  "
          f"t({n-1}) = {t:+.2f}, p = {p_val:.4f}")
    d = asym.mean() / asym.std() if asym.std() > 0 else 0
    print(f"     Cohen's d = {d:+.2f}")

    if p_val < 0.05:
        which = "positive" if asym.mean() > 0 else "negative"
        print(f"\n     Signed: {which} signals move the probe more.")
    else:
        print("\n     No significant asymmetry. With n=30 this rules out")
        print("     large effects, not small ones -- report the CI, not")
        print("     'no difference'.")

    ci = 1.96 * asym.std() / np.sqrt(n)
    print(f"     95% CI on asymmetry: [{asym.mean()-ci:+.3f}, "
          f"{asym.mean()+ci:+.3f}]")

    # sanity: do the signals move the probe in the expected directions at all?
    tu, pu = paired_t(up)
    td, pd = paired_t(down)
    print(f"\nmanipulation check")
    print(f"  pos vs baseline: t = {tu:+.2f}, p = {pu:.4f}")
    print(f"  neg vs baseline: t = {td:+.2f}, p = {pd:.4f}")
    if pu > 0.05 and pd > 0.05:
        print("  Neither signal moves the probe. Nothing to compare --")
        print("  the signals are too subtle or the probe is mispositioned.")

    per_concept = defaultdict(list)
    for p, row in zip(complete, [r for r in meta if r["condition"] == "pos"]):
        per_concept[row["concept"]].append(p["pos"] - p["baseline"])
    spread = [np.mean(v) for v in per_concept.values()]
    if len(spread) > 3:
        print(f"\nper-concept effect spread: {min(spread):+.2f} to "
              f"{max(spread):+.2f}")
        print("  wide spread means expertise may not be one direction")


def paired_t(x):
    """One-sample t on paired differences, no scipy dependency."""
    n = len(x)
    if n < 2 or x.std() == 0:
        return 0.0, 1.0
    t = x.mean() / (x.std(ddof=1) / np.sqrt(n))
    # normal approximation to the p-value; fine at n=30
    from math import erf, sqrt
    p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
    return t, p


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "analyse"
    if cmd == "build":
        build()
    elif cmd == "extract":
        extract()
    else:
        analyse()
