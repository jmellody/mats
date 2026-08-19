"""End-to-end pipeline.

  python run.py sweep       -- train probes, print layer-by-layer accuracy
  python run.py experiment  -- run the 5 conditions at the chosen layer

Set MODEL and ELICITATION below. Run the whole thing twice, once with
ELICITATION = None and once with the string, and check the results survive
both. If they only appear with the elicitation prompt, the probe may be
reading the prompt rather than the model's view of the user.
"""

import sys
from collections import defaultdict

import numpy as np

from extract import extract_dataset, load_extracted, load_model
from probe import (best_layer, fit_final_probe, layer_sweep, print_sweep,
                   score)

MODEL = "Qwen/Qwen2.5-3B-Instruct"
ELICITATION = None
# ELICITATION = "\nHow much does this user know about statistics? I think they know"

LAYER = None  # set after running the sweep


def sweep():
    model, tok = load_model(MODEL)
    acts, meta = extract_dataset(
        "data/probe_train.jsonl", "data/probe_train_acts.npz",
        model=model, tok=tok, elicitation=ELICITATION,
    )
    labels = [m["label"] for m in meta]
    n_per_class = min(labels.count(0), labels.count(1))
    if n_per_class < 30:
        print(f"\n!! only {n_per_class} per class -- accuracy here is noise. "
              f"Generate more stimuli before believing this.\n")
    results = layer_sweep(acts, labels, n_splits=min(5, n_per_class))
    print_sweep(results)


def experiment(layer=None):
    train_acts, train_meta = load_extracted("data/probe_train_acts.npz")
    labels = [m["label"] for m in train_meta]

    if layer is None:
        layer = LAYER or best_layer(layer_sweep(train_acts, labels))["layer"]
    print(f"using layer {layer}")

    probe = fit_final_probe(train_acts, labels, layer)

    model, tok = load_model(MODEL)
    acts, meta = extract_dataset(
        "data/experiment.jsonl", "data/experiment_acts.npz",
        model=model, tok=tok, elicitation=ELICITATION, per_turn=True,
    )
    p = score(probe, acts, layer)

    # trajectory: probe value after each user turn, per condition
    traj = defaultdict(lambda: defaultdict(list))
    for m, v in zip(meta, p):
        traj[m["condition"]][m["n_turns"]].append(v)

    print(f"\n{'condition':<14} " + " ".join(f"t{t:<5}" for t in sorted(
        {m['n_turns'] for m in meta})))
    finals = {}
    for cond in ["baseline", "pos_only", "neg_only", "pos_then_neg", "neg_then_pos"]:
        if cond not in traj:
            continue
        ts = sorted(traj[cond])
        means = [np.mean(traj[cond][t]) for t in ts]
        finals[cond] = means[-1]
        print(f"{cond:<14} " + " ".join(f"{m:<6.3f}" for m in means))

    if {"baseline", "pos_only", "neg_only"} <= finals.keys():
        up = finals["pos_only"] - finals["baseline"]
        down = finals["neg_only"] - finals["baseline"]
        print(f"\nRQ1  up {up:+.3f}   down {down:+.3f}   "
              f"asymmetry {abs(up) - abs(down):+.3f}")
        print("     ceiling check -- baseline sits at "
              f"{finals['baseline']:.3f}; if that is near 0 or 1 the "
              "asymmetry may just be headroom")

    if {"neg_only", "pos_then_neg", "pos_only"} <= finals.keys():
        from_neutral = finals["neg_only"] - finals["baseline"]
        from_established = finals["pos_then_neg"] - finals["pos_only"]
        print(f"\nRQ2  neg from neutral {from_neutral:+.3f}   "
              f"neg after established {from_established:+.3f}")
        print("     smaller second number = representation resists revision")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "sweep"
    if cmd == "sweep":
        sweep()
    elif cmd == "experiment":
        experiment(int(sys.argv[2]) if len(sys.argv) > 2 else None)
