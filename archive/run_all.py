"""Run the whole pipeline for one model.

    python run_all.py                    # everything, stops at layer choice
    python run_all.py --layer 27         # skip the pause, use this layer
    python run_all.py --from dose        # resume from a stage
    python run_all.py --stages           # list stage names and exit

Stages, in order:

    stimuli     build probe_train / dose / contra jsonl from pairs.json
    train       extract activations for probe training
    sweep       per-layer grouped CV, choose the probe layer
    dose        extract + analyse the dose-response experiment
    contra      extract + analyse the contradiction experiment
    figures     write the plots

Everything is keyed to config.MODEL_TAG, so switching models is two env vars
and a rerun. Extraction is resumable, so an interrupted stage picks up where it
stopped rather than restarting.

WHY THERE IS A PAUSE AT `sweep`

The probe layer is a judgement call, not an automatic one. The script prints
the accuracy-by-layer curve and its own pick, then stops. Look at the shape
before accepting it: accuracy that peaks in the first few layers means the
probe found vocabulary rather than expertise, and everything downstream would
be measuring the wrong thing. Pass --layer to skip the pause once you have
seen the curve.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

import config

STAGES = ["stimuli", "train", "sweep", "dose", "contra", "figures"]
LAYER_FILE = "chosen_layer.json"


def banner(name):
    print(f"\n{'=' * 68}\n  {name.upper()}   [{config.MODEL_TAG}]\n{'=' * 68}",
          flush=True)


PY = f'"{sys.executable}"'

def sh(cmd):
    print(f"  $ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        sys.exit(f"\nFAILED at: {cmd}")


def stage_stimuli():
    banner("stimuli")
    if not os.path.exists("data/pairs.json"):
        sys.exit("data/pairs.json missing. Run make_stimuli.py generate first.")
    sh(f"{sys.executable} make_stimuli.py build")
    sh(f"{sys.executable} dose.py build")
    sh(f"{sys.executable} contradiction.py build")


def stage_train():
    banner("train")
    import acts
    out = config.data_path("probe_train_acts.npz")
    t = time.time()
    a, m = acts.extract("data/probe_train.jsonl", out)
    print(f"  {a.shape} in {(time.time() - t) / 60:.1f} min")


def stage_sweep(force_layer=None):
    banner("sweep")
    import acts
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import GroupKFold
    from probe import make_probe

    a, m = acts.load(config.data_path("probe_train_acts.npz"))
    labels = np.array([r["label"] for r in m])
    groups = np.array([r.get("group", r["id"]) for r in m])
    n_groups = len(set(groups))
    print(f"  {a.shape[0]} samples, {n_groups} groups, "
          f"{a.shape[1]} layers")

    results = []
    for layer in range(a.shape[1]):
        X = a[:, layer, :]
        accs, f1s = [], []
        for tr, te in GroupKFold(n_splits=min(5, n_groups)).split(
                X, labels, groups):
            p = make_probe(config.C)
            p.fit(X[tr], labels[tr])
            pred = p.predict(X[te])
            accs.append(accuracy_score(labels[te], pred))
            f1s.append(f1_score(labels[te], pred, average="macro"))
        results.append({"layer": layer, "acc": float(np.mean(accs)),
                        "sd": float(np.std(accs)), "f1": float(np.mean(f1s))})

    print(f"\n  {'layer':>5} {'acc':>7} {'sd':>6} {'f1':>7}")
    for r in results:
        bar = "#" * int(max(0, r["acc"] - 0.5) * 100)
        print(f"  {r['layer']:>5} {r['acc']:>7.3f} {r['sd']:>6.3f} "
              f"{r['f1']:>7.3f}  {bar}")

    usable = results[:int(len(results) * 0.85)]
    best = max(usable, key=lambda r: r["acc"])
    json.dump({"layer": best["layer"], "acc": best["acc"],
               "model": config.MODEL, "sweep": results},
              open(config.data_path(LAYER_FILE), "w"), indent=2)

    depth = best["layer"] / (a.shape[1] - 1)
    print(f"\n  suggested layer {best['layer']} "
          f"(acc {best['acc']:.3f}, {depth:.0%} depth)")

    if min(r["acc"] for r in results) < 0.45:
        print("\n  WARNING: below-chance accuracy somewhere. Check that the")
        print("  meta carries a 'group' field per stimulus pair -- minimal")
        print("  pairs straddling a CV split produce exactly this.")
    if max(r["acc"] for r in results[:6]) >= best["acc"] - 0.02:
        print("\n  WARNING: early layers as good as the peak. The probe may")
        print("  be reading vocabulary, not expertise. Check stimulus")
        print("  matching before trusting anything downstream.")

    if force_layer is None:
        print(f"\n  Look at the curve. If it looks right, rerun with:")
        print(f"    python run_all.py --from dose --layer {best['layer']}")
        sys.exit(0)


def resolve_layer(cli_layer):
    if cli_layer is not None:
        return cli_layer
    p = config.data_path(LAYER_FILE)
    if os.path.exists(p):
        return json.load(open(p))["layer"]
    return config.LAYER


def stage_dose(layer):
    banner("dose")
    env = f"PROBE_LAYER={layer}"
    sh(f"{env} {sys.executable} dose.py extract" if os.name != "nt"
       else f"set PROBE_LAYER={layer} && {sys.executable} dose.py extract")
    sh(f"set PROBE_LAYER={layer} && {sys.executable} dose.py analyse"
       if os.name == "nt"
       else f"{env} {sys.executable} dose.py analyse")


def stage_contra(layer):
    banner("contradiction")
    if os.name == "nt":
        sh(f"set PROBE_LAYER={layer} && {sys.executable} contradiction.py extract")
        sh(f"set PROBE_LAYER={layer} && {sys.executable} contradiction.py analyse")
    else:
        sh(f"PROBE_LAYER={layer} {sys.executable} contradiction.py extract")
        sh(f"PROBE_LAYER={layer} {sys.executable} contradiction.py analyse")


def stage_figures(layer):
    banner("figures")
    if os.name == "nt":
        sh(f"set PROBE_LAYER={layer} && {sys.executable} figures.py")
    else:
        sh(f"PROBE_LAYER={layer} {sys.executable} figures.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="stimuli", choices=STAGES)
    ap.add_argument("--only", default=None, choices=STAGES)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--stages", action="store_true")
    a = ap.parse_args()

    if a.stages:
        print("\n".join(STAGES))
        return

    config.setup()
    print(f"model  {config.MODEL}")
    print(f"tag    {config.MODEL_TAG}")
    print(f"device {config.device()}   batch {config.batch_size()}")

    todo = [a.only] if a.only else STAGES[STAGES.index(a.start):]
    layer = resolve_layer(a.layer)

    t0 = time.time()
    for s in todo:
        if s == "stimuli":
            stage_stimuli()
        elif s == "train":
            stage_train()
        elif s == "sweep":
            stage_sweep(a.layer)
            layer = resolve_layer(a.layer)
        elif s == "dose":
            stage_dose(layer)
        elif s == "contra":
            stage_contra(layer)
        elif s == "figures":
            stage_figures(layer)

    print(f"\n{'=' * 68}")
    print(f"  done in {(time.time() - t0) / 60:.1f} min   layer {layer}")
    print(f"  results under data/{config.MODEL_TAG}/")


if __name__ == "__main__":
    main()
