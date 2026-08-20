"""End-to-end smoke test on a tiny model and a tiny stimulus set.

    $env:PROBE_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
    $env:PROBE_TAG="smoke"
    python smoke.py

Checks that every code path runs and produces correctly shaped output. Does NOT
check that results are meaningful -- with 6 concepts and a 0.5B model they will
not be. The point is to find import errors, path bugs, and shape mismatches on
a laptop instead of on a rented GPU.

THE TEST THAT MATTERS MOST is the batching check. config.batch_size() returns 1
on CPU, so the batched last-token read never runs during normal laptop use --
but it is what will run on GPU. If padding is on the wrong side, position -1 is
a pad token for every sequence shorter than the longest in its batch, and you
get plausible-looking garbage with no error. This forces a batch and asserts
that batched and unbatched extraction agree.
"""

import json
import os
import shutil
import sys

import numpy as np

import config
import acts

FAILURES = []


def check(name, condition, detail=""):
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {name}" + (f"  -- {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)
    return condition


def section(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def tiny_stimuli():
    """Six hand-written pairs. Enough to exercise every path."""
    pairs = [
        {"concept": "p-value", "keep": True,
         "correct": "I'm treating the p-value as how incompatible my data are with the null model rather than as a probability the null is true.",
         "incorrect": "I'm treating the p-value as the probability the null is true rather than as how incompatible my data are with the null model."},
        {"concept": "confidence interval", "keep": True,
         "correct": "I'm reading the interval as a range of parameter values my data are consistent with, not as a probability statement about this one interval.",
         "incorrect": "I'm reading the interval as a probability statement about this one interval, not as a range of parameter values my data are consistent with."},
        {"concept": "power", "keep": True,
         "correct": "My sample is small so a null result here tells me very little, since the study was underpowered for the effect size I care about.",
         "incorrect": "My sample is small so a null result here settles the question, since the study was underpowered for the effect size I care about."},
        {"concept": "multiple comparisons", "keep": True,
         "correct": "I ran this across eight outcomes so I should correct before treating any single significant result as meaningful on its own.",
         "incorrect": "I ran this across eight outcomes so I can treat each single significant result as meaningful on its own without correcting."},
        {"concept": "regression to the mean", "keep": True,
         "correct": "I picked these groups on a baseline score so some of the change I see is regression to the mean rather than a real effect.",
         "incorrect": "I picked these groups on a baseline score so none of the change I see is regression to the mean rather than a real effect."},
        {"concept": "heteroskedasticity", "keep": True,
         "correct": "The residuals fan out at the high end so my standard errors are probably too small and the intervals too narrow.",
         "incorrect": "The residuals fan out at the high end so my standard errors are probably too large and the intervals too wide."},
    ]
    wrappers = [
        "I've got survey responses from last quarter and I'm trying to work out whether the difference between two groups is worth reporting.",
        "We collected readings from about sixty sensors over three months and I need to figure out what to do with them.",
        "I have outcome data for a programme across four sites and I'm trying to work out whether it did anything.",
    ]
    os.makedirs("data", exist_ok=True)
    json.dump({"pairs": pairs, "wrappers": wrappers},
              open("data/pairs_smoke.json", "w"), indent=2)
    return pairs, wrappers


def build_probe_train(pairs, wrappers, out):
    rows = []
    for i, p in enumerate(pairs):
        for j, w in enumerate(wrappers):
            for label, key in [(1, "correct"), (0, "incorrect")]:
                rows.append({
                    "id": f"tr_{i}_{j}_{label}", "label": label,
                    "concept": p["concept"], "group": f"pair_{i}",
                    "turns": [
                        {"role": "user", "content": w},
                        {"role": "assistant", "content": config.ACK},
                        {"role": "user", "content": p[key]},
                        {"role": "assistant", "content": config.ACK},
                        {"role": "user", "content": config.FOLLOWUP},
                    ],
                })
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows


def main():
    section("0. CONFIG")
    dev = config.setup()
    print(f"  model    {config.MODEL}")
    print(f"  tag      {config.MODEL_TAG}")
    print(f"  device   {dev}")
    print(f"  dtype    {config.dtype()}")
    print(f"  batch    {config.batch_size()}")
    check("model tag is not a real experiment tag",
          "smoke" in config.MODEL_TAG.lower(),
          "set PROBE_TAG=smoke or you will overwrite real activations")

    d = config.data_path("probe_train_acts.npz")
    check("data_path creates a tagged directory", config.MODEL_TAG in d, d)

    section("1. STIMULI")
    pairs, wrappers = tiny_stimuli()
    train_file = config.data_path("smoke_train.jsonl")
    rows = build_probe_train(pairs, wrappers, train_file)
    check("stimulus file written", os.path.exists(train_file),
          f"{len(rows)} conversations")
    lens = {len(r["turns"]) for r in rows}
    check("all conversations same length", len(lens) == 1, str(lens))

    section("2. MODEL LOAD")
    model, tok = acts.load_model()
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    print(f"  {n_layers} layers, d_model {d_model}")
    check("tokenizer pads on the left", tok.padding_side == "left",
          f"got {tok.padding_side} -- batched reads would hit pad tokens")
    check("pad token set", tok.pad_token is not None)

    section("3. BATCHING (the GPU-only code path)")
    texts = [acts.format_conversation(r["turns"], tok) for r in rows[:6]]
    tok_lens = [len(tok(t)["input_ids"]) for t in texts]
    print(f"  token lengths in batch: {tok_lens}")
    check("batch has ragged lengths", len(set(tok_lens)) > 1,
          "padding is not exercised if all lengths are equal")

    batched = acts.batch_activations(model, tok, texts)
    single = np.stack([acts.batch_activations(model, tok, [t])[0]
                       for t in texts])
    check("batched shape correct",
          batched.shape == (6, n_layers + 1, d_model), str(batched.shape))
    diff = np.abs(batched - single).max()
    rel = diff / (np.abs(single).max() + 1e-9)
    check("batched == unbatched", rel < 0.02,
          f"max relative diff {rel:.4f} -- if this fails, padding side is wrong")

    section("4. EXTRACTION + RESUME")
    out = config.data_path("smoke_acts.npz")
    shard = out.replace(".npz", "_partial.jsonl")
    for f in (out, shard):
        if os.path.exists(f):
            os.remove(f)

    a1, m1 = acts.extract(train_file, out, model=model, tok=tok)
    check("activation shape", a1.shape == (len(rows), n_layers + 1, d_model),
          str(a1.shape))
    check("shard written", os.path.exists(shard))

    a2, m2 = acts.extract(train_file, out, model=model, tok=tok)
    check("resume reuses cache", np.allclose(a1, a2),
          "second run should read the shard, not recompute")

    a3, m3 = acts.load(out)
    check("npz round-trips", np.allclose(a1, a3) and len(m3) == len(m1))

    section("5. PER-TURN EXTRACTION")
    pt_out = config.data_path("smoke_perturn.npz")
    for f in (pt_out, pt_out.replace(".npz", "_partial.jsonl")):
        if os.path.exists(f):
            os.remove(f)
    ap, mp = acts.extract(train_file, pt_out, per_turn=True,
                          model=model, tok=tok)
    turns = sorted({r["n_turns"] for r in mp})
    check("per-turn yields multiple prefixes", len(turns) == 3, str(turns))
    check("per-turn count matches", ap.shape[0] == len(rows) * 3, str(ap.shape))

    section("6. PROBE")
    from probe import layer_sweep, make_probe
    labels = [r["label"] for r in m1]
    groups = [r["group"] for r in m1]
    check("labels balanced", labels.count(1) == labels.count(0),
          f"{labels.count(1)}/{labels.count(0)}")

    p = make_probe(config.C)
    p.fit(a1[:, 1, :], labels)
    margins = p.decision_function(a1[:, 1, :])
    check("decision_function returns graded values",
          np.mean(np.abs(margins) > 20) < 0.5,
          f"range [{margins.min():.1f}, {margins.max():.1f}]")

    try:
        from resweep import sweep_grouped
        res = sweep_grouped(a1, labels, groups, n_splits=3)
        check("grouped sweep runs", len(res) == n_layers + 1)
    except ImportError:
        print("  [SKIP] resweep archived; grouped CV lives in probe.py now")

    section("RESULT")
    if FAILURES:
        print(f"  {len(FAILURES)} FAILURES:")
        for f in FAILURES:
            print(f"    - {f}")
        print("\n  Fix these before renting a GPU.")
        sys.exit(1)
    print("  All checks passed. Pipeline is GPU-ready.")
    print("\n  Clean up smoke artifacts with:")
    print(f"    rm -r data/{config.MODEL_TAG}  (or Remove-Item -Recurse)")


if __name__ == "__main__":
    main()
