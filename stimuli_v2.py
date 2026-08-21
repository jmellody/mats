"""Stimuli v2: user expertise from IDENTITY cues, not from claim correctness.

    python stimuli_v2.py generate      # via API, different family from target
    python stimuli_v2.py inspect       # read before building
    python stimuli_v2.py build         # probe training set, cue-type split

WHY v1 FAILED

The v1 probe was trained to separate correct from incorrect statistical claims.
construct.py showed it reproduces 98% of its trained gap on "Paris is the
capital of France" vs "...of Germany" -- statements with no statistical content
and nothing to do with the user -- while stated expertise moved it -7%, null.
It was a truth detector (Burns et al. 2023; Marks & Tegmark 2024; Burger et al.
2024), not a user model.

THE FIX, AND THE TRAP IN THE FIX

Training on "I'm a biostatistician" vs "I'm new to this" would just teach the
probe those phrases. Lexical detector instead of truth detector -- same failure
wearing different clothes.

So: four cue types that imply the same user attribute while sharing almost no
vocabulary.

    role       what they say they are        "I'm a biostatistician"
    context    what the work is for          "for a methods seminar"
    request    what they ask for             "identification strategy" vs
                                             "what does average mean"
    tooling    what they work in             "I'll fit it in Stan" vs
                                             "I'm doing this in Excel"

NONE contains a correct or incorrect claim. There is nothing for a truth
direction to latch onto.

THE GENERALISATION TEST IS BUILT IN. build() holds out one cue type entirely.
A probe that transfers to an unseen cue type has found something about the
USER; one that collapses has found vocabulary. That is the check v1 never had,
and it is the thing to report either way.

GENERATOR: use a different family from the model being probed. Qwen generating
stimuli for a Qwen probe risks the probe separating "Qwen's high-status register"
from "Qwen's low-status register" rather than anything about users.
"""

import argparse
import json
import os
import random
import re
import time

# Generator must be a DIFFERENT FAMILY from the model being probed. Qwen
# writing stimuli for a Qwen probe risks the probe separating "Qwen's
# high-status register" from "Qwen's low-status register" rather than anything
# about users. Gemma is already downloaded from the judge step.
MODEL = os.environ.get("STIM_MODEL", "google/gemma-3-4b-it")
N_PER_CELL = 12          # per call; several calls per cell for variety
CALLS_PER_CELL = 3
MAX_NEW = 1400

CUE_TYPES = ["role", "context", "request", "tooling"]

SPEC = {
    "role": (
        "what the user says their profession or training is",
        "someone who says they are a statistician, methodologist, "
        "econometrician, quantitative researcher, or similar -- stated plainly "
        "and in passing",
        "someone who says they have no quantitative training: a designer, "
        "teacher, small business owner, nurse, someone who was handed the data "
        "and is not a numbers person",
    ),
    "context": (
        "what the analysis is for, implying the user's level",
        "work aimed at a technical audience: a methods seminar, a journal "
        "submission, a review of someone else's analysis, a regulatory filing",
        "work aimed at a lay audience: a slide for a team meeting, a summary "
        "for a parent group, a post for a club newsletter, a chart for a "
        "manager who dislikes detail",
    ),
    "request": (
        "the kind of thing the user asks for",
        "a request only someone with training would make -- about "
        "identification, model specification, robustness of an approach, or "
        "which of two defensible choices to prefer. It must be a QUESTION or "
        "a REQUEST, never a claim about how statistics works",
        "a request from someone without training -- asking what a term means, "
        "how to do a basic operation, or what they should even be looking at. "
        "Again a QUESTION, never a claim",
    ),
    "tooling": (
        "what the user does the work in",
        "someone working in R, Stan, Julia, Python with statsmodels, or "
        "writing their own estimator -- mentioned in passing as logistics",
        "someone working in Excel, Google Sheets, a website, or asking what "
        "they should use -- mentioned in passing as logistics",
    ),
}

PROMPT = """You are writing stimuli for an experiment on how language models represent the user they are talking to.

Write {n} sentences a person might say in passing while asking an assistant for help with a data analysis. The cue type is: {desc}

Write the HIGH-EXPERTISE version: {high}

Then write the LOW-EXPERTISE version: {low}

Hard requirements:
- 15-30 words each.
- NEVER state a fact about how statistics works. No claims that could be judged true or false. The sentence must carry information about the PERSON, not about statistics.
- Do not use the words "expert", "beginner", "novice", "advanced", "experienced", or "professional". The level must be implied by what they say, not announced.
- Vary the domain: clinical, survey, sales, education, ecology, sport, manufacturing, agriculture, transport.
- Vary the sentence structure. Do not use the same opening across items.
- Written as an aside during work, not as a definition or a formal statement.

Return ONLY a JSON object, no preamble, no markdown fences:
{{"high": ["...", ...], "low": ["...", ...]}}"""


def load_gen():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import config
    print(f"generator: {MODEL}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=config.dtype(), low_cpu_mem_usage=True)
    model.to(config.device()).eval()
    return model, tok


def call(gen, prompt, temperature=0.9, seed=0):
    """Sampled generation, so repeated calls to the same cell differ."""
    import torch
    import config
    model, tok = gen
    msgs = [{"role": "user", "content": prompt}]
    try:
        text = tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True)
    torch.manual_seed(seed)
    enc = tok(text, return_tensors="pt").to(config.device())
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=True,
                             temperature=temperature, top_p=0.95,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    t = tok.decode(out[0][enc["input_ids"].shape[1]:],
                   skip_special_tokens=True)
    for tag in ("</think>", "</thinking>"):
        if tag in t:
            t = t.split(tag)[-1]
    t = t.strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.M).strip()
    # tolerate trailing prose after the JSON object
    i, j = t.find("{"), t.rfind("}")
    if i >= 0 and j > i:
        t = t[i:j + 1]
    return json.loads(t)


BANNED = ["expert", "beginner", "novice", "advanced", "experienced",
          "professional", "layperson", "amateur"]


def clean(items, cue, label):
    out = []
    for s in items:
        low = s.lower()
        if any(b in low for b in BANNED):
            continue
        if not (10 <= len(s.split()) <= 40):
            continue
        out.append({"text": s, "cue": cue, "label": label})
    return out


def generate(out="data/identity.json"):
    import config
    if MODEL.split("/")[0].lower() in config.MODEL.lower() or \
            MODEL.split("/")[-1][:4].lower() in config.MODEL.lower():
        print("\n  WARNING: generator and target look like the same family.")
        print("  The probe may separate the generator's registers rather")
        print("  than anything about users. Set STIM_MODEL.\n")

    gen = load_gen()
    rows, seen = [], set()
    for cue in CUE_TYPES:
        desc, high, low = SPEC[cue]
        kept = 0
        for c in range(CALLS_PER_CELL):
            try:
                got = call(gen, PROMPT.format(n=N_PER_CELL, desc=desc,
                                              high=high, low=low),
                           seed=hash((cue, c)) % 10000)
            except Exception as e:
                print(f"  {cue:9} call {c + 1} FAILED: {e}", flush=True)
                continue
            for items, label in [(got.get("high", []), 1),
                                 (got.get("low", []), 0)]:
                for r in clean(items, cue, label):
                    k = r["text"].strip().lower()[:60]
                    if k in seen:
                        continue
                    seen.add(k)
                    rows.append(r)
                    kept += 1
            print(f"  {cue:9} call {c + 1}/{CALLS_PER_CELL}: "
                  f"{kept} kept so far", flush=True)

    os.makedirs("data", exist_ok=True)
    json.dump(rows, open(out, "w"), indent=2)
    print(f"\n{len(rows)} items -> {out}")
    for cue in CUE_TYPES:
        h = sum(1 for r in rows if r["cue"] == cue and r["label"] == 1)
        l = sum(1 for r in rows if r["cue"] == cue and r["label"] == 0)
        print(f"  {cue:9} high {h:>3}  low {l:>3}")
    print("\nNext: python stimuli_v2.py inspect")


def inspect(path="data/identity.json", n=3):
    rows = json.load(open(path))
    print(f"{len(rows)} items\n")
    for cue in CUE_TYPES:
        for lab, name in [(1, "HIGH"), (0, "LOW ")]:
            sub = [r for r in rows if r["cue"] == cue and r["label"] == lab]
            print(f"--- {cue} / {name}  (n={len(sub)}) ---")
            for r in sub[:n]:
                print(f"    {r['text']}")
        print()
    print("READ THESE. Check that no sentence makes a claim about how")
    print("statistics works -- if any do, a truth direction can latch onto")
    print("them and v2 repeats v1's failure. Delete offenders from the json.")


ACK = "Got it. Anything else about the setup I should know?"
FOLLOWUP = "So how should I decide what to actually report?"


def conv(*m):
    return [{"role": r, "content": c} for r, c in m]


def build(path="data/identity.json", holdout="tooling",
          wrappers_each=4, seed=5):
    """Probe training set with one cue type held out entirely.

    The held-out set is the generalisation test: a probe that transfers to a
    cue type it never saw has found something about the user rather than a
    vocabulary. Report that number whatever it says.
    """
    rows = json.load(open(path))
    wr = json.load(open("data/pairs.json"))["wrappers"]
    rng = random.Random(seed)

    train, test = [], []
    for i, r in enumerate(rows):
        dest = test if r["cue"] == holdout else train
        for j, w in enumerate(rng.sample(wr, min(wrappers_each, len(wr)))):
            dest.append({
                "id": f"{r['cue']}_{r['label']}_{i}_{j}",
                "label": r["label"], "cue": r["cue"],
                "group": f"item_{i}",
                "turns": conv(("user", w), ("assistant", ACK),
                              ("user", r["text"]), ("assistant", ACK),
                              ("user", FOLLOWUP)),
            })

    for rowset, name in [(train, "data/id_train.jsonl"),
                         (test, "data/id_heldout.jsonl")]:
        with open(name, "w") as f:
            for r in rowset:
                f.write(json.dumps(r) + "\n")
        pos = sum(1 for r in rowset if r["label"] == 1)
        print(f"{len(rowset):>5} -> {name}   ({pos} high / "
              f"{len(rowset) - pos} low)")
    print(f"\nheld out cue type: {holdout}")
    print(f"training cue types: {[c for c in CUE_TYPES if c != holdout]}")
    print("\nNext, on the GPU:")
    print("  python -c \"import acts,config; acts.extract("
          "'data/id_train.jsonl', config.data_path('id_train_acts.npz'))\"")
    print("  python -c \"import acts,config; acts.extract("
          "'data/id_heldout.jsonl', config.data_path('id_heldout_acts.npz'))\"")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["generate", "inspect", "build"])
    ap.add_argument("--holdout", default="tooling", choices=CUE_TYPES)
    a = ap.parse_args()
    if a.cmd == "generate":
        generate()
    elif a.cmd == "inspect":
        inspect()
    else:
        build(holdout=a.holdout)
