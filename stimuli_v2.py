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

CUE_TYPES = ["role", "context", "stance", "tooling"]

SPEC = {
    "role": (
        "the job or training the user names for themselves",
        "someone who names a quantitative job or training in passing -- "
        "statistician, methodologist, econometrician, biostatistician, "
        "quantitative researcher. Use NO statistical vocabulary beyond the job "
        "title itself. The rest of the sentence must be plain English about "
        "their data",
        "someone who names a non-quantitative job in passing -- designer, "
        "teacher, nurse, shop owner, coach, someone a colleague handed the "
        "data to. Plain English throughout",
    ),
    "context": (
        "who the OUTPUT is for -- the audience or venue only",
        "output for a technical audience: a methods seminar, a journal "
        "submission, a peer reviewer, a regulator, a stats reading group. "
        "Describe ONLY the audience or venue. Do NOT name any statistical "
        "method, model, or technical term anywhere in the sentence",
        "output for a lay audience: a team meeting, a parents' evening, a club "
        "newsletter, a manager who dislikes detail, a customer email. Describe "
        "ONLY the audience or venue",
    ),
    "stance": (
        "the user's RELATIONSHIP to the technical choices in their analysis, "
        "with the SAME technical vocabulary appearing on both sides",
        "someone who made the analytic choice themselves and can say why, or "
        "names a cost they accepted. They own the decision",
        "someone reporting what they were told to do, or asking which option "
        "to pick, or unsure what the terms refer to. They do not own the "
        "decision",
    ),
    "tooling": (
        "which software the user works in, mentioned incidentally",
        "someone working in R, Stan, Julia, or Python. Mention ONLY the "
        "software name plus something mundane about files, scripts, or setup. "
        "Never use the word 'logistics'. No statistical terms",
        "someone working in Excel or Google Sheets, or asking which tool they "
        "should use. Mention ONLY the software plus something mundane about "
        "files or setup",
    ),
}

STANCE_PROMPT = """You are writing stimuli for an experiment on how language models represent the user they are talking to.

Write {n} MINIMAL PAIRS. Each pair is two sentences a person might say in passing while working on their own data analysis.

The two sentences in a pair must contain THE SAME technical terms. Choose one or two statistical terms and use them in both sentences.

- "high": the speaker made the analytic choice themselves and can say why, or names a cost they accepted for it. They own the decision.
- "low": the speaker is reporting what someone else told them to do, or asking which option to pick, or unsure what the terms refer to. They do not own the decision.

Hard requirements:
- The same technical vocabulary appears in BOTH sentences. Never jargon in one and plain language in the other.
- Within 3 words of each other in length. 18-32 words each.
- NEITHER sentence may state a fact about how statistics works. No claim that could be judged true or false. Both describe the speaker's own situation, or ask a question.
- Do not use the words expert, beginner, novice, advanced, experienced, professional, or confused.
- Vary the domain: clinical, survey, sales, education, ecology, sport, manufacturing, agriculture, transport.

Return ONLY a JSON object, no preamble, no markdown fences:
{{"pairs": [{{"high": "...", "low": "..."}}, ...]}}"""

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


# Explicit level markers, and hedges that are themselves a lexical shortcut.
# "I'm not sure" separates the classes without the model representing anything
# about the user, so it has to go.
BANNED = ["expert", "beginner", "novice", "advanced", "experienced",
          "professional", "layperson", "amateur", "confused",
          "i don't understand", "i dont understand", "no idea",
          "clueless", "out of my depth"]


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
            tmpl = STANCE_PROMPT if cue == "stance" else PROMPT
            try:
                got = call(gen, tmpl.format(n=N_PER_CELL, desc=desc,
                                            high=high, low=low),
                           seed=hash((cue, c)) % 10000)
            except Exception as e:
                print(f"  {cue:9} call {c + 1} FAILED: {e}", flush=True)
                continue
            if cue == "stance":
                pairs = got.get("pairs", [])
                got = {"high": [p["high"] for p in pairs if "high" in p],
                       "low": [p["low"] for p in pairs if "low" in p]}
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

# Probe-training conversations are padded to the SAME 9 turns the dynamics
# experiments use. In v1 the probe was fit on 5-turn conversations and applied
# to 7-turn ones; baseline drifted 9.2 while the effect was 1.06, because the
# inputs were far outside the distribution the probe had seen.
PAD = [
    "I put the whole thing together over the weekend and it's all in one place now.",
    "It took a while to collect but I finally have everything I need in the file.",
    "I've been meaning to look at this properly for a couple of weeks now.",
    "There's quite a lot of it, more than I expected when I started.",
    "I tidied it up a bit yesterday so it's easier to work with now.",
    "It's all in one folder along with the notes I made at the time.",
]


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
            # signal in the LAST slot, filler before it -- same shape as the
            # dose experiment's dose-1 condition
            p1, p2 = PAD[(i + j) % len(PAD)], PAD[(i + j + 1) % len(PAD)]
            dest.append({
                "id": f"{r['cue']}_{r['label']}_{i}_{j}",
                "label": r["label"], "cue": r["cue"],
                "group": f"item_{i}",
                "turns": conv(("user", w), ("assistant", ACK),
                              ("user", p1), ("assistant", ACK),
                              ("user", p2), ("assistant", ACK),
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
