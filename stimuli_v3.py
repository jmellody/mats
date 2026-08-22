"""Stimuli v3: expertise carried by AGENCY, with vocabulary held constant.

    python stimuli_v3.py generate     # local HF model, different family
    python stimuli_v3.py filter       # automated lexical-overlap cuts
    python stimuli_v3.py inspect      # read before building
    python stimuli_v3.py build        # probe training set

WHY v3

  v1  correct vs incorrect claims        -> learned statement TRUTH
                                            (98% of gap on Paris/France)
  v2  identity cues (role, tooling...)   -> learned partly VOCABULARY
                                            (AUC 0.757 at layer 1)

Both failures have the same shape: the label was perfectly predictable from
something other than a user model. v3 removes the shortcut by construction.

THE DESIGN

Each item is a PAIR sharing the same technical vocabulary and the same domain.
Only the speaker's relationship to the work differs:

  high: I fit the model in Stan and tightened the priors after the first run
        came back too wide.
  low:  A colleague set the Stan model up for me; I just press run and copy
        the numbers into the report.

"Stan", "model", "run" appear in both. A probe cannot separate these on
vocabulary, so whatever accuracy it achieves must come from something else.

FOUR AGENCY CONTRASTS, all lexically matched:

  decision    made the choice and can say why   |  was told to, or is asking
  diagnosis   knows why the problem arose       |  describes the symptom only
  tradeoff    names a cost they accepted        |  wants the single right answer
  fluency     uses the tool to do their work    |  operates it as instructed

WHAT THIS CANNOT DO

It cannot make expertise non-lexical. Any reliable signal is carried by SOME
words -- "I decided" versus "someone told me" is itself a lexical difference.
The claim is narrower and honest: the DOMAIN vocabulary is matched, so the
probe cannot succeed by detecting statistical jargon. Measure what remains at
layer 1 and report it as the shallow baseline rather than treating it as a
flaw.

BANNED, because they are lexical shortcuts of their own: hedges ("not sure",
"confused"), explicit level markers, and question marks in the high condition
only.
"""

import argparse
import json
import os
import random
import re

MODEL = os.environ.get("STIM_MODEL", "google/gemma-3-12b-it")
PAIRS_PER_CALL = 8
CALLS_PER_CONTRAST = 6
MAX_NEW = 2000

CONTRASTS = {
    "decision": (
        "the speaker made an analytic choice themselves and can say what "
        "drove it",
        "the speaker was told which choice to make by someone else, or is "
        "asking which to pick",
    ),
    "diagnosis": (
        "the speaker knows why a problem in their analysis arose and what it "
        "implies",
        "the speaker describes the same symptom without knowing what causes "
        "it",
    ),
    "tradeoff": (
        "the speaker names a cost they knowingly accepted for their approach",
        "the speaker wants to know which option is simply the right one",
    ),
    "fluency": (
        "the speaker uses a tool as an instrument of their own work, changing "
        "it as needed",
        "the speaker operates the same tool as instructed, without changing it",
    ),
}

PROMPT = """You are writing stimuli for an experiment on how language models represent the user they are talking to.

Write {n} MINIMAL PAIRS. Each pair is two sentences a person might say in passing while working on their own data analysis.

The contrast is AGENCY:
  "high" = {high}
  "low"  = {low}

THE CRITICAL REQUIREMENT: both sentences in a pair must use THE SAME technical vocabulary and the SAME domain. Pick one or two statistical or software terms, and use those same terms in both sentences. The pair must be impossible to tell apart by which technical words appear.

Example of a good pair:
  high: "I fit the model in Stan and tightened the priors after the first run came back too wide."
  low:  "A colleague set the Stan model up for me; I just press run and copy the numbers into the report."

Both contain Stan, model, run. Neither could be classified by jargon alone.

Other hard requirements:
- 18-32 words each, within 4 words of each other.
- NEITHER sentence may state a fact about how statistics works. No claim that could be judged true or false. Both describe what the speaker did or is doing.
- Do not use: expert, beginner, novice, advanced, experienced, professional, confused, "not sure", "no idea", "don't understand".
- Do not put a question mark in the high sentence only. Either both ask something or neither does.
- Vary the domain across pairs: clinical, survey, sales, education, ecology, sport, manufacturing, agriculture, transport, logistics.

Return ONLY a JSON object, no preamble, no markdown fences:
{{"pairs": [{{"high": "...", "low": "..."}}, ...]}}"""

BANNED = ["expert", "beginner", "novice", "advanced", "experienced",
          "professional", "layperson", "amateur", "confused",
          "not sure", "no idea", "don't understand", "dont understand",
          "clueless", "out of my depth", "i'm new", "im new"]


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


def call(gen, prompt, seed=0):
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
                             temperature=0.9, top_p=0.95,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    t = tok.decode(out[0][enc["input_ids"].shape[1]:],
                   skip_special_tokens=True)
    for tag in ("</think>", "</thinking>"):
        if tag in t:
            t = t.split(tag)[-1]
    t = re.sub(r"^```(?:json)?|```$", "", t.strip(), flags=re.M).strip()
    i, j = t.find("{"), t.rfind("}")
    if i >= 0 and j > i:
        t = t[i:j + 1]
    return json.loads(t)


def generate(out="data/agency_raw.json"):
    import config
    gen = load_gen()
    rows, seen = [], set()
    for c, (high, low) in CONTRASTS.items():
        for k in range(CALLS_PER_CONTRAST):
            try:
                got = call(gen, PROMPT.format(n=PAIRS_PER_CALL, high=high,
                                              low=low),
                           seed=abs(hash((c, k))) % 100000)
            except Exception as e:
                print(f"  {c:10} call {k+1} FAILED: {e}", flush=True)
                continue
            for p in got.get("pairs", []):
                if "high" not in p or "low" not in p:
                    continue
                key = p["high"].strip().lower()[:60]
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"contrast": c, "high": p["high"],
                             "low": p["low"]})
            print(f"  {c:10} call {k+1}/{CALLS_PER_CONTRAST}: "
                  f"{len(rows)} total", flush=True)

    os.makedirs("data", exist_ok=True)
    json.dump(rows, open(out, "w"), indent=2)
    print(f"\n{len(rows)} raw pairs -> {out}")
    print("Next: python stimuli_v3.py filter")


def toks(s):
    return set(re.findall(r"[a-z][a-z'-]*", s.lower()))


STOP = toks("i my me the a an and or but of to in on for with that this it "
            "is are was were be been so as at from by we our us they them")


# Words that look domain-technical: long, or in a known jargon list. The
# point of the design is that these appear in BOTH halves of a pair, so a
# probe cannot classify on jargon. General content words need not match.
JARGON = toks(
    "model models regression variance covariance residual residuals priors "
    "posterior bayesian stan julia python r spss stata excel sheets "
    "coefficient coefficients estimate estimates interval intervals "
    "significance significant pvalue anova ttest chisquare bootstrap "
    "imputation imputed weighting weighted clustered clustering "
    "specification identification robustness covariate covariates "
    "confounder stratified matching propensity likelihood mle glm lmer "
    "crossvalidation overfitting regularisation regularization lasso ridge "
    "sample samples sampling distribution normality skew outlier outliers")


def technical(s):
    """Domain terms in a sentence: known jargon, or any long content word."""
    t = toks(s) - STOP
    return {w for w in t if w in JARGON or len(w) > 9}


def flags(p):
    h, l = p["high"], p["low"]
    f = []
    lh, ll = len(h.split()), len(l.split())
    if abs(lh - ll) > 4:
        f.append("length")
    if not (14 <= lh <= 40 and 14 <= ll <= 40):
        f.append("range")
    low_all = (h + " " + l).lower()
    if any(b in low_all for b in BANNED):
        f.append("banned")
    if ("?" in l) != ("?" in h):
        f.append("qmark")

    # THE CORE CHECK: technical terms must be shared, not just present
    th, tl = technical(h), technical(l)
    shared = th & tl
    union = th | tl
    tech_ov = len(shared) / max(len(union), 1)
    p["tech_shared"] = sorted(shared)
    p["tech_only_high"] = sorted(th - tl)
    p["tech_only_low"] = sorted(tl - th)
    p["overlap"] = round(tech_ov, 3)
    if not shared:
        f.append("no_shared_tech")
    elif tech_ov < 0.5:
        f.append(f"tech_{tech_ov:.2f}")
    return f


def filter_pairs(inp="data/agency_raw.json", out="data/agency.json"):
    from collections import Counter
    rows = json.load(open(inp))
    kept, reasons = [], Counter()
    for p in rows:
        f = flags(p)
        if f:
            for x in f:
                reasons[x.split("_")[0]] += 1
            continue
        p["keep"] = True
        kept.append(p)
    kept.sort(key=lambda p: -p["overlap"])
    json.dump(kept, open(out, "w"), indent=2)
    print(f"kept {len(kept)} of {len(rows)}")
    for r, n in reasons.most_common():
        print(f"  dropped for {r}: {n}")
    if kept:
        ov = [p["overlap"] for p in kept]
        print(f"\ntechnical-term overlap: median "
              f"{sorted(ov)[len(ov)//2]:.2f}, min {min(ov):.2f}")
        print("  (fraction of domain terms appearing in BOTH halves;")
        print("   v2 pairs had no such requirement at all)")
    print(f"\n-> {out}   sorted best-first")
    print("Next: python stimuli_v3.py inspect")


def inspect(path="data/agency.json", n=4):
    rows = json.load(open(path))
    print(f"{len(rows)} pairs\n")
    for c in CONTRASTS:
        sub = [p for p in rows if p["contrast"] == c]
        print(f"=== {c}  (n={len(sub)}) ===")
        for p in sub[:n]:
            print(f"  tech overlap {p['overlap']:.2f}   "
                  f"shared: {', '.join(p['tech_shared']) or '(none)'}")
            if p["tech_only_high"] or p["tech_only_low"]:
                print(f"    UNMATCHED  high-only {p['tech_only_high']}  "
                      f"low-only {p['tech_only_low']}")
            print(f"    HIGH  {p['high']}")
            print(f"    LOW   {p['low']}")
        print()
    print("CHECK: could you classify these by which technical words appear?")
    print("If yes for a pair, delete it -- that is the v2 failure returning.")


ACK = "Got it. Anything else about the setup I should know?"
FOLLOWUP = "So how should I decide what to actually report?"
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


def build(path="data/agency.json", holdout="fluency", wrappers_each=4,
          seed=13):
    """9-turn conversations, signal in the last slot, matching the dynamics
    experiments exactly. One contrast type held out for the generalisation
    test."""
    rows = [p for p in json.load(open(path)) if p.get("keep", True)]
    wr = json.load(open("data/pairs.json"))["wrappers"]
    rng = random.Random(seed)

    train, test = [], []
    for i, p in enumerate(rows):
        dest = test if p["contrast"] == holdout else train
        for j, w in enumerate(rng.sample(wr, min(wrappers_each, len(wr)))):
            p1, p2 = PAD[(i + j) % len(PAD)], PAD[(i + j + 1) % len(PAD)]
            for lab, key in [(1, "high"), (0, "low")]:
                dest.append({
                    "id": f"{p['contrast']}_{lab}_{i}_{j}",
                    "label": lab, "cue": p["contrast"],
                    "group": f"pair_{i}",
                    "turns": conv(("user", w), ("assistant", ACK),
                                  ("user", p1), ("assistant", ACK),
                                  ("user", p2), ("assistant", ACK),
                                  ("user", p[key]), ("assistant", ACK),
                                  ("user", FOLLOWUP)),
                })

    for rs, name in [(train, "data/ag_train.jsonl"),
                     (test, "data/ag_heldout.jsonl")]:
        with open(name, "w") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")
        pos = sum(1 for r in rs if r["label"] == 1)
        print(f"{len(rs):>5} -> {name}   ({pos} high / {len(rs)-pos} low)")
    print(f"\nheld out contrast: {holdout}")
    print(f"turn counts: "
          f"{set(len(r['turns']) for r in train + test)}  (must be {{9}})")
    print("\nBoth halves of a pair share a group id, so pair-grouped CV keeps")
    print("them on the same side of every split.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["generate", "filter", "inspect", "build"])
    ap.add_argument("--holdout", default="fluency", choices=list(CONTRASTS))
    a = ap.parse_args()
    if a.cmd == "generate":
        generate()
    elif a.cmd == "filter":
        filter_pairs()
    elif a.cmd == "inspect":
        inspect()
    else:
        build(holdout=a.holdout)
