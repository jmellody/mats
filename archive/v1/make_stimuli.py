"""Generate minimal-pair stimuli at scale via Gemini's OpenAI-compatible API.

    python make_stimuli.py generate            # ~500 pairs, free tier, ~15 min
    python make_stimuli.py filter              # automated quality cuts
    python make_stimuli.py review              # top 60 -> tsv for hand rating
    python make_stimuli.py build               # assemble jsonl

Needs GEMINI_API_KEY from aistudio.google.com (free, no card).
    pip install openai

TWO SETS, DIFFERENT STANDARDS -- this is the whole design:

  Probe training (~470 pairs): noisy is fine. Its job is to produce a working
  probe, and held-out accuracy tells you whether it worked. Automated filters
  only, no hand review.

  Experiment (30 pairs): must be airtight. There is no downstream check that
  catches a bad pair here -- an unmatched pair produces a clean, wrong result.
  Hand-rate these. 30 is plenty because each pair is its own control across the
  5 conditions, so you run paired tests.

The failure mode to watch for: the incorrect statement is obviously wrong to
anyone while the correct one needs real training. That is an unmatched pair and
it manufactures an asymmetry that has nothing to do with user models.
"""

import argparse
import json
import os
import random
import re
import time
from collections import Counter

from dotenv import load_dotenv
load_dotenv()

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")

PAIRS_PER_CALL = 5
SLEEP = 4.5  # free tier is ~15 rpm; back off if you see 429s

CONCEPTS = [
    # inference
    "p-value interpretation", "confidence interval interpretation",
    "null result interpretation", "effect size vs significance",
    "statistical power", "multiple comparisons", "stopping rules",
    "one-tailed vs two-tailed tests", "type I and type II errors",
    "prior probability", "likelihood vs probability", "Bayes factor",
    # estimation and variance
    "standard error vs standard deviation", "sampling distribution",
    "central limit theorem", "bootstrapping", "degrees of freedom",
    "Welch's t-test", "heteroskedasticity", "variance inflation",
    "shrinkage estimators", "margin of error",
    # causal
    "correlation and causation", "confounding variables", "collider bias",
    "selection bias", "survivorship bias", "regression to the mean",
    "Simpson's paradox", "ecological fallacy", "instrumental variables",
    "difference-in-differences", "propensity score matching",
    "randomisation and balance", "intention-to-treat", "mediation vs moderation",
    "counterfactual reasoning", "backdoor paths",
    # modelling
    "overfitting", "cross-validation", "regularisation", "interaction effects",
    "multicollinearity", "residual diagnostics", "link functions",
    "clustered data", "random vs fixed effects", "model selection criteria",
    "extrapolation beyond data range", "categorical encoding",
    # measurement and data
    "measurement reliability", "construct validity", "censored data",
    "missing data mechanisms", "imputation", "outlier handling",
    "base rates", "sensitivity and specificity", "aggregation and granularity",
    "response bias in surveys", "weighting survey samples", "attrition",
    "floor and ceiling effects", "test-retest reliability",
    "operationalisation", "measurement error attenuation",
    "data leakage", "temporal ordering in panel data",
]

CONCEPTS_PER_CALL = 5
PAIRS_PER_CONCEPT = 5

PAIR_PROMPT = """You are constructing controlled stimuli for an experiment on how language models represent a user's expertise.

For EACH of these concepts, write {n} DIFFERENT minimal pairs:
{concept_list}

Each pair is two sentences a person might say in passing while working on their own data analysis.
- "correct": uses the concept accurately.
- "incorrect": contains a clear, specific misconception about the SAME concept.

Hard requirements for every pair:
- Both sentences use the SAME technical vocabulary. Never jargon in one and plain language in the other.
- Within 3 words of each other in length. Aim for 20-30 words each.
- Same grammatical structure; same opening words where possible.
- They differ by exactly ONE semantic element. Not tone, not confidence, not register.
- No hedging in either ("I think", "I'm not sure"). The only signal is whether the content is right or wrong.
- Written as an aside during work, not as a definition or a question.
- Each sentence must be embedded in the person's own ongoing analysis: reference their data, their result, their decision. Use "I" or "my". Never state a general definition of the concept.

Return ONLY a JSON array, no preamble, no markdown fences. Each object must include its concept:
[{{"concept": "...", "correct": "...", "incorrect": "..."}}, ...]"""


def client():
    from openai import OpenAI
    return OpenAI(
        api_key=os.environ["GEMINI_API_KEY"],
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )


def call(cl, prompt, max_tokens=16000, retries=3):
    for attempt in range(retries):
        try:
            r = cl.chat.completions.create(
                model=MODEL, max_tokens=max_tokens,
                reasoning_effort="low",
                messages=[{"role": "user", "content": prompt}],
            )
            if r.choices[0].finish_reason == "length":
                raise ValueError("hit max_tokens - raise it or lower PAIRS_PER_CALL")
            text = r.choices[0].message.content.strip()
            text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
            return json.loads(text)
        except json.JSONDecodeError:
            if attempt == retries - 1:
                raise
            time.sleep(2)
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                time.sleep(30)
            elif attempt == retries - 1:
                raise
            else:
                time.sleep(5)
    raise RuntimeError("unreachable")


def generate(out="data/pairs_raw.json"):
    cl = client()
    pairs = []
    batches = [CONCEPTS[i:i + CONCEPTS_PER_CALL]
               for i in range(0, len(CONCEPTS), CONCEPTS_PER_CALL)]
    os.makedirs("data", exist_ok=True)

    for i, batch in enumerate(batches):
        listing = "\n".join(f"- {c}" for c in batch)
        try:
            got = call(cl, PAIR_PROMPT.format(concept_list=listing,
                                              n=PAIRS_PER_CONCEPT))
            for p in got:
                if all(k in p for k in ("concept", "correct", "incorrect")):
                    pairs.append(p)
            print(f"  [{i+1}/{len(batches)}] +{len(got)} (total {len(pairs)})",
                  flush=True)
        except Exception as e:
            print(f"  [{i+1}/{len(batches)}] FAILED: {e}", flush=True)
            if "PerDay" in str(e):
                print("  daily quota gone -- saving what we have, resume tomorrow")
                break
        json.dump({"pairs": pairs, "wrappers": []}, open(out, "w"), indent=2)
        time.sleep(SLEEP)

    try:
        wrappers = call(cl, WRAPPER_PROMPT.format(n=25))
    except Exception:
        wrappers = []
        print("  wrappers failed -- rerun later, or write 25 by hand")

    json.dump({"pairs": pairs, "wrappers": wrappers}, open(out, "w"), indent=2)
    print(f"\n{len(pairs)} pairs, {len(wrappers)} wrappers -> {out}")


# --- automated quality control -----------------------------------------------

HEDGES = ["i think", "i'm not sure", "i am not sure", "probably", "maybe",
          "i guess", "i'm confused", "not certain", "i believe", "perhaps",
          "does that", "right?", "is that correct"]


def tok(s):
    return set(re.findall(r"[a-z]+", s.lower()))


def pair_flags(p):
    """Return list of reasons to drop. Empty list = keeps."""
    c, w = p["correct"], p["incorrect"]
    flags = []
    lc, lw = len(c.split()), len(w.split())
    if abs(lc - lw) > 3:
        flags.append(f"length_{lc - lw:+d}")
    if not (14 <= lc <= 40 and 14 <= lw <= 40):
        flags.append("out_of_range")
    tc, tw = tok(c), tok(w)
    jac = len(tc & tw) / max(len(tc | tw), 1)
    if jac < 0.65:
        flags.append(f"vocab_{jac:.2f}")
    low = (c + " " + w).lower()
    if any(h in low for h in HEDGES):
        flags.append("hedge")
    if c.strip().lower() == w.strip().lower():
        flags.append("identical")
    p["overlap"] = round(jac, 3)
    p["len_diff"] = lc - lw
    return flags


def filter_pairs(inp="data/pairs_raw.json", out="data/pairs.json"):
    d = json.load(open(inp))
    kept, dropped, reasons = [], 0, Counter()
    seen = set()

    for p in d["pairs"]:
        flags = pair_flags(p)
        key = p["correct"].strip().lower()[:60]
        if key in seen:
            flags.append("duplicate")
        if flags:
            dropped += 1
            for f in flags:
                reasons[f.split("_")[0]] += 1
            continue
        seen.add(key)
        p["keep"] = True
        kept.append(p)

    kept.sort(key=lambda p: (-p["overlap"], abs(p["len_diff"])))
    json.dump({"pairs": kept, "wrappers": d["wrappers"]}, open(out, "w"), indent=2)

    print(f"kept {len(kept)}, dropped {dropped}")
    for r, n in reasons.most_common():
        print(f"  {r}: {n}")
    print(f"\n-> {out}   (sorted best-first by vocabulary overlap)")
    print("Next: python make_stimuli.py review")


def review(path="data/pairs.json", out="data/review.tsv", n=60):
    """Top n pairs by overlap, shuffled and blinded, for hand rating.

    Rate each statement in isolation: how central is this to understanding the
    concept, 1-5. Two raters. Then drop any pair whose two ratings differ by
    more than 1 -- that gap is exactly the confound that would fake your result.
    Keep the best 30. Mark the rest "keep": false in pairs.json.
    """
    d = json.load(open(path))
    top = d["pairs"][:n]
    rows = []
    for i, p in enumerate(top):
        rows.append((f"{i}a", p["concept"], p["correct"], len(p["correct"].split())))
        rows.append((f"{i}b", p["concept"], p["incorrect"], len(p["incorrect"].split())))
    random.Random(0).shuffle(rows)
    with open(out, "w") as f:
        f.write("id\tconcept\tstatement\twords\tcentrality_r1\tcentrality_r2\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\t\t\n")
    print(f"{len(rows)} statements from {len(top)} pairs -> {out}")
    print("Valence is not in the file, deliberately. Rate blind.")
    print("Then keep your best 30 and set the rest to \"keep\": false in pairs.json")


# --- assembly ----------------------------------------------------------------

def conv(*msgs):
    return [{"role": r, "content": c} for r, c in msgs]


ACK = "Got it. Anything else about the setup I should know?"
FOLLOWUP = "So how should I decide what to actually report?"
FILLER = "I pulled the data last week and it's all in one spreadsheet."


def build(path="data/pairs.json", n_experiment=30, wrappers_per_pair=3, seed=0):
    d = json.load(open(path))
    curated = [p for p in d["pairs"] if p.get("keep", True)]
    wrappers = d["wrappers"]
    rng = random.Random(seed)

    exp_pairs = curated[:n_experiment]
    exp_concepts = {p["concept"] for p in exp_pairs}
    # disjoint by CONCEPT, not just by pair -- otherwise the probe trains on a
    # sibling sentence about the same idea and the experiment is circular
    train_pairs = [p for p in d["pairs"] if p["concept"] not in exp_concepts]
    rng.shuffle(train_pairs)

    print(f"experiment: {len(exp_pairs)} pairs across {len(exp_concepts)} concepts")
    print(f"probe train: {len(train_pairs)} pairs, concepts disjoint from experiment")

    train = []
    for i, p in enumerate(train_pairs):
        for w in rng.sample(wrappers, min(wrappers_per_pair, len(wrappers))):
            for label, key in [(1, "correct"), (0, "incorrect")]:
                train.append({
                    "id": f"tr_{i}_{label}_{len(train)}", "label": label,
                    "concept": p["concept"], "group": f"pair_{i}",
                    "turns": conv(("user", w), ("assistant", ACK),
                                  ("user", p[key]), ("assistant", ACK),
                                  ("user", FOLLOWUP)),
                })

    exp = []
    for i, p in enumerate(exp_pairs):
        w = wrappers[i % len(wrappers)]
        pos, neg = p["correct"], p["incorrect"]
        for cond, (s1, s2) in {
            "baseline": (FILLER, FILLER),
            "pos_only": (FILLER, pos),
            "neg_only": (FILLER, neg),
            "pos_then_neg": (pos, neg),
            "neg_then_pos": (neg, pos),
        }.items():
            exp.append({
                "id": f"{cond}_{i}", "condition": cond, "pair": i,
                "concept": p["concept"],
                "turns": conv(("user", w), ("assistant", ACK),
                              ("user", s1), ("assistant", ACK),
                              ("user", s2), ("assistant", ACK),
                              ("user", FOLLOWUP)),
            })

    for rows, out in [(train, "data/probe_train.jsonl"),
                      (exp, "data/experiment.jsonl")]:
        with open(out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {len(rows)} -> {out}")

    n = sum(1 for r in train if r["label"] == 1)
    print(f"\n{n} per class for the probe")
    hrs = len(train) * 8 / 3600
    print(f"rough CPU extraction time: {hrs:.1f}h for training set "
          f"(+{len(exp) * 3 * 8 / 3600:.1f}h for experiment, per-turn)")
    if hrs > 6:
        print("that's an overnight run -- or cut wrappers_per_pair to 2")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["generate", "filter", "review", "build"])
    ap.add_argument("--n-experiment", type=int, default=30)
    ap.add_argument("--wrappers-per-pair", type=int, default=3)
    a = ap.parse_args()
    if a.cmd == "generate":
        generate()
    elif a.cmd == "filter":
        filter_pairs()
    elif a.cmd == "review":
        review()
    else:
        build(n_experiment=a.n_experiment, wrappers_per_pair=a.wrappers_per_pair)
