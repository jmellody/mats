"""Rate stimulus statements for centrality using an LLM judge.

    python rate_pairs.py                 # rate all pairs in data/pairs.json
    python rate_pairs.py --spot-check 20 # write a subset for you to rate by hand

Why this exists: your correct and incorrect statements must be equally
diagnostic of expertise. If the incorrect one is a glaring error any layperson
would catch while the correct one requires real training, you will "find" an
asymmetry that is entirely an artifact of your stimuli. This scores both halves
of every pair and drops the mismatched ones.

HONEST LIMITATIONS, put these in your writeup:

- An LLM judge is not two independent human raters. Two passes with shuffled
  order measures self-consistency, not inter-rater reliability.
- Do not judge with the model you generated with. Correlated errors: models
  rate their own output as better matched than it is.
- Do not judge with the model you are probing (Qwen). Same reason, worse.
- Spot-check at least 20 pairs by hand and report the agreement. That is what
  makes the automated ratings credible rather than a shortcut.

Judge model is set by JUDGE_MODEL / JUDGE_PROVIDER env vars.
"""

import argparse
import json
import os
import random
import re
import time
from collections import defaultdict
from dotenv import load_dotenv
load_dotenv()

PROVIDER = os.environ.get("JUDGE_PROVIDER", "gemini")
JUDGE_MODEL = os.environ.get(
    "JUDGE_MODEL",
    "claude-haiku-4-5-20251001" if PROVIDER == "anthropic" else "gemini-3.1-flash-lite",
)
BATCH = 20

RATE_PROMPT = """You are helping validate experimental stimuli. Below are statements someone might make while working on their own data analysis.

For each, rate ONE thing: how much does this statement reveal about the speaker's grasp of statistics?

Rate 1-5 on CENTRALITY:
1 = reveals almost nothing about their understanding; could be said by anyone
3 = reveals moderate engagement with the underlying idea
5 = strongly diagnostic; only someone who has really thought about this concept would say something in this territory

Important: rate how DIAGNOSTIC the statement is, not whether it is correct. A confidently stated misconception about a subtle point can be just as diagnostic as a correct statement about it. Do not reward correctness and do not penalise error. You are rating how much signal the statement carries, in either direction.

Statements:
{items}

Return ONLY a JSON array, no preamble, no markdown fences:
[{{"id": "...", "centrality": N}}, ...]"""


def get_client():
    if PROVIDER == "anthropic":
        import anthropic
        return anthropic.Anthropic()
    from openai import OpenAI
    return OpenAI(
        api_key=os.environ["GEMINI_API_KEY"],
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )


def call(cl, prompt, max_tokens=4000, retries=3):
    for attempt in range(retries):
        try:
            if PROVIDER == "anthropic":
                r = cl.messages.create(
                    model=JUDGE_MODEL, max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}])
                text = "".join(b.text for b in r.content if b.type == "text")
            else:
                r = cl.chat.completions.create(
                    model=JUDGE_MODEL, max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}])
                if r.choices[0].finish_reason == "length":
                    raise ValueError("truncated")
                text = r.choices[0].message.content
            text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
            return json.loads(text)
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(20 if "429" in str(e) else 3)
    raise RuntimeError("unreachable")


def flatten(pairs):
    out = []
    for i, p in enumerate(pairs):
        out.append({"id": f"{i}a", "pair": i, "valence": "correct",
                    "text": p["correct"], "concept": p["concept"]})
        out.append({"id": f"{i}b", "pair": i, "valence": "incorrect",
                    "text": p["incorrect"], "concept": p["concept"]})
    return out


def rate_pass(cl, items, seed):
    """One rating pass over all items, in a shuffled order."""
    shuffled = items[:]
    random.Random(seed).shuffle(shuffled)
    scores = {}
    for i in range(0, len(shuffled), BATCH):
        chunk = shuffled[i:i + BATCH]
        listing = "\n".join(f'{it["id"]}: {it["text"]}' for it in chunk)
        try:
            got = call(cl, RATE_PROMPT.format(items=listing))
            for g in got:
                if "id" in g and "centrality" in g:
                    scores[str(g["id"])] = float(g["centrality"])
            print(f"    batch {i//BATCH + 1}: {len(got)} rated", flush=True)
        except Exception as e:
            print(f"    batch {i//BATCH + 1} FAILED: {e}", flush=True)
        time.sleep(1)
    return scores


def main(path="data/pairs.json", n_top=60, max_gap=1.0, keep_n=30, passes=2):
    d = json.load(open(path))
    pairs = d["pairs"][:n_top]
    items = flatten(pairs)
    cl = get_client()

    print(f"judge: {JUDGE_MODEL} ({PROVIDER})")
    print(f"rating {len(items)} statements from {len(pairs)} pairs, "
          f"{passes} passes\n")

    all_scores = defaultdict(list)
    for p in range(passes):
        print(f"  pass {p + 1}")
        for k, v in rate_pass(cl, items, seed=p).items():
            all_scores[k].append(v)

    # collapse to per-pair gaps
    results = []
    for i, pair in enumerate(pairs):
        a, b = all_scores.get(f"{i}a", []), all_scores.get(f"{i}b", [])
        if not a or not b:
            continue
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        spread = max(
            (max(a) - min(a)) if len(a) > 1 else 0,
            (max(b) - min(b)) if len(b) > 1 else 0,
        )
        results.append({
            "idx": i, "concept": pair["concept"],
            "correct_score": round(ma, 2), "incorrect_score": round(mb, 2),
            "gap": round(abs(ma - mb), 2), "self_consistency_spread": spread,
        })

    results.sort(key=lambda r: (r["gap"], r["self_consistency_spread"]))
    keep = {r["idx"] for r in results[:keep_n] if r["gap"] <= max_gap}

    for i, p in enumerate(d["pairs"]):
        p["keep"] = i in keep
        if i < len(pairs):
            m = next((r for r in results if r["idx"] == i), None)
            if m:
                p["ratings"] = m
    json.dump(d, open(path, "w"), indent=2)

    print(f"\n{'gap':>5}  {'corr':>5}  {'incorr':>6}  concept")
    for r in results[:keep_n + 5]:
        mark = "keep" if r["idx"] in keep else "drop"
        print(f"{r['gap']:>5.2f}  {r['correct_score']:>5.2f}  "
              f"{r['incorrect_score']:>6.2f}  {mark}  {r['concept']}")

    dropped_gap = [r for r in results if r["gap"] > max_gap]
    print(f"\nkept {len(keep)}, {len(dropped_gap)} exceeded gap of {max_gap}")

    flagged = [r for r in results if r["self_consistency_spread"] >= 2]
    if flagged:
        print(f"{len(flagged)} pairs where the judge disagreed with itself by "
              f"2+ across passes -- treat those ratings as unreliable")

    print("\nNow spot-check by hand:  python rate_pairs.py --spot-check 20")


def spot_check(n, path="data/pairs.json", out="data/spot_check.tsv"):
    """Blind subset for you to rate by hand, to validate the judge."""
    d = json.load(open(path))
    rated = [(i, p) for i, p in enumerate(d["pairs"]) if "ratings" in p]
    sample = random.Random(7).sample(rated, min(n, len(rated)))
    rows = []
    for i, p in sample:
        rows.append((f"{i}a", p["concept"], p["correct"]))
        rows.append((f"{i}b", p["concept"], p["incorrect"]))
    random.Random(7).shuffle(rows)
    with open(out, "w") as f:
        f.write("id\tconcept\tstatement\tyour_centrality_1_to_5\n")
        for r in rows:
            f.write("\t".join(r) + "\t\n")
    print(f"{len(rows)} statements -> {out}")
    print("Rate blind, then compare to the judge scores in pairs.json.")
    print("Report the correlation in your writeup -- that is what makes the")
    print("automated ratings defensible rather than a shortcut.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spot-check", type=int, default=0)
    ap.add_argument("--keep", type=int, default=30)
    ap.add_argument("--max-gap", type=float, default=1.0)
    ap.add_argument("--passes", type=int, default=2)
    a = ap.parse_args()
    if a.spot_check:
        spot_check(a.spot_check)
    else:
        main(keep_n=a.keep, max_gap=a.max_gap, passes=a.passes)
