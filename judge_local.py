"""Local LLM judge: does steering change the expertise level a response assumes?

    python judge_local.py                      # rate steer_results.json
    python judge_local.py --spot 15            # blind subset to rate by hand
    JUDGE_MODEL=<hf-id> python judge_local.py  # pick a different judge

No API keys. Runs a Hugging Face model on the GPU you already have.

CHOOSING A JUDGE MODEL

Use a DIFFERENT FAMILY from the model being judged. Qwen judging Qwen output
risks correlated blind spots -- a model tends to find its own register
unremarkable, which is exactly the axis being measured here.

Candidates, roughly 4-8B so they fit alongside nothing else:
    google/gemma-3-4b-it            Gemma 3; requires accepting the license
                                    on the model page while logged in
    meta-llama/Llama-3.2-3B-Instruct   gated; request access on HF
    microsoft/Phi-4-mini-instruct   usually ungated
    HuggingFaceTB/SmolLM3-3B        ungated, small

Check the model page before running. If a model is gated, `hf auth login`
with a read token from huggingface.co/settings/tokens.

HOW THE RATING WORKS

Rather than generating text and parsing a number out of it -- which fails
unpredictably and adds a sampling step you would then have to control for --
this reads the logits over the digit tokens "1".."7" at the first generated
position and takes their softmax. That gives a deterministic rating and, as a
bonus, a graded expected value rather than an integer, which is more sensitive
to small shifts.

BLINDING

The judge sees one response at a time with no alpha, no direction, and no other
response from the same prompt. It cannot infer an intended ordering.

A NULL IS THE RESULT. Do not hunt for a judge that produces movement.
"""

import argparse
import json
import os
import random

import numpy as np
import torch

import config

JUDGE = os.environ.get("JUDGE_MODEL", "google/gemma-3-4b-it")
EXCERPT = 1600
LEVELS = [1, 2, 3, 4, 5, 6, 7]

RUBRIC = """Rate this assistant response on ONE dimension: what level of statistical background does it assume the reader has?

1 = assumes no background; defines every term; heavy analogies; avoids notation
2 = assumes very little; explains standard terms when used
3 = assumes some familiarity; explains the less common terms
4 = assumes a typical intro-course background; uses standard terms undefined
5 = assumes solid working knowledge; uses technical vocabulary freely
6 = assumes substantial training; discusses subtleties without scaffolding
7 = assumes an expert peer; terse, notation-heavy, engages with edge cases

Judge ONLY the level of assumed background. Ignore length, quality,
correctness, formatting, and helpfulness.

Response to rate:
---
{text}
---

Answer with a single digit 1-7 and nothing else."""


def load_judge():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"judge model: {JUDGE}")
    tok = AutoTokenizer.from_pretrained(JUDGE)
    model = AutoModelForCausalLM.from_pretrained(
        JUDGE, dtype=config.dtype(), low_cpu_mem_usage=True)
    model.to(config.device()).eval()
    return model, tok


def digit_ids(tok):
    """Token ids for the digits 1-7 as they appear at the start of a reply.

    Some tokenizers emit a leading space; try both and keep whichever resolves
    to a single token.
    """
    ids = {}
    for d in LEVELS:
        cands = [str(d), f" {str(d)}"]
        for c in cands:
            t = tok.encode(c, add_special_tokens=False)
            if len(t) == 1:
                ids[d] = t[0]
                break
        else:
            ids[d] = tok.encode(str(d), add_special_tokens=False)[0]
    return ids


@torch.no_grad()
def rate(model, tok, ids, text):
    """Expected level under the model's distribution over digits 1-7."""
    msgs = [{"role": "user",
             "content": RUBRIC.format(text=text[:EXCERPT])}]
    try:
        prompt = tok.apply_chat_template(msgs, tokenize=False,
                                         add_generation_prompt=True,
                                         enable_thinking=False)
    except TypeError:
        prompt = tok.apply_chat_template(msgs, tokenize=False,
                                         add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt").to(config.device())
    logits = model(**enc).logits[0, -1, :].float()
    sel = torch.tensor([logits[ids[d]] for d in LEVELS])
    p = torch.softmax(sel, dim=0).numpy()
    return float(np.dot(p, LEVELS)), p


def norm_p(t):
    from math import erf, sqrt
    return 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))


def load_rows():
    path = config.data_path("steer_results.json")
    rows = json.load(open(path))
    for i, r in enumerate(rows):
        r["_id"] = f"r{i}"
    return rows, path


def main():
    import logs
    logs.start("judge")
    rows, path = load_rows()
    print(f"{len(rows)} responses   generator was {config.MODEL}")
    if JUDGE.split("/")[0].lower() in config.MODEL.lower():
        print("\n  WARNING: judge and generator look like the same family.")
        print("  Correlated blind spots are likely. Set JUDGE_MODEL to a")
        print("  different lineage.\n")

    model, tok = load_judge()
    ids = digit_ids(tok)

    # shuffled so any drift in the loop is not aligned with condition
    order = rows[:]
    random.Random(0).shuffle(order)
    ent = []
    for i, r in logs.progress(order, "rating"):
        lvl, p = rate(model, tok, ids, r["text"])
        r["level"] = lvl
        r["level_probs"] = [round(float(x), 4) for x in p]
        ent.append(-float(np.sum(p * np.log(p + 1e-9))))

    json.dump(rows, open(path, "w"), indent=2)
    print(f"\nratings written back to {path}")
    print(f"mean judge entropy over the 7 levels: {np.mean(ent):.2f} nats")
    if np.mean(ent) > 1.6:
        print("  ^ near-uniform; the judge is barely discriminating. Try a")
        print("    larger judge model before believing a null.")

    analyse(rows)


def analyse(rows=None):
    if rows is None:
        rows, _ = load_rows()
    scored = [r for r in rows if "level" in r]
    if not scored:
        print("no ratings yet -- run without --analyse first")
        return

    allv = np.array([r["level"] for r in scored])
    print(f"\noverall level: mean {allv.mean():.2f}  "
          f"range {allv.min():.2f}-{allv.max():.2f}  sd {allv.std():.2f}")

    for dname in sorted({r["direction"] for r in scored}):
        sub = [r for r in scored if r["direction"] == dname]
        alphas = sorted({r["alpha"] for r in sub})
        print("\n" + "=" * 66)
        print(f"DIRECTION: {dname}")
        print("=" * 66)
        print(f"  {'alpha':>7} {'level':>9} {'sd':>7} {'n':>4}")
        for a in alphas:
            v = np.array([r["level"] for r in sub if r["alpha"] == a])
            print(f"  {a:>+7.1f} {v.mean():>9.3f} {v.std():>7.3f} {len(v):>4}")

        slopes = []
        for pr in {r["prompt"] for r in sub}:
            pts = sorted((r["alpha"], r["level"])
                         for r in sub if r["prompt"] == pr)
            x = np.array([a for a, _ in pts])
            y = np.array([v for _, v in pts])
            if len(set(x)) > 1:
                slopes.append(np.polyfit(x, y, 1)[0])
        slopes = np.array(slopes)
        n = len(slopes)
        se = slopes.std(ddof=1) / np.sqrt(n) if n > 1 else 0
        t = slopes.mean() / se if se else 0
        ci = 1.96 * se
        print(f"\n  slope {slopes.mean():+.4f} level per alpha step")
        print(f"  t({n - 1}) = {t:+.2f}  p = {norm_p(t):.4f}  "
              f"CI [{slopes.mean() - ci:+.4f}, {slopes.mean() + ci:+.4f}]")
        print(f"  {int((slopes > 0).sum())}/{n} prompts with positive slope")

    print("\n" + "=" * 66)
    print("READING THIS")
    print("=" * 66)
    print("  Positive slope = steering toward expert raises the assumed")
    print("  background, so the direction IS used and the keyword metrics")
    print("  simply missed it.")
    print()
    print("  A tight null, from a judge that used a real range across")
    print("  responses, is the stronger claim: decodable but not used to")
    print("  control this behaviour. With KL < 0.01 at 15+ SD")
    print("  perturbations, that is a well-evidenced reading/control gap.")
    print()
    print("  Validate before concluding either way:")
    print("    python judge_local.py --spot 15")


def spot(n):
    rows, _ = load_rows()
    scored = [r for r in rows if "level" in r]
    if not scored:
        print("run the judge first")
        return
    sample = random.Random(11).sample(scored, min(n, len(scored)))
    out = config.data_path("judge_spotcheck.tsv")
    with open(out, "w") as f:
        f.write("id\tyour_level_1_to_7\tresponse\n")
        for r in sample:
            t = r["text"][:900].replace("\t", " ").replace("\n", " | ")
            f.write(f'{r["_id"]}\t\t{t}\n')
    print(f"{len(sample)} responses -> {out}")
    print("Rate 1-7 without looking at the judge's numbers, then correlate.")
    print("Report that correlation -- it is what makes the automated")
    print("ratings credible rather than a shortcut.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spot", type=int, default=0)
    ap.add_argument("--analyse", action="store_true")
    a = ap.parse_args()
    if a.spot:
        spot(a.spot)
    elif a.analyse:
        analyse()
    else:
        main()
