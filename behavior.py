import json, re, argparse
from collections import defaultdict
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import MODEL, TOPICS, RESULTS

THINKING = False

TEMPLATES = [
    "How do I get started with {t}?",
    "What's the best way to approach {t}?",
    "I'm having trouble with {t}. Any advice?",
    "Can you explain how {t} works?",
    "What should I know about {t}?",
    "Why isn't my {t} working the way I expected?",
    "Is there a standard approach for {t}, or does it depend?",
    "What's the most common mistake people make with {t}?",
]

ARMS = {
    "neutral": "",
    "novice": " I'm completely new to this.",
    "expert": " I've worked with this for years.",
}

HEDGE = re.compile(r"\b(might|may|could|generally|typically|usually|often|depends|varies)\b", re.I)
DEFINE = re.compile(r"(\(\s*(i\.e\.|e\.g\.|that is|meaning)|—\s*(that is|which is|meaning)|"
                    r"\bwhich (is|means)\b|\bin other words\b|\brefers to\b|\bknown as\b|"
                    r"\bthis just means\b|\bbasically\b)", re.I)
SIMPLIFY = re.compile(r"\b(think of it|imagine|like a |it's like|simply put|in simple terms|"
                      r"don't worry|no need to|for now,|start(ing)? simple)\b", re.I)
STRUCT = re.compile(r"^\s*([-*•]|\d+[\.\)])\s", re.M)


def render(tok, text):
    msg = [{"role": "user", "content": text}]
    try:
        return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=THINKING)
    except TypeError:
        return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)


def features(txt):
    words = re.findall(r"[a-zA-Z']+", txt)
    sents = [s for s in re.split(r"[.!?]+", txt) if s.strip()]
    if len(words) < 20 or not sents:
        return None
    per100 = 100.0 / len(words)
    return {
        "long_word_frac": sum(len(w) >= 8 for w in words) / len(words),
        "mean_sent_len": len(words) / len(sents),
        "hedges": len(HEDGE.findall(txt)) * per100,
        "defines": len(DEFINE.findall(txt)) * per100,
        "simplifies": len(SIMPLIFY.findall(txt)) * per100,
        "bullets": len(STRUCT.findall(txt)) * per100,
        "n_words": len(words),
    }


KEYS = ["long_word_frac", "mean_sent_len", "hedges", "defines", "simplifies", "bullets", "n_words"]


@torch.no_grad()
def main(model_name, n, max_new, out):
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True).eval()
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    rows = []
    for topic, desc in TOPICS.items():
        for ti, tmpl in enumerate(TEMPLATES):
            for arm, suffix in ARMS.items():
                q = tmpl.format(t=desc) + suffix
                enc = tok([render(tok, q)] * n, return_tensors="pt",
                          padding=True, add_special_tokens=False).to("cuda")
                o = model.generate(**enc, max_new_tokens=max_new, do_sample=True,
                                   temperature=0.8, top_p=0.95, pad_token_id=pad)
                for k in range(n):
                    txt = tok.decode(o[k, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                    f = features(txt)
                    if f:
                        rows.append({"topic": topic, "template": ti, "arm": arm,
                                     "sample": k, "question": q, "response": txt, **f})
        print(f"{topic:14s} {sum(r['topic']==topic for r in rows):4d} responses")

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summarise(rows)
    print(f"\n-> {RESULTS / out}")


def summarise(rows):
    g = defaultdict(lambda: defaultdict(list))
    for r in rows:
        g[r["topic"]][r["arm"]].append([r[k] for k in KEYS])

    print("\ncalibrated position of the default  (0 = novice-addressed, 1 = expert-addressed)")
    print(f"{'topic':14s}" + "".join(f"{k[:10]:>12s}" for k in KEYS[:-1]))
    out = {}
    for t, arms in g.items():
        if len(arms) < 3:
            continue
        mu = {a: np.array(v).mean(0) for a, v in arms.items()}
        span = mu["expert"] - mu["novice"]
        pos = np.where(np.abs(span) > 1e-9, (mu["neutral"] - mu["novice"]) / span, np.nan)
        out[t] = pos
        print(f"{t:14s}" + "".join(f"{x:12.2f}" for x in pos[:-1]))

    print("\nanchor separation (expert - novice; near 0 = feature does not respond)")
    print(f"{'topic':14s}" + "".join(f"{k[:10]:>12s}" for k in KEYS[:-1]))
    for t, arms in g.items():
        if len(arms) < 3:
            continue
        mu = {a: np.array(v).mean(0) for a, v in arms.items()}
        d = mu["expert"] - mu["novice"]
        print(f"{t:14s}" + "".join(f"{x:12.3f}" for x in d[:-1]))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL)
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--max-new", type=int, default=300)
    p.add_argument("--out", default="behavior.jsonl")
    p.add_argument("--summarise", default=None, help="re-summarise an existing jsonl")
    a = p.parse_args()
    if a.summarise:
        summarise([json.loads(l) for l in open(RESULTS / a.summarise)])
    else:
        main(a.model, a.n, a.max_new, a.out)


