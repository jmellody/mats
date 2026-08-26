import json, argparse
from collections import defaultdict
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import roc_auc_score
from config import MODEL, TOPICS, DATA, RESULTS

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

# disjoint paraphrase sets: A builds the direction, B tests it
NOVICE = {
    "A": [" I'm completely new to this.",
          " I've never done this before.",
          " I'm a total beginner here."],
    "B": [" I don't know the first thing about it.",
          " This is all brand new to me.",
          " I'm just starting out with it."],
}
EXPERT = {
    "A": [" I've worked with this for years.",
          " I've done this professionally for a while.",
          " I'm pretty experienced with this."],
    "B": [" I've been deep in this for a long time.",
          " I do this for a living, day to day.",
          " I know this area very well."],
}


def render(tok, text):
    msg = [{"role": "user", "content": text}]
    try:
        return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=THINKING)
    except TypeError:
        return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)


def build_prompts():
    rows = []
    for topic, desc in TOPICS.items():
        for ti, tmpl in enumerate(TEMPLATES):
            base = tmpl.format(t=desc)
            rows.append({"topic": topic, "template": ti, "arm": "neutral",
                         "set": "-", "text": base})
            for s in ("A", "B"):
                for j, suf in enumerate(NOVICE[s]):
                    rows.append({"topic": topic, "template": ti, "arm": "novice",
                                 "set": s, "text": base + suf})
                for j, suf in enumerate(EXPERT[s]):
                    rows.append({"topic": topic, "template": ti, "arm": "expert",
                                 "set": s, "text": base + suf})
    # suffix-only control: no topic at all
    for s in ("A", "B"):
        for suf in NOVICE[s]:
            rows.append({"topic": "_ctrl", "template": -1, "arm": "novice",
                         "set": s, "text": suf.strip()})
        for suf in EXPERT[s]:
            rows.append({"topic": "_ctrl", "template": -1, "arm": "expert",
                         "set": s, "text": suf.strip()})
    for i, r in enumerate(rows):
        r["idx"] = i
    return rows


@torch.no_grad()
def extract(model_name, rows, bs=32):
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True).eval()
    print("read position:", repr(render(tok, "test")[-60:]))

    out = []
    for i in range(0, len(rows), bs):
        prompts = [render(tok, r["text"]) for r in rows[i:i + bs]]
        enc = tok(prompts, return_tensors="pt", padding=True,
                  add_special_tokens=False).to("cuda")
        hs = model(**enc, output_hidden_states=True).hidden_states[1:]
        out.append(torch.stack([h[:, -1, :] for h in hs], 1).float().cpu().numpy())
        print(f"  {min(i+bs, len(rows))}/{len(rows)}", end="\r")
    return np.concatenate(out)


def direction(a, b):
    d = a.mean(0) - b.mean(0)
    return d / np.linalg.norm(d)


def analyse(acts, rows):
    arm = np.array([r["arm"] for r in rows])
    top = np.array([r["topic"] for r in rows])
    st = np.array([r["set"] for r in rows])
    topics = [t for t in TOPICS]
    nL = acts.shape[1]

    # layer choice: cross-set AUC on anchors (build on A, test on B)
    aucs = []
    for L in range(nL):
        X = acts[:, L, :]
        m = (top != "_ctrl")
        d = direction(X[m & (arm == "expert") & (st == "A")],
                      X[m & (arm == "novice") & (st == "A")])
        te = m & (st == "B") & (arm != "neutral")
        aucs.append(roc_auc_score((arm[te] == "expert").astype(int), X[te] @ d))
    aucs = np.array(aucs)
    L = int(aucs.argmax())
    print(f"\nanchor AUC (build A, test B): L1={aucs[0]:.3f}  peak L{L}={aucs[L]:.3f}")

    X = acts[:, L, :]

    # suffix-only control
    c = top == "_ctrl"
    dc = direction(X[c & (arm == "expert")], X[c & (arm == "novice")])
    m = top != "_ctrl"
    dm = direction(X[m & (arm == "expert")], X[m & (arm == "novice")])
    print(f"cos(topic-free suffix dir, full dir) = {dc @ dm:.3f}")

    # per-topic position, separately for each paraphrase set
    pos = {}
    for s in ("A", "B"):
        p = {}
        for t in topics:
            mt = top == t
            d = direction(X[mt & (arm == "expert") & (st == s)],
                          X[mt & (arm == "novice") & (st == s)])
            lo = (X[mt & (arm == "novice") & (st == s)] @ d).mean()
            hi = (X[mt & (arm == "expert") & (st == s)] @ d).mean()
            neu = X[mt & (arm == "neutral")] @ d
            p[t] = (neu - lo) / (hi - lo)
        pos[s] = p

    print(f"\n{'topic':14s}{'set A':>10s}{'set B':>10s}")
    for t in sorted(topics, key=lambda x: pos["A"][x].mean()):
        print(f"{t:14s}{pos['A'][t].mean():10.2f}{pos['B'][t].mean():10.2f}")

    from scipy.stats import spearmanr
    a = [pos["A"][t].mean() for t in topics]
    b = [pos["B"][t].mean() for t in topics]
    r = spearmanr(a, b)
    print(f"\nparaphrase-set agreement: rho={r.statistic:.3f} p={r.pvalue:.4f}")
    return L, pos, aucs


def bow_control(rows, topics):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    a = [r for r in rows if r["arm"] != "neutral" and r["topic"] != "_ctrl" and r["set"] == "A"]
    n = [r for r in rows if r["arm"] == "neutral"]
    V = TfidfVectorizer(min_df=2)
    clf = LogisticRegression(max_iter=1000).fit(
        V.fit_transform([r["text"] for r in a]), [r["arm"] == "expert" for r in a])
    s = clf.decision_function(V.transform([r["text"] for r in n]))
    tn = np.array([r["topic"] for r in n])
    print("\nBoW ordering on neutral prompts")
    for t in sorted(topics, key=lambda x: s[tn == x].mean()):
        print(f"  {t:14s}{s[tn == t].mean():8.3f}")
    return {t: s[tn == t] for t in topics}


def main(model_name, out):
    rows = build_prompts()
    print(f"{len(rows)} prompts")
    DATA.mkdir(exist_ok=True)
    with open(DATA / "probe2_prompts.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    acts = extract(model_name, rows)
    np.savez_compressed(DATA / out, acts=acts)
    L, pos, aucs = analyse(acts, rows)
    bow_control(rows, list(TOPICS))
    RESULTS.mkdir(exist_ok=True)
    np.savez(RESULTS / "probe2_positions.npz", layer=L, aucs=aucs,
             **{f"{s}_{t}": v for s, p in pos.items() for t, v in p.items()})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL)
    p.add_argument("--out", default="acts_probe2.npz")
    a = p.parse_args()
    main(a.model, a.out)


