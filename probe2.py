import json, argparse
from collections import defaultdict
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from config import MODEL, TOPICS, DATA, RESULTS

THINKING = False

# presupposition held constant within family
FAMILIES = {
    # phrased so both anchors stay coherent: a NEW instance, not a first attempt
    "pre": [
        "I'm about to get into {t} again. What should I know going in?",
        "I've got something coming up involving {t}. Where should I start?",
        "I'm going to be dealing with {t} shortly. What should I sort out first?",
        "I'm looking at {t} for something new. What should I think about first?",
        "I need to take another run at {t}. What do I need to line up?",
    ],

    "post": [
        "I tried {t} and it didn't go how I expected. Any ideas?",
        "My {t} isn't working right. What should I check?",
        "I've been doing {t} but something's off. Any thoughts?",
        "I ran into a problem with {t}. What's likely going on?",
        "{t} isn't behaving the way I thought. Where should I look?",
    ],
}

# disjoint paraphrase sets: build direction on one, test on the other
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
        for fam, tmpls in FAMILIES.items():
            for ti, tmpl in enumerate(tmpls):
                base = tmpl.format(t=desc)
                rows.append(dict(topic=topic, family=fam, template=ti,
                                 arm="neutral", set="-", text=base))
                for s in ("A", "B"):
                    for suf in NOVICE[s]:
                        rows.append(dict(topic=topic, family=fam, template=ti,
                                         arm="novice", set=s, text=base + suf))
                    for suf in EXPERT[s]:
                        rows.append(dict(topic=topic, family=fam, template=ti,
                                         arm="expert", set=s, text=base + suf))
    for s in ("A", "B"):                     # suffix-only control, no topic
        for suf in NOVICE[s]:
            rows.append(dict(topic="_ctrl", family="_ctrl", template=-1,
                             arm="novice", set=s, text=suf.strip()))
        for suf in EXPERT[s]:
            rows.append(dict(topic="_ctrl", family="_ctrl", template=-1,
                             arm="expert", set=s, text=suf.strip()))
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
        enc = tok([render(tok, r["text"]) for r in rows[i:i + bs]], return_tensors="pt",
                  padding=True, add_special_tokens=False).to("cuda")
        hs = model(**enc, output_hidden_states=True).hidden_states[1:]
        out.append(torch.stack([h[:, -1, :] for h in hs], 1).float().cpu().numpy())
        print(f"  {min(i+bs, len(rows))}/{len(rows)}", end="\r")
    print()
    return np.concatenate(out)


def direction(a, b):
    d = a.mean(0) - b.mean(0)
    return d / np.linalg.norm(d)


def positions(X, meta, topic, family, set_, keep_set):
    """Calibrated position of each neutral prompt, anchors from `keep_set`."""
    m = (meta["topic"] == topic) & (meta["family"] == family)
    a_ = meta["arm"]
    tr = m & (set_ == keep_set)
    d = direction(X[tr & (a_ == "expert")], X[tr & (a_ == "novice")])
    lo = (X[tr & (a_ == "novice")] @ d).mean()
    hi = (X[tr & (a_ == "expert")] @ d).mean()
    neu = m & (a_ == "neutral")
    return (X[neu] @ d - lo) / (hi - lo), meta["template"][neu]


def analyse(acts, rows, seed=0):
    meta = {k: np.array([r[k] for r in rows]) for k in
            ("topic", "family", "template", "arm", "set")}
    arm, top, st = meta["arm"], meta["topic"], meta["set"]
    topics, fams = list(TOPICS), list(FAMILIES)
    rng = np.random.default_rng(seed)

    aucs = []
    for L in range(acts.shape[1]):
        X = acts[:, L, :]
        m = top != "_ctrl"
        d = direction(X[m & (arm == "expert") & (st == "A")],
                      X[m & (arm == "novice") & (st == "A")])
        te = m & (st == "B") & (arm != "neutral")
        aucs.append(roc_auc_score((arm[te] == "expert").astype(int), X[te] @ d))
    aucs = np.array(aucs)
    L = int(aucs.argmax())
    print(f"anchor AUC (build A / test B):  L0={aucs[0]:.3f}   peak L{L}={aucs[L]:.3f}")

    X = acts[:, L, :]
    c = top == "_ctrl"
    m = ~c
    print(f"cos(suffix-only dir, full dir) = "
          f"{direction(X[c & (arm=='expert')], X[c & (arm=='novice')]) @ direction(X[m & (arm=='expert')], X[m & (arm=='novice')]):.3f}")

    pos = {}   # (topic, family, set) -> array over templates
    for t in topics:
        for f in fams:
            for s in ("A", "B"):
                v, ti = positions(X, meta, t, f, st, s)
                pos[(t, f, s)] = v[np.argsort(ti)]

    both = {(t, f): np.concatenate([pos[(t, f, "A")], pos[(t, f, "B")]])
            for t in topics for f in fams}

    print(f"\n{'topic':14s}" + "".join(f"{f:>12s}" for f in fams))
    for t in sorted(topics, key=lambda x: both[(x, fams[0])].mean()):
        print(f"{t:14s}" + "".join(f"{both[(t,f)].mean():12.2f}" for f in fams))

    pre = np.concatenate([both[(t, "pre")] for t in topics])
    post = np.concatenate([both[(t, "post")] for t in topics])
    print(f"\npre  {pre.mean():+.3f}  (sd {pre.std(ddof=1):.3f})")
    print(f"post {post.mean():+.3f}  (sd {post.std(ddof=1):.3f})")
    print(f"presupposition effect = {post.mean()-pre.mean():+.3f}")

    print("\nvariance decomposition, within family")
    for f in fams:
        v = {t: both[(t, f)] for t in topics}
        within = np.mean([x.var(ddof=1) for x in v.values()])
        n = np.mean([len(x) for x in v.values()])
        mu = np.array([x.mean() for x in v.values()])
        true_b = max(mu.var(ddof=1) - within / n, 0.0)
        obs = mu.var(ddof=1)
        flat, sz = np.concatenate(list(v.values())), [len(x) for x in v.values()]
        null = np.array([np.var([y.mean() for y in
                                 np.split(rng.permutation(flat), np.cumsum(sz)[:-1])])
                         for _ in range(4000)])
        print(f"  {f:5s} within={within:.4f}  between(obs)={obs:.5f}  "
              f"floor={within/n:.5f}  true={true_b:.5f}  p={(null>=obs).mean():.3f}")

    print("\nparaphrase-set stability (same neutral text, different anchors)")
    for f in fams:
        a = [pos[(t, f, "A")].mean() for t in topics]
        b = [pos[(t, f, "B")].mean() for t in topics]
        pv = np.mean([np.var(np.stack([pos[(t, f, "A")], pos[(t, f, "B")]]), axis=0).mean()
                      for t in topics])
        r = spearmanr(a, b)
        print(f"  {f:5s} var(A vs B) = {pv:.4f}   rho = {r.statistic:+.2f} (p={r.pvalue:.3f})")

    print("\ntemplate means within family (residual after presupposition control)")
    for f in fams:
        v = np.stack([both[(t, f)].reshape(2, -1).mean(0) for t in topics])
        for i, tm in enumerate(FAMILIES[f]):
            print(f"  {f:5s} {v[:, i].mean():+.2f}  {tm}")
    return L, aucs, pos, both, meta


def bow_control(rows):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    a = [r for r in rows if r["arm"] != "neutral" and r["topic"] != "_ctrl" and r["set"] == "A"]
    n = [r for r in rows if r["arm"] == "neutral"]
    V = TfidfVectorizer(min_df=2)
    clf = LogisticRegression(max_iter=1000).fit(
        V.fit_transform([r["text"] for r in a]), [r["arm"] == "expert" for r in a])
    s = clf.decision_function(V.transform([r["text"] for r in n]))
    tn = np.array([r["topic"] for r in n])
    out = {t: s[tn == t] for t in TOPICS}
    print("\nBoW on neutral prompts (should be flat across topics)")
    for t in sorted(out, key=lambda x: out[x].mean()):
        print(f"  {t:14s}{out[t].mean():8.3f}")
    print(f"  range = {max(v.mean() for v in out.values()) - min(v.mean() for v in out.values()):.4f}")
    return out


def main(model_name, out):
    rows = build_prompts()
    print(f"{len(rows)} prompts")
    DATA.mkdir(exist_ok=True); RESULTS.mkdir(exist_ok=True)
    with open(DATA / "probe2_prompts.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    acts = extract(model_name, rows)
    np.savez_compressed(DATA / out, acts=acts)
    L, aucs, pos, both, meta = analyse(acts, rows)
    bow = bow_control(rows)
    np.savez(RESULTS / "probe2_positions.npz", layer=L, aucs=aucs,
             **{f"{t}|{f}|{s}": v for (t, f, s), v in pos.items()},
             **{f"bow|{t}": v for t, v in bow.items()})
    print(f"\n-> {RESULTS / 'probe2_positions.npz'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL)
    p.add_argument("--out", default="acts_probe2.npz")
    a = p.parse_args()
    main(a.model, a.out)

