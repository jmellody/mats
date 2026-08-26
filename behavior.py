import json, re, time, argparse
from collections import defaultdict
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import MODEL, TOPICS, RESULTS
from probe2 import FAMILIES, NOVICE, EXPERT, THINKING

HEDGE = re.compile(r"\b(might|may|could|generally|typically|usually|often|depends|varies)\b", re.I)
DEFINE = re.compile(r"(\(\s*(i\.e\.|e\.g\.|that is|meaning)|—\s*(that is|which is|meaning)|"
                    r"\bwhich (is|means)\b|\bin other words\b|\brefers to\b|\bknown as\b|"
                    r"\bthis just means\b|\bbasically\b)", re.I)
SIMPLIFY = re.compile(r"\b(think of it|imagine|like a |it's like|simply put|in simple terms|"
                      r"don't worry|no need to|for now,|start(ing)? simple)\b", re.I)
STRUCT = re.compile(r"^\s*([-*•]|\d+[\.\)])\s", re.M)
KEYS = ["long_word_frac", "mean_sent_len", "hedges", "defines", "simplifies", "bullets", "n_words"]


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
    p = 100.0 / len(words)
    return {"long_word_frac": sum(len(w) >= 8 for w in words) / len(words),
            "mean_sent_len": len(words) / len(sents),
            "hedges": len(HEDGE.findall(txt)) * p,
            "defines": len(DEFINE.findall(txt)) * p,
            "simplifies": len(SIMPLIFY.findall(txt)) * p,
            "bullets": len(STRUCT.findall(txt)) * p,
            "n_words": len(words)}


def build_jobs(n_neutral, n_anchor):
    """One job = (metadata, prompt text, n samples to draw)."""
    jobs = []
    for topic, desc in TOPICS.items():
        for fam, tmpls in FAMILIES.items():
            for ti, tmpl in enumerate(tmpls):
                stem = tmpl.format(t=desc)
                base = dict(topic=topic, family=fam, template=ti)
                jobs.append((dict(base, arm="neutral", set="-", para=-1), stem, n_neutral))
                for s in ("A", "B"):
                    for j, suf in enumerate(NOVICE[s]):
                        jobs.append((dict(base, arm="novice", set=s, para=j), stem + suf, n_anchor))
                    for j, suf in enumerate(EXPERT[s]):
                        jobs.append((dict(base, arm="expert", set=s, para=j), stem + suf, n_anchor))
    return jobs


def job_key(m):
    return f"{m['topic']}|{m['family']}|{m['template']}|{m['arm']}|{m['set']}|{m['para']}"


@torch.no_grad()
def main(model_name, n_neutral, n_anchor, max_new, out, resume):
    path = RESULTS / out
    RESULTS.mkdir(exist_ok=True)
    done = defaultdict(int)
    if resume and path.exists():
        for l in open(path):
            done[job_key(json.loads(l))] += 1
        print(f"resuming: {sum(done.values())} responses already on disk")

    jobs = build_jobs(n_neutral, n_anchor)
    todo = [(m, t, n - done[job_key(m)]) for m, t, n in jobs if n - done[job_key(m)] > 0]
    total = sum(n for _, _, n in todo)
    print(f"{len(jobs)} prompts, {total} generations to run")

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True).eval()
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    f = open(path, "a")
    t0, seen = time.time(), 0
    for k, (meta, text, n) in enumerate(todo):
        enc = tok([render(tok, text)] * n, return_tensors="pt",
                  padding=True, add_special_tokens=False).to("cuda")
        o = model.generate(**enc, max_new_tokens=max_new, do_sample=True,
                           temperature=0.8, top_p=0.95, pad_token_id=pad)
        for i in range(n):
            txt = tok.decode(o[i, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            ft = features(txt)
            if ft:
                f.write(json.dumps({**meta, "sample": i, "question": text,
                                    "response": txt, **ft}) + "\n")
        f.flush()
        seen += n
        if k % 20 == 0 or k == len(todo) - 1:
            el = time.time() - t0
            eta = el / max(seen, 1) * (total - seen)
            print(f"  {seen}/{total}  elapsed {el/60:.0f}m  eta {eta/60:.0f}m", flush=True)
    f.close()
    summarise([json.loads(l) for l in open(path)])


def summarise(rows):
    g = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        for k in KEYS[:-1]:
            g[k][(r["topic"], r["family"])][r["arm"]].append(r[k])

    print(f"\n{len(rows)} responses\n\nanchor separation (expert - novice)")
    consistent = []
    for k in KEYS[:-1]:
        sep = [np.mean(v["expert"]) - np.mean(v["novice"]) for v in g[k].values()
               if v["expert"] and v["novice"]]
        frac = np.mean([s > 0 for s in sep])
        print(f"  {k:16s} {np.mean(sep):8.3f}   same sign {frac*100:3.0f}%")
        if frac >= 0.9:
            consistent.append(k)
    print(f"\ncomposite over: {consistent}")

    def calib(k, nboot=2000, seed=0):
        """Position per (topic, family), with a bootstrap CI on the anchor gap.
        A cell is 'informative' if that CI excludes zero."""
        rng = np.random.default_rng(seed)
        out = {}
        for key, v in g[k].items():
            nov, exp, neu = (np.array(v["novice"]), np.array(v["expert"]),
                             np.array(v["neutral"]))
            if len(nov) < 3 or len(exp) < 3 or len(neu) < 3:
                continue
            gaps = (rng.choice(exp, (nboot, len(exp))).mean(1)
                    - rng.choice(nov, (nboot, len(nov))).mean(1))
            lo_ci, hi_ci = np.percentile(gaps, [2.5, 97.5])
            gap = exp.mean() - nov.mean()
            out[key] = dict(pos=(neu - nov.mean()) / gap if gap != 0 else np.nan,
                            gap=gap, gap_lo=lo_ci, gap_hi=hi_ci,
                            informative=bool(lo_ci > 0))
        return out

    per = {k: calib(k) for k in consistent}
    for k in consistent:
        bad = [f"{t}/{f}" for (t, f), d in per[k].items() if not d["informative"]]
        print(f"  {k}: anchor gap CI includes 0 in {len(bad)}/{len(per[k])} cells {bad}")

    print("\nper-feature position spread")
    for k in consistent:
        v = [np.clip(per[k][key]["pos"], -1, 2).mean()
             for key in per[k] if per[k][key]["informative"]]
        print(f"  {k:16s} mean {np.mean(v):+.3f}  sd {np.std(v):.3f}  "
              f"range [{min(v):+.2f}, {max(v):+.2f}]  n={len(v)}")




    pos, nfeat = {}, {}
    for key in set().union(*(set(v) for v in per.values())):
        vals = [np.clip(per[k][key]["pos"], -1, 2).mean()
                for k in consistent if key in per[k] and per[k][key]["informative"]]
        if len(vals) >= 2:
            pos[key] = float(np.mean(vals))
            nfeat[key] = len(vals)

    print(f"\n{'topic':14s}" + "".join(f"{f:>12s}" for f in FAMILIES))
    for t in TOPICS:
        print(f"{t:14s}" + "".join(
            f"{pos[(t,f)]:9.2f}({nfeat[(t,f)]})" if (t, f) in pos else f"{'--':>12s}"
            for f in FAMILIES))
    for fam in FAMILIES:
        v = np.array([pos[(t, fam)] for t in TOPICS if (t, fam) in pos])
        print(f"\n{fam:5s} mean {v.mean():+.3f}  sd {v.std(ddof=1):.3f}  n={len(v)}")

    np.savez(RESULTS / "behavior_positions.npz",
             feats=np.array(consistent),
             **{f"{t}|{f}": np.array([v]) for (t, f), v in pos.items()})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL)
    p.add_argument("--n-neutral", type=int, default=8)
    p.add_argument("--n-anchor", type=int, default=2)
    p.add_argument("--max-new", type=int, default=250)
    p.add_argument("--out", default="behavior.jsonl")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--summarise", default=None)
    a = p.parse_args()
    if a.summarise:
        summarise([json.loads(l) for l in open(RESULTS / a.summarise)])
    else:
        main(a.model, a.n_neutral, a.n_anchor, a.max_new, a.out, not a.no_resume)


