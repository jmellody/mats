"""Steer along the expertise direction with no expertise stated in the prompt.

Adds alpha * d at a single layer (compounding across layers destroys generation).
If the direction is causally relevant, features should move monotonically with
alpha, in the same direction they move for the stated anchors.
"""
import json, argparse
from collections import defaultdict
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import MODEL, TOPICS, RESULTS
from probe2 import FAMILIES, NOVICE, EXPERT, render
from behavior import features

FEATS = ["long_word_frac", "mean_sent_len", "defines", "hedges", "simplifies"]


@torch.no_grad()
def build_direction(model, tok, layer, batch=32):
    texts, y = [], []
    for topic, desc in TOPICS.items():
        for fam, tmpls in FAMILIES.items():
            for tmpl in tmpls:
                stem = tmpl.format(t=desc)
                for s in ("A", "B"):
                    for suf in NOVICE[s]:
                        texts.append(stem + suf); y.append(0)
                    for suf in EXPERT[s]:
                        texts.append(stem + suf); y.append(1)
    y = np.array(y)
    acts = []
    for i in range(0, len(texts), batch):
        enc = tok([render(tok, t) for t in texts[i:i + batch]], return_tensors="pt",
                  padding=True, add_special_tokens=False).to("cuda")
        hs = model(**enc, output_hidden_states=True).hidden_states[layer]
        acts.append(hs[:, -1, :].float().cpu().numpy())
    a = np.concatenate(acts)
    d = a[y == 1].mean(0) - a[y == 0].mean(0)
    return d / np.linalg.norm(d), float(np.linalg.norm(a, axis=1).mean())


@torch.no_grad()
def suffix_direction(model, tok, layer):
    """Direction from the bare suffixes with no question attached."""
    texts, y = [], []
    for s in ("A", "B"):
        for suf in NOVICE[s]:
            texts.append(suf.strip()); y.append(0)
        for suf in EXPERT[s]:
            texts.append(suf.strip()); y.append(1)
    y = np.array(y)
    enc = tok([render(tok, t) for t in texts], return_tensors="pt",
              padding=True, add_special_tokens=False).to("cuda")
    hs = model(**enc, output_hidden_states=True).hidden_states[layer]
    a = hs[:, -1, :].float().cpu().numpy()
    d = a[y == 1].mean(0) - a[y == 0].mean(0)
    return d / np.linalg.norm(d)


def make_hook(vec, alpha):
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h = h + alpha * vec
        return (h,) + out[1:] if isinstance(out, tuple) else h
    return hook


@torch.no_grad()
def main(model_name, layer, alphas, n, max_new, out, project_out):
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True).eval()
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    print(f"building direction at layer {layer} ...", flush=True)
    d, scale = build_direction(model, tok, layer)
    s = suffix_direction(model, tok, layer)
    print(f"  |resid| ~ {scale:.1f}   cos(d, suffix-only) = {d @ s:+.3f}")
    if project_out:
        d = d - (d @ s) * s
        d = d / np.linalg.norm(d)
        print(f"  projected out suffix component; new cos = {d @ s:+.4f}")

    vec = torch.tensor(d, dtype=torch.bfloat16, device="cuda") * scale
    layers = model.model.layers if hasattr(model, "model") else model.layers

    prompts = [(t, f, ti, tmpl.format(t=desc))
               for t, desc in TOPICS.items()
               for f, tmpls in FAMILIES.items()
               for ti, tmpl in enumerate(tmpls)]
    print(f"{len(prompts)} prompts x {len(alphas)} alphas x {n} samples "
          f"= {len(prompts)*len(alphas)*n} generations", flush=True)

    RESULTS.mkdir(exist_ok=True)
    fh = open(RESULTS / out, "w")
    for alpha in alphas:
        h = layers[layer].register_forward_hook(make_hook(vec, alpha)) if alpha else None
        try:
            for topic, fam, ti, text in prompts:
                enc = tok([render(tok, text)] * n, return_tensors="pt",
                          padding=True, add_special_tokens=False).to("cuda")
                o = model.generate(**enc, max_new_tokens=max_new, do_sample=True,
                                   temperature=0.8, top_p=0.95, pad_token_id=pad)
                for i in range(n):
                    txt = tok.decode(o[i, enc["input_ids"].shape[1]:],
                                     skip_special_tokens=True)
                    ft = features(txt)
                    if ft:
                        fh.write(json.dumps(dict(
                            alpha=alpha, topic=topic, family=fam, template=ti,
                            sample=i, question=text, response=txt,
                            ascii_frac=sum(c.isascii() for c in txt) / max(len(txt), 1),
                            **ft)) + "\n")
                fh.flush()
        finally:
            if h is not None:
                h.remove()
        print(f"alpha {alpha:+.2f} done", flush=True)
    fh.close()
    summarise([json.loads(l) for l in open(RESULTS / out)])


def summarise(rows):
    g = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for k in FEATS + ["ascii_frac"]:
            if k in r:
                g[r["alpha"]][k].append(r[k])
    alphas = sorted(g)
    base = {k: np.mean(g[0.0][k]) for k in FEATS} if 0.0 in g else None

    print(f"\n{len(rows)} responses\n{'alpha':>7s}" +
          "".join(f"{k[:11]:>17s}" for k in FEATS) + f"{'ascii':>8s}{'n':>7s}")
    for a in alphas:
        line = f"{a:+7.2f}"
        for k in FEATS:
            v = np.mean(g[a][k])
            line += f"{v:10.3f}" + (f"({v-base[k]:+.2f})" if base else "       ")
        line += f"{np.mean(g[a].get('ascii_frac', [1])):8.2f}{len(g[a][FEATS[0]]):7d}"
        print(line)

    print("\nmonotonic increasing in alpha?")
    for k in FEATS:
        v = np.array([np.mean(g[a][k]) for a in alphas])
        d = np.diff(v)
        tag = "yes" if all(d > 0) else ("yes (dec)" if all(d < 0) else "no")
        print(f"  {k:16s} {tag:10s} {np.round(v, 3)}")
    print("\nCheck the ascii column — anything below ~0.98 means generation is "
          "degrading and the feature movement is an artifact.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--alphas", type=float, nargs="+",
                   default=[-0.8, -0.4, 0.0, 0.4, 0.8])
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--max-new", type=int, default=250)
    p.add_argument("--out", default="steering.jsonl")
    p.add_argument("--project-out", action="store_true",
                   help="remove the suffix-only component from the direction")
    p.add_argument("--summarise", default=None)
    a = p.parse_args()
    if a.summarise:
        summarise([json.loads(l) for l in open(RESULTS / a.summarise)])
    else:
        main(a.model, a.layer, sorted(a.alphas), a.n, a.max_new, a.out, a.project_out)


