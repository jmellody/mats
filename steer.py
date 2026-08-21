"""Steering: does the model USE the expertise direction, or just encode it?

    python steer.py pilot          # 3 prompts, eyeball the text
    python steer.py run            # full sweep + behavioural scoring
    python steer.py analyse

WHY THIS IS THE IMPORTANT EXPERIMENT

Everything else in this project is correlational. A probe reading 0.74 says a
direction CORRELATES with user expertise; it does not say the model uses that
direction when deciding how to explain something. Steering is the causal test:
add alpha * direction to the residual stream during generation and see whether
the output actually changes.

A NULL HERE IS STILL A RESULT. TalkTuner found their best READING probes were
not their best CONTROL probes and had to train separate ones. Landing in the
same place independently is a real finding about what linear probes on this
attribute capture. Report it either way.

TWO DIRECTIONS ARE TESTED

  logistic   the probe's own weight vector, optimised to discriminate
  meandiff   mean(correct) - mean(incorrect), optimised for nothing

The logistic direction often reads better and steers worse, because
discriminating and moving are different objectives. Trying both costs one extra
generation pass and pre-empts an obvious reviewer question.

BEHAVIOURAL MEASURES

Probe value moving is not evidence of use -- of course it moves, you added the
direction to it. What matters is whether the TEXT changes in the way a shifted
user model predicts: more jargon and less scaffolding when steered toward
expert, the reverse when steered toward novice.
"""

import json
import re
import sys
from collections import defaultdict

import os

import numpy as np
import torch

import config
import logs
from acts import load, load_model
from probe import (fit_final_probe, make_probe, margin_scale,
                   mean_diff_direction, probe_direction)

ALPHAS = [-3.0, -1.5, 0.0, 1.5, 3.0]   # in units of training-margin SD
N_PROMPTS = 12
UNIT_FRAC = 0.02     # fraction of mean activation norm, per layer per alpha step
BAND = 6             # steer across config.LAYER-BAND .. config.LAYER
MAX_NEW = 450
OUT = os.environ.get("STEER_OUT", "steer_v2_results.json")

# Neutral requests: the user asks for an explanation without revealing how much
# they know. Any change in how these are answered has to come from steering.
PROMPTS = [
    "I've got two groups in my data and the difference looks big. How should I decide whether to report it?",
    "My model isn't fitting well. What should I look at first?",
    "How should I handle the missing values in my dataset?",
    "I want to know if the change I made actually had an effect. How do I check?",
    "What's the right way to summarise this data for a report?",
    "Some of my measurements look way off from the rest. What should I do with them?",
    "How many observations do I need for this to be worth analysing?",
    "I have data collected at several sites. Does that change how I should analyse it?",
    "My results changed when I added another variable. What does that mean?",
    "How do I tell whether the relationship I'm seeing is real?",
    "I need to compare more than two groups. What's the approach?",
    "The data isn't shaped the way I expected. Does that matter?",
]

# --- behavioural scoring -----------------------------------------------------

JARGON = [
    "heteroskedastic", "homoskedastic", "multicollinearity", "collinearity",
    "confidence interval", "p-value", "null hypothesis", "type i", "type ii",
    "statistical power", "effect size", "cohen", "bonferroni", "welch",
    "bootstrap", "quantile", "residual", "variance", "covariance",
    "regression", "coefficient", "estimator", "unbiased", "asymptotic",
    "likelihood", "posterior", "prior", "bayesian", "frequentist",
    "degrees of freedom", "standard error", "confounder", "confounding",
    "instrumental variable", "propensity", "stratif", "clustered",
    "random effect", "fixed effect", "interaction term", "main effect",
    "distribution", "parametric", "nonparametric", "imputation",
    "cross-validation", "overfitting", "regularis", "regulariz", "shrinkage",
    "specificity", "sensitivity", "roc", "auc", "kurtosis", "skew",
]

DEFINING = [
    "which means", "that is,", "in other words", "i.e.", "refers to",
    "basically", "put simply", "think of it as", "essentially",
    "to put it another way", "what this means is", "in simple terms",
    "meaning that", "or rather", "for example", "for instance",
]

SCAFFOLD = [
    "as you know", "you probably know", "you may know", "you'll recall",
    "as you're aware", "familiar with", "you likely know", "no doubt you",
    "as you'd expect", "of course,",
]


def score_text(t):
    low = t.lower()
    words = re.findall(r"[a-z][a-z'-]*", low)
    nw = max(len(words), 1)
    sents = [s for s in re.split(r"[.!?]+", t) if s.strip()]
    return {
        "words": len(words),
        "jargon_per_100": 100 * sum(low.count(j) for j in JARGON) / nw,
        "defining_per_100": 100 * sum(low.count(d) for d in DEFINING) / nw,
        "scaffold_per_100": 100 * sum(low.count(s) for s in SCAFFOLD) / nw,
        "sent_len": nw / max(len(sents), 1),
        "long_words_pct": 100 * sum(len(w) > 8 for w in words) / nw,
    }



def chat(prompt, tok):
    """Format a single user turn, disabling the reasoning trace if the
    tokenizer supports it.

    Qwen3.5 is a reasoning model: without this, every generation opens with
    "Here's a thinking process..." and the behavioural scorer measures the
    chain of thought rather than the answer the user would see.
    """
    msgs = [{"role": "user", "content": prompt}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True)


def strip_thinking(t):
    """Drop any reasoning trace that survived enable_thinking=False."""
    for tag in ("</think>", "</thinking>"):
        if tag in t:
            t = t.split(tag)[-1]
    return t.strip()


# --- steering hook -----------------------------------------------------------

class Steerer:
    """Adds alpha * direction to the residual stream at EVERY layer in a band.

    Single-layer steering at the probe layer was measured (kl_multilayer.py) to
    change the output distribution by only 0.0012 nats/token -- three orders of
    magnitude below the divergence temperature sampling introduces. It could
    not have tested whether the direction controls behaviour, because it barely
    perturbed behaviour at all.

    Band L20-26 gives 0.083 nats/token: meaningful perturbation, still well
    short of the 1-3 nats where the model stops functioning. Wider bands
    (L13-26 at 1.29, L6-31 at 4.64) destroy the forward pass -- the model
    assigns its own prior output ~40x less probability per token -- and are the
    same failure mode as an over-large alpha, reached by stacking layers.

    Hooks stay registered across generation steps, so every generated token is
    affected, not just the prompt.
    """

    def __init__(self, model, layers, direction):
        if isinstance(layers, int):
            layers = [layers]
        self.blocks = [model.model.layers[i] for i in layers]
        self.layers = list(layers)
        self.v = torch.tensor(direction, dtype=torch.float32)
        self.alpha = 0.0
        self.handles = []

    def _hook(self, mod, inp, out):
        if self.alpha == 0.0:
            return out
        h = out[0] if isinstance(out, tuple) else out
        v = self.v.to(h.device, h.dtype)
        h = h + self.alpha * v
        return (h,) + out[1:] if isinstance(out, tuple) else h

    def __enter__(self):
        self.handles = [b.register_forward_hook(self._hook)
                        for b in self.blocks]
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()
        self.handles = []


@torch.no_grad()
def generate(model, tok, text, steerer, alpha, max_new=MAX_NEW):
    steerer.alpha = alpha
    enc = tok(text, return_tensors="pt").to(config.device())
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    steerer.alpha = 0.0
    txt = tok.decode(out[0][enc["input_ids"].shape[1]:],
                     skip_special_tokens=True)
    return strip_thinking(txt)


def band():
    lo = max(config.LAYER - BAND, 0)
    return list(range(lo, config.LAYER + 1))


TRAIN_ACTS = os.environ.get("STEER_TRAIN", "id_train_acts.npz")


def setup():
    tr, trm = load(config.data_path(TRAIN_ACTS))
    labels = np.array([m["label"] for m in trm])
    probe = make_probe(config.C)
    probe.fit(tr[:, config.LAYER, :], labels)
    scale = margin_scale(probe, tr)
    dirs = {
        "logistic": probe_direction(probe),
        "meandiff": mean_diff_direction(tr, labels),
    }
    cos = float(dirs["logistic"] @ dirs["meandiff"])
    print(f"cos(logistic, meandiff) = {cos:+.3f}")
    print(f"probe train set: {TRAIN_ACTS}")
    print(f"training margin SD = {scale:.2f}")
    b = band()
    print(f"steering band: L{b[0]}-L{b[-1]} ({len(b)} layers)")
    # activation norm sets the meaningful scale for alpha
    norm = float(np.linalg.norm(tr[:, config.LAYER, :], axis=1).mean())
    print(f"mean activation norm at layer {config.LAYER} = {norm:.1f}")
    return probe, dirs, norm


def pilot():
    """Three prompts at three strengths. Read the text yourself before
    trusting any automated score."""
    logs.start("steer_pilot")
    probe, dirs, norm = setup()
    model, tok = load_model()
    unit = norm * UNIT_FRAC

    for dname in ("meandiff", "logistic"):
        st = Steerer(model, band(), dirs[dname])
        with st:
            for p in PROMPTS[:2]:
                print(f"\n{'=' * 70}\n[{dname}] {p}\n{'=' * 70}")
                for a in [-2.0, 0.0, 2.0]:
                    out = generate(model, tok, text=chat(p, tok),
                                   steerer=st, alpha=a * unit, max_new=300)
                    sc = score_text(out)
                    print(f"\n--- alpha {a:+.1f} | jargon "
                          f"{sc['jargon_per_100']:.1f} | defining "
                          f"{sc['defining_per_100']:.1f} | {sc['words']}w ---")
                    print(out[:700] if out else "(EMPTY -- steering too strong)")
    print("\nIf the three outputs look identical, steering is not working:")
    print("raise the alpha unit, or try a different layer.")


def run():
    logs.start("steer")
    probe, dirs, norm = setup()
    model, tok = load_model()
    unit = norm * UNIT_FRAC
    rows = []

    for dname, v in dirs.items():
        st = Steerer(model, band(), v)
        with st:
            jobs = [(p, a) for p in PROMPTS[:N_PROMPTS] for a in ALPHAS]
            for i, (p, a) in logs.progress(jobs, f"gen[{dname}]"):
                text = chat(p, tok)
                out = generate(model, tok, text, st, a * unit)
                rows.append({"direction": dname, "prompt": p, "alpha": a,
                             "text": out, **score_text(out)})

    json.dump(rows, open(config.data_path(OUT), "w"), indent=2)
    print(f"\n{len(rows)} generations -> {config.data_path(OUT)}")
    analyse()


def analyse():
    rows = json.load(open(config.data_path(OUT)))
    metrics = ["jargon_per_100", "defining_per_100", "scaffold_per_100",
               "words", "sent_len", "long_words_pct"]

    print(f"\nmodel {config.MODEL}   layer {config.LAYER}")
    for dname in sorted({r["direction"] for r in rows}):
        sub = [r for r in rows if r["direction"] == dname]
        print("\n" + "=" * 74)
        print(f"DIRECTION: {dname}")
        print("=" * 74)
        alphas = sorted({r["alpha"] for r in sub})
        print(f"  {'metric':20}" + "".join(f"{a:>+9.1f}" for a in alphas)
              + f"{'slope':>10}{'p':>9}")
        for mname in metrics:
            by = defaultdict(list)
            for r in sub:
                by[r["alpha"]].append(r[mname])
            means = [np.mean(by[a]) for a in alphas]
            # per-prompt slope, so each prompt is its own control
            slopes = []
            for p in {r["prompt"] for r in sub}:
                pts = sorted((r["alpha"], r[mname])
                             for r in sub if r["prompt"] == p)
                x = np.array([a for a, _ in pts])
                y = np.array([v for _, v in pts])
                if len(set(x)) > 1:
                    slopes.append(np.polyfit(x, y, 1)[0])
            slopes = np.array(slopes)
            n = len(slopes)
            se = slopes.std(ddof=1) / np.sqrt(n) if n > 1 else 0
            t = slopes.mean() / se if se else 0
            from math import erf, sqrt
            pv = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
            print(f"  {mname:20}" + "".join(f"{m:>9.2f}" for m in means)
                  + f"{slopes.mean():>+10.3f}{pv:>9.4f}")

        deg = sum(1 for r in sub if r["words"] < 15)
        if deg:
            print(f"\n  {deg}/{len(sub)} outputs under 15 words -- steering may")
            print("  be degrading generation rather than shifting register.")
            print("  Lower the alpha unit and rerun.")

    print("\n" + "=" * 74)
    print("HOW TO READ THIS")
    print("=" * 74)
    print("  Steering toward EXPERT (positive alpha) predicts: jargon up,")
    print("  defining down, scaffolding down, words down.")
    print("  Toward NOVICE (negative alpha): the reverse.")
    print()
    print("  If jargon slope is significant and outputs are not degraded,")
    print("  the direction is causally used -- that is the finding this")
    print("  whole project was missing.")
    print()
    print("  If nothing moves, say so: the direction is decodable but not")
    print("  demonstrably used. That is TalkTuner's reading/control gap and")
    print("  it is worth reporting.")
    print()
    print("  READ THE ACTUAL TEXT before believing any of these numbers.")
    print(f"  It is all in {config.data_path(OUT)}.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    {"pilot": pilot, "run": run, "analyse": analyse}[cmd]()
