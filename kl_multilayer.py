"""Does the steering null survive multi-layer intervention?

    export PROBE_LAYER=26
    python kl_multilayer.py

~10 min. Teacher-forced, no generation.

THE PROBLEM WITH THE SINGLE-LAYER RESULT

kl_seq.py steered at layer 26 only -- the layer where the probe reads best --
and found mean KL of 0.0013 nats/token. Two explanations, and they are very
different findings:

  REAL NULL      the direction does not control output behaviour
  WEAK PROBE     a single edit at a late layer is too small an intervention.
                 Layer 26 of 32 is where the representation is most READABLE,
                 which is not necessarily where it is USED. By then the
                 computation that sets register may already be done, and a
                 single-layer edit gets washed out by what follows.

Most steering work applies the direction across a BAND of layers rather than
one, precisely because single-layer edits attenuate. This tests several
configurations:

    single       L26 only                     (what was run before)
    band_late    L20-26                       edit while the read is forming
    band_mid     L13-26                       from mid-stack
    band_early   L6-19                        before the representation
                                              consolidates
    all_past_6   L6-31                        everything downstream of early

If KL stays near 0.001 everywhere, the null is robust and the single-layer
result was not the limitation. If a band produces substantially more
divergence, the earlier null was about the intervention, not the model, and
should not be reported as a finding.

CAVEAT ON THE BANDS: applying the same vector at several layers is not
principled -- the "same" direction in a different layer's basis may mean
something else. Treat a positive result here as evidence that the single-layer
test was underpowered, not as evidence about a specific layer's role.
"""

import json
from collections import defaultdict

import os

import os

import numpy as np
import torch

import config
from acts import load, load_model
from probe import make_probe, mean_diff_direction, probe_direction

UNIT_FRAC = 0.02
ALPHA = 3.0
MAX_TOK = 320


class MultiSteerer:
    """Adds alpha * direction at every layer in `layers`."""

    def __init__(self, model, layers, direction):
        self.blocks = [model.model.layers[i] for i in layers]
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


def chat(prompt, tok):
    msgs = [{"role": "user", "content": prompt}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True)


@torch.no_grad()
def seq_logprobs(model, tok, prompt_text, response_text):
    p_ids = tok(prompt_text, return_tensors="pt").input_ids
    r_ids = tok(response_text, return_tensors="pt",
                add_special_tokens=False).input_ids[:, :MAX_TOK]
    ids = torch.cat([p_ids, r_ids], dim=1).to(config.device())
    logits = model(input_ids=ids).logits[0].float()
    start = p_ids.shape[1] - 1
    lp = torch.log_softmax(logits[start:start + r_ids.shape[1], :], dim=-1)
    return lp.cpu(), r_ids[0]


def main():
    tr, trm = load(config.data_path(os.environ.get("STEER_TRAIN","ag_train_acts.npz")))
    labels = np.array([m["label"] for m in trm])
    probe = make_probe(config.C)
    probe.fit(tr[:, config.LAYER, :], labels)
    norm = float(np.linalg.norm(tr[:, config.LAYER, :], axis=1).mean())
    unit = norm * UNIT_FRAC
    v = mean_diff_direction(tr, labels)

    rows = json.load(open(config.data_path(os.environ.get("STEER_OUT","steer_v3_results.json"))))
    base = {r["prompt"]: r["text"] for r in rows
            if r["alpha"] == 0.0 and r["direction"] == "logistic"}
    prompts = sorted(base)

    model, tok = load_model()
    n_layers = model.config.num_hidden_layers
    L = config.LAYER

    configs = {
        "single    L%d" % L: [L],
        "band_late L%d-%d" % (max(L - 6, 0), L): list(range(max(L - 6, 0), L + 1)),
        "band_mid  L%d-%d" % (max(L - 13, 0), L): list(range(max(L - 13, 0), L + 1)),
        "band_early L6-%d" % max(L - 7, 7): list(range(6, max(L - 7, 7) + 1)),
        "all_past_6 L6-%d" % (n_layers - 1): list(range(6, n_layers)),
    }

    print(f"model {config.MODEL}   {n_layers} layers   probe layer {L}")
    print(f"per-layer perturbation = {ALPHA * unit:.2f} "
          f"vs activation norm {norm:.1f}")
    print(f"{len(prompts)} prompts, teacher-forced, alpha = +/-{ALPHA}\n")

    results = {}
    for name, layers in configs.items():
        st = MultiSteerer(model, layers, v)
        kls, dlp = [], []
        with st:
            for p in prompts:
                ptext, rtext = chat(p, tok), base[p]
                st.alpha = 0.0
                lp0, ids = seq_logprobs(model, tok, ptext, rtext)
                p0 = lp0.exp()
                own0 = lp0[torch.arange(len(ids)), ids].sum().item()
                for sgn in (-1, 1):
                    st.alpha = sgn * ALPHA * unit
                    lp1, _ = seq_logprobs(model, tok, ptext, rtext)
                    st.alpha = 0.0
                    kls.append(float((p0 * (lp0 - lp1)).sum(-1).mean()))
                    own1 = lp1[torch.arange(len(ids)), ids].sum().item()
                    dlp.append((own1 - own0) / len(ids))
        results[name] = (len(layers), float(np.mean(kls)),
                         float(np.mean(dlp)))
        print(f"  {name:22} {len(layers):>3} layers   "
              f"mean KL {np.mean(kls):.4f}   d logp/tok {np.mean(dlp):+.4f}",
              flush=True)

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    single = results[[k for k in results if k.startswith("single")][0]][1]
    best = max(results.items(), key=lambda kv: kv[1][1])
    print(f"  single layer:  {single:.4f} nats/token")
    print(f"  best config:   {best[1][1]:.4f}  ({best[0].strip()})")
    ratio = best[1][1] / single if single else float("inf")
    print(f"  ratio: {ratio:.1f}x")

    if best[1][1] < 0.05:
        print("\n  NULL IS ROBUST. Even applying the direction across the")
        print("  whole stack leaves the output distribution essentially")
        print("  unchanged. The single-layer result was not the limitation.")
        print("  Report the steering null with this as supporting evidence.")
    elif ratio > 5:
        print("\n  SINGLE-LAYER WAS THE LIMITATION. Multi-layer intervention")
        print("  produces substantially more divergence. Do not report the")
        print("  single-layer null as a finding -- rerun the behavioural")
        print("  sweep with the band that worked, and say which one.")
    else:
        print("\n  Some increase, still small. Report the range across")
        print("  configurations rather than a single number.")

    print("\n  Note for the writeup: the per-layer perturbation is held")
    print("  constant across configurations, so a band applies more total")
    print("  perturbation than a single layer. That is the intended")
    print("  comparison -- but say so, rather than implying the")
    print("  interventions are matched in magnitude.")


if __name__ == "__main__":
    main()
