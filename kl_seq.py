"""Sequence-level KL: does steering change predictions across a whole response?

    export PROBE_LAYER=26
    python kl_seq.py

~5 min. No generation -- teacher-forced forward passes over text you already
have.

WHY THIS EXISTS

steer_check.py measured KL at the FIRST generated token only. That is a weak
test: a perturbation could leave token 1 almost unchanged and still compound
across 280 tokens into a different response. Single-token KL cannot see that.

This takes each unsteered generation, feeds it back through the model under
each steering strength, and measures KL between the steered and unsteered
next-token distributions at EVERY position. Three things fall out:

  MEAN KL       average divergence per token across the whole response
  MAX KL        the worst single position -- catches localised divergence that
                a mean would wash out
  POSITION TREND whether divergence grows with depth into the response. If
                early positions are unaffected but later ones diverge, the
                perturbation compounds and single-token KL was misleading.

Also reported: the change in total log-probability the model assigns to its own
unsteered output. If steering made the model prefer a different continuation,
that number should drop. If it does not, the steered model would have produced
much the same text.

INTERPRETING THE SCALE

KL is in nats per token. For reference, resampling at temperature 1.0 vs greedy
produces per-token divergences on the order of 1-3 nats. Values under ~0.05
mean the distribution is barely perturbed; the model would generate nearly the
same text.
"""

import json
from collections import defaultdict

import numpy as np
import torch

import config
from acts import load, load_model
from probe import make_probe, mean_diff_direction, probe_direction

ALPHAS = [-3.0, -1.5, 1.5, 3.0]
UNIT_FRAC = 0.02
MAX_TOK = 320


class Steerer:
    def __init__(self, model, layer, direction):
        self.block = model.model.layers[layer]
        self.v = torch.tensor(direction, dtype=torch.float32)
        self.alpha = 0.0
        self.handle = None

    def _hook(self, mod, inp, out):
        if self.alpha == 0.0:
            return out
        h = out[0] if isinstance(out, tuple) else out
        v = self.v.to(h.device, h.dtype)
        h = h + self.alpha * v
        return (h,) + out[1:] if isinstance(out, tuple) else h

    def __enter__(self):
        self.handle = self.block.register_forward_hook(self._hook)
        return self

    def __exit__(self, *a):
        if self.handle:
            self.handle.remove()


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
    """Log-probs at every position of the response, teacher-forced."""
    p_ids = tok(prompt_text, return_tensors="pt").input_ids
    r_ids = tok(response_text, return_tensors="pt",
                add_special_tokens=False).input_ids[:, :MAX_TOK]
    ids = torch.cat([p_ids, r_ids], dim=1).to(config.device())
    logits = model(input_ids=ids).logits[0].float()
    # positions predicting each response token
    start = p_ids.shape[1] - 1
    end = start + r_ids.shape[1]
    lp = torch.log_softmax(logits[start:end, :], dim=-1)
    return lp.cpu(), r_ids[0]


def main():
    tr, trm = load(config.data_path("probe_train_acts.npz"))
    labels = np.array([m["label"] for m in trm])
    probe = make_probe(config.C)
    probe.fit(tr[:, config.LAYER, :], labels)
    norm = float(np.linalg.norm(tr[:, config.LAYER, :], axis=1).mean())
    unit = norm * UNIT_FRAC
    dirs = {"logistic": probe_direction(probe),
            "meandiff": mean_diff_direction(tr, labels)}

    rows = json.load(open(config.data_path("steer_results.json")))
    base = {r["prompt"]: r["text"] for r in rows
            if r["alpha"] == 0.0 and r["direction"] == "logistic"}
    prompts = sorted(base)
    print(f"model {config.MODEL}   layer {config.LAYER}")
    print(f"{len(prompts)} prompts, teacher-forced over their unsteered text")
    print(f"perturbation at alpha 3 = {3 * unit:.2f} "
          f"vs activation norm {norm:.1f}\n")

    model, tok = load_model()

    for dname, v in dirs.items():
        st = Steerer(model, config.LAYER, v)
        out = defaultdict(lambda: defaultdict(list))
        with st:
            for p in prompts:
                ptext, rtext = chat(p, tok), base[p]
                st.alpha = 0.0
                lp0, ids = seq_logprobs(model, tok, ptext, rtext)
                p0 = lp0.exp()
                own0 = lp0[torch.arange(len(ids)), ids].sum().item()

                for a in ALPHAS:
                    st.alpha = a * unit
                    lp1, _ = seq_logprobs(model, tok, ptext, rtext)
                    st.alpha = 0.0
                    kl = (p0 * (lp0 - lp1)).sum(dim=-1).numpy()
                    own1 = lp1[torch.arange(len(ids)), ids].sum().item()
                    out[a]["mean"].append(float(kl.mean()))
                    out[a]["max"].append(float(kl.max()))
                    n = len(kl)
                    out[a]["first"].append(float(kl[:n // 4].mean()))
                    out[a]["last"].append(float(kl[-n // 4:].mean()))
                    out[a]["dlogp"].append((own1 - own0) / n)

        print("=" * 72)
        print(f"DIRECTION: {dname}")
        print("=" * 72)
        print(f"  {'alpha':>7} {'mean KL':>10} {'max KL':>10} "
              f"{'first 25%':>11} {'last 25%':>10} {'d logp/tok':>12}")
        for a in ALPHAS:
            d = out[a]
            print(f"  {a:>+7.1f} {np.mean(d['mean']):>10.4f} "
                  f"{np.mean(d['max']):>10.4f} {np.mean(d['first']):>11.4f} "
                  f"{np.mean(d['last']):>10.4f} {np.mean(d['dlogp']):>12.4f}")

        big = max(np.mean(out[a]["mean"]) for a in ALPHAS)
        growth = (np.mean(out[3.0]["last"]) - np.mean(out[3.0]["first"]))
        print(f"\n  largest mean KL: {big:.4f} nats/token")
        if big < 0.05:
            print("  Distribution barely perturbed. The steered model would")
            print("  have produced nearly the same text throughout the whole")
            print("  response, not just at the first token.")
        elif big < 0.3:
            print("  Modest perturbation. Some drift, but far below the")
            print("  divergence that temperature sampling introduces.")
        else:
            print("  Substantial perturbation. The behavioural null is then")
            print("  about the MEASURE, not the mechanism.")

        print(f"  first->last 25% change at alpha +3: {growth:+.4f}")
        if growth > 0.02:
            print("  Divergence GROWS through the response -- the effect")
            print("  compounds, and first-token KL understated it.")
        else:
            print("  Divergence does not grow with depth, so first-token KL")
            print("  was not misleading.")
        print()

    print("=" * 72)
    print("FOR THE WRITEUP")
    print("=" * 72)
    print("  Quote mean KL per token across the full response, not the")
    print("  single-token number. State the perturbation size in both")
    print("  activation-norm and probe-SD terms, so a reader can see the")
    print("  representation moved a lot while the output distribution did")
    print("  not.")


if __name__ == "__main__":
    main()
