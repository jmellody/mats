"""Does steering actually move the representation?

    PROBE_LAYER=26 python steer_check.py

~2 min. Answers the question that decides how to write up the steering null.

THE TWO KINDS OF NULL

  PLUMBING NULL   the hook is not doing anything. The representation does not
                  move, so of course the text does not. Nothing to report
                  except a bug.

  SCIENTIFIC NULL the representation moves as intended and the text still does
                  not change. That is the real claim: the direction is
                  decodable but not used to control this behaviour. It is
                  TalkTuner's reading/control gap, found independently.

The behavioural sweep cannot distinguish these. This can.

WHAT IT MEASURES

1. Does the probe margin at the steered layer shift with alpha? It must, since
   the direction is added to that very layer -- if not, the hook is broken.

2. Does the shift SURVIVE to later layers? This is the interesting one. Adding
   a vector at layer 26 mechanically changes layer 26. Whether layers 27+ still
   carry the shifted value tells you whether the perturbation propagates or
   gets washed out by subsequent computation. A perturbation that does not
   survive cannot influence the output, and that would explain the behavioural
   null mechanically.

3. Does next-token prediction change at all? KL divergence between steered and
   unsteered output distributions. If KL is ~0 the model is not responding to
   the intervention in any way.
"""

import numpy as np
import torch

import config
from acts import load, load_model
from probe import make_probe, mean_diff_direction, probe_direction

ALPHAS = [-3.0, -1.5, 0.0, 1.5, 3.0]
UNIT_FRAC = 0.4

PROMPTS = [
    "I've got two groups in my data and the difference looks big. How should I decide whether to report it?",
    "My model isn't fitting well. What should I look at first?",
    "How should I handle the missing values in my dataset?",
    "I want to know if the change I made actually had an effect. How do I check?",
]


def fmt(prompt, tok):
    msgs = [{"role": "user", "content": prompt}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True)


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


@torch.no_grad()
def probe_and_logits(model, tok, text):
    enc = tok(text, return_tensors="pt").to(config.device())
    out = model(**enc, output_hidden_states=True)
    acts = torch.stack([h[0, -1, :] for h in out.hidden_states]).float().cpu()
    logits = out.logits[0, -1, :].float().cpu()
    return acts.numpy(), torch.log_softmax(logits, dim=-1)


def main():
    tr, trm = load(config.data_path("probe_train_acts.npz"))
    labels = np.array([m["label"] for m in trm])
    probe = make_probe(config.C)
    probe.fit(tr[:, config.LAYER, :], labels)
    scale = probe.decision_function(tr[:, config.LAYER, :]).std()
    norm = float(np.linalg.norm(tr[:, config.LAYER, :], axis=1).mean())
    unit = norm * UNIT_FRAC

    dirs = {"logistic": probe_direction(probe),
            "meandiff": mean_diff_direction(tr, labels)}

    model, tok = load_model()
    n_layers = model.config.num_hidden_layers
    print(f"model {config.MODEL}   steering at layer {config.LAYER} "
          f"of {n_layers}")
    print(f"alpha unit = {unit:.2f}  (training margin SD = {scale:.2f})\n")

    # a probe for a LATER layer, to test propagation
    late = min(config.LAYER + 4, n_layers - 1)
    probe_late = make_probe(config.C)
    probe_late.fit(tr[:, late, :], labels)

    for dname, v in dirs.items():
        st = Steerer(model, config.LAYER, v)
        rows = []
        with st:
            for a in ALPHAS:
                marg, marg_late, kls = [], [], []
                for p in PROMPTS:
                    text = fmt(p, tok)
                    st.alpha = 0.0
                    _, base_lp = probe_and_logits(model, tok, text)
                    st.alpha = a * unit
                    acts, lp = probe_and_logits(model, tok, text)
                    st.alpha = 0.0
                    marg.append(probe.decision_function(
                        acts[config.LAYER][None, :])[0])
                    marg_late.append(probe_late.decision_function(
                        acts[late][None, :])[0])
                    kl = torch.sum(lp.exp() * (lp - base_lp)).item()
                    kls.append(kl)
                rows.append((a, np.mean(marg), np.mean(marg_late),
                             np.mean(kls)))

        print("=" * 68)
        print(f"DIRECTION: {dname}")
        print("=" * 68)
        print(f"  {'alpha':>7} {'margin@L' + str(config.LAYER):>12} "
              f"{'margin@L' + str(late):>12} {'KL vs alpha=0':>15}")
        for a, m, ml, kl in rows:
            print(f"  {a:>+7.1f} {m:>12.3f} {ml:>12.3f} {kl:>15.4f}")

        span = max(r[1] for r in rows) - min(r[1] for r in rows)
        span_late = max(r[2] for r in rows) - min(r[2] for r in rows)
        kl_max = max(r[3] for r in rows)

        print(f"\n  margin span at L{config.LAYER}: {span:.2f} "
              f"({span / scale:.1f} training SDs)")
        print(f"  margin span at L{late}:  {span_late:.2f} "
              f"({span_late / scale:.1f} training SDs)")
        print(f"  max KL on next token:  {kl_max:.4f}")

        if span < 0.5:
            print("\n  HOOK IS NOT WORKING. The margin barely moves at the")
            print("  layer being steered. This is a plumbing bug, not a")
            print("  finding. Check the hook is registered on the right")
            print("  module and that alpha is reaching it.")
        elif span_late < span * 0.25:
            print("\n  PERTURBATION DOES NOT PROPAGATE. The representation")
            print("  shifts at L{} but is largely washed out by L{}."
                  .format(config.LAYER, late))
            print("  That mechanically explains the behavioural null: the")
            print("  intervention never reaches the output. Try steering at")
            print("  an earlier layer, or at several layers at once.")
        elif kl_max < 0.01:
            print("\n  REPRESENTATION MOVES, OUTPUT DOES NOT. The steered")
            print("  representation propagates but next-token predictions are")
            print("  essentially unchanged. This is the scientific null:")
            print("  decodable but not used.")
        else:
            print("\n  Representation moves, propagates, and shifts next-token")
            print("  predictions. The behavioural null is then about the")
            print("  MEASURE, not the mechanism -- the model's output changes")
            print("  but not along the axes being scored. Use an LLM judge on")
            print("  explanation level rather than keyword counts.")
        print()

    print("=" * 68)
    print("WHAT TO DO WITH THIS")
    print("=" * 68)
    print("  A steering null is only worth reporting if the hook demonstrably")
    print("  moved the representation. If it did, say so explicitly in the")
    print("  writeup with these numbers -- otherwise a reviewer cannot tell")
    print("  your null from a bug, and neither could you.")


if __name__ == "__main__":
    main()
