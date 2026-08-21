"""Dynamics experiments on the v2 (identity-cue) probe.

    python dynamics_v2.py build
    python dynamics_v2.py extract
    python dynamics_v2.py dose
    python dynamics_v2.py contra
    python dynamics_v2.py order

WHAT CHANGED FROM v1

The v1 experiments built their signal from correct vs incorrect statistical
claims. construct.py showed that probe was a truth detector: 98% of its trained
gap was reproduced by "Paris is the capital of France" vs "...of Germany",
while stated expertise moved it -7%. Every dynamics result from v1 is therefore
about how the model accumulates evidence of STATEMENT TRUTH, not about a user
model.

Here the signal is identity cues -- job title, audience, stance, software --
from data/identity.json. The v2 probe trained on these shows truth
contamination of -2%, stated expertise +73%, and transfers to a held-out cue
type at AUC 0.806.

The DESIGNS are unchanged. Only the signal source swaps. That is deliberate:
running the same experiments on both probes is what lets you say the earlier
results were about truth and these are about the user.

  dose     0/1/2/3 identity cues of one valence, filler in unused slots
  contra   hhh / hhl / lll / llh -- history vs most recent cue
  order    hl vs lh, count-matched, plus separated variants

FILLER is deliberately neutral on expertise: mundane logistics that any user
might say. Given the v1 ceiling problem, dose() prints a headroom check before
reporting anything.
"""

import json
import random
import sys
from collections import defaultdict
from math import erf, sqrt

import numpy as np

import config
from acts import extract as extract_acts
from acts import load as load_acts
from acts import load_model
from probe import make_probe

BOOT = 10000
MAX_DOSE = 3

FILLERS = [
    "I put the whole thing together over the weekend and it's all in one place now.",
    "It took a while to collect but I finally have everything I need in the file.",
    "I've been meaning to look at this properly for a couple of weeks now.",
    "There's quite a lot of it, more than I expected when I started.",
    "I tidied it up a bit yesterday so it's easier to work with now.",
    "It's all in one folder along with the notes I made at the time.",
]

FILES = {
    "dose": ("data/v2_dose.jsonl", "v2_dose_acts.npz"),
    "contra": ("data/v2_contra.jsonl", "v2_contra_acts.npz"),
    "order": ("data/v2_order.jsonl", "v2_order_acts.npz"),
}

CONTRA = {"hhh": "hhh", "hhl": "hhl", "lll": "lll", "llh": "llh"}
# 3 signal slots everywhere, so every conversation in every experiment is 9
# turns -- and matches the padded probe-training conversations. The v1 dose
# experiment scored 7-turn conversations with a probe fit on 5-turn ones and
# the baseline drifted 9.2 while the effect was 1.06. Never again.
ORDER = {"hl": "fhl", "lh": "flh", "hh": "fhh", "ll": "fll",
         "hl_gap": "hfl", "lh_gap": "lfh"}


def conv(*m):
    return [{"role": r, "content": c} for r, c in m]


def pools(path="data/identity.json", seed=7):
    """Cue sentences split by label, shuffled. Cue types are interleaved so a
    single conversation can mix them -- that is intentional: it stops any one
    conversation being carried by one cue type's vocabulary."""
    rows = json.load(open(path))
    rng = random.Random(seed)
    hi = [r["text"] for r in rows if r["label"] == 1]
    lo = [r["text"] for r in rows if r["label"] == 0]
    rng.shuffle(hi)
    rng.shuffle(lo)
    return hi, lo, rng


def build():
    hi, lo, rng = pools()
    wr = json.load(open("data/pairs.json"))["wrappers"]
    n_units = min(len(hi), len(lo)) // MAX_DOSE
    print(f"{len(hi)} high cues, {len(lo)} low cues -> {n_units} units")

    def sig(kind, unit_hi, unit_lo, k):
        return unit_hi[k] if kind == "h" else unit_lo[k]

    def assemble(pattern, unit_hi, unit_lo, w, fill):
        turns = [("user", w), ("assistant", config.ACK)]
        fi = ki = 0
        for ch in pattern:
            if ch == "f":
                s = fill[fi]
                fi += 1
            else:
                s = sig(ch, unit_hi, unit_lo, ki)
                ki += 1
            turns += [("user", s), ("assistant", config.ACK)]
        turns.append(("user", config.FOLLOWUP))
        return conv(*turns)

    dose_rows, contra_rows, order_rows = [], [], []
    for i in range(n_units):
        uh = hi[i * MAX_DOSE:(i + 1) * MAX_DOSE]
        ul = lo[i * MAX_DOSE:(i + 1) * MAX_DOSE]
        w = wr[i % len(wr)]
        fill = [FILLERS[(i + k) % len(FILLERS)] for k in range(4)]

        # dose: signals in the LAST slots, filler pads the front
        for arm, ch in [("hi", "h"), ("lo", "l")]:
            for d in range(MAX_DOSE + 1):
                if arm == "lo" and d == 0:
                    continue
                pat = "f" * (MAX_DOSE - d) + ch * d
                dose_rows.append({
                    "id": f"{arm}_{d}_{i}", "arm": arm if d else "both",
                    "dose": d, "pair": i,
                    "turns": assemble(pat, uh, ul, w, fill)})

        # contradiction: three signals, mixed valence
        for cond, pat in CONTRA.items():
            contra_rows.append({
                "id": f"{cond}_{i}", "condition": cond, "pair": i,
                "turns": assemble(pat, uh, ul, w, fill)})

        # order: two signals, count-matched, adjacent and separated
        for cond, pat in ORDER.items():
            order_rows.append({
                "id": f"{cond}_{i}", "condition": cond, "pair": i,
                "turns": assemble(pat, uh, ul, w, fill)})

    for rows, key in [(dose_rows, "dose"), (contra_rows, "contra"),
                      (order_rows, "order")]:
        path = FILES[key][0]
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        lens = {len(r["turns"]) for r in rows}
        print(f"  {key:7} {len(rows):>5} rows -> {path}   turns {lens}")


def extract():
    model, tok = load_model()
    for key, (path, acts) in FILES.items():
        print(f"\n--- {key} ---")
        extract_acts(path, config.data_path(acts), model=model, tok=tok,
                     per_turn=False)


def fit():
    a, m = load_acts(config.data_path("id_train_acts.npz"))
    y = np.array([r["label"] for r in m])
    p = make_probe(config.C)
    p.fit(a[:, config.LAYER, :], y)
    s = p.decision_function(a[:, config.LAYER, :])
    return p, s, y


def norm_p(t):
    return 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))


def report(x, label):
    n = len(x)
    m, sd = x.mean(), x.std(ddof=1)
    se = sd / sqrt(n) if n > 1 else 0
    t = m / se if se else 0.0
    ci = 1.96 * se
    rng = np.random.default_rng(0)
    b = np.percentile([rng.choice(x, n, replace=True).mean()
                       for _ in range(BOOT)], [2.5, 97.5])
    print(f"  {label:24} {m:+.4f}  t = {t:+6.2f}  p = {norm_p(t):.4f}  "
          f"CI [{m-ci:+.4f}, {m+ci:+.4f}]  boot [{b[0]:+.4f}, {b[1]:+.4f}]")
    return m


def headroom(probe, train_s, train_y, exp_s, label):
    top = np.percentile(train_s[train_y == 1], 95)
    bot = np.percentile(train_s[train_y == 0], 5)
    print(f"  probe range on training cues: [{bot:+.2f}, {top:+.2f}]")
    print(f"  {label} mean: {exp_s.mean():+.2f}   "
          f"room above {top - exp_s.mean():+.2f}, "
          f"below {exp_s.mean() - bot:+.2f}")
    if min(top - exp_s.mean(), exp_s.mean() - bot) < 1.0:
        print("  WARNING: little room in one direction -- the v1 dose")
        print("  experiment failed exactly this check. Interpret with care.")
    else:
        print("  Both directions have room.")


def dose():
    probe, ts, ty = fit()
    a, meta = load_acts(config.data_path(FILES["dose"][1]))
    s = probe.decision_function(a[:, config.LAYER, :])
    print(f"model {config.MODEL}   layer {config.LAYER}   v2 identity probe\n")

    vals = defaultdict(lambda: defaultdict(dict))
    for r, v in zip(meta, s):
        if r["dose"] == 0:
            vals[r["pair"]]["hi"][0] = v
            vals[r["pair"]]["lo"][0] = v
        else:
            vals[r["pair"]][r["arm"]][r["dose"]] = v

    d = np.arange(MAX_DOSE + 1)
    curves = {"hi": [], "lo": []}
    slopes = {"hi": [], "lo": []}
    for _, arms in sorted(vals.items()):
        if not all(set(d) <= set(arms[k]) for k in ("hi", "lo")):
            continue
        for k in ("hi", "lo"):
            y = np.array([arms[k][j] for j in d])
            curves[k].append(y)
            slopes[k].append(np.polyfit(d, y, 1)[0])
    H, L = np.stack(curves["hi"]), np.stack(curves["lo"])
    bh, bl = np.array(slopes["hi"]), np.array(slopes["lo"])
    n = len(bh)

    print("=" * 70)
    print("HEADROOM")
    print("=" * 70)
    headroom(probe, ts, ty, np.concatenate([H[:, 0], L[:, 0]]), "dose-0")

    print("\n" + "=" * 70)
    print(f"TRAJECTORY  (n = {n} units)")
    print("=" * 70)
    print(f"  {'dose':>5} " + " ".join(f"{k:>9}" for k in d))
    for k, Y in [("hi", H), ("lo", L)]:
        print(f"  {k:>5} " + " ".join(f"{v:>+9.3f}" for v in Y.mean(0)))

    print("\n" + "=" * 70)
    print("SLOPES AND VALENCE ASYMMETRY")
    print("=" * 70)
    report(bh, "high-cue slope")
    report(bl, "low-cue slope")
    report(np.abs(bh) - np.abs(bl), "|b_hi| - |b_lo|")

    print("\n" + "=" * 70)
    print("SATURATION")
    print("=" * 70)
    for k, Y in [("hi", H), ("lo", L)]:
        first, last = Y[:, 1] - Y[:, 0], Y[:, MAX_DOSE] - Y[:, MAX_DOSE - 1]
        print(f"\n  {k} arm:")
        report(first, "  1st cue step")
        report(last, "  3rd cue step")
        report(np.abs(first) - np.abs(last), "  |1st| - |3rd|")
    print("\n  positive = later cues matter less (saturating)")


def contra():
    probe, ts, ty = fit()
    a, meta = load_acts(config.data_path(FILES["contra"][1]))
    s = probe.decision_function(a[:, config.LAYER, :])
    print(f"model {config.MODEL}   layer {config.LAYER}   v2 identity probe\n")

    v = defaultdict(dict)
    for r, x in zip(meta, s):
        v[r["pair"]][r["condition"]] = x
    keep = sorted(p for p, c in v.items() if set(CONTRA) <= c.keys())
    g = lambda k: np.array([v[p][k] for p in keep])
    hhh, hhl, lll, llh = g("hhh"), g("hhl"), g("lll"), g("llh")
    n = len(keep)

    print(f"n = {n} units\n")
    print("=" * 70)
    print("CONDITION MEANS")
    print("=" * 70)
    for name, arr in [("hhh  H H H", hhh), ("hhl  H H L", hhl),
                      ("llh  L L H", llh), ("lll  L L L", lll)]:
        print(f"  {name:12} {arr.mean():+.3f}   sd {arr.std():.3f}")

    print("\n" + "=" * 70)
    print("RETENTION -- both conditions end on the same cue valence")
    print("=" * 70)
    report(hhl - lll, "hhl - lll (both end L)")
    report(hhh - llh, "hhh - llh (both end H)")
    comb = ((hhl - lll) + (hhh - llh)) / 2
    m = report(comb, "mean retention")
    rec = ((hhh - hhl) + (llh - lll)) / 2
    mr = report(rec, "final-cue effect")
    if mr:
        print(f"\n  retention / recency = {m / mr:.2f}")
        print("  CAVEAT: retention sums TWO earlier cues against ONE recent")
        print("  one, so part of any ratio above 1 is accumulation rather")
        print("  than primacy. order() holds count constant.")


def order():
    probe, ts, ty = fit()
    a, meta = load_acts(config.data_path(FILES["order"][1]))
    s = probe.decision_function(a[:, config.LAYER, :])
    print(f"model {config.MODEL}   layer {config.LAYER}   v2 identity probe\n")

    v = defaultdict(dict)
    for r, x in zip(meta, s):
        v[r["pair"]][r["condition"]] = x
    keep = sorted(p for p, c in v.items() if set(ORDER) <= c.keys())
    g = lambda k: np.array([v[p][k] for p in keep])
    hl, lh, hh, ll = g("hl"), g("lh"), g("hh"), g("ll")
    hlg, lhg = g("hl_gap"), g("lh_gap")

    print(f"n = {len(keep)} units\n")
    print("=" * 70)
    print("CONDITION MEANS")
    print("=" * 70)
    for name, arr in [("hh  H H", hh), ("hl  H L", hl), ("lh  L H", lh),
                      ("ll  L L", ll), ("hl_gap  H . . L", hlg),
                      ("lh_gap  L . . H", lhg)]:
        print(f"  {name:18} {arr.mean():+.3f}   sd {arr.std():.3f}")

    print("\n" + "=" * 70)
    print("ORDER, COUNT-MATCHED -- the clean primacy test")
    print("=" * 70)
    m = report(hl - lh, "hl - lh")
    print("  > 0 = FIRST cue has more influence (primacy)")
    print("  < 0 = LAST cue has more influence (recency)")
    comp = report(hh - ll, "hh - ll (composition)")
    if comp:
        print(f"\n  order / composition = {m / comp:.3f}")

    print("\n" + "=" * 70)
    print("SEPARATION")
    print("=" * 70)
    report(hlg - lhg, "hl_gap - lh_gap")
    report((hl - lh) - (hlg - lhg), "adjacent - separated")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "dose"
    {"build": build, "extract": extract, "dose": dose,
     "contra": contra, "order": order}[cmd]()
