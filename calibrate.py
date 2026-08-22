"""Pick filler that sits mid-range on the probe, BEFORE running anything.

    export PROBE_LAYER=<N>
    python calibrate.py build
    python calibrate.py extract
    python calibrate.py analyse

THE PROBLEM THIS EXISTS TO PREVENT

Three experiments in this project were compromised by where the filler landed:

  v1 dose   technical filler sat at +3.37 against a 95th percentile of +3.57
            for genuinely expert text. The positive arm had 0.20 of headroom,
            so it could not move. Result: a large, highly significant
            asymmetry (-0.82, t = -7.78) pointing OPPOSITE to the 3B result.
            Units with headroom moved +2.04; units without moved +0.10.

  v2 dose   mundane filler sat at -4.51; trained low-expertise cues sat at
            -4.70. Statistically the same place. The low arm could not move
            because the model had already concluded non-expert. Result:
            low-cue slope +0.014, p = 0.94 -- uninterpretable as "negative
            evidence is ignored".

Both were found afterwards, by which point the experiment had been run and
interpreted. This finds it beforehand.

WHAT IT DOES

Scores several candidate filler families against the trained probe and reports
where each sits relative to the probe's own range. Pick the one closest to the
midpoint between the high and low training means -- that leaves room to move in
both directions.

Ten minutes. It would have saved two experiments.
"""

import json
import sys

import numpy as np

import config
from acts import extract as extract_acts
from acts import load as load_acts
from acts import load_model
from probe import make_probe

STIM = "data/calib.jsonl"
ACTS = "calib_acts.npz"
TRAIN = "ag_train_acts.npz"      # v3 probe; override for other probes

# Candidate filler families, spanning the register range on purpose.
CANDIDATES = {
    "mundane": [
        "I put the whole thing together over the weekend and it's all in one place now.",
        "It took a while to collect but I finally have everything I need in the file.",
        "I've been meaning to look at this properly for a couple of weeks now.",
        "There's quite a lot of it, more than I expected when I started.",
        "I tidied it up a bit yesterday so it's easier to work with now.",
        "It's all in one folder along with the notes I made at the time.",
    ],
    "procedural": [
        "I saved the file out this morning and the columns are all named consistently.",
        "The rows are one per observation and the grouping variables are coded as text.",
        "I kept the original and the cleaned version separate in case I need to go back.",
        "Everything runs from a single file so I can regenerate it whenever I need to.",
        "The summaries are already written out alongside the counts for each group.",
        "I have not changed any of the default settings since I set it up.",
    ],
    "technical": [
        "I ran the models last night and exported the coefficient tables into a spreadsheet.",
        "The estimates and their intervals are saved in one table with the cell counts.",
        "I fit it with the package defaults and have not touched the convergence settings.",
        "All the variables are on their original scales and the summaries are printed.",
        "The diagnostic plots are saved next to the model output from the last run.",
        "I split the data by group and computed the summaries for each one separately.",
    ],
    "logistical": [
        "The deadline for this moved up a week so I am working through it faster than planned.",
        "I am picking this up from where someone else left off a few months ago.",
        "There are two other datasets I will need to bring in at some point as well.",
        "I have about an hour a day to work on this alongside everything else.",
        "The person who normally handles this is away until the end of the month.",
        "I will need to hand the results over to someone else once it is done.",
    ],
}

ACK = "Got it. Anything else about the setup I should know?"
FOLLOWUP = "So how should I decide what to actually report?"


def conv(*m):
    return [{"role": r, "content": c} for r, c in m]


def build(out=STIM, wrappers_each=6):
    wr = json.load(open("data/pairs.json"))["wrappers"]
    rows = []
    for fam, items in CANDIDATES.items():
        for i, s in enumerate(items):
            for j in range(wrappers_each):
                w = wr[(i * wrappers_each + j) % len(wr)]
                # same 9-turn shape as everything else; all three signal
                # slots are filler from this family
                p2 = items[(i + 1) % len(items)]
                p3 = items[(i + 2) % len(items)]
                rows.append({
                    "id": f"{fam}_{i}_{j}", "family": fam,
                    "turns": conv(("user", w), ("assistant", ACK),
                                  ("user", s), ("assistant", ACK),
                                  ("user", p2), ("assistant", ACK),
                                  ("user", p3), ("assistant", ACK),
                                  ("user", FOLLOWUP)),
                })
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} conversations -> {out}")
    print(f"turn counts: {set(len(r['turns']) for r in rows)}  (must be {{9}})")
    print(f"families: {list(CANDIDATES)}")


def extract():
    model, tok = load_model()
    extract_acts(STIM, config.data_path(ACTS), model=model, tok=tok,
                 per_turn=False)


def main():
    import os
    train = os.environ.get("CALIB_TRAIN", TRAIN)
    a, m = load_acts(config.data_path(train))
    y = np.array([r["label"] for r in m])
    probe = make_probe(config.C)
    probe.fit(a[:, config.LAYER, :], y)
    ts = probe.decision_function(a[:, config.LAYER, :])

    hi_m, lo_m = ts[y == 1].mean(), ts[y == 0].mean()
    hi_95 = np.percentile(ts[y == 1], 95)
    lo_5 = np.percentile(ts[y == 0], 5)
    mid = (hi_m + lo_m) / 2

    ea, meta = load_acts(config.data_path(ACTS))
    s = probe.decision_function(ea[:, config.LAYER, :])

    print(f"model {config.MODEL}   layer {config.LAYER}   probe {train}\n")
    print("=" * 70)
    print("PROBE SCALE")
    print("=" * 70)
    print(f"  low cues   mean {lo_m:+.2f}   5th pct {lo_5:+.2f}")
    print(f"  high cues  mean {hi_m:+.2f}   95th pct {hi_95:+.2f}")
    print(f"  midpoint   {mid:+.2f}   <- filler should land near here")

    print("\n" + "=" * 70)
    print("CANDIDATE FILLER FAMILIES")
    print("=" * 70)
    print(f"  {'family':12} {'mean':>8} {'sd':>6} {'room up':>9} "
          f"{'room down':>10} {'|dist to mid|':>14}")
    best, bestd = None, 1e9
    for fam in CANDIDATES:
        v = s[[i for i, r in enumerate(meta) if r["family"] == fam]]
        up, down = hi_95 - v.mean(), v.mean() - lo_5
        d = abs(v.mean() - mid)
        if d < bestd:
            best, bestd = fam, d
        print(f"  {fam:12} {v.mean():>+8.2f} {v.std():>6.2f} {up:>9.2f} "
              f"{down:>10.2f} {d:>14.2f}")

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    for fam in CANDIDATES:
        v = s[[i for i, r in enumerate(meta) if r["family"] == fam]]
        up, down = hi_95 - v.mean(), v.mean() - lo_5
        if min(up, down) < 1.0:
            side = "ceiling" if up < down else "floor"
            print(f"  {fam:12} REJECT -- against the {side} "
                  f"({min(up, down):.2f} of room)")
        elif abs(v.mean() - mid) < 1.5:
            print(f"  {fam:12} GOOD -- near midpoint, room both ways")
        else:
            print(f"  {fam:12} usable, but off-centre by "
                  f"{abs(v.mean() - mid):.2f}")

    print(f"\n  Use: {best}")
    print("  Copy that family into the FILLERS list in the dynamics script")
    print("  before building any experiment.")
    print()
    print("  If NO family lands near the midpoint, that is itself a finding:")
    print("  the model's default read of an unknown user sits at one end of")
    print("  the scale, and no neutral text will move it to the middle.")
    print("  Report it, and interpret the dynamics with the floor or ceiling")
    print("  stated up front rather than discovered afterwards.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "analyse"
    {"build": build, "extract": extract, "analyse": main}[cmd]()
