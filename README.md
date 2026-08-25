# User-expertise representations in LLMs

Do language models build an internal representation of how much a user knows,
and how does it update as evidence accumulates?

## Findings so far (Qwen2.5-3B-Instruct, layer 27)

| # | Result | Statistic |
|---|--------|-----------|
| 1 | Correct vs incorrect use of a concept is linearly decodable | d = 0.62, p = 0.0006 (paired, n=30) |
| 2 | Positive and negative evidence move the representation in opposite directions | slopes +0.19 / −0.29, both p < 0.0002 |
| 3 | **Valence asymmetry is null** | −0.03, CI [−0.13, +0.06], n=92 |
| 4 | Strong saturation: the first signal does nearly all the work | 1st step +1.21, 3rd step −0.31, p < 0.0001 |
| 5 | Earlier evidence survives contradiction | +0.87, t = 14.3, mirrored in both directions |
| 6 | Primacy slightly outweighs recency | retention/recency = 1.34 |

Plus one methodological result: **probe-based asymmetry claims depend on the
choice of neutral reference condition, which has no principled definition.**
Orthogonalising the probe direction against technical register reversed which
side of neutral the positive and negative conditions fell on, while leaving the
positive-negative contrast untouched. The dose-response design was adopted
specifically to remove the anchor from the analysis.

### What is NOT shown

All of the above is correlational. Nothing here demonstrates the model *uses*
this direction. Steering with behavioural validation is the next step and the
main reason to move to GPU.

## Pipeline

```
make_stimuli.py generate   # minimal pairs via LLM, matched vocab and length
make_stimuli.py filter     # automated cuts: length, vocab overlap, hedging
rate_pairs.py              # LLM-judged centrality, blind, two passes
make_stimuli.py build      # split by concept: probe-train vs experiment

acts.extract(...)          # batched, resumable, CPU or GPU
probe.py                   # grouped CV, per-layer sweep, directions

dose.py                    # experiment 1: slopes, no anchor needed
contradiction.py           # experiment 2: retention under contradiction
figures.py                 # publication figures
```

## Design decisions worth preserving

**Minimal pairs.** Correct and incorrect statements share vocabulary, length,
and structure, differing by one semantic element. Without this the probe
separates classes on keywords rather than expertise.

**Pair-grouped cross-validation.** Both halves of a pair must stay on the same
side of a CV split. A random split gave systematically *below-chance* accuracy:
the probe learned the wording from one half and confidently mispredicted the
near-identical other half.

**Matched conversation scaffolding.** Every conversation, in training and in
every experiment, is read at the same final turn. A 7-turn experiment against a
5-turn training set put the probe far out of distribution -- baseline drifted
9.2 while the effect was 1.06.

**Margins, not probabilities.** `predict_proba` on 2048-dim logistic regression
saturates to 0/1 and destroys the graded signal. Use `decision_function` with
C=0.01.

**Concept-level disjointness.** Probe-training and experiment concepts do not
overlap, so the probe never scores a sibling sentence about an idea it trained on.

## Next steps

1. **Steering** (needs GPU). Add the layer-27 direction to the residual stream,
   generate responses, measure jargon density and unprompted definitions.
   Establishes whether the direction is causal or merely decodable.
2. **Count-matched order test.** `+ −` vs `− +`, one signal each. Isolates order
   from accumulation -- the current retention result confounds the two.
3. **Multi-model.** At least two families and two sizes before treating any of
   this as a property of language models rather than of one checkpoint.
4. **Token-span pooling.** Mean over the signal sentence rather than the final
   token. Probe accuracy is 0.67; dilution across intervening turns is the
   likely cause.

## Structure

Superseded scripts live in `archive/` and in git history. The evolution is in
the commit log, not in the working tree.

```
config.py           single source of truth for MODEL, LAYER, C, device
acts.py             extraction (batched, resumable)
probe.py            probe training and directions
make_stimuli.py     stimulus generation and curation
rate_pairs.py       LLM-judge centrality rating
dose.py             experiment 1
contradiction.py    experiment 2
figures.py          figures
archive/            superseded: run.py, rq1.py, neutral_baseline.py,
                    final_stats.py, resweep.py, fast_extract.py,
                    add_wrappers.py, diagnose.py, robustness.py,
                    orthogonalize.py
```
