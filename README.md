# Default assumed user expertise

Does an LLM's prior about who it's talking to depend on the topic?

Before a user says anything about themselves, the model already has a default
picture of them. This measures where that default sits — per topic, calibrated
against expert and novice anchors from the same topic.

## Setup

```bash
git clone <repo> && cd userprior
bash setup_vast.sh
hf auth login                  # gated repos
```

## Run

```bash
python generate.py --model <generator-id>   # -> data/prompts.jsonl
python extract.py --model <hf-model-id>     # -> data/acts.npz
jupyter lab explore.ipynb
```

Swap `--model` / `--out` to compare model families:

```bash
python extract.py --model Qwen/Qwen2.5-7B-Instruct --out acts_qwen.npz
```

## Design

Every prompt is a **single user turn**, read at the last token with
`add_generation_prompt=True`. Three cells per topic:

| cell | what it is |
|---|---|
| `expert` | expertise evident from how they ask, never stated |
| `novice` | beginner evident from how they ask, never stated |
| `neutral` | no expertise signal at all |

No assistant turns anywhere, so assistant register can't confound the readout,
and neutral prompts sit at exactly the same read position as the anchors.

Per topic: direction = mean(expert) − mean(novice), fit on half the prompts,
AUC on the held-out half. Neutral prompts are scored and rescaled so 0 = novice
anchor, 1 = expert anchor. That rescaling is what makes topics comparable —
raw projections across different directions are not.

## Checks

- **AUC gate** — drop topics below 0.75. A default measured on a
  non-discriminating direction is noise.
- **Permutation test** — shuffle topic labels on neutral prompts for a null on
  between-topic variance.
- **Read position** — `extract.py` prints the prompt tail; confirm it ends in an
  open assistant header.
- **Just ask** — prompt the model directly for the asker's expertise and compare
  to the probe. If asking works as well, the probe isn't earning its keep.
- **Second model** — does the topic ordering replicate?

## Layout

```
config.py         model, topics, constants
generate.py       prompt generation, local HF model
extract.py        activations -> data/acts.npz
analyze.py        directions, AUC, calibration
explore.ipynb     layer sweep, main figure, permutation test
```

## Next

- Behavioral: does response register track the readout across topics?
- Within-topic spread over paraphrases = confidence of the prior
- Update dynamics: evidence confirming vs. contradicting the default

## Related work

- Chen et al., *Designing a Dashboard for Transparency and Control of
  Conversational AI* (arXiv:2406.07882) — user attributes are linearly
  represented; probes best at reading were not best for steering
- *Which Institutional Frameworks Do Chatbots Assume?* (arXiv:2606.00333) —
  same audit-the-default move, applied to jurisdiction
- *ExPerT* (arXiv:2607.01242) — user expertise varies per query and is hard to
  infer from query text alone
