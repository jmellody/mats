# What Level of Expertise Do LLMs Assume, and Does It Depend on the Topic?

Probing where an LLM's default reading of user expertise lives in the residual stream — and whether that representation causally shapes its response.

## Motivation

Before a user says anything about themselves, the model still has to situate its answer somewhere. Where? The model's read of who's asking shapes what it says — level of detail, which caveats survive, tone — and that read is a variable affecting model behavior that isn't usually controlled for or reported. This project measures the default assumed expertise of an unknown user, tests whether it varies by topic and question framing, and checks whether the internal direction is causally connected to output behavior via activation steering.

## Summary of Findings

1. **The default is novice-leaning in every condition tested** — 0.13 to 0.51 on a scale where 0 is a stated beginner and 1 a stated expert, never approaching the expert end.
2. **Question framing matters more than topic** — whether the question presupposes prior engagement ("my X isn't working right") vs. none ("I'm about to get into X") shifts the model's estimate by 0.189, further than switching topics entirely.
3. **Topic matters, but modestly** — between-topic SD is 0.041 (pre-engagement) and 0.059 (post-engagement). Python and ML read as most expert-assumed; sourdough and chemotherapy as least.
4. **The direction steers** — adding the expertise direction at a single layer, with no expertise stated in the prompt, moves four of five behavioral features monotonically in the expected direction.

## Experiments

### Experiment 1 (Probe): `probe2.py`

The core experiment. Each of 10 topics × 2 framing families × 5 templates produces a fixed question stem. The stem appears in three variants: bare (neutral), plus a stated-novice suffix ("I'm completely new to this"), plus a stated-expert suffix ("I've worked with this for years"). Everything before the suffix is identical; the suffix says "this" rather than naming the domain, so it is constant across topics. A linear direction is built from the anchored prompts (expert minus novice mean activations) at the best layer (19, selected by held-out AUC on disjoint paraphrase sets A/B). Neutral prompts — containing no suffix — are projected onto this direction and rescaled per topic and family so 0 = novice anchor and 1 = expert anchor.

Construct validity checks include variance decomposition, paraphrase-set stability (building on set A, scoring on set B), a bag-of-words control on the neutral prompts, and a suffix-only control direction.

### Experiment 2 (Behavioral): `behavior.py`

Same stimuli, but measuring generated responses rather than activations. The model produces full responses to anchored and neutral prompts, and six surface features are extracted: long-word fraction, mean sentence length, hedging rate, inline definitions, simplification language, and bullet usage. Three features (long_word_frac, mean_sent_len, defines) clear 90% sign consistency across the 20 cells and form a composite. The novice-leaning default and framing effect both replicate, and topic ordering converges with Experiment 1 under post-engagement framing (ρ = 0.76).

### Experiment 3 (Steering): `steer.py`

Adds the expertise direction to the residual stream at layer 19 during generation, with no expertise stated in the prompt. The direction is scaled relative to the mean residual norm. At α in [−0.8, +0.8], generation stays fluent and four of five features shift monotonically: pushing toward expert produces longer words, more hedging, and more inline definitions; pushing toward novice produces more analogy and reassurance. Mean sentence length is non-monotonic. An optional `--project-out` flag removes the suffix-only component of the direction to confirm the effect isn't driven by the literal anchor wording.

### Earlier attempt: `generate.py` + `extract.py`

An initial approach generated naturalistic expert-sounding and novice-sounding questions (via Llama-3.1-8B-Instruct) where expertise was encoded in *what* was asked, not in self-description. This failed — the probe separated held-out examples at AUC ~1 by layer 0, and a TF-IDF logistic regression matched it. The direction was picking up lexical confounds (vocabulary, question structure) rather than an expertise representation. This led to the redesign in `probe2.py`, which holds the prompt text constant and varies only an appended self-description. The `generate.py` and `extract.py` scripts remain in the repo as a record of the first attempt.

## Repo Structure

```
├── config.py           # Central config: models, topics, labels, paths
├── generate.py         # Prompt generation for initial (failed) approach
├── extract.py          # Activation extraction for initial approach
├── probe2.py           # Main experiment: anchored probe with validity checks
├── behavior.py         # Behavioral feature extraction from generated responses
├── steer.py            # Activation steering experiments
├── analyze.py          # Analysis and visualization
├── preflight.py        # Sanity checks
├── explore.ipynb       # Exploratory analysis notebook
├── setup_vast.sh       # GPU instance setup (Vast.ai)
├── requirements.txt    # Python dependencies
├── results/            # Experiment outputs (.npz, .jsonl)
└── data/               # Generated prompts and activations (not committed)
```

## Topics

Ten topics chosen to span variation in technicality, stakes, and demographic association:

| Key | Topic |
|-----|-------|
| `sourdough` | Sourdough bread baking |
| `chemo` | Chemotherapy treatment decisions |
| `python` | Python programming |
| `mortgage` | Mortgage refinancing |
| `carrepair` | Car engine repair |
| `musictheory` | Music theory and composition |
| `taxlaw` | Personal income tax law |
| `houseplants` | Houseplant care |
| `ml` | Machine learning research |
| `fitness` | Strength training programming |

## Models

- **Subject model:** Qwen/Qwen3.5-9B — all probing, behavioral generation, and steering runs against this model.
- **Generator model:** meta-llama/Llama-3.1-8B-Instruct — used only in the initial prompt generation approach (`generate.py`), deliberately different from the subject model to prevent circularity.

## Setup

**Hardware:** Requires a GPU with ≥24 GB VRAM (e.g., A100, 4090). The `setup_vast.sh` script configures a [Vast.ai](https://vast.ai) instance.

**Install:**

```bash
pip install -r requirements.txt
huggingface-cli login
```

## Running the Experiments

```bash
# Experiment 1: Anchored probe with construct validity analysis
python probe2.py

# Experiment 2: Behavioral measurement
python behavior.py

# Experiment 3: Activation steering (layer 19 is the probe's peak)
python steer.py --layer 19
```

Each script accepts `--help` for full argument details. Key defaults (model, topics, batch size) are set in `config.py`.

To rerun the initial (failed) approach for reference:

```bash
python generate.py
python extract.py
```

## Design Decisions

- **Fixed stems, varied suffixes.** The successful probe holds question text constant and appends a self-description suffix, so any topic variation in neutral-prompt positioning must come from the question content, not from lexical differences between anchors.
- **Disjoint paraphrase sets.** Anchor suffixes are split into sets A and B. The direction is built on one and tested on the other, so probe accuracy isn't inflated by lexical overlap.
- **Presupposition families.** Templates come in "pre" (prospective: "I'm about to…") and "post" (retrospective: "I tried…") variants. Splitting on presupposition cut within-topic variance roughly sevenfold and revealed a domain effect that was undetectable when framings were pooled.
- **Single-layer steering.** Compounding the intervention across layers destroyed generation; restricting to the peak probe layer opened a wide stable window (α ∈ [−0.8, +0.8]).
- **Suffix projection.** `steer.py --project-out` removes the component of the expertise direction that aligns with bare suffix phrases, isolating the content-driven signal.

## Limitations

1. One model, one attribute, ten topics. No cross-model replication.
2. Synthetic stimuli — real conversations don't come in template families.
3. Pre-engagement anchors are strained ("where do I begin?" + "I've worked with this for years" is somewhat incoherent), so the framing effect of 0.189 is likely an upper bound.
4. The behavioral readout is crude — three of six features worked, and they disagree on absolute position.
5. Steering was lightly sampled (three α values, window found by hand).

## Citation

```
@misc{mellody2026expertise,
  author = {Mellody, James},
  title  = {What Level of Expertise Do LLMs Assume, and Does It Depend on the Topic?},
  year   = {2026},
  url    = {https://github.com/jmellody/mats}
}
```

## License

MIT
