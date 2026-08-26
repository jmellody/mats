import json, re, argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import TOPICS, LABELS, N_PER_CELL, GEN_MODEL, DATA, SEED

SPEC = {
    "expert": (
        "The asker knows this subject deeply, but writes in PLAIN EVERYDAY LANGUAGE. "
        "Their expertise shows ONLY in what the question takes for granted and how "
        "specific the problem is: they ask about a trade-off, an interaction between "
        "two factors, a case where the usual advice breaks down, or something they "
        "have already tried that did not work. They never state credentials."
    ),
    "novice": (
        "The asker is new to this subject and writes in PLAIN EVERYDAY LANGUAGE. "
        "Their inexperience shows ONLY in what the question asks about: a basic "
        "definition, a first step, or a general 'how do I start' framing. "
        "They never say they are a beginner."
    ),
    "neutral": (
        "A plain question about the subject that gives no signal either way. "
        "Equally plausible from someone who has done this for years or someone "
        "starting today."
    ),
}

RULES = (
    "HARD RULES, apply to every message:\n"
    "- 18 to 28 words. Count them.\n"
    "- NO technical terms, NO jargon, NO field-specific vocabulary of any kind. "
    "Use words a 12-year-old would know.\n"
    "- Same casual register throughout: lowercase starts and contractions are fine, "
    "and should appear about equally often in all messages.\n"
    "- No self-description: no age, job, education, experience level, or credentials.\n"
    "- The ONLY difference between an expert and a novice message is WHICH QUESTION "
    "is being asked, never HOW it is worded.\n"
)

EXAMPLES = """Worked examples of the contrast (note: same words, different question):

sourdough
  expert:  "when my kitchen gets colder, is it better to feed the starter less often or use less flour each time?"
  novice:  "how often am i supposed to feed my starter? mine has been sitting out for two days."

python
  expert:  "if two parts of my program change the same list at the same time, what do people normally do about that?"
  novice:  "how do i add something to the end of a list? i keep getting an error."
"""


BATCH = 5


def prompt(topic_desc, label, n, seen):
    p = (f"Write {n} distinct opening messages a person might send to an AI assistant "
         f"about {topic_desc}.\n\n{SPEC[label]}\n\n{RULES}\n{EXAMPLES}")
    if seen:
        p += "\nDo NOT repeat these:\n" + "\n".join(f"- {s}" for s in seen)
    return p + "\n\nReturn ONLY a JSON array of strings."



def parse(txt):
    txt = re.sub(r"^```(?:json)?|```$", "", txt.strip(), flags=re.M)
    m = re.search(r"\[.*\]", txt, re.S)
    if not m:
        return []
    try:
        return [s.strip() for s in json.loads(m.group()) if isinstance(s, str) and s.strip()]
    except Exception:
        return []


@torch.no_grad()
def run(tok, model, text, temperature):
    enc = tok.apply_chat_template(
        [{"role": "user", "content": text}],
        return_tensors="pt", add_generation_prompt=True, return_dict=True).to(model.device)
    n = enc["input_ids"].shape[1]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    out = model.generate(**enc, max_new_tokens=4096, do_sample=True,
                         temperature=temperature, top_p=0.95,
                         pad_token_id=pad)
    return tok.decode(out[0, n:], skip_special_tokens=True)


def main(out, model_name, temperature):
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda").eval()

    rows = []
    for topic in TOPICS:
        for label in LABELS:
            seen, calls = [], 0
            while len(seen) < N_PER_CELL and calls < 12:
                calls += 1
                seen += [s for s in parse(run(tok, model, prompt(TOPICS[topic], label, BATCH, seen), temperature))
                         if s not in seen]
            seen = seen[:N_PER_CELL]
            rows += [{"topic": topic, "label": label, "idx": i, "text": t}
                     for i, t in enumerate(seen)]
            print(f"{topic:14s} {label:8s} {len(seen):3d}  ({calls} calls)")

    DATA.mkdir(exist_ok=True)
    with open(DATA / out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} rows -> {DATA / out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="prompts.jsonl")
    p.add_argument("--model", default=GEN_MODEL)
    p.add_argument("--temperature", type=float, default=1.0)
    a = p.parse_args()
    main(a.out, a.model, a.temperature)
