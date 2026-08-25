import json, re, argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import TOPICS, LABELS, N_PER_CELL, GEN_MODEL, DATA, SEED

SPEC = {
    "expert": "The asker is clearly an expert: precise terminology used correctly, "
              "assumes background, asks about an edge case or a live debate in the field. "
              "They do NOT state their credentials.",
    "novice": "The asker is clearly a beginner: everyday vocabulary, asks about a basic "
              "concept, may be slightly confused about fundamentals. "
              "They do NOT state that they are a beginner.",
    "neutral": "The asker's expertise is completely unknowable: a plain, short question "
               "with no technical vocabulary, no naive framing, and no self-disclosure. "
               "It should be equally plausible from an expert or a beginner.",
}
BATCH = 10


def prompt(topic_desc, label, n, seen):
    p = (f"Write {n} distinct opening messages a person might send to an AI assistant "
         f"about {topic_desc}.\n\n{SPEC[label]}\n\n"
         "Vary length, phrasing, and subtopic. Never mention age, gender, job, or education.")
    if seen:
        p += "\n\nDo NOT repeat these:\n" + "\n".join(f"- {s}" for s in seen[-15:])
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
    ids = tok.apply_chat_template(
        [{"role": "user", "content": text}],
        return_tensors="pt", add_generation_prompt=True).to(model.device)
    out = model.generate(ids, max_new_tokens=2048, do_sample=True,
                         temperature=temperature, top_p=0.95,
                         pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


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
