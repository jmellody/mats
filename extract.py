import json, argparse
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import MODEL, DATA, BATCH_SIZE


def load_model(name):
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    return tok, model


def render(tok, text):
    return tok.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
    )


@torch.no_grad()
def extract(tok, model, texts, bs=BATCH_SIZE):
    out = []
    for i in range(0, len(texts), bs):
        prompts = [render(tok, t) for t in texts[i:i + bs]]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        hs = model(**enc, output_hidden_states=True).hidden_states[1:]
        out.append(torch.stack([h[:, -1, :] for h in hs], 1).float().cpu().numpy())
    return np.concatenate(out)


def main(inp, out, model_name):
    rows = [json.loads(l) for l in open(DATA / inp)]
    tok, model = load_model(model_name)
    print(repr(render(tok, "test")[-60:]))
    acts = extract(tok, model, [r["text"] for r in rows])
    np.savez_compressed(
        DATA / out, acts=acts,
        topic=np.array([r["topic"] for r in rows]),
        label=np.array([r["label"] for r in rows]),
        idx=np.array([r["idx"] for r in rows]),
    )
    print(acts.shape, "->", DATA / out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", default="prompts.jsonl")
    p.add_argument("--out", default="acts.npz")
    p.add_argument("--model", default=MODEL)
    a = p.parse_args()
    main(a.inp, a.out, a.model)
