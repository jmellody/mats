"""Extract last-token residual stream activations from a chat model."""

import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"


def load_model(name=DEFAULT_MODEL, dtype=torch.float32, device="cpu"):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=dtype, low_cpu_mem_usage=True
    )
    model.to(device)
    model.eval()
    return model, tok


def format_conversation(turns, tok, elicitation=None):
    """turns: list of {"role": "user"|"assistant", "content": str}

    elicitation: optional string appended after the conversation to push the
    model to consolidate its read of the user. Leaving this None is the
    cleaner condition -- run both and compare.
    """
    text = tok.apply_chat_template(turns, tokenize=False, add_generation_prompt=True)
    if elicitation:
        text = text + elicitation
    return text


@torch.no_grad()
def get_activations(model, tok, text, device="cpu"):
    """Returns [n_layers + 1, d_model] float32 numpy array.

    Index 0 is the embedding output; index i is the residual stream after
    transformer block i. Taken at the final token position.
    """
    inputs = tok(text, return_tensors="pt").to(device)
    out = model(**inputs, output_hidden_states=True)
    acts = torch.stack([h[0, -1, :] for h in out.hidden_states])
    return acts.float().cpu().numpy()


def turn_prefixes(turns):
    """Yield (n_turns_included, prefix) for cumulative prefixes of a conversation.

    Cuts after each user message, so each prefix ends where the model would be
    about to respond. That's the point where its read of the user matters.
    """
    for i, t in enumerate(turns):
        if t["role"] == "user":
            yield i + 1, turns[: i + 1]


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_dataset(path, out_path, model=None, tok=None, elicitation=None,
                    device="cpu", per_turn=False, model_name=DEFAULT_MODEL):
    """Extract activations for every conversation in a jsonl file.

    Each line needs: {"id": str, "label": int, "turns": [...], ...}
    Any extra fields are carried through into the metadata.

    per_turn=False -> one activation per conversation (full conversation)
    per_turn=True  -> one activation per user turn prefix (for trajectories)
    """
    if model is None:
        model, tok = load_model(model_name, device=device)

    rows = load_jsonl(path)
    acts, meta = [], []

    for r in rows:
        if per_turn:
            items = list(turn_prefixes(r["turns"]))
        else:
            items = [(len(r["turns"]), r["turns"])]
        for n, prefix in items:
            text = format_conversation(prefix, tok, elicitation)
            acts.append(get_activations(model, tok, text, device))
            m = {k: v for k, v in r.items() if k != "turns"}
            m["n_turns"] = n
            meta.append(m)
        print(f"  done {r['id']}", flush=True)

    acts = np.stack(acts)
    np.savez_compressed(out_path, acts=acts, meta=json.dumps(meta))
    print(f"saved {acts.shape} -> {out_path}")
    return acts, meta


def load_extracted(path):
    d = np.load(path, allow_pickle=False)
    return d["acts"], json.loads(str(d["meta"]))
