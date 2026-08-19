"""Activation extraction. Batched, resumable, CPU or GPU.

    from acts import extract
    extract("data/dose.jsonl", "data/dose_acts.npz")

Replaces extract.py and fast_extract.py. Differences that matter for scaling:

  - BATCHED. On GPU this is the whole speedup. Requires left-padding so that
    position -1 is the true final token for every sequence in the batch;
    right-padding would silently read pad tokens and produce garbage that
    looks plausible.
  - RESUMABLE. Writes a shard file every N conversations. Killing a run costs
    at most N items, not the whole run.
  - DEVICE-AGNOSTIC. bf16 on GPU, fp32 on CPU, set in config.
"""

import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config


def load_model(name=None):
    name = name or config.MODEL
    tok = AutoTokenizer.from_pretrained(name)
    # left-padding is required for batched last-token reads
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=config.dtype(), low_cpu_mem_usage=True)
    model.to(config.device()).eval()
    return model, tok


def format_conversation(turns, tok):
    return tok.apply_chat_template(turns, tokenize=False,
                                   add_generation_prompt=True)


@torch.no_grad()
def batch_activations(model, tok, texts):
    """[batch, n_layers+1, d_model] at the final token of each sequence."""
    enc = tok(texts, return_tensors="pt", padding=True).to(config.device())
    out = model(**enc, output_hidden_states=True)
    # left-padded, so -1 is the real final token for every row
    acts = torch.stack([h[:, -1, :] for h in out.hidden_states], dim=1)
    return acts.float().cpu().numpy()


def turn_prefixes(turns):
    for i, t in enumerate(turns):
        if t["role"] == "user":
            yield i + 1, turns[: i + 1]


def _shard(out):
    return out.replace(".npz", "_partial.jsonl")


def _load_shard(path):
    done = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done[r["_key"]] = r
    return done


def extract(inp, out, per_turn=False, save_every=32, model=None, tok=None):
    with open(inp) as f:
        rows = [json.loads(l) for l in f if l.strip()]

    items = []
    for r in rows:
        spans = list(turn_prefixes(r["turns"])) if per_turn else \
            [(len(r["turns"]), r["turns"])]
        for n, prefix in spans:
            meta = {k: v for k, v in r.items() if k != "turns"}
            meta["n_turns"] = n
            meta["_key"] = f"{r['id']}@{n}"
            items.append((meta, prefix))

    shard = _shard(out)
    done = _load_shard(shard)
    todo = [(m, p) for m, p in items if m["_key"] not in done]
    print(f"{config.device()} | {len(done)} cached, {len(todo)} to extract")

    if todo:
        if model is None:
            model, tok = load_model()
        bs = config.batch_size()
        buf = []
        with open(shard, "a") as f:
            for i in range(0, len(todo), bs):
                chunk = todo[i:i + bs]
                texts = [format_conversation(p, tok) for _, p in chunk]
                acts = batch_activations(model, tok, texts)
                for (meta, _), a in zip(chunk, acts):
                    rec = dict(meta)
                    rec["acts"] = a.astype(np.float32).tolist()
                    buf.append(json.dumps(rec))
                if len(buf) >= save_every or i + bs >= len(todo):
                    f.write("\n".join(buf) + "\n")
                    f.flush()
                    buf = []
                    print(f"  {min(i + bs, len(todo))}/{len(todo)}", flush=True)
        done = _load_shard(shard)

    keys = [m["_key"] for m, _ in items]
    recs = [dict(done[k]) for k in keys if k in done]
    acts = np.stack([np.array(r.pop("acts"), dtype=np.float32) for r in recs])
    np.savez_compressed(out, acts=acts, meta=json.dumps(recs))
    print(f"saved {acts.shape} -> {out}")
    return acts, recs


def load(path):
    d = np.load(path, allow_pickle=False)
    return d["acts"], json.loads(str(d["meta"]))
