"""Run this first. Nothing downstream is meaningful if these checks fail."""
import argparse
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from config import MODEL

EXPERT = "For a hybrid varietal starter, is the acetic/lactic ratio driving my crumb structure or is that mostly hydration?"
NOVICE = "my bread came out really flat and dense. what did i do wrong?"


def render(tok, text, thinking=None):
    kw = {} if thinking is None else {"enable_thinking": thinking}
    try:
        return tok.apply_chat_template([{"role": "user", "content": text}],
                                       tokenize=False, add_generation_prompt=True, **kw)
    except TypeError:
        return tok.apply_chat_template([{"role": "user", "content": text}],
                                       tokenize=False, add_generation_prompt=True)


def main(name, thinking):
    cfg = AutoConfig.from_pretrained(name, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)

    print("== config")
    for k in ["model_type", "num_hidden_layers", "hidden_size", "architectures"]:
        print(f"  {k}: {getattr(cfg, k, getattr(getattr(cfg, 'text_config', cfg), k, '?'))}")

    print("\n== read position")
    for t in ([None] if thinking is None else [True, False]):
        r = render(tok, "test", t)
        print(f"  enable_thinking={t}: ...{r[-70:]!r}")

    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True).eval()

    print("\n== hidden states")
    ids = tok(render(tok, "test", thinking), return_tensors="pt", add_special_tokens=False).to("cuda")
    with torch.no_grad():
        hs = model(**ids, output_hidden_states=True).hidden_states
    print(f"  {len(hs) - 1} layers, d_model={hs[-1].shape[-1]}")

    print("\n== does it adapt? (the check that decides whether there is anything to probe)")
    for lbl, q in [("EXPERT", EXPERT), ("NOVICE", NOVICE)]:
        i = tok(render(tok, q, thinking), return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.no_grad():
            o = model.generate(**i, max_new_tokens=180, do_sample=False,
                               pad_token_id=tok.pad_token_id or tok.eos_token_id)
        print(f"\n--- {lbl}\n{tok.decode(o[0, i['input_ids'].shape[1]:], skip_special_tokens=True)}")

    print("\nIf both answers read the same, this model has no user model to probe. Go bigger.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL)
    p.add_argument("--thinking", type=lambda s: s.lower() == "true", default=None)
    a = p.parse_args()
    main(a.model, a.thinking)
