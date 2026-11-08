"""Isolates what makes generation slow on MPS: dtype, score capture, hidden state capture, batch size."""
import json
import time

import torch

from knowwhen.models.prompts import closed_prompt
from knowwhen.models.runner import encode, load_model
from knowwhen.paths import DATA

rows = [json.loads(l) for l in open(DATA / "processed" / "questions.jsonl")][:32]


def run(model, tok, batch, **flags):
    enc = encode(tok, [closed_prompt(r["question"]) for r in rows[:batch]], "mps")
    torch.mps.synchronize()
    t = time.time()
    with torch.inference_mode():
        model.generate(**enc, max_new_tokens=12, min_new_tokens=12, do_sample=False, temperature=None, top_p=None, top_k=None,
                       pad_token_id=tok.pad_token_id, return_dict_in_generate=True, **flags)
    torch.mps.synchronize()
    return time.time() - t


for dtype in (torch.bfloat16, torch.float16):
    t = time.time()
    model, tok = load_model("CohereLabs/tiny-aya-global", dtype=dtype)
    print(dtype, "load", round(time.time() - t, 1), "cache", model.config.use_cache, flush=True)
    run(model, tok, 1)  # warm up kernels and page in weights
    for batch in (1, 8, 16):
        for name, flags in (("plain", {}), ("scores", dict(output_scores=True)), ("hidden", dict(output_hidden_states=True)),
                            ("both", dict(output_scores=True, output_hidden_states=True))):
            s = run(model, tok, batch, **flags)
            print(f"{dtype} batch={batch} {name}: {s:.2f}s, {s / 12:.3f}s per step, {batch / s:.2f} q/s at 12 tokens", flush=True)
    del model
    torch.mps.empty_cache()
