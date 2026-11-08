"""Checks on the real model that the cached preamble leaves answers unchanged."""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from knowwhen.models.prompts import closed_prompt
from knowwhen.models.runner import build_prefix, generate_batch, load_model, select_layers
from knowwhen.paths import DATA


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="CohereLabs/aya-expanse-8b")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--n", type=int, default=32)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(DATA / "processed" / "questions.jsonl")]
    rows = rows[:: len(rows) // args.n][: args.n]   # spread over the file, which is grouped by dataset
    model, tok = load_model(args.model, device=args.device, dtype=getattr(torch, args.dtype))
    layers = select_layers(model.config.num_hidden_layers + 1)
    prompts = [closed_prompt(r["question"]) for r in rows]
    prefix = build_prefix(model, tok)
    same, gap = 0, 0.0
    for s in range(0, len(prompts), 8):
        a = generate_batch(model, tok, prompts[s : s + 8], layers)
        b = generate_batch(model, tok, prompts[s : s + 8], layers, prefix=prefix)
        same += sum(x == y for x, y in zip(a.answers, b.answers))
        gap = max(gap, float(np.abs(a.pre - b.pre).max() / (np.abs(a.pre).max() + 1e-9)))
    print(f"preamble length {prefix.length}; identical answers {same}/{len(prompts)}; largest relative state gap {gap:.4f}", flush=True)
    if same < 0.9 * len(prompts):
        raise SystemExit("preamble cache changes too many answers on this model; do not run with it")


if __name__ == "__main__":
    main()
