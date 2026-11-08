"""Discovers which heads and MLPs carry a passage answer to the output, with CircuitKIT EAP-IG."""
from __future__ import annotations

import argparse
import json

import pandas as pd

import circuitkit as ck  # noqa: F401, must come before transformer_lens
from circuitkit import Pipeline

from knowwhen.paths import ROOT, inside_project


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="CohereLabs/tiny-aya-global")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--sparsity", type=float, default=0.9)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ig-steps", type=int, default=3)
    ap.add_argument("--scope", default="both", choices=["both", "mlp", "heads"])
    ap.add_argument("--precision", default="bfloat16")
    ap.add_argument("--evaluate", type=int, default=0, help="examples for the faithfulness check, 0 to skip")
    ap.add_argument("--tag", default="tiny")
    args = ap.parse_args()
    out = inside_project(ROOT / "experiments" / "circuit" / f"{args.tag}_seed{args.seed}")
    out.mkdir(parents=True, exist_ok=True)
    # The custom CSV path attributes over every row, so the subset is cut here. Long prompts are
    # left out because TransformerLens keeps a per head result that overflows a T4 beyond about 100 tokens.
    rows = pd.read_csv(ROOT / "experiments" / "circuit" / "grounding_pairs.csv")
    rows = rows[rows.clean_prompt.str.len() < 380].sample(frac=1, random_state=args.seed).head(args.n)
    subset = out / "pairs.csv"
    rows.to_csv(subset, index=False)
    pipe = Pipeline.from_custom_data(args.model, subset, clean_prompt="{clean_prompt}", clean_answer="{clean_answer}",
                                     corrupt_prompt="{corrupt_prompt}", corrupt_answer="{corrupt_answer}", task_name="passage_grounding",
                                     precision=args.precision, device=args.device, output_dir=str(out))
    # Attribution only needs gradients to reach the activations. With every weight trainable, the
    # backward pass also builds weight gradients, which do not fit on a T4. Node scores are identical
    # with only the embedding left trainable (checked on GPT-2).
    model = pipe._ensure_model()
    model.requires_grad_(False)
    model.embed.W_E.requires_grad_(True)
    pipe.discover(algorithm="eap-ig", level="node", sparsity=args.sparsity, n_examples=args.n, batch_size=args.batch, scope=args.scope, seed=args.seed, ig_steps=args.ig_steps)
    print("discovered", type(pipe._circuit), flush=True)
    print(pipe._circuit, flush=True)
    if args.evaluate:
        pipe.evaluate(n_examples=args.evaluate)
        print("evaluation", getattr(pipe, "_evaluation", None), flush=True)
    (out / "done.json").write_text(json.dumps(vars(args)))


if __name__ == "__main__":
    main()
