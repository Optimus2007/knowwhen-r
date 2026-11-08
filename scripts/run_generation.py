"""Generate answers and cache activations. Safe to stop and resume."""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from knowwhen.models.prompts import closed_prompt, rag_prompt
from knowwhen.models.runner import build_prefix, generate_batch, load_model, select_layers, split_on_oom
from knowwhen.paths import ARTIFACTS, DATA, ENV, check_storage, inside_project
from knowwhen.text import contains_answer, exact_match, is_refusal


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="CohereLabs/tiny-aya-global")
    ap.add_argument("--condition", choices=["closed", "rag"], required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="first N questions only, for timing tests")
    ap.add_argument("--passages", type=int, default=3)
    ap.add_argument("--retrieved", default="retrieved.jsonl", help="passage file in data/processed")
    ap.add_argument("--tag", default="", help="separate output name, for tests")
    ap.add_argument("--device", default="mps", help="mps, cuda, or auto to split across GPUs")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--model-path", default="", help="load weights from this folder; --model still names the outputs")
    ap.add_argument("--questions", default="questions.jsonl", help="question file in data/processed")
    ap.add_argument("--max-hours", type=float, default=0, help="stop cleanly after this long, so a limited session can resume later")
    args = ap.parse_args()
    check_storage()
    torch.manual_seed(0)

    rows = [json.loads(l) for l in open(DATA / "processed" / args.questions)]
    if args.limit:
        rows = rows[: args.limit]
    index = {r["id"]: i for i, r in enumerate(rows)}
    passages = {}
    if args.condition == "rag":
        for line in open(DATA / "processed" / args.retrieved):
            r = json.loads(line)
            passages[r["id"]] = r["passages"][: args.passages]

    name = args.model.split("/")[-1] + (f"-{args.tag}" if args.tag else "")
    run_dir = inside_project(ARTIFACTS / "runs" / name)
    act_dir = inside_project(ENV / "acts" / name)
    run_dir.mkdir(parents=True, exist_ok=True)
    act_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / f"{args.condition}.jsonl"
    done = {json.loads(l)["id"] for l in open(out_path)} if out_path.exists() else set()
    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(rows)} questions, {len(todo)} to do", flush=True)
    if not todo:
        return

    model, tok = load_model(args.model_path or args.model, device=args.device, dtype=getattr(torch, args.dtype))
    prefix = build_prefix(model, tok)
    n_states = model.config.num_hidden_layers + 1
    layers = select_layers(n_states)
    hidden = model.config.hidden_size
    maps = {}
    for key in ("pre", "last", "mean"):
        path = act_dir / f"{args.condition}_{key}.npy"
        mode = "r+" if path.exists() else "w+"
        maps[key] = np.lib.format.open_memmap(path, mode=mode, dtype=np.float16, shape=(len(rows), len(layers), hidden))
    (act_dir / "meta.json").write_text(json.dumps(dict(model=args.model, layers=layers, hidden=hidden, ids=[r["id"] for r in rows])))

    def prompt(r):
        return closed_prompt(r["question"]) if args.condition == "closed" else rag_prompt(r["question"], passages[r["id"]])

    # Sort in large blocks by length to reduce padding, keeping the overall order mixed.
    block = 256
    ordered = []
    for s in range(0, len(todo), block):
        ordered += sorted(todo[s : s + block], key=lambda r: len(prompt(r)))

    start, n_tok, seen = time.time(), 0, 0
    with out_path.open("a") as f:

        def store(chunk, out):
            nonlocal n_tok, seen
            # float16 can overflow where bfloat16 does not. Stop rather than store bad activations.
            if not (np.isfinite(out.pre).all() and np.isfinite(out.mean).all()):
                raise SystemExit(f"non finite activations at batch starting {chunk[0]['id']}")
            for i, r in enumerate(chunk):
                j = index[r["id"]]
                maps["pre"][j], maps["last"][j], maps["mean"][j] = out.pre[i], out.last[i], out.mean[i]
                a = out.answers[i]
                f.write(json.dumps(dict(id=r["id"], answer=a, correct=contains_answer(a, r["answers"]), em=exact_match(a, r["answers"]),
                                        refusal=is_refusal(a), **out.stats[i]), ensure_ascii=False) + "\n")
                n_tok += out.stats[i]["n_tokens"]
            seen += len(chunk)
            for m in maps.values():
                m.flush()
            f.flush()

        for s in range(0, len(ordered), args.batch):
            chunk = ordered[s : s + args.batch]
            if args.max_hours and time.time() - start > args.max_hours * 3600:
                print(f"stopped at time limit with {len(ordered) - seen} left", flush=True)
                return
            split_on_oom(chunk, lambda c: generate_batch(model, tok, [prompt(r) for r in c], layers, prefix=prefix), store,
                         lambda m: print(m, flush=True))
            if (s // args.batch) % 10 == 0:
                el = time.time() - start
                print(f"{seen}/{len(ordered)} {seen / el:.2f} q/s {n_tok / el:.1f} tok/s eta {(len(ordered) - seen) / (seen / el) / 60:.0f} min", flush=True)
    print("finished", time.time() - start, flush=True)


if __name__ == "__main__":
    main()
