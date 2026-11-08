"""Two baselines that need the model but no training: P(Yes) self rating and self consistency.

  --mode yes          one forward pass per question asking whether the model can answer it, read
                      before any answer exists. This is the plain P(Yes) signal from Chen et al.,
                      Query-Level Uncertainty, last token and last layer only, not their weighted
                      Internal Confidence.
  --mode consistency  five sampled closed book answers per non training question.

Both are safe to stop and resume. Scoring happens later, in the analysis.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time

import torch

from knowwhen.models.runner import build_prefix, load_model, encode, split_on_oom
from knowwhen.models.prompts import closed_prompt
from knowwhen.paths import ARTIFACTS, DATA, check_storage, inside_project

ASK = "Can you answer the following question correctly? Respond only with 'Yes' or 'No'.\n\nQuestion: {question}"
SAMPLES, TEMPERATURE = 5, 0.7


@torch.inference_mode()
def p_yes(model, tok, prefix, questions, yes, no):
    enc = tok([ASK.format(question=q) + prefix.tail for q in questions], return_tensors="pt", padding=True,
              add_special_tokens=False).to(model.device)
    ids, part = enc["input_ids"], enc["attention_mask"]
    position = prefix.length + (part.long().cumsum(-1) - 1).clamp(min=0)
    mask = torch.cat([part.new_ones(ids.shape[0], prefix.length), part], dim=1)
    cache = copy.deepcopy(prefix.cache)
    cache.batch_repeat_interleave(ids.shape[0])
    logits = model(input_ids=ids, attention_mask=mask, position_ids=position, past_key_values=cache, use_cache=True, logits_to_keep=1).logits
    pair = logits[:, -1, [yes, no]].float()
    return torch.softmax(pair, dim=-1)[:, 0].tolist()


@torch.inference_mode()
def sample(model, tok, questions):
    enc = encode(tok, [closed_prompt(q) for q in questions], str(model.device))
    out = model.generate(**enc, do_sample=True, temperature=TEMPERATURE, top_p=1.0, top_k=0, num_return_sequences=SAMPLES, max_new_tokens=32)
    text = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return [[t.strip() for t in text[i * SAMPLES:(i + 1) * SAMPLES]] for i in range(len(questions))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="CohereLabs/tiny-aya-global")
    ap.add_argument("--mode", choices=["yes", "consistency"], required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--max-hours", type=float, default=0)
    args = ap.parse_args()
    check_storage()
    torch.manual_seed(0)

    rows = {}
    for line in open(DATA / "processed" / "questions.jsonl"):
        r = json.loads(line)
        rows[r["id"]] = r   # the last copy of a repeated ID, matching the final rows
    rows = list(rows.values())
    if args.mode == "consistency":
        rows = [r for r in rows if r["split"] != "train"]
    if args.limit:
        rows = rows[: args.limit]

    name = args.model.split("/")[-1] + (f"-{args.tag}" if args.tag else "")
    run_dir = inside_project(ARTIFACTS / "runs" / name)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / ("p_yes.jsonl" if args.mode == "yes" else "samples.jsonl")
    done = {json.loads(l)["id"] for l in open(path)} if path.exists() else set()
    # Shuffled, so a time limit leaves a random remainder, then sorted in blocks to reduce padding.
    pending = [r for r in rows if r["id"] not in done]
    random.Random(0).shuffle(pending)
    todo = [r for s in range(0, len(pending), 256) for r in sorted(pending[s : s + 256], key=lambda r: len(r["question"]))]
    print(f"{args.mode}: {len(rows)} questions, {len(todo)} to do", flush=True)
    if not todo:
        return

    model, tok = load_model(args.model, device=args.device, dtype=getattr(torch, args.dtype))
    if args.mode == "yes":
        prefix = build_prefix(model, tok)
        yes = tok.encode("Yes", add_special_tokens=False)[0]
        no = tok.encode("No", add_special_tokens=False)[0]
        run = lambda c: p_yes(model, tok, prefix, [r["question"] for r in c], yes, no)
        key = "p_yes"
    else:
        run = lambda c: sample(model, tok, [r["question"] for r in c])
        key = "samples"

    start, seen = time.time(), 0
    with path.open("a") as f:

        def store(chunk, out):
            nonlocal seen
            for r, o in zip(chunk, out):
                f.write(json.dumps({"id": r["id"], key: o}, ensure_ascii=False) + "\n")
            f.flush()
            seen += len(chunk)

        for s in range(0, len(todo), args.batch):
            if args.max_hours and time.time() - start > args.max_hours * 3600:
                print(f"stopped at time limit with {len(todo) - seen} left", flush=True)
                return
            split_on_oom(todo[s : s + args.batch], run, store, lambda m: print(m, flush=True))
            if (s // args.batch) % 50 == 0:
                print(f"{seen}/{len(todo)} {seen / (time.time() - start):.2f} q/s", flush=True)
    print("finished", time.time() - start, flush=True)


if __name__ == "__main__":
    main()
