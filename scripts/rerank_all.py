"""Keyword top 100, then a cross encoder picks the top 5. Uses no gold answers. Safe to stop and resume."""
from __future__ import annotations

import json
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from knowwhen.paths import DATA, WIKI_INDEX, check_storage, inside_project
from knowwhen.retrieval.wiki import WikiRetriever

MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
POOL, KEEP = 100, 5


def main() -> None:
    check_storage()
    rows, seen = [], set()
    for line in open(DATA / "processed" / "questions.jsonl"):
        x = json.loads(line)
        if x["id"] not in seen:
            seen.add(x["id"])
            rows.append(x)
    out = inside_project(DATA / "processed" / "retrieved_reranked.jsonl")
    done = {json.loads(l)["id"] for l in open(out)} if out.exists() else set()
    todo = [x for x in rows if x["id"] not in done]
    print(len(rows), "questions,", len(todo), "to do", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).to("mps").eval()
    w = WikiRetriever(WIKI_INDEX)
    start = time.time()
    with out.open("a") as f:
        for n, x in enumerate(todo):
            passages = w.search(x["question"], POOL)
            scores = []
            with torch.inference_mode():
                for s in range(0, len(passages), 50):
                    chunk = passages[s : s + 50]
                    enc = tok([x["question"]] * len(chunk), [f"{p.title}. {p.text}" for p in chunk], truncation=True, max_length=256, padding=True, return_tensors="pt").to("mps")
                    scores.append(model(**enc).logits.squeeze(-1).float().cpu().numpy())
            s = np.concatenate(scores) if scores else np.array([])
            top = np.argsort(-s)[:KEEP]
            f.write(json.dumps(dict(id=x["id"], passages=[dict(title=passages[j].title, text=passages[j].text, score=float(s[j])) for j in top], pool=len(passages)), ensure_ascii=False) + "\n")
            if n % 50 == 0:
                f.flush()
                el = time.time() - start
                print(f"{n}/{len(todo)} {(n + 1) / el:.2f} q/s eta {(len(todo) - n) / ((n + 1) / el) / 60:.0f} min", flush=True)
    print("finished", time.time() - start, flush=True)


if __name__ == "__main__":
    main()
