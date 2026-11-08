"""Does a cross encoder lift the answer into the top passages? Validation rows only."""
from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from knowwhen.paths import DATA, WIKI_INDEX
from knowwhen.retrieval.wiki import WikiRetriever
from knowwhen.text import contains_answer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--pool", type=int, default=100)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    final = json.load(open(DATA / "processed" / "final_rows.json"))
    val = [r["id"] for r in final if r["split"] == "val"]
    q = {}
    for line in open(DATA / "processed" / "questions.jsonl"):
        x = json.loads(line)
        q[x["id"]] = x
    random.Random(13).shuffle(val)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(args.device).eval()
    w = WikiRetriever(WIKI_INDEX)
    res, start, pairs = {}, time.time(), 0
    for n, i in enumerate(val[: args.n]):
        passages = w.search(q[i]["question"], args.pool)
        if not passages:
            continue
        scores = []
        with torch.inference_mode():
            for s in range(0, len(passages), 50):
                chunk = passages[s : s + 50]
                enc = tok([q[i]["question"]] * len(chunk), [f"{p.title}. {p.text}" for p in chunk], truncation=True, max_length=256, padding=True, return_tensors="pt").to(args.device)
                scores.append(model(**enc).logits.squeeze(-1).float().cpu().numpy())
        order = np.argsort(-np.concatenate(scores))
        pairs += len(passages)
        hit = [contains_answer(p.text, q[i]["answers"]) for p in passages]
        for name in ("all", q[i]["dataset"]):
            res.setdefault(name, []).append((any(hit[:3]), any(hit[:5]), any(hit[j] for j in order[:3]), any(hit[j] for j in order[:5]), any(hit)))
        if n % 100 == 0:
            print(n, round(pairs / (time.time() - start)), "pairs/s", flush=True)
    for name, v in res.items():
        a = np.mean(np.array(v, float), axis=0).round(3)
        print(name, len(v), dict(bm25_3=a[0], bm25_5=a[1], rerank_3=a[2], rerank_5=a[3], pool=a[4]))


if __name__ == "__main__":
    main()
