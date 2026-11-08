"""How often the gold answer is in the top k passages, on validation rows only. Sets the ceiling for reranking."""
from __future__ import annotations

import json
import random

import numpy as np

from knowwhen.paths import DATA, WIKI_INDEX
from knowwhen.retrieval.wiki import WikiRetriever
from knowwhen.text import contains_answer

KS = [3, 5, 10, 20, 50, 100]


def main() -> None:
    final = json.load(open(DATA / "processed" / "final_rows.json"))
    val = [r["id"] for r in final if r["split"] == "val"]
    q = {}
    for line in open(DATA / "processed" / "questions.jsonl"):
        x = json.loads(line)
        q[x["id"]] = x
    random.Random(13).shuffle(val)
    ids = val[:800]
    w = WikiRetriever(WIKI_INDEX)
    hits = {}
    for n, i in enumerate(ids):
        passages = w.search(q[i]["question"], max(KS))
        first = next((j for j, p in enumerate(passages) if contains_answer(p.text, q[i]["answers"])), None)
        hits.setdefault(q[i]["dataset"], []).append(first)
        hits.setdefault("all", []).append(first)
        if n % 200 == 0:
            print(n, flush=True)
    for name, f in hits.items():
        print(name, len(f), {k: round(float(np.mean([x is not None and x < k for x in f])), 3) for k in KS})


if __name__ == "__main__":
    main()
