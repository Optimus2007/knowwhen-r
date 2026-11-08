"""Run BM25 once for every question and save the passages, so retrieval is never repeated."""
from __future__ import annotations

import json
import multiprocessing as mp
import time

from knowwhen.paths import DATA, WIKI_INDEX, check_storage
from knowwhen.retrieval.wiki import WikiRetriever
from knowwhen.text import contains_answer

K = 5
_r = None


def init():
    global _r
    _r = WikiRetriever(WIKI_INDEX)


def work(row):
    t = time.time()
    hits = _r.search(row["question"], k=K)
    return dict(id=row["id"], seconds=time.time() - t, passages=[dict(title=h.title, text=h.text, score=h.score) for h in hits],
                answer_in_top3=any(contains_answer(h.text, row["answers"]) for h in hits[:3]))


def main():
    check_storage()
    rows = [json.loads(l) for l in open(DATA / "processed" / "questions.jsonl")]
    out = DATA / "processed" / "retrieved.jsonl"
    done = {json.loads(l)["id"] for l in open(out)} if out.exists() else set()
    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(todo)} to retrieve", flush=True)
    with mp.Pool(4, initializer=init) as pool, out.open("a") as f:
        for i, res in enumerate(pool.imap(work, todo, chunksize=8)):
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            if i % 500 == 0:
                f.flush()
                print(i, round(res["seconds"], 2), flush=True)


if __name__ == "__main__":
    main()
