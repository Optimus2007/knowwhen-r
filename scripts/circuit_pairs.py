"""Builds counterfactual passage pairs for circuit discovery, from training questions only.

Each pair uses a question that Tiny Aya got wrong closed book and right with passages, so the answer
has to come from the text. The corrupted side is the same passage with the answer replaced by
another question's answer of equal token length.
"""
from __future__ import annotations

import csv
import json
import random
import re

from transformers import AutoTokenizer

from knowwhen.paths import ARTIFACTS, DATA, ROOT, inside_project

TEMPLATE = "Use the passage to answer the question with a short phrase only.\n\nPassage: {passage}\n\nQuestion: {question}\nAnswer:"
OUT = inside_project(ROOT / "experiments" / "circuit" / "grounding_pairs.csv")


def window(text: str, start: int, end: int, size: int = 70) -> str:
    return text[max(0, start - size): end + size].strip()


def main() -> None:
    tok = AutoTokenizer.from_pretrained("CohereLabs/tiny-aya-global")
    n = lambda s: len(tok(s, add_special_tokens=False).input_ids)
    questions = {}
    for line in open(DATA / "processed" / "questions.jsonl"):
        r = json.loads(line)
        questions[r["id"]] = r
    load = lambda c: {json.loads(l)["id"]: json.loads(l) for l in open(ARTIFACTS / "runs" / "tiny-aya-global-t4" / f"{c}.jsonl")}
    closed, rag = load("closed"), load("rag")
    passages = {json.loads(l)["id"]: json.loads(l)["passages"][:3] for l in open(DATA / "processed" / "retrieved.jsonl")}

    usable = []
    for i, q in questions.items():
        if q["split"] != "train" or i not in rag or closed[i]["correct"] or not rag[i]["correct"]:
            continue
        for a in q["answers"]:
            hit = next(((p, m) for p in passages[i] for m in [re.search(rf"\b{re.escape(a)}\b", p["text"])] if m), None)
            if hit and rag[i]["answer"] and a.lower() in rag[i]["answer"].lower():
                usable.append(dict(id=i, dataset=q["dataset"], question=q["question"], answer=a, text=hit[0]["text"], span=hit[1].span()))
                break
    print("usable", len(usable))
    rng = random.Random(13)
    rng.shuffle(usable)
    by_len = {}
    for u in usable:
        by_len.setdefault(n(" " + u["answer"]), []).append(u)
    rows = []
    for u in usable:
        pool = [v for v in by_len[n(" " + u["answer"])] if v["dataset"] == u["dataset"] and v["answer"].lower() != u["answer"].lower()]
        rng.shuffle(pool)
        for v in pool[:20]:
            passage = window(u["text"], *u["span"])
            if v["answer"].lower() in passage.lower():
                continue
            swapped = re.sub(rf"\b{re.escape(u['answer'])}\b", v["answer"], passage)
            clean, corrupt = TEMPLATE.format(passage=passage, question=u["question"]), TEMPLATE.format(passage=swapped, question=u["question"])
            a, b = tok(" " + u["answer"], add_special_tokens=False).input_ids, tok(" " + v["answer"], add_special_tokens=False).input_ids
            if n(clean) == n(corrupt) and a[0] != b[0]:
                rows.append(dict(id=u["id"], dataset=u["dataset"], clean_prompt=clean, clean_answer=" " + u["answer"], corrupt_prompt=corrupt, corrupt_answer=" " + v["answer"]))
                break
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    import collections
    print(len(rows), collections.Counter(r["dataset"] for r in rows))
    print(rows[0])


if __name__ == "__main__":
    main()
