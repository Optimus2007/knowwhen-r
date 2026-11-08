"""Multilingual transfer set from MKQA, added after the test run. See docs/analysis_plan.md.

MKQA (Longpre et al. 2021, CC BY-SA 3.0) translates NQ questions into 26 languages with answers in
each language. We keep the same questions in six languages both models support, so each language
is compared on identical facts. Choices, fixed before any generation:

  languages   en, es, de, fr, ru, tr
  answers     entity and short_phrase types only; numbers and dates are written differently
              across languages and string matching would mislabel them
  gold        the language's own answer and aliases, plus the English ones, since a model may
              answer a foreign question in English
  overlap     any question whose normalized English text matches a question in our own set is
              dropped, because MKQA comes from NQ
  sample      600 questions, seed 13, so 3,600 prompts per model
"""
from __future__ import annotations

import gzip
import json
import random

from knowwhen.paths import DATA, inside_project
from knowwhen.text import normalize

LANGS = ("en", "es", "de", "fr", "ru", "tr")
TYPES = {"entity", "short_phrase"}
N, SEED = 600, 13


def golds(answers: list[dict]) -> list[str]:
    out = []
    for a in answers:
        if a.get("type") in TYPES and a.get("text"):
            out += [a["text"], *a.get("aliases", [])]
    return out


def main():
    ours = {normalize(json.loads(l)["question"]) for l in open(DATA / "processed" / "questions.jsonl")}
    keep, overlap = [], 0
    for line in gzip.open(DATA / "raw" / "mkqa" / "mkqa.jsonl.gz", "rt"):
        r = json.loads(line)
        en = r["answers"]["en"]
        if not en or any(a.get("type") not in TYPES for a in en):
            continue
        if normalize(r["queries"]["en"]) in ours:
            overlap += 1
            continue
        if all(golds(r["answers"][l]) and r["queries"].get(l) for l in LANGS):
            keep.append(r)
    eligible = len(keep)
    random.Random(SEED).shuffle(keep)
    keep = keep[:N]
    out = inside_project(DATA / "processed" / "mkqa_questions.jsonl")
    with out.open("w") as f:
        for r in keep:
            english = golds(r["answers"]["en"])
            for l in LANGS:
                gold = list(dict.fromkeys(golds(r["answers"][l]) + english))
                f.write(json.dumps(dict(id=f"mkqa:{r['example_id']}:{l}", dataset="mkqa", lang=l, question=r["queries"][l],
                                        answers=gold, split="mkqa"), ensure_ascii=False) + "\n")
    print(json.dumps(dict(eligible_after_overlap=eligible, dropped_overlap=overlap, sampled=len(keep), rows=len(keep) * len(LANGS))))


if __name__ == "__main__":
    main()
