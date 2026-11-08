"""BM25 retrieval over Wikipedia passages with Tantivy.

Each article contributes its first passages of about 100 words, which keeps the index
manageable while covering the lead sections where most facts appear. SQLite FTS5 was tried
first, but ranking took 14 to 107 seconds per query on this hardware.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import tantivy

PASSAGE_WORDS = 100
MAX_PASSAGES_PER_ARTICLE = 4
TITLE_BOOST = 4.0

_STOP = frozenset(
    "a an and are as at be been but by did do does for from had has have he her his how i in into is it its "
    "many much of on or she that the their them they this to was were what when where which who whom whose "
    "why will with would you name called".split()
)
_WORD = re.compile(r"[^\W_]+", re.UNICODE)


@dataclass
class Passage:
    title: str
    text: str
    score: float


def split_passages(text: str) -> list[str]:
    """Cut article text into fixed size word windows, keeping only the first few."""
    words = text.split()
    limit = PASSAGE_WORDS * MAX_PASSAGES_PER_ARTICLE
    return [" ".join(words[i : i + PASSAGE_WORDS]) for i in range(0, min(len(words), limit), PASSAGE_WORDS)]


def question_terms(question: str) -> list[str]:
    """Lowercased content words of a question, without repeats."""
    terms = []
    for word in _WORD.findall(question.lower()):
        if word not in _STOP and word not in terms:
            terms.append(word)
    return terms


def build_schema() -> tantivy.Schema:
    b = tantivy.SchemaBuilder()
    b.add_text_field("title", stored=True, tokenizer_name="en_stem")
    b.add_text_field("body", stored=True, tokenizer_name="en_stem")
    return b.build()


class WikiRetriever:
    def __init__(self, path: Path):
        self.index = tantivy.Index.open(str(path))
        self.searcher = self.index.searcher()

    def search(self, question: str, k: int = 3) -> list[Passage]:
        terms = question_terms(question)
        if not terms:
            return []
        # Terms are quoted so that words like NOT cannot act as query operators.
        text = " ".join(f'"{t}"' for t in terms)
        query = self.index.parse_query(text, ["title", "body"], field_boosts={"title": TITLE_BOOST})
        hits = self.searcher.search(query, k).hits
        out = []
        for score, address in hits:
            doc = self.searcher.doc(address)
            out.append(Passage(title=doc["title"][0], text=doc["body"][0], score=float(score)))
        return out
