import tantivy

from knowwhen.retrieval.wiki import WikiRetriever, build_schema, question_terms, split_passages


def test_terms_drop_stopwords():
    terms = question_terms('Who wrote "Hamlet" AND NOT Macbeth?')
    assert "hamlet" in terms and "who" not in terms


def test_split_passages_caps_length():
    parts = split_passages(" ".join(["w"] * 1000))
    assert len(parts) == 4 and all(len(p.split()) == 100 for p in parts)


def _index(path, docs):
    index = tantivy.Index(build_schema(), path=str(path))
    w = index.writer(heap_size=50_000_000, num_threads=1)
    for t, b in docs:
        w.add_document(tantivy.Document(title=t, body=b))
    w.commit()
    w.wait_merging_threads()


def test_search_ranks_title_match(tmp_path):
    _index(tmp_path, [("Paris", "Paris is the capital of France."), ("Dog", "Dogs are animals kept as pets."),
                      ("Berlin", "Berlin is the capital of Germany.")])
    assert WikiRetriever(tmp_path).search("What is the capital of France?", k=2)[0].title == "Paris"


def test_operator_words_are_safe(tmp_path):
    _index(tmp_path, [("Logic", "AND OR NOT are operators")])
    assert WikiRetriever(tmp_path).search('Is "AND" NOT OR (logic)?', k=1)[0].title == "Logic"
