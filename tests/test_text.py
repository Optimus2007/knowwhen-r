from knowwhen.text import contains_answer, exact_match, is_refusal, normalize


def test_normalize():
    assert normalize("The Beatles!") == "beatles"
    assert normalize("Café  au  lait") == "cafe au lait"


def test_contains_whole_tokens_only():
    assert contains_answer("It was Paris, France", ["paris"])
    assert not contains_answer("comparison", ["paris"])
    assert not contains_answer("1999", ["99"])


def test_exact():
    assert exact_match("the Paris", ["Paris"])
    assert not exact_match("Paris France", ["Paris"])


def test_refusal():
    assert is_refusal("I don't know")
    assert is_refusal("  ")
    assert not is_refusal("Paris")
