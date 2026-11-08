"""Answer normalization and correctness matching."""
from __future__ import annotations

import re
import string
import unicodedata

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans({c: " " for c in string.punctuation + "‘’“”–—"})


def normalize(text: str) -> str:
    """Lowercase, strip accents, punctuation and articles, squeeze spaces (SQuAD style)."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().translate(_PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def contains_answer(prediction: str, answers: list[str]) -> bool:
    """True if any gold alias occurs in the prediction as a whole token span."""
    padded = f" {normalize(prediction)} "
    for answer in answers:
        gold = normalize(answer)
        if gold and f" {gold} " in padded:
            return True
    return False


def exact_match(prediction: str, answers: list[str]) -> bool:
    pred = normalize(prediction)
    return any(pred == normalize(a) for a in answers)


_REFUSAL = re.compile(
    r"\b(i don.?t know|i do not know|i.?m not sure|i am not sure|cannot (answer|determine|find)|"
    r"can.?t (answer|determine|find)|unable to (answer|determine|find)|no information|not enough information|"
    r"unknown|not (provided|mentioned|specified|available))\b",
    re.I,
)


def is_refusal(prediction: str) -> bool:
    """Flag answers where the model itself declines. Counted apart from wrong answers."""
    return bool(_REFUSAL.search(prediction)) or not prediction.strip()
