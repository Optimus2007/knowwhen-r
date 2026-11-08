"""Pure helpers for the exploratory combination experiments.

None of these read gold answers. `cross_path_features` looks only at the two generated answers
and the retrieved text, so it is available at run time.
"""
from __future__ import annotations

import numpy as np

from knowwhen.text import contains_answer, is_refusal


def agree(a: str, b: str) -> bool:
    """True if one answer contains the other as whole tokens. A refusal agrees with nothing."""
    if is_refusal(a) or is_refusal(b):
        return False
    return contains_answer(a, [b]) or contains_answer(b, [a])


def grounded(answer: str, passages: str) -> bool:
    """True if the answer text appears in the retrieved passages."""
    return not is_refusal(answer) and contains_answer(passages, [answer])


def cross_path_features(closed: list[str], rag: list[str], passages: list[str]) -> np.ndarray:
    rows = [(agree(c, r), grounded(r, p), grounded(c, p), is_refusal(r), is_refusal(c)) for c, r, p in zip(closed, rag, passages)]
    return np.asarray(rows, dtype=np.float64)


def em_prior(p: np.ndarray, source_prior: float, steps: int = 200) -> float:
    """Saerens style estimate of the positive rate on unlabelled data, given calibrated source scores."""
    pi = source_prior
    for _ in range(steps):
        a = p * pi / source_prior
        b = (1 - p) * (1 - pi) / (1 - source_prior)
        pi = float(np.mean(a / (a + b)))
    return pi


def shift_prior(p: np.ndarray, source_prior: float, target_prior: float) -> np.ndarray:
    a = p * target_prior / source_prior
    b = (1 - p) * (1 - target_prior) / (1 - source_prior)
    return a / (a + b)


def conformal_cut(conf: np.ndarray, wrong: np.ndarray, alpha: float) -> float:
    """Lowest confidence cut whose wrong answer rate is bounded by alpha, by conformal risk control.

    The loss is 1 when a question is answered wrongly. It can only fall as the cut rises, which
    is what the guarantee needs. The guarantee assumes calibration and new data are exchangeable,
    so it is not expected to hold on a shifted dataset.
    """
    n = len(conf)
    for t in np.concatenate([[-np.inf], np.unique(conf)]):
        risk = float(np.sum(wrong & (conf >= t)))
        if (risk + 1) / (n + 1) <= alpha:
            return float(t)
    return float("inf")
