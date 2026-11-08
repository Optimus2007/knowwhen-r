"""Detection, calibration and selective prediction metrics.

Scores are probabilities that the answer is correct, unless a function says otherwise.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def detection(correct: np.ndarray, p_correct: np.ndarray) -> dict:
    """Wrong answers are the positive class, as they are what we want to flag."""
    wrong = 1 - correct.astype(int)
    if wrong.min() == wrong.max():
        return dict(auroc=float("nan"), auprc=float("nan"), base_error=float(wrong.mean()))
    risk = 1 - p_correct
    return dict(auroc=float(roc_auc_score(wrong, risk)), auprc=float(average_precision_score(wrong, risk)),
                base_error=float(wrong.mean()))


def calibration(correct: np.ndarray, p_correct: np.ndarray, bins: int = 15) -> dict:
    correct = correct.astype(float)
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(p_correct, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = which == b
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - p_correct[m].mean())
    return dict(ece=float(ece), brier=float(np.mean((p_correct - correct) ** 2)))


def risk_coverage(correct: np.ndarray, confidence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Answer the most confident items first. Returns coverage and risk at each cut."""
    order = np.argsort(-confidence, kind="stable")
    errors = 1 - correct[order].astype(float)
    n = np.arange(1, len(errors) + 1)
    return n / len(errors), np.cumsum(errors) / n


def aurc(correct: np.ndarray, confidence: np.ndarray) -> float:
    """Area under the risk coverage curve. Lower is better."""
    return float(risk_coverage(correct, confidence)[1].mean())


def risk_at_coverage(correct: np.ndarray, confidence: np.ndarray, coverage: float) -> float:
    cov, risk = risk_coverage(correct, confidence)
    return float(risk[min(np.searchsorted(cov, coverage), len(risk) - 1)])
