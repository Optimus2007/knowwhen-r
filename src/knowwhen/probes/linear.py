"""Probes on cached activations, with calibration fitted on separate data."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


@dataclass
class Temperature:
    """One scale parameter on the logit. Preserves ranking, so AUROC is untouched."""
    t: float = 1.0

    def fit(self, logits: np.ndarray, y: np.ndarray) -> "Temperature":
        def nll(log_t: float) -> float:
            z = logits / np.exp(log_t)
            return float(np.mean(np.logaddexp(0, z) - y * z))

        self.t = float(np.exp(minimize_scalar(nll, bounds=(-4, 4), method="bounded").x))
        return self

    def __call__(self, logits: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-logits / self.t))


class LinearProbe:
    """Standardized L2 logistic regression. Class weights stay off so outputs remain probabilities."""

    def __init__(self, c: float = 0.01, seed: int = 0):
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(C=c, max_iter=300, random_state=seed)
        self.temperature = Temperature()

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LinearProbe":
        self.clf.fit(self.scaler.fit_transform(x.astype(np.float32)), y)
        return self

    def logits(self, x: np.ndarray) -> np.ndarray:
        return self.clf.decision_function(self.scaler.transform(x.astype(np.float32))).astype(np.float64)

    def calibrate(self, x: np.ndarray, y: np.ndarray) -> "LinearProbe":
        self.temperature.fit(self.logits(x), y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.temperature(self.logits(x))
