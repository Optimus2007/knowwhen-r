import numpy as np

from knowwhen.evaluation.metrics import aurc, calibration, detection, risk_at_coverage
from knowwhen.probes.linear import LinearProbe, Temperature


def test_perfect_detector():
    y = np.array([1, 1, 0, 0])
    assert detection(y, np.array([0.9, 0.8, 0.2, 0.1]))["auroc"] == 1.0


def test_calibration_of_exact_probabilities():
    rng = np.random.default_rng(0)
    p = rng.random(200000)
    y = rng.random(200000) < p
    assert calibration(y, p)["ece"] < 0.01


def test_aurc_prefers_good_ranking():
    y = np.array([1, 1, 1, 0, 0])
    good, bad = np.array([5, 4, 3, 2, 1.0]), np.array([1, 2, 3, 4, 5.0])
    assert aurc(y, good) < aurc(y, bad)
    assert risk_at_coverage(y, good, 0.6) == 0.0


def test_temperature_undoes_overconfidence():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 1, 50000)
    y = (rng.random(50000) < 1 / (1 + np.exp(-z))).astype(float)
    assert abs(Temperature().fit(z * 3, y).t - 3) < 0.15


def test_probe_learns_a_planted_direction_and_keeps_ranking_after_calibration():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3000, 40))
    y = (x[:, 0] + 0.3 * rng.normal(size=3000) > 0).astype(int)
    probe = LinearProbe(c=0.1).fit(x[:2000], y[:2000])
    raw = probe.logits(x[2000:])
    probe.calibrate(x[2000:2500], y[2000:2500])
    p = probe.predict(x[2000:])
    assert detection(y[2000:], p)["auroc"] > 0.9
    assert (np.argsort(raw) == np.argsort(p)).all()
