import numpy as np

from knowwhen.policies.combine import agree, conformal_cut, cross_path_features, em_prior, grounded, shift_prior


def test_agree_handles_wording_and_refusals():
    assert agree("Paris.", "The city of Paris")
    assert not agree("Paris", "London")
    assert not agree("Unknown.", "Unknown.")


def test_grounded_needs_whole_tokens():
    assert grounded("Paris", "He moved to Paris in 1900.")
    assert not grounded("Par", "He moved to Paris in 1900.")
    assert not grounded("Not mentioned.", "Not mentioned anywhere.")


def test_feature_shape():
    assert cross_path_features(["a1"], ["a1"], ["x a1 y"]).tolist() == [[1, 1, 1, 0, 0]]


def test_em_recovers_a_shifted_prior():
    rng = np.random.default_rng(0)
    y = rng.random(40000) < 0.2
    z = rng.normal(np.where(y, 1.0, -1.0))
    # Scores that are calibrated for a 50% source prior.
    p = 1 / (1 + np.exp(-2 * z))
    assert abs(em_prior(p, 0.5) - 0.2) < 0.02


def test_shift_prior_is_identity_without_shift():
    p = np.array([0.1, 0.5, 0.9])
    assert np.allclose(shift_prior(p, 0.3, 0.3), p)


def test_conformal_cut_controls_risk_on_exchangeable_data():
    rng = np.random.default_rng(1)
    misses = []
    for _ in range(200):
        conf = rng.random(800)
        wrong = rng.random(800) > conf
        cut = conformal_cut(conf[:400], wrong[:400], 0.2)
        misses.append(float((wrong[400:] & (conf[400:] >= cut)).mean()))
    assert np.mean(misses) <= 0.2 + 0.005
