import numpy as np

from knowwhen.evaluation.audit import ABSTAIN_TEXT, compare, paired_test, replay
from knowwhen.policies.decision import Costs

C = Costs(wrong=2, abstain=0, retrieve=0.1)


def _run(outputs, retrieved, name):
    n = len(outputs)
    return replay([f"q{i}" for i in range(n)], [f"question {i}" for i in range(n)], [["Paris", "City of Light"]] * n, outputs, retrieved, C, name)


def test_replay_scores_match_hand_calculation():
    run = _run(["Paris", "London", ABSTAIN_TEXT, "It is Paris."], [False, False, False, True], "a")
    assert np.isclose(run.headline["utility"], (1 - 2 + 0 + 0.9) / 4)
    assert np.isclose(run.headline["wrong_answer"], 0.25)
    assert np.isclose(run.headline["answered"], 0.75)
    assert np.isclose(run.headline["retrieved"], 0.25)


def test_paired_comparison_detects_a_real_improvement():
    n = 200
    base = _run(["London"] * n, [False] * n, "before")
    better = _run(["Paris"] * n, [False] * n, "after")
    out = compare(base, better)
    assert out["n"] == n and out["delta"] == 3.0 and out["ci_low"] > 0 and out["p_value"] < 0.05


def test_paired_test_has_correct_size_and_power():
    rng = np.random.default_rng(0)
    false_alarms = 0
    for seed in range(60):  # no true effect: about 5% of tests should fire
        b = rng.normal(size=400)
        c = b + rng.normal(size=400)
        false_alarms += paired_test(list(zip("x" * 400, b)), list(zip("x" * 400, c)), resamples=500, seed=seed)["p_value"] < 0.05
    assert false_alarms <= 9
    b = rng.normal(size=2000)
    out = paired_test(list(zip("x" * 2000, b)), list(zip("x" * 2000, b + 0.2 + rng.normal(0, 0.5, 2000))))
    assert out["p_value"] < 0.01 and out["ci_low"] > 0.1
