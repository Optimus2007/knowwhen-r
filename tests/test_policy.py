import numpy as np

from knowwhen.policies.decision import (ABSTAIN, ANSWER, RETRIEVE_ANSWER, RETRIEVE_FALLBACK, Costs, oracle, per_item_utility,
                                        score, tuned_thresholds, two_thresholds, value_policy)

C = Costs(wrong=2, abstain=0, retrieve=0.1)


def test_utility_values():
    actions = np.array([ANSWER, ANSWER, ABSTAIN, RETRIEVE_ANSWER, RETRIEVE_FALLBACK])
    y0 = np.array([1, 0, 1, 0, 1])
    y1 = np.array([0, 0, 0, 1, 0])
    assert np.allclose(per_item_utility(actions, y0, y1, C), [1, -2, 0, 0.9, 0.9])


def test_rates_add_up():
    actions = np.array([ANSWER, ABSTAIN, RETRIEVE_ANSWER, ANSWER])
    o = score(actions, np.array([1, 0, 0, 0]), np.array([0, 0, 1, 0]), C)
    assert o.coverage == 0.75 and o.abstain_rate == 0.25 and o.retrieval_rate == 0.25
    assert np.isclose(o.wrong_rate + o.correct_rate, o.coverage)


def test_value_policy_extremes():
    # Sure of itself: answer. Hopeless either way: abstain. Evidence helps: retrieve and use it.
    p0 = np.array([0.99, 0.05, 0.1])
    before = np.array([0.99, 0.05, 0.9])
    after = np.array([0.99, 0.05, 0.9])
    assert value_policy(p0, before, after, C).tolist() == [ANSWER, ABSTAIN, RETRIEVE_ANSWER]


def test_value_policy_stage_two_abstains_after_bad_retrieval():
    a = value_policy(np.array([0.1]), np.array([0.9]), np.array([0.1]), C)
    assert a[0] == 3  # paid for retrieval, the evidence did not help, so abstain


def test_oracle_dominates_any_policy():
    rng = np.random.default_rng(0)
    y0, y1 = rng.integers(0, 2, 500), rng.integers(0, 2, 500)
    best = score(oracle(y0, y1, C), y0, y1, C).utility
    for a in (np.zeros(500, int), np.full(500, RETRIEVE_ANSWER), rng.integers(0, 3, 500)):
        assert score(a, y0, y1, C).utility <= best


def test_tuned_thresholds_recover_a_clean_rule():
    rng = np.random.default_rng(1)
    s = rng.random(4000)
    y0 = (s > 0.7).astype(int)
    y1 = (s > 0.3).astype(int)
    high, low = tuned_thresholds(s, y0, y1, C)
    assert abs(high - 0.7) < 0.05 and abs(low - 0.3) < 0.05
    assert score(two_thresholds(s, high, low), y0, y1, C).wrong_rate < 0.02
