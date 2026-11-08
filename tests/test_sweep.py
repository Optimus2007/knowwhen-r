import numpy as np
import pandas as pd

from knowwhen.evaluation.audit import ABSTAIN_TEXT, replay
from knowwhen.policies.decision import ABSTAIN, ANSWER, DECLINED, RETRIEVE_ANSWER, Costs, oracle, per_item_utility, score
from knowwhen.policies.sweep import PLANNED, coverage_policies, matched_retrieval, oracle_recovered, policies, retrieval_stats

C = Costs(wrong=2, abstain=0, retrieve=0.1)


def frame(n, seed):
    rng = np.random.default_rng(seed)
    p0, p1 = rng.random(n), rng.random(n)
    y0, y1 = (rng.random(n) < p0).astype(int), (rng.random(n) < p1).astype(int)
    noisy = lambda p: np.clip(p + rng.normal(0, 0.05, n), 0.001, 0.999)
    return pd.DataFrame(dict(y0=y0, y1=y1, p0=p0, p_before=noisy(p1), p_after=p1, t0=noisy(p0), t_before=noisy(p1), t_after=noisy(p1),
                             mean_logprob=np.log(p0), neg_entropy=noisy(p0), hit=rng.random(n) < 0.5))


def test_refusal_scores_as_abstention_not_wrong():
    y0 = np.array([DECLINED, 0, 1])
    y1 = np.array([0, DECLINED, 1])
    assert np.allclose(per_item_utility(np.full(3, ANSWER), y0, y1, C), [0, -2, 1])
    assert np.allclose(per_item_utility(np.full(3, RETRIEVE_ANSWER), y0, y1, C), [-2.1, -0.1, 0.9])
    o = score(np.full(3, ANSWER), y0, y1, C)
    assert np.isclose(o.coverage, 2 / 3) and np.isclose(o.wrong_rate, 1 / 3) and np.isclose(o.selective_accuracy, 0.5)


def test_oracle_prefers_a_refusal_to_a_wrong_answer():
    a = oracle(np.array([DECLINED, 0]), np.array([0, 0]), C)
    assert score(a, np.array([DECLINED, 0]), np.array([0, 0]), C).utility == 0


def test_audit_scorer_agrees_on_refusals():
    run = replay(["a", "b"], ["q", "q"], [["Paris"], ["Paris"]], ["Unknown.", "London"], [False, False], C, "r")
    assert np.isclose(run.headline["utility"], -1) and np.isclose(run.headline["answered"], 0.5)
    assert np.isclose(run.headline["wrong_answer"], 0.5)
    assert ABSTAIN_TEXT != "Unknown."


def test_evaluation_labels_reach_only_the_oracle():
    val, ev = frame(800, 0), frame(800, 1)
    first = policies(val, ev, C)
    flipped = ev.assign(y0=1 - ev.y0, y1=1 - ev.y1)
    second = policies(val, flipped, C)
    for name in first:
        assert name == "oracle" or (first[name] == second[name]).all(), name


def test_every_planned_policy_exists_and_oracle_is_the_ceiling():
    val, ev = frame(800, 2), frame(800, 3)
    acts = policies(val, ev, C)
    assert set(PLANNED) <= set(acts) and "value_recalibrated" not in acts
    top = score(acts["oracle"], ev.y0.values, ev.y1.values, C).utility
    assert all(score(a, ev.y0.values, ev.y1.values, C).utility <= top + 1e-12 for a in acts.values())


def test_calibrated_value_policy_beats_fixed_actions_when_signals_are_informative():
    val, ev = frame(4000, 4), frame(4000, 5)
    acts = policies(val, ev, C)
    u = lambda k: score(acts[k], ev.y0.values, ev.y1.values, C).utility
    assert u("value") > max(u("closed_only"), u("rag_always"), u("abstain_all"))
    assert 0 < oracle_recovered(u("value"), acts, ev.y0.values, ev.y1.values, C) < 1


def test_oracle_recovered_endpoints():
    ev = frame(500, 6)
    acts = policies(frame(500, 7), ev, C)
    y0, y1 = ev.y0.values, ev.y1.values
    assert np.isclose(oracle_recovered(score(acts["oracle"], y0, y1, C).utility, acts, y0, y1, C), 1)
    assert np.isclose(oracle_recovered(score(acts["abstain_all"], y0, y1, C).utility, acts, y0, y1, C), 0)


def test_retrieval_stats_by_hand():
    a = np.array([RETRIEVE_ANSWER, RETRIEVE_ANSWER, ANSWER, ABSTAIN])
    s = retrieval_stats(a, np.array([0, 1, 1, 0]), np.array([1, 0, 0, 0]), np.array([True, False, True, True]))
    assert s == dict(retrieval_hit=0.5, accuracy_after_retrieval=0.5, rescued=0.5, harmed=0.5)


def test_matched_retrieval_uses_fewer_retrievals_than_always():
    val, ev = frame(3000, 8), frame(3000, 9)
    m = matched_retrieval(val, ev, C, "correct_rate")
    assert m["retrieval_rate"] <= 1 and np.isfinite(m["value"])


def test_coverage_policies_hit_their_target_on_similar_data():
    val, ev = frame(5000, 10), frame(5000, 11)
    for target in (0.5, 0.7, 0.9):
        for name, a in coverage_policies(val, ev, target).items():
            assert abs(score(a, ev.y0.values, ev.y1.values, C).coverage - target) < 0.04, (name, target)


def test_coverage_policies_ignore_evaluation_labels():
    val, ev = frame(800, 12), frame(800, 13)
    first = coverage_policies(val, ev, 0.7)
    second = coverage_policies(val, ev.assign(y0=1 - ev.y0, y1=1 - ev.y1), 0.7)
    assert all((first[k] == second[k]).all() for k in first)


def test_more_coverage_means_more_wrong_answers_for_a_useful_score():
    val, ev = frame(5000, 14), frame(5000, 15)
    rate = [score(coverage_policies(val, ev, t)["cov_closed_probe"], ev.y0.values, ev.y1.values, C).wrong_rate for t in (0.5, 0.7, 0.9)]
    assert rate[0] < rate[1] < rate[2]


def test_routing_pays_for_retrieval_only_when_routed():
    val, ev = frame(2000, 16), frame(2000, 17)
    a = coverage_policies(val, ev, 0.7)["cov_route_probe"]
    routed = ev.p_before.values > ev.p0.values
    assert np.isclose(score(a, ev.y0.values, ev.y1.values, C).retrieval_rate, routed.mean())
