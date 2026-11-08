"""Builds every policy for one fold and cost setting, using validation rows for all tuning.

`val` and `ev` are frames with y0 and y1 (1, 0 or DECLINED) and the scores below, all oriented so
higher means safer. Evaluation labels are read only by the oracle; a test holds this in place.

  p0, p_before, p_after   activation probes: closed answer, forecast of the retrieved answer made
                          before retrieval, and retrieved answer judged after retrieval
  t0, t_before, t_after   the same three estimates from token statistics
  mean_logprob, neg_entropy   raw closed book uncertainty
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from knowwhen.policies.decision import (ABSTAIN, ANSWER, RETRIEVE_ABSTAIN, RETRIEVE_ANSWER, Costs, fixed_abstain, fixed_retrieve, oracle, per_item_utility,
                                        resolve, score, tuned_thresholds, two_thresholds, value_policy)

# The comparisons fixed in docs/analysis_plan.md. Everything else is labelled exploratory.
COVERAGES = [0.5, 0.7, 0.9]
PLANNED = ["closed_only", "rag_always", "fixed_abstain_05", "fixed_retrieve_05", "two_cut_probe", "two_cut_logprob", "value", "oracle"]


def best_cut(s: np.ndarray, y0: np.ndarray, y1: np.ndarray, costs: Costs, rule, grid: int = 41) -> float:
    """Single threshold for `rule`, chosen by utility on validation data."""
    cuts = np.concatenate([[-np.inf], np.unique(np.quantile(s, np.linspace(0, 1, grid))), [np.inf]])
    return float(cuts[int(np.argmax([per_item_utility(rule(s, c), y0, y1, costs).mean() for c in cuts]))])


def policies(val: pd.DataFrame, ev: pd.DataFrame, costs: Costs) -> dict[str, np.ndarray]:
    vy0, vy1 = val.y0.values, val.y1.values
    n = len(ev)
    out = dict(closed_only=np.full(n, ANSWER), rag_always=np.full(n, RETRIEVE_ANSWER), abstain_all=np.full(n, ABSTAIN))
    out["fixed_abstain_05"] = fixed_abstain(ev.p0.values, 0.5)
    out["fixed_retrieve_05"] = fixed_retrieve(ev.p0.values, 0.5)
    out["tuned_abstain"] = fixed_abstain(ev.p0.values, best_cut(val.p0.values, vy0, vy1, costs, fixed_abstain))
    out["tuned_retrieve"] = fixed_retrieve(ev.p0.values, best_cut(val.p0.values, vy0, vy1, costs, fixed_retrieve))
    for name, col in (("two_cut_probe", "p0"), ("two_cut_logprob", "mean_logprob"), ("two_cut_entropy", "neg_entropy"), ("two_cut_token_stats", "t0")):
        out[name] = two_thresholds(ev[col].values, *tuned_thresholds(val[col].values, vy0, vy1, costs))
    out["value"] = value_policy(ev.p0.values, ev.p_before.values, ev.p_after.values, costs)
    out["value_no_after"] = value_policy(ev.p0.values, ev.p_before.values, ev.p_before.values, costs)
    out["value_token_stats"] = value_policy(ev.t0.values, ev.t_before.values, ev.t_after.values, costs)
    if "p0_rc" in ev and ev.p0_rc.notna().all():
        out["value_recalibrated"] = value_policy(ev.p0_rc.values, ev.p_before_rc.values, ev.p_after_rc.values, costs)
    out["oracle"] = oracle(ev.y0.values, ev.y1.values, costs)
    return out


def oracle_recovered(utility: float, actions: dict[str, np.ndarray], y0: np.ndarray, y1: np.ndarray, costs: Costs) -> float:
    """Share of the gap from the best single fixed action to the oracle. Below zero means worse than that action."""
    base = max(score(actions[k], y0, y1, costs).utility for k in ("closed_only", "rag_always", "abstain_all"))
    top = score(actions["oracle"], y0, y1, costs).utility
    return float((utility - base) / (top - base)) if top > base else float("nan")


def retrieval_stats(actions: np.ndarray, y0: np.ndarray, y1: np.ndarray, hit: np.ndarray) -> dict:
    """How retrieval behaved on the questions where this policy paid for it."""
    _, retrieved = resolve(actions, y0, y1)
    used = actions == RETRIEVE_ANSWER
    rate = lambda mask: float(mask.mean()) if mask.size else float("nan")
    return dict(retrieval_hit=rate(hit[retrieved]), accuracy_after_retrieval=rate(y1[used] == 1),
                rescued=rate((y0[used] != 1) & (y1[used] == 1)), harmed=rate((y0[used] == 1) & (y1[used] != 1)))


def matched_retrieval(val: pd.DataFrame, ev: pd.DataFrame, costs: Costs, metric: str = "utility") -> dict:
    """Fewest retrievals, without abstention, that match RAG always on validation data.

    Abstention is excluded on purpose: with a wrong penalty, abstaining alone can beat RAG always
    and the comparison would say nothing about retrieval.
    """
    vy0, vy1 = val.y0.values, val.y1.values
    target = getattr(score(np.full(len(val), RETRIEVE_ANSWER), vy0, vy1, costs), metric)
    cuts = np.concatenate([np.unique(np.quantile(val.p0.values, np.linspace(0, 1, 101))), [np.inf]])
    cut = next(c for c in cuts if getattr(score(fixed_retrieve(val.p0.values, c), vy0, vy1, costs), metric) >= target)
    got = score(fixed_retrieve(ev.p0.values, cut), ev.y0.values, ev.y1.values, costs)
    rag = score(np.full(len(ev), RETRIEVE_ANSWER), ev.y0.values, ev.y1.values, costs)
    return dict(metric=metric, cut=float(cut), retrieval_rate=got.retrieval_rate, value=getattr(got, metric), rag_always_value=getattr(rag, metric))


def _route(frame: pd.DataFrame, closed: str, before: str, after: str) -> tuple[np.ndarray, np.ndarray]:
    """Whether each question goes to retrieval, and the confidence in the answer it ends with."""
    routed = frame[before].values > frame[closed].values
    return routed, np.where(routed, frame[after].values, frame[closed].values)


def coverage_policies(val: pd.DataFrame, ev: pd.DataFrame, target: float) -> dict[str, np.ndarray]:
    """Policies that answer their most confident `target` share of questions, ignoring costs.

    Exploratory, added after the first end to end test: at the headline costs a weak model is
    told to abstain on nearly everything. The cut is a validation quantile, so coverage reached
    on evaluation rows is measured rather than assumed. Refusals lower it further.
    """
    q = 1 - target
    out = {}
    for name, col in (("cov_closed_probe", "p0"), ("cov_closed_logprob", "mean_logprob"), ("cov_closed_token_stats", "t0")):
        out[name] = np.where(ev[col].values >= np.quantile(val[col].values, q), ANSWER, ABSTAIN)
    out["cov_rag_probe"] = np.where(ev.p_after.values >= np.quantile(val.p_after.values, q), RETRIEVE_ANSWER, RETRIEVE_ABSTAIN)
    for name, cols in (("cov_route_probe", ("p0", "p_before", "p_after")), ("cov_route_token_stats", ("t0", "t_before", "t_after"))):
        cut = np.quantile(_route(val, *cols)[1], q)
        routed, conf = _route(ev, *cols)
        keep = conf >= cut
        out[name] = np.where(routed, np.where(keep, RETRIEVE_ANSWER, RETRIEVE_ABSTAIN), np.where(keep, ANSWER, ABSTAIN))
    return out
