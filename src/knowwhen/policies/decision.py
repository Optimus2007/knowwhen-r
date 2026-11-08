"""Answer, retrieve or abstain policies, scored by realized utility.

A correct answer earns 1, a wrong answer costs `wrong`, abstaining costs `abstain`, and
every retrieval costs `retrieve` whether or not it helps. Correctness arrays hold 1 for right,
0 for wrong and DECLINED when the model itself refused, which is scored as an abstention. Counting
refusals as wrong would punish the always answer baselines and flatter any abstaining policy.

The comparison that matters is between policies given equal tuning data. `tuned_thresholds`
searches two cut points on validation data at the given costs. `value_policy` sets nothing
by search; its actions follow from probabilities and costs, so any gain over a tuned single
score has to come from its extra signals, not from extra fitting.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ANSWER, RETRIEVE_ANSWER, ABSTAIN, RETRIEVE_ABSTAIN, RETRIEVE_FALLBACK = 0, 1, 2, 3, 4
DECLINED = -1


@dataclass(frozen=True)
class Costs:
    wrong: float = 2.0
    abstain: float = 0.0
    retrieve: float = 0.1


@dataclass
class Outcome:
    utility: float
    wrong_rate: float
    correct_rate: float
    coverage: float
    abstain_rate: float
    retrieval_rate: float
    selective_accuracy: float


def resolve(actions: np.ndarray, y0: np.ndarray, y1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """What each action ended in (1, 0 or DECLINED), and whether it paid for retrieval."""
    closed = (actions == ANSWER) | (actions == RETRIEVE_FALLBACK)
    rag = actions == RETRIEVE_ANSWER
    retrieved = rag | (actions == RETRIEVE_ABSTAIN) | (actions == RETRIEVE_FALLBACK)
    return np.where(closed, y0, np.where(rag, y1, DECLINED)), retrieved


def score(actions: np.ndarray, y0: np.ndarray, y1: np.ndarray, costs: Costs) -> Outcome:
    """Realized result of each action against the true correctness of both answers."""
    result, retrieved = resolve(actions, y0, y1)
    right, wrong = result == 1, result == 0
    answered = right | wrong
    per_item = per_item_utility(actions, y0, y1, costs)
    n = len(actions)
    return Outcome(utility=float(per_item.mean()), wrong_rate=float(wrong.sum() / n), correct_rate=float(right.sum() / n),
                   coverage=float(answered.mean()), abstain_rate=float(1 - answered.mean()), retrieval_rate=float(retrieved.mean()),
                   selective_accuracy=float(right.sum() / max(answered.sum(), 1)))


def per_item_utility(actions: np.ndarray, y0: np.ndarray, y1: np.ndarray, costs: Costs) -> np.ndarray:
    result, retrieved = resolve(actions, y0, y1)
    return np.where(result == 1, 1.0, np.where(result == 0, -costs.wrong, -costs.abstain)) - costs.retrieve * retrieved


def expected_answer_value(p: np.ndarray, costs: Costs) -> np.ndarray:
    return p - (1 - p) * costs.wrong


def fixed_abstain(p0: np.ndarray, t: float) -> np.ndarray:
    return np.where(p0 >= t, ANSWER, ABSTAIN)


def fixed_retrieve(p0: np.ndarray, t: float) -> np.ndarray:
    return np.where(p0 >= t, ANSWER, RETRIEVE_ANSWER)


def two_thresholds(score0: np.ndarray, high: float, low: float) -> np.ndarray:
    """Answer above high, retrieve between, abstain below low. The retrieved answer is always used."""
    return np.where(score0 >= high, ANSWER, np.where(score0 >= low, RETRIEVE_ANSWER, ABSTAIN))


def tuned_thresholds(score0: np.ndarray, y0: np.ndarray, y1: np.ndarray, costs: Costs, grid: int = 41) -> tuple[float, float]:
    """Best high and low cut points on validation data, for any score oriented so higher means safer."""
    cuts = np.unique(np.quantile(score0, np.linspace(0, 1, grid)))
    cuts = np.concatenate([[-np.inf], cuts, [np.inf]])
    best, pair = -np.inf, (np.inf, -np.inf)
    for i, low in enumerate(cuts):
        for high in cuts[i:]:
            u = per_item_utility(two_thresholds(score0, high, low), y0, y1, costs).mean()
            if u > best:
                best, pair = u, (float(high), float(low))
    return pair


def value_policy(p0: np.ndarray, p_rag_before: np.ndarray, p_rag_after: np.ndarray, costs: Costs) -> np.ndarray:
    """Two stage expected utility policy.

    Stage one sees the closed book answer and a forecast of how the retrieved answer would do.
    Stage two runs only after retrieval and picks between the retrieved answer, the original
    answer and abstaining, using the probe read from the retrieval pass.
    """
    v_answer = expected_answer_value(p0, costs)
    v_abstain = np.full_like(p0, -costs.abstain)
    # Retrieval keeps the option to fall back, so it is worth the best of its outcomes, less its cost.
    v_retrieve = np.maximum.reduce([expected_answer_value(p_rag_before, costs), v_answer, v_abstain]) - costs.retrieve
    first = np.argmax(np.stack([v_answer, v_retrieve, v_abstain]), axis=0)  # 0 answer, 1 retrieve, 2 abstain

    v_after = expected_answer_value(p_rag_after, costs)
    second = np.argmax(np.stack([v_after, v_answer, v_abstain]), axis=0)
    stage_two = np.choose(second, [RETRIEVE_ANSWER, RETRIEVE_FALLBACK, RETRIEVE_ABSTAIN])
    return np.where(first == 0, ANSWER, np.where(first == 2, ABSTAIN, stage_two))


def oracle(y0: np.ndarray, y1: np.ndarray, costs: Costs) -> np.ndarray:
    """Best action with knowledge of both outcomes. An upper bound, not a usable method."""
    value = lambda y: np.where(y == 1, 1.0, np.where(y == 0, -costs.wrong, -costs.abstain))
    v_answer = value(y0)
    v_rag = value(y1) - costs.retrieve
    v_abstain = np.full(len(y0), -costs.abstain)
    pick = np.argmax(np.stack([v_answer, v_rag, v_abstain]), axis=0)
    return np.choose(pick, [ANSWER, RETRIEVE_ANSWER, ABSTAIN])
