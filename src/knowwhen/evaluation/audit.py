"""Before and after evaluation through AuditKIT.

Each policy's final answers are replayed as precomputed outputs, so nothing is regenerated.
The scorer re-derives correctness from the output text and gold aliases, which gives an
independent check on the numbers from `policies.decision`. AuditKIT aligns runs by sample ID;
the significance test is our own, for the reason given in `paired_test`.
"""
from __future__ import annotations

from typing import Any

import auditkit as ak
import numpy as np
from auditkit.metric import Metric
from auditkit.score import Score
from auditkit.types import Direction, ScoreKind

from knowwhen.policies.decision import Costs
from knowwhen.text import contains_answer, is_refusal

ABSTAIN_TEXT = "I am not confident enough to answer this."


class PolicyOutcome(Metric):
    """Per question result of a policy: utility, wrong answer, answered, correct."""

    name = "policy_outcome"
    kind = ScoreKind.BENCHMARK
    direction = Direction.MAXIMIZE

    def __init__(self, costs: Costs):
        self.costs = costs

    def identity(self) -> dict:
        return dict(name=self.name, wrong=self.costs.wrong, abstain=self.costs.abstain, retrieve=self.costs.retrieve)

    def score(self, sample: ak.Sample, output: str, context: Any = None) -> list[Score]:
        declined = output == ABSTAIN_TEXT
        right = not declined and contains_answer(output, sample.metadata["answers"])
        # The model's own refusal counts as abstaining, matching `policies.decision`.
        answered = right or not (declined or is_refusal(output))
        wrong = answered and not right
        retrieved = bool(sample.metadata["retrieved"])
        utility = (1.0 if right else -self.costs.wrong if wrong else -self.costs.abstain) - self.costs.retrieve * retrieved
        return [
            Score(name="utility", value=utility, direction=Direction.MAXIMIZE),
            Score(name="wrong_answer", value=float(wrong), direction=Direction.MINIMIZE),
            Score(name="correct_answer", value=float(right), direction=Direction.MAXIMIZE),
            Score(name="answered", value=float(answered), direction=Direction.MAXIMIZE),
            Score(name="retrieved", value=float(retrieved), direction=Direction.MINIMIZE),
        ]


def replay(ids: list[str], questions: list[str], answers: list[list[str]], outputs: list[str], retrieved: list[bool],
           costs: Costs, name: str):
    """Scores recorded outputs and returns an AuditKIT RunResult."""
    samples = [
        ak.Sample(id=i, input=q, target=a[0], actual_output=o, metadata=dict(answers=a, retrieved=bool(r)))
        for i, q, a, o, r in zip(ids, questions, answers, outputs, retrieved)
    ]
    return ak.evaluate(samples, model="precomputed", scorers=[PolicyOutcome(costs)], tags=[name])


def paired_test(baseline: list[tuple[str, float]], candidate: list[tuple[str, float]], resamples: int = 2000, seed: int = 0) -> dict:
    """Paired bootstrap confidence interval, and a sign flip test of zero mean difference.

    AuditKIT 1.1.3's `paired_bootstrap` is not used for significance: it never centres the
    differences under the null, so its p value stays near 0.5 whatever the effect size (see
    docs/toolkit_integration.md). AuditKIT's `score_pairs` is still used to align runs by ID.
    """
    d = np.array([c - b for (_, b), (_, c) in zip(baseline, candidate)], dtype=float)
    rng = np.random.default_rng(seed)
    n, observed = len(d), d.mean()
    idx = rng.integers(0, n, size=(resamples, n))
    means = d[idx].mean(1)
    flips = rng.choice([-1.0, 1.0], size=(resamples, n))
    null = (flips * d).mean(1)
    p = (np.sum(np.abs(null) >= abs(observed) - 1e-12) + 1) / (resamples + 1)
    return dict(n=n, delta=float(observed), ci_low=float(np.quantile(means, 0.025)), ci_high=float(np.quantile(means, 0.975)), p_value=float(p))


def compare(baseline, candidate, metric: str = "utility", resamples: int = 2000) -> dict:
    """Aligns two replayed runs by sample ID with AuditKIT, then tests the paired difference."""
    from auditkit._bootstrap import score_pairs

    pairs = score_pairs(baseline, candidate, metric)
    return paired_test([("", b) for b, _ in pairs], [("", c) for _, c in pairs], resamples=resamples)
