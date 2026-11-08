"""Added after the test run: unseen datasets at exactly matched coverage.

The planned cut is a quantile of validation rows from other datasets, so systems reach different
coverage on a held out dataset. Here the cut is a quantile of each system's own confidence on the
evaluation rows. That uses no labels, but it needs the whole unlabelled batch, so it is a fair
comparison of ranking and not something a deployed system gets for free.
"""
from __future__ import annotations

import argparse

import pandas as pd

from knowwhen.evaluation.audit import paired_test
from knowwhen.paths import ARTIFACTS, RESULTS, inside_project
from knowwhen.policies.decision import Costs, resolve, score
from knowwhen.policies.sweep import coverage_policies

BASES = ("cov_closed_probe", "cov_rag_probe", "cov_route_token_stats")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    p = pd.read_pickle(ARTIFACTS / "policy" / f"final_{args.tag}_seed0.pkl")
    rows, tests = [], []
    for fold, f in p.groupby("fold", sort=False):
        e = f[f.role == "eval"]
        y0, y1 = e.y0.values, e.y1.values
        for target in (0.5, 0.7, 0.9):
            acts = coverage_policies(e, e, target)
            for name, a in acts.items():
                rows.append(dict(fold=fold, target_coverage=target, policy=name, **vars(score(a, y0, y1, Costs(wrong=2.0, abstain=0.0, retrieve=0.1)))))
            wrong = lambda n: list(zip(e["id"], (resolve(acts[n], y0, y1)[0] == 0).astype(float)))
            for b in BASES:
                tests.append(dict(fold=fold, target_coverage=target, baseline=b, **paired_test(wrong(b), wrong("cov_route_probe"))))
    d = inside_project(RESULTS / "final")
    t, g = pd.DataFrame(rows), pd.DataFrame(tests)
    t.to_csv(d / f"{args.tag}_matched_unseen.csv", index=False)
    g.to_csv(d / f"{args.tag}_matched_unseen_tests.csv", index=False)
    pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_rows", 100)
    print(args.tag)
    x = t[t.target_coverage == 0.7]
    print(x.pivot(index="fold", columns="policy", values="wrong_rate").round(3).to_string())
    print(x.pivot(index="fold", columns="policy", values="coverage").round(3).to_string())
    print(g[g.target_coverage.isin([0.7, 0.9])][["fold", "target_coverage", "baseline", "delta", "ci_low", "ci_high"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
