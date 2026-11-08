"""Averages the test run over seeds and applies Holm's correction to the coverage family.

The family is fixed in docs/analysis_plan.md: routing against three rivals at 70 and 90% coverage,
in domain, on the wrong answer share, corrected per model and per seed. A test only counts as
confirmed if it holds on every seed, so the largest adjusted p value across seeds is reported.
"""
from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd

from knowwhen.paths import RESULTS, inside_project

RIVALS = ["cov_closed_probe", "cov_rag_probe", "cov_route_token_stats"]


def holm(p):
    """Holm adjusted p values, in the input order."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]).clip(max=1)
    out = np.empty_like(adj)
    out[order] = adj
    return out


def seeds(pattern):
    """Every file matching a results pattern with {seed} in it, with the seed as a column."""
    d = RESULTS / pattern.split("/")[0]
    rx = re.compile(pattern.split("/")[1].replace("{seed}", r"(\d+)") + "$")
    hits = [(f, rx.match(f.name)) for f in sorted(d.iterdir()) if not f.name.startswith("._")]
    return pd.concat([pd.read_csv(f).assign(seed=int(m[1])) for f, m in hits if m])


def spread(frame, keys, values):
    g = frame.groupby(keys, sort=False)[values]
    out = g.agg(["mean", "std", "count"]).round(4)
    return out.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["t4_rerank", "aya8b_rerank"])
    args = ap.parse_args()
    out = inside_project(RESULTS / "final")
    out.mkdir(exist_ok=True)
    pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_rows", 300)
    for tag in args.tags:
        t = seeds(f"policy/final_{tag}_seed{{seed}}_coverage_tests.csv")
        fam = t[(t.fold == "in_domain") & t.target_coverage.isin([0.7, 0.9]) & t.baseline.isin(RIVALS)].copy()
        fam["p_holm"] = fam.groupby("seed").p_value.transform(holm)
        table = fam.groupby(["target_coverage", "baseline"], sort=False).agg(
            delta=("delta", "mean"), ci_low=("ci_low", "mean"), ci_high=("ci_high", "mean"), worst_p_holm=("p_holm", "max"), seeds=("seed", "nunique")).reset_index()
        table["confirmed"] = table.worst_p_holm < 0.05
        table.to_csv(out / f"{tag}_coverage_family.csv", index=False)
        print(tag, "\n", table.round(4).to_string(index=False))

        spread(seeds(f"policy/final_{tag}_seed{{seed}}_coverage.csv"), ["fold", "target_coverage", "policy"], "wrong_rate").to_csv(out / f"{tag}_coverage_wrong.csv", index=False)
        c = seeds(f"policy/final_{tag}_seed{{seed}}_coverage.csv").groupby(["fold", "target_coverage", "policy"], sort=False)[["coverage", "wrong_rate", "correct_rate", "selective_accuracy", "retrieval_rate", "utility"]].mean().round(4).reset_index()
        c.to_csv(out / f"{tag}_coverage_mean.csv", index=False)
        p = seeds(f"policy/final_{tag}_seed{{seed}}_policies.csv")
        p.groupby(["fold", "group", "wrong_cost", "retrieve_cost", "policy"], sort=False).mean(numeric_only=True).round(4).reset_index().to_csv(out / f"{tag}_policies_mean.csv", index=False)
        a = seeds(f"policy/final_{tag}_seed{{seed}}_audit_tests.csv")
        a.groupby(["fold", "candidate", "baseline", "metric"], sort=False).agg(delta=("delta", "mean"), ci_low=("ci_low", "mean"), ci_high=("ci_high", "mean"), worst_p=("p_value", "max")).round(4).reset_index().to_csv(out / f"{tag}_primary_tests.csv", index=False)
        for cond in ("closed", "rag"):
            d = seeds(f"probes/{cond}_final_{tag}_seed{{seed}}.csv")
            d.groupby(["fold", "method"], sort=False).mean(numeric_only=True).round(4).reset_index().to_csv(out / f"{tag}_probes_{cond}_mean.csv", index=False)
            print(cond, "seed spread of macro AUROC, largest:", d.groupby(["fold", "method"]).macro_auroc.std().max().round(4))
        try:
            b = seeds(f"exploratory/baselines_final_{tag}_seed{{seed}}.csv")
            b.groupby(["fold", "method"], sort=False).mean(numeric_only=True).round(4).reset_index().to_csv(out / f"{tag}_baselines_mean.csv", index=False)
            g = seeds(f"exploratory/baselines_final_{tag}_seed{{seed}}_gaps.csv")
            g.groupby(["fold", "candidate", "baseline"], sort=False).mean(numeric_only=True).round(4).reset_index().to_csv(out / f"{tag}_baselines_gaps_mean.csv", index=False)
        except ValueError:
            print("no baselines for", tag)


if __name__ == "__main__":
    main()
