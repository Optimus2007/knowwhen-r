"""Added after the test run: how much do test results move with the training rows?

Refits the frozen in domain probes on random subsets of the training rows. Layer, position and
strength stay at the saved choice. Nothing here feeds back into the system.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from eval_policy import load  # noqa: E402

from knowwhen.paths import RESULTS, inside_project  # noqa: E402
from knowwhen.policies.decision import Costs, score  # noqa: E402
from knowwhen.policies.sweep import coverage_policies  # noqa: E402
from knowwhen.probes.linear import LinearProbe  # noqa: E402
from knowwhen.probes.select import STATS, feats, stat_probe  # noqa: E402

TARGETS = [0.5, 0.7, 0.9]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--subsets", type=int, default=5)
    ap.add_argument("--share", type=float, default=0.9)
    args = ap.parse_args()
    c, r, train, val, test = load(True)
    chosen = json.load(open(RESULTS / "policy" / f"final_{args.tag}_seed0_chosen.json"))["in_domain"]
    both = val.union(test)
    rows = []
    for s in range(args.subsets):
        rng = np.random.default_rng(100 + s)
        tr = pd.Index(rng.choice(train, int(args.share * len(train)), replace=False))
        out = pd.DataFrame(index=both)
        for pname, tname, cond, frame, label in (("p0", "t0", "closed", c, "correct"), ("p_before", "t_before", "closed", c, "correct1"), ("p_after", "t_after", "rag", r, "correct1")):
            k = chosen[pname]
            y = c[label].values.astype(int)
            x = lambda i: feats(cond, k["position"], k["layer_index"], frame.loc[i, "row"].values)
            probe = LinearProbe(c=k["c"]).fit(x(tr), y[tr]).calibrate(x(val), y[val])
            out[pname] = probe.predict(x(both))
            scaler, clf = stat_probe(frame.loc[tr].assign(correct=y[tr]))
            out[tname] = clf.predict_proba(scaler.transform(frame.loc[both, STATS].values))[:, 1]
        for col in ("y0", "y1", "mean_logprob"):
            out[col] = c.loc[both, col]
        v, e = out.loc[val], out.loc[test]
        for target in TARGETS:
            for name, a in coverage_policies(v, e, target).items():
                rows.append(dict(subset=s, target_coverage=target, policy=name, **vars(score(a, e.y0.values, e.y1.values, Costs(wrong=2.0, abstain=0.0, retrieve=0.1)))))
        print("subset", s, flush=True)
    t = pd.DataFrame(rows)
    d = inside_project(RESULTS / "final")
    d.mkdir(exist_ok=True)
    t.to_csv(d / f"{args.tag}_stability.csv", index=False)
    pd.set_option("display.width", 180, "display.max_rows", 100)
    print(t.groupby(["target_coverage", "policy"], sort=False)[["wrong_rate", "coverage"]].agg(["mean", "std", "min", "max"]).round(4).to_string())
    w = t.pivot_table(index=["target_coverage", "subset"], columns="policy", values="wrong_rate")
    gaps = pd.DataFrame({b: w["cov_route_probe"] - w[b] for b in ("cov_closed_probe", "cov_rag_probe", "cov_route_token_stats")})
    print(gaps.groupby("target_coverage").agg(["mean", "min", "max"]).round(4).to_string())


if __name__ == "__main__":
    main()
