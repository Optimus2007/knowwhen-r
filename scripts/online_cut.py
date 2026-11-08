"""Added after the test run: a label free answer cut that follows the incoming questions.

The planned cut is a quantile of validation confidence from other datasets, so on a new dataset a
70% target landed anywhere from 56% to 83%. Here questions arrive one at a time. The cut for each
question is the (1 - target) quantile of the confidence of the previous W questions, so it uses no
labels and never looks at the question it is deciding on. The stream starts with validation rows
from the training datasets and then switches to the new dataset, which is the shift a deployed
system would see. Results are averaged over random stream orders.

Development mode (default) reads validation only tables and sweeps W. The final mode takes the
single W fixed in docs/analysis_plan.md and runs once on the test tables.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from knowwhen.paths import ARTIFACTS, RESULTS, inside_project
from knowwhen.policies.decision import ABSTAIN, ANSWER, RETRIEVE_ABSTAIN, RETRIEVE_ANSWER, Costs, score
from knowwhen.policies.sweep import _route

WINDOWS = (50, 100, 200, 400, 800)
TARGETS = (0.5, 0.7, 0.9)
ORDERS = 20
COSTS = Costs(wrong=2.0, abstain=0.0, retrieve=0.1)


def stream_keep(warm: np.ndarray, stream: np.ndarray, target: float, window: int) -> np.ndarray:
    """Whether to answer each stream item, using only confidences seen before it."""
    history = list(warm[-window:])
    keep = np.empty(len(stream), dtype=bool)
    q = 1 - target
    for i, c in enumerate(stream):
        keep[i] = c >= np.quantile(history[-window:], q)
        history.append(c)
    return keep


def systems(frame: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """For each system: whether it retrieves, and the confidence it answers with."""
    out = {
        "closed_probe": (np.zeros(len(frame), bool), frame.p0.values),
        "rag_probe": (np.ones(len(frame), bool), frame.p_after.values),
    }
    out["route_probe"] = _route(frame, "p0", "p_before", "p_after")
    out["route_token_stats"] = _route(frame, "t0", "t_before", "t_after")
    return out


def actions(routed: np.ndarray, keep: np.ndarray) -> np.ndarray:
    return np.where(routed, np.where(keep, RETRIEVE_ANSWER, RETRIEVE_ABSTAIN), np.where(keep, ANSWER, ABSTAIN))


def run(p: pd.DataFrame, windows, folds=None) -> pd.DataFrame:
    rows = []
    for fold, f in p.groupby("fold", sort=False):
        if folds and fold not in folds:
            continue
        v, e = f[f.role == "val"], f[f.role == "eval"]
        sv, se = systems(v), systems(e)
        y0, y1 = e.y0.values, e.y1.values
        for name in se:
            routed_e, conf_e = se[name]
            conf_v = sv[name][1]
            for target in TARGETS:
                planned = conf_e >= np.quantile(conf_v, 1 - target)
                exact = conf_e >= np.quantile(conf_e, 1 - target)
                for label, keep in (("planned", planned), ("exact_batch", exact)):
                    rows.append(dict(fold=fold, system=name, target=target, cut=label, window=0, order=0,
                                     **vars(score(actions(routed_e, keep), y0, y1, COSTS))))
                for w in windows:
                    for o in range(ORDERS):
                        rng = np.random.default_rng(1000 + o)
                        wi, ei = rng.permutation(len(conf_v)), rng.permutation(len(conf_e))
                        keep = np.empty(len(conf_e), bool)
                        keep[ei] = stream_keep(conf_v[wi], conf_e[ei], target, w)
                        rows.append(dict(fold=fold, system=name, target=target, cut="online", window=w, order=o,
                                         **vars(score(actions(routed_e, keep), y0, y1, COSTS))))
    return pd.DataFrame(rows)


def summarize(t: pd.DataFrame) -> pd.DataFrame:
    t = t.assign(cov_error=(t.coverage - t.target).abs())
    return (t.groupby(["fold", "system", "target", "cut", "window"])
             .agg(coverage=("coverage", "mean"), cov_sd=("coverage", "std"), cov_error=("cov_error", "mean"),
                  wrong_rate=("wrong_rate", "mean"), wrong_sd=("wrong_rate", "std"))
             .reset_index())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--window", type=int, help="the single window fixed in the analysis plan; final mode only")
    args = ap.parse_args()
    if args.final and not args.window:
        ap.error("--final needs the --window fixed in docs/analysis_plan.md")
    stem = "final" if args.final else "dev"
    p = pd.read_pickle(ARTIFACTS / "policy" / f"{stem}_{args.tag}_seed0.pkl")
    windows = (args.window,) if args.final else WINDOWS
    t = run(p, windows)
    s = summarize(t)
    d = inside_project(RESULTS / ("final" if args.final else "exploratory"))
    t.to_csv(d / f"{args.tag}_online_cut_{stem}_raw.csv", index=False)
    s.to_csv(d / f"{args.tag}_online_cut_{stem}.csv", index=False)
    pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_rows", 400)
    unseen = s[s.fold != "in_domain"]
    print(args.tag, stem)
    print("Mean absolute coverage error over unseen folds, targets 0.7 and 0.9")
    x = unseen[unseen.target.isin([0.7, 0.9])]
    print(x.groupby(["system", "cut", "window"])[["cov_error", "cov_sd", "wrong_rate"]].mean().round(4).to_string())
    print(json.dumps({"rows": len(t)}))


if __name__ == "__main__":
    main()
