"""Planned check: does the probe only read how famous the subject is? PopQA test rows by popularity.

Uses saved test predictions, so nothing is refit. Quartiles come from page views alone, no labels.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from knowwhen.paths import ARTIFACTS, DATA, RESULTS, inside_project


def auc(y, s):
    return roc_auc_score(y, s) if 0 < y.mean() < 1 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    pop = {q["id"]: q["meta"]["s_pop"] for q in map(json.loads, open(DATA / "processed" / "questions.jsonl")) if q["dataset"] == "popqa"}
    p = pd.read_pickle(ARTIFACTS / "policy" / f"final_{args.tag}_seed0.pkl")
    rows = []
    for fold in ("in_domain", "without_popqa"):
        e = p[(p.fold == fold) & (p.role == "eval") & (p.dataset == "popqa")].copy()
        e["pop"] = np.log1p(e["id"].map(pop).astype(float))
        e["quartile"] = pd.qcut(e["pop"], 4, labels=["q1_rare", "q2", "q3", "q4_famous"])
        for name, g in [("all", e)] + [(str(k), v) for k, v in e.groupby("quartile", observed=True)]:
            y0, y1 = (g.y0 == 1).astype(int).values, (g.y1 == 1).astype(int).values
            rows.append(dict(fold=fold, quartile=name, n=len(g), closed_acc=y0.mean(), rag_acc=y1.mean(), closed_probe=auc(y0, g.p0.values), closed_stats=auc(y0, g.t0.values), closed_popularity=auc(y0, g["pop"].values),
                             rag_probe=auc(y1, g.p_after.values), rag_stats=auc(y1, g.t_after.values), rag_popularity=auc(y1, g["pop"].values)))
    t = pd.DataFrame(rows)
    t.to_csv(inside_project(RESULTS / "final") / f"{args.tag}_popqa_popularity.csv", index=False)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(args.tag)
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
