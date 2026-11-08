"""Probe evaluation following docs/analysis_plan.md.

Default is development mode: test rows are dropped on load, and half of validation stands in for
test. `--final` uses the real test split and should run once the design is frozen. Selection and
calibration only ever see the training datasets of each fold.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from knowwhen.evaluation.data import load_frame
from knowwhen.evaluation.metrics import aurc, calibration, detection
from knowwhen.paths import RESULTS, inside_project
from knowwhen.probes.linear import LinearProbe
from knowwhen.probes.select import STATS, feats, macro_auroc, select_probe, stat_probe

DATASETS = ["nq", "triviaqa", "hotpotqa", "popqa"]


def report(name, fold, ev, p):
    y = ev.correct.values.astype(int)
    out = dict(method=name, fold=fold, n=len(ev), macro_auroc=macro_auroc(y, p, ev.dataset.values), aurc=aurc(y, p),
               **detection(y, p), **calibration(y, p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--condition", default="closed")
    ap.add_argument("--tag", default="dev")
    args = ap.parse_args()
    f = load_frame(args.condition)
    if args.final:
        train, val, test = f[f.split == "train"], f[f.split == "val"], f[f.split == "test"]
    else:
        f = f[f.split != "test"].reset_index(drop=True)
        v = f[f.split == "val"]
        half = np.arange(len(v)) % 2 == 0
        train, val, test = f[f.split == "train"], v[half], v[~half]
    print(f"train {len(train)} val {len(val)} eval {len(test)} final={args.final}", flush=True)

    rows, chosen = [], {}
    for fold in ["in_domain"] + [f"without_{d}" for d in DATASETS]:
        held = fold.replace("without_", "")
        tr = train if fold == "in_domain" else train[train.dataset != held]
        va = val if fold == "in_domain" else val[val.dataset != held]
        ev = test if fold == "in_domain" else test[test.dataset == held]
        t = time.time()
        probe, info = select_probe(tr, va, args.condition, args.seed)
        chosen[fold] = info
        x = feats(args.condition, info["position"], info["layer_index"], ev.row.values)
        p = probe.predict(x)
        rows.append(report("activation_probe", fold, ev, p))
        # Activations plus token statistics, using the same position and layer.
        s = StandardScaler().fit(tr[STATS].values)
        comb = lambda d: np.hstack([feats(args.condition, info["position"], info["layer_index"], d.row.values), s.transform(d[STATS].values) * 10])
        both = LinearProbe(c=info["c"], seed=args.seed).fit(comb(tr), tr.correct.values.astype(int)).calibrate(comb(va), va.correct.values.astype(int))
        rows.append(report("activation_plus_stats", fold, ev, both.predict(comb(ev))))
        sc, clf = stat_probe(tr)
        rows.append(report("token_stats_logistic", fold, ev, clf.predict_proba(sc.transform(ev[STATS].values))[:, 1]))
        for col, sign in (("mean_logprob", 1), ("min_logprob", 1), ("mean_entropy", -1), ("first_entropy", -1)):
            # Ranks are what matter for these unsupervised scores; a logistic map keeps them in (0, 1).
            rows.append(report(col, fold, ev, 1 / (1 + np.exp(-sign * ev[col].values))))
        print(fold, info, f"{time.time() - t:.0f}s", flush=True)

    # Source check: can the same features name the dataset? High accuracy is a warning sign.
    info = chosen["in_domain"]
    xt, xe = feats(args.condition, info["position"], info["layer_index"], train.row.values), feats(args.condition, info["position"], info["layer_index"], test.row.values)
    sc = StandardScaler().fit(xt)
    src = LogisticRegression(C=1e-3, max_iter=300).fit(sc.transform(xt), train.dataset.values)
    source_acc = float((src.predict(sc.transform(xe)) == test.dataset.values).mean())

    out = pd.DataFrame(rows)
    pd.set_option("display.width", 170)
    print(out.pivot(index="method", columns="fold", values="macro_auroc").round(3))
    print(out[out.fold == "in_domain"][["method", "macro_auroc", "auroc", "auprc", "ece", "brier", "aurc"]].round(3).to_string(index=False))
    print("source prediction accuracy", round(source_acc, 3))
    d = inside_project(RESULTS / "probes")
    d.mkdir(parents=True, exist_ok=True)
    stem = f"{args.condition}_{'final' if args.final else 'dev'}_{args.tag}_seed{args.seed}"
    out.to_csv(d / f"{stem}.csv", index=False)
    (d / f"{stem}.json").write_text(json.dumps(dict(chosen=chosen, source_accuracy=source_acc, final=args.final), indent=1))


if __name__ == "__main__":
    main()
