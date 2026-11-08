"""Development only: small MLP against the linear probe, feature scaling, and training set size.

Uses the position and layer already chosen by eval_probes.py for each fold, so nothing here is
selected on the rows it is scored on. Test rows are dropped on load.
"""
from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from knowwhen.evaluation.data import load_frame
from knowwhen.evaluation.metrics import calibration
from knowwhen.paths import RESULTS, inside_project
from knowwhen.probes.linear import LinearProbe, Temperature
from knowwhen.probes.select import feats, macro_auroc

DATASETS = ["nq", "triviaqa", "hotpotqa", "popqa"]
warnings.filterwarnings("ignore")


def mlp_scores(xt, yt, xv, yv, xe, seed):
    """Width and penalty are picked on validation rows, then temperature is fitted there too."""
    s = StandardScaler().fit(xt)
    xt, xv, xe = s.transform(xt), s.transform(xv), s.transform(xe)
    best = None
    for width in (32, 128):
        for alpha in (1.0, 30.0):
            clf = MLPClassifier((width,), alpha=alpha, early_stopping=True, validation_fraction=0.1, n_iter_no_change=5, max_iter=60,
                                batch_size=128, learning_rate_init=1e-3, random_state=seed).fit(xt, yt)
            score = -np.mean(np.logaddexp(0, logit(clf, xv)) - yv * logit(clf, xv))
            if best is None or score > best[0]:
                best = (score, clf, width, alpha)
    _, clf, width, alpha = best
    t = Temperature().fit(logit(clf, xv), yv)
    return t(logit(clf, xe)), dict(width=width, alpha=alpha)


def logit(clf, x):
    p = np.clip(clf.predict_proba(x)[:, 1], 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chosen", required=True, help="json written by eval_probes.py")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    chosen = json.load(open(args.chosen))["chosen"]
    f = load_frame("closed")
    f = f[f.split != "test"].reset_index(drop=True)
    v = f[f.split == "val"]
    half = np.arange(len(v)) % 2 == 0
    train, val, test = f[f.split == "train"], v[half], v[~half]

    rows = []
    for fold, info in chosen.items():
        held = fold.replace("without_", "")
        tr = train if fold == "in_domain" else train[train.dataset != held]
        va = val if fold == "in_domain" else val[val.dataset != held]
        ev = test if fold == "in_domain" else test[test.dataset == held]
        X = lambda d: feats("closed", info["position"], info["layer_index"], d.row.values)
        xt, xv, xe = X(tr), X(va), X(ev)
        yt, yv, ye = (d.correct.values.astype(int) for d in (tr, va, ev))
        add = lambda name, seed, p, **k: rows.append(dict(variant=name, fold=fold, seed=seed, macro_auroc=macro_auroc(ye, p, ev.dataset.values), **calibration(ye, p), **k))
        for seed in range(args.seeds):
            add("linear", seed, LinearProbe(c=info["c"], seed=seed).fit(xt, yt).calibrate(xv, yv).predict(xe))
            l2 = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
            add("linear_l2_rows", seed, LinearProbe(c=info["c"], seed=seed).fit(l2(xt), yt).calibrate(l2(xv), yv).predict(l2(xe)))
            p, extra = mlp_scores(xt, yt, xv, yv, xe, seed)
            add("mlp", seed, p, **extra)
            order = np.random.default_rng(seed).permutation(len(xt))
            for n in (500, 1000, 2500, 5000):
                i = order[:n]
                add(f"linear_n{n}", seed, LinearProbe(c=info["c"], seed=seed).fit(xt[i], yt[i]).calibrate(xv, yv).predict(xe))
            print(fold, seed, flush=True)

    out = pd.DataFrame(rows)
    d = inside_project(RESULTS / "exploratory")
    out.to_csv(d / f"probe_variants_dev_{args.tag}.csv", index=False)
    pd.set_option("display.width", 170)
    print(out.groupby(["variant", "fold"]).macro_auroc.mean().unstack().round(3).to_string())
    print(out.groupby(["variant", "fold"]).ece.mean().unstack().round(3).to_string())


if __name__ == "__main__":
    main()
