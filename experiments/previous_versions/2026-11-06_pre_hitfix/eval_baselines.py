"""Development only: P(Yes) and self consistency against probes, on closed book correctness.

Two comparisons. Before generation: P(Yes) against a probe on the last prompt token, also before
any answer exists. After generation: self consistency against the chosen activation probe.
Test rows are dropped on load.
"""
from __future__ import annotations

import argparse
import collections
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from knowwhen.evaluation.data import MODEL, layers, load_frame
from knowwhen.paths import ARTIFACTS, RESULTS, inside_project
from knowwhen.probes.linear import LinearProbe
from knowwhen.probes.select import feats, macro_auroc, select_probe
from knowwhen.text import normalize

DATASETS = ["nq", "triviaqa", "hotpotqa", "popqa"]


def read(name, key):
    return {json.loads(l)["id"]: json.loads(l)[key] for l in open(ARTIFACTS / "runs" / MODEL / name)}


def agreement(greedy, samples):
    """Share of samples matching the greedy answer, and share held by the most common sample."""
    norm = [normalize(s) for s in samples]
    top = collections.Counter(norm).most_common(1)[0][1] / len(norm)
    return sum(n == normalize(greedy) for n in norm) / len(norm), top


def pre_probe(tr, va, seed):
    y, yv, g = tr.correct.values.astype(int), va.correct.values.astype(int), va.dataset.values
    best = max(((macro_auroc(yv, LinearProbe(c=1e-3, seed=seed).fit(feats("closed", "pre", li, tr.row.values), y).logits(feats("closed", "pre", li, va.row.values)), g), li)
                for li in range(1, len(layers()), 2)))
    li = best[1]
    return LinearProbe(c=1e-3, seed=seed).fit(feats("closed", "pre", li, tr.row.values), y), li


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    f = load_frame("closed")
    f = f[f.split != "test"].reset_index(drop=True)
    f["p_yes"] = f["id"].map(read("p_yes.jsonl", "p_yes"))
    samples = read("samples.jsonl", "samples")
    both = [agreement(a, samples[i]) if i in samples else (np.nan, np.nan) for i, a in zip(f["id"], f["answer"])]
    f["agree_greedy"], f["agree_top"] = zip(*both)
    v = f[f.split == "val"]
    half = np.arange(len(v)) % 2 == 0
    train, val, test = f[f.split == "train"], v[half], v[~half]

    rows = []
    for fold in ["in_domain"] + [f"without_{d}" for d in DATASETS]:
        held = fold.replace("without_", "")
        tr = train if fold == "in_domain" else train[train.dataset != held]
        va = val if fold == "in_domain" else val[val.dataset != held]
        ev = test if fold == "in_domain" else test[test.dataset == held]
        y = ev.correct.values.astype(int)
        pre, li = pre_probe(tr, va, args.seed)
        post, info = select_probe(tr, va, "closed", args.seed)
        scores = dict(pre_probe=pre.logits(feats("closed", "pre", li, ev.row.values)), p_yes=ev.p_yes.values, activation_probe=post.logits(feats("closed", info["position"], info["layer_index"], ev.row.values)),
                      agree_greedy=ev.agree_greedy.values, agree_top=ev.agree_top.values)
        for name, s in scores.items():
            ok = ~np.isnan(s)
            rows.append(dict(method=name, fold=fold, n=int(ok.sum()), macro_auroc=macro_auroc(y[ok], s[ok], ev.dataset.values[ok]), pooled_auroc=roc_auc_score(y[ok], s[ok])))
        print(fold, flush=True)
    out = pd.DataFrame(rows)
    d = inside_project(RESULTS / "exploratory")
    out.to_csv(d / f"baselines_dev_{args.tag}.csv", index=False)
    print(out.pivot(index="method", columns="fold", values="macro_auroc").round(3).to_string())
    print(out[out.fold == "in_domain"][["method", "n", "pooled_auroc"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
