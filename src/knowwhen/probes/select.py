"""Choosing a probe on validation data only. Shared by the probe and policy evaluations."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from knowwhen.evaluation.data import activations, layers
from knowwhen.probes.linear import LinearProbe

POSITIONS = ["pre", "last", "mean"]
STATS = ["mean_logprob", "min_logprob", "sum_logprob", "first_entropy", "mean_entropy", "max_entropy", "n_tokens"]
C_GRID = [1e-4, 1e-3, 1e-2]


def macro_auroc(y, score, groups) -> float:
    """Mean AUROC within each dataset, so that difficulty gaps between datasets do not count as skill."""
    vals = [roc_auc_score(y[groups == g], score[groups == g]) for g in np.unique(groups) if 0 < y[groups == g].mean() < 1]
    return float(np.mean(vals))


def feats(condition, position, li, rows):
    return np.asarray(activations(condition, position)[rows, li], dtype=np.float32)


def select_probe(tr, va, condition, seed):
    """Picks position, layer and strength by within dataset AUROC on validation rows.

    `tr` and `va` need columns row, correct and dataset. The label may come from a different
    condition than the activations, which is how the pre retrieval forecast is trained.
    """
    best = (-1.0, None)
    y, yv, g = tr.correct.values.astype(int), va.correct.values.astype(int), va.dataset.values
    for pos in POSITIONS:
        for li in range(1, len(layers()), 2):
            xt, xv = feats(condition, pos, li, tr.row.values), feats(condition, pos, li, va.row.values)
            s = macro_auroc(yv, LinearProbe(c=1e-3, seed=seed).fit(xt, y).logits(xv), g)
            if s > best[0]:
                best = (s, (pos, li))
    pos, li = best[1]
    xt, xv = feats(condition, pos, li, tr.row.values), feats(condition, pos, li, va.row.values)
    scored = [(macro_auroc(yv, LinearProbe(c=c, seed=seed).fit(xt, y).logits(xv), g), c) for c in C_GRID]
    score, c = max(scored)
    probe = LinearProbe(c=c, seed=seed).fit(xt, y).calibrate(xv, yv)
    return probe, dict(position=pos, layer=layers()[li], layer_index=li, c=c, val_macro_auroc=score)


def stat_probe(tr):
    scaler = StandardScaler().fit(tr[STATS].values)
    clf = LogisticRegression(C=1.0, max_iter=500).fit(scaler.transform(tr[STATS].values), tr.correct.values.astype(int))
    return scaler, clf
