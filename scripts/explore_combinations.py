"""Exploratory combinations from the literature, on development rows only. See docs/analysis_plan.md.

Each variant gives a calibrated chance that the closed answer and the RAG answer are right. The
question then goes to the likelier answer, and the top share by confidence is answered. Probes are
fitted on train rows, and stackers, bands and cuts on validation rows. The test split is never loaded.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from eval_policy import DATASETS, load  # noqa: E402

from knowwhen.evaluation.audit import paired_test  # noqa: E402
from knowwhen.evaluation.data import MODEL, layers, passage_file  # noqa: E402
from knowwhen.evaluation.metrics import calibration  # noqa: E402
from knowwhen.paths import DATA, RESULTS, inside_project  # noqa: E402
from knowwhen.policies.combine import conformal_cut, cross_path_features, em_prior, shift_prior  # noqa: E402
from knowwhen.policies.decision import ABSTAIN, ANSWER, RETRIEVE_ABSTAIN, RETRIEVE_ANSWER, RETRIEVE_FALLBACK, Costs, resolve, score  # noqa: E402
from knowwhen.probes.linear import LinearProbe  # noqa: E402
from knowwhen.probes.select import STATS, feats, macro_auroc  # noqa: E402

BAND_LAYERS = [8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
BANDS = {"L16": [16], "L14_18": [14, 16, 18], "L12_20": [12, 14, 16, 18, 20], "L8_28": BAND_LAYERS}
COVERAGES, ALPHAS = [0.5, 0.7, 0.9], [0.2, 0.3, 0.4]
COSTS = Costs()
VARIANTS = ["single_layer", "layer_band", "stats_only", "cross_path_only", "stats_cross", "band_stats", "band_stats_cross", "all_with_delta"]


def stack(cols, va, ev, y, seed=0):
    """Calibrated logistic combiner, fitted on validation rows only."""
    xv, xe = (np.column_stack([c[i] for c in cols]) for i in (va, ev))
    s = StandardScaler().fit(xv)
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed).fit(s.transform(xv), y[va])
    return clf.predict_proba(s.transform(xv))[:, 1], clf.predict_proba(s.transform(xe))[:, 1]


def layer_logits(cond, rows, tr, y):
    out = {}
    for layer in BAND_LAYERS:
        li = layers().index(layer)
        x = feats(cond, "mean", li, rows)
        out[layer] = LinearProbe(c=1e-3).fit(x[tr], y[tr]).logits(x)
    return out


def best_band(logits, va, y, groups):
    scored = {k: macro_auroc(y[va], np.mean([logits[l][va] for l in v], axis=0), groups[va]) for k, v in BANDS.items()}
    name = max(scored, key=scored.get)
    return name, np.mean([logits[l] for l in BANDS[name]], axis=0), np.asarray(logits[16])


def fold_scores(c, r, xp, tr, va, ev):
    """Index arrays are positions into the fold's own rows. Returns estimates per variant for val and eval."""
    rows = c.row.values
    y0, y1 = (c.correct.values.astype(int), c.correct1.values.astype(int))
    g = c.dataset.values
    li = layers().index(16)
    delta = feats("rag", "mean", li, rows) - feats("closed", "mean", li, rows)
    stats = np.hstack([c[STATS].values, r[STATS].values])
    cols_stats = [stats[:, j] for j in range(stats.shape[1])]
    cols_cross = [xp[:, j] for j in range(xp.shape[1])]
    est, info = {}, {}
    for key, cond, y in (("closed", "closed", y0), ("rag", "rag", y1)):
        logits = layer_logits(cond, rows, tr, y)
        band_name, band, single = best_band(logits, va, y, g)
        dl = LinearProbe(c=1e-3).fit(delta[tr], y[tr]).logits(delta)
        info[key] = band_name
        for name, cols in (("single_layer", [single]), ("layer_band", [band]), ("stats_only", cols_stats), ("cross_path_only", cols_cross),
                           ("stats_cross", cols_stats + cols_cross), ("band_stats", [band] + cols_stats),
                           ("band_stats_cross", [band] + cols_stats + cols_cross), ("all_with_delta", [band, dl] + cols_stats + cols_cross)):
            est[(name, key)] = stack(cols, va, ev, y)
    return est, info


def route(p0, p1):
    use_rag = p1 > p0
    return use_rag, np.where(use_rag, p1, p0)


def actions(use_rag, keep):
    """Both answers were needed to route, so every path here has paid for retrieval."""
    return np.where(keep, np.where(use_rag, RETRIEVE_ANSWER, RETRIEVE_FALLBACK), RETRIEVE_ABSTAIN)


def main():
    c, r, train, val, test = load(False)
    assert not (c.split == "test").any()
    passages = {}
    for line in open(DATA / "processed" / passage_file()):
        q = json.loads(line)
        passages[q["id"]] = " ".join(p["text"] for p in q["passages"][:3])
    xp = cross_path_features(c.answer.tolist(), r.answer.tolist(), [passages[i] for i in c.id])
    print("rows", len(train), len(val), len(test), "model", MODEL, flush=True)

    det, cov, tests, conf_rows, em_rows, matched, matched_tests = [], [], [], [], [], [], []
    for fold in ["in_domain"] + [f"without_{d}" for d in DATASETS]:
        t = time.time()
        held = fold.replace("without_", "")
        ds = c.dataset.values
        if fold == "in_domain":
            tr, va, ev, hv = train.values, val.values, test.values, np.array([], int)
        else:
            tr, va = train.values[ds[train] != held], val.values[ds[val] != held]
            ev, hv = test.values[ds[test] == held], val.values[ds[val] == held]
        est, info = fold_scores(c, r, xp, tr, va, ev)
        y0v, y1v, y0e, y1e = c.y0.values[va], c.y1.values[va], c.y0.values[ev], c.y1.values[ev]
        ge = ds[ev]
        for name in VARIANTS:
            for key, y in (("closed", c.correct.values[ev].astype(int)), ("rag", c.correct1.values[ev].astype(int))):
                p = est[(name, key)][1]
                det.append(dict(fold=fold, variant=name, answer=key, macro_auroc=macro_auroc(y, p, ge), **calibration(y, p)))
        outcomes = {}
        for target in COVERAGES:
            for name in VARIANTS:
                rv, cv = route(est[(name, "closed")][0], est[(name, "rag")][0])
                re_, ce = route(est[(name, "closed")][1], est[(name, "rag")][1])
                a = actions(re_, ce >= np.quantile(cv, 1 - target))
                outcomes[(name, target)] = a
                cov.append(dict(fold=fold, target=target, system=f"route_{name}", **vars(score(a, y0e, y1e, COSTS))))
            lp = c.mean_logprob.values
            cut = np.quantile(lp[va], 1 - target)
            a = np.where(lp[ev] >= cut, ANSWER, ABSTAIN)
            outcomes[("closed_logprob", target)] = a
            cov.append(dict(fold=fold, target=target, system="closed_logprob", **vars(score(a, y0e, y1e, COSTS))))
            lr = r.mean_logprob.values
            a = np.where(lr[ev] >= np.quantile(lr[va], 1 - target), RETRIEVE_ANSWER, RETRIEVE_ABSTAIN)
            outcomes[("rag_logprob", target)] = a
            cov.append(dict(fold=fold, target=target, system="rag_logprob", **vars(score(a, y0e, y1e, COSTS))))
            wrong = lambda k: list(zip(c.id.values[ev], (resolve(outcomes[(k, target)], y0e, y1e)[0] == 0).astype(float)))
            for cand, base in (("band_stats_cross", "stats_cross"), ("band_stats_cross", "rag_logprob"), ("band_stats_cross", "closed_logprob"),
                               ("all_with_delta", "band_stats_cross"), ("stats_cross", "stats_only"), ("layer_band", "single_layer")):
                tests.append(dict(fold=fold, target=target, candidate=cand, baseline=base, **paired_test(wrong(base), wrong(cand))))
        # Same share answered by every system, so wrong rates compare fairly. Each cut is a quantile of
        # that system's own unlabelled scores on these rows, so no labels are involved.
        for target in COVERAGES:
            per = {}
            for name in VARIANTS:
                re_, ce = route(est[(name, "closed")][1], est[(name, "rag")][1])
                per[name] = actions(re_, ce >= np.quantile(ce, 1 - target))
            lr = r.mean_logprob.values[ev]
            per["rag_logprob"] = np.where(lr >= np.quantile(lr, 1 - target), RETRIEVE_ANSWER, RETRIEVE_ABSTAIN)
            lc = c.mean_logprob.values[ev]
            per["closed_logprob"] = np.where(lc >= np.quantile(lc, 1 - target), ANSWER, ABSTAIN)
            for name, a in per.items():
                matched.append(dict(fold=fold, target=target, system=name, **vars(score(a, y0e, y1e, COSTS))))
            wrong = lambda k: list(zip(c.id.values[ev], (resolve(per[k], y0e, y1e)[0] == 0).astype(float)))
            for cand, base in (("band_stats_cross", "stats_cross"), ("band_stats", "stats_only"), ("band_stats_cross", "rag_logprob"), ("band_stats_cross", "single_layer"),
                               ("all_with_delta", "band_stats_cross"), ("layer_band", "single_layer"), ("band_stats_cross", "band_stats")):
                matched_tests.append(dict(fold=fold, target=target, candidate=cand, baseline=base, **paired_test(wrong(base), wrong(cand))))
        # Conformal cut on the routed confidence: does the promised bound on wrong answers hold?
        for name in ("band_stats_cross", "stats_cross"):
            rv, cv = route(est[(name, "closed")][0], est[(name, "rag")][0])
            re_, ce = route(est[(name, "closed")][1], est[(name, "rag")][1])
            wrong_v = np.where(rv, y1v == 0, y0v == 0)
            for alpha in ALPHAS:
                keep = ce >= conformal_cut(cv, wrong_v, alpha)
                o = score(actions(re_, keep), y0e, y1e, COSTS)
                conf_rows.append(dict(fold=fold, variant=name, alpha=alpha, wrong_rate=o.wrong_rate, coverage=o.coverage, selective_accuracy=o.selective_accuracy, bound_held=o.wrong_rate <= alpha))
        # Label free prior correction on the held out dataset, using its unlabelled validation rows.
        if len(hv):
            name = "band_stats_cross"
            for key, label in (("closed", "correct"), ("rag", "correct1")):
                yv = c[label].values[va].astype(int)
                ye = c[label].values[ev].astype(int)
                pv, pe = est[(name, key)]
                # Eval scores are used without their labels. A deployed system would see these rows unlabelled.
                em_rows.append(dict(fold=fold, answer=key, source_prior=float(yv.mean()), eval_prior=float(ye.mean()), estimated_prior=em_prior(pe, float(yv.mean())),
                                    ece_before=calibration(ye, pe)["ece"], ece_after=calibration(ye, shift_prior(pe, float(yv.mean()), em_prior(pe, float(yv.mean()))))["ece"]))
        print(fold, info, f"{time.time() - t:.0f}s", flush=True)

    d = inside_project(RESULTS / "exploratory")
    d.mkdir(parents=True, exist_ok=True)
    frames = dict(detection=pd.DataFrame(det), coverage=pd.DataFrame(cov), tests=pd.DataFrame(tests), conformal=pd.DataFrame(conf_rows), prior=pd.DataFrame(em_rows),
                  matched=pd.DataFrame(matched), matched_tests=pd.DataFrame(matched_tests))
    for k, v in frames.items():
        v.to_csv(d / f"combinations_dev_{MODEL}_{k}.csv", index=False)
    pd.set_option("display.width", 190, "display.max_columns", 30, "display.max_rows", 200)
    x = frames["detection"]
    print(x[x.answer == "rag"].pivot(index="variant", columns="fold", values="macro_auroc").round(3).loc[VARIANTS])
    print(x[x.answer == "closed"].pivot(index="variant", columns="fold", values="macro_auroc").round(3).loc[VARIANTS])
    y = frames["coverage"]
    for target in COVERAGES:
        print("target", target)
        print(y[y.target == target].pivot(index="system", columns="fold", values="wrong_rate").round(3))
        print(y[y.target == target].pivot(index="system", columns="fold", values="coverage").round(3))
    print(frames["tests"][frames["tests"].target == 0.7].round(4).to_string(index=False))
    z = frames["matched"]
    for target in COVERAGES:
        print("matched", target)
        print(z[z.target == target].pivot(index="system", columns="fold", values="wrong_rate").round(3))
    print(frames["matched_tests"][frames["matched_tests"].target == 0.7].round(4).to_string(index=False))
    print(frames["conformal"].round(3).to_string(index=False))
    print(frames["prior"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
