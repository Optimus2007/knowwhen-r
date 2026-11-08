"""Decision policy evaluation following docs/analysis_plan.md.

Stage one fits the three activation probes and three token statistic models per fold and saves
their predictions. Stage two replays every policy at every cost setting from those predictions.
Default is development mode, as in eval_probes.py: test rows are dropped on load and half of
validation stands in for test. `--final` uses the real test split and should run once.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from knowwhen.evaluation.audit import ABSTAIN_TEXT, compare, paired_test, replay
from knowwhen.evaluation.data import load_frame, passage_file
from knowwhen.paths import ARTIFACTS, DATA, RESULTS, inside_project
from knowwhen.policies.decision import ANSWER, DECLINED, RETRIEVE_ANSWER, RETRIEVE_FALLBACK, Costs, per_item_utility, resolve, score
from knowwhen.policies.sweep import COVERAGES, PLANNED, coverage_policies, matched_retrieval, oracle_recovered, policies, retrieval_stats
from knowwhen.probes.select import STATS, feats, select_probe, stat_probe
from knowwhen.text import contains_answer

DATASETS = ["nq", "triviaqa", "hotpotqa", "popqa"]
WRONG, RETRIEVE = [1.0, 2.0, 4.0], [0.05, 0.1, 0.2]
HEADLINE = Costs(wrong=2.0, abstain=0.0, retrieve=0.1)
AGAINST = ["two_cut_probe", "two_cut_logprob", "closed_only", "rag_always"]


def labels(frame: pd.DataFrame) -> np.ndarray:
    return np.where(frame.correct, 1, np.where(frame.refusal, DECLINED, 0))


def load(final: bool):
    """Closed and RAG frames row aligned on the questions present in both, and the three index sets."""
    c, r = load_frame("closed"), load_frame("rag")
    ids = sorted(set(c.id) & set(r.id))
    c, r = (f.set_index("id").loc[ids].reset_index() for f in (c, r))
    assert (c.row == r.row).all() and (c.split == r.split).all()
    if not final:
        keep = (c.split != "test").values
        c, r = c[keep].reset_index(drop=True), r[keep].reset_index(drop=True)
    assert final or not (c.split == "test").any()
    c["y0"], c["y1"] = labels(c), labels(r)
    c["correct1"] = r.correct.values
    passages = {}
    for line in open(DATA / "processed" / passage_file()):
        q = json.loads(line)
        passages[q["id"]] = " ".join(p["text"] for p in q["passages"][:3])
    c["hit"] = [contains_answer(passages[i], g) for i, g in zip(c.id, c.gold)]
    if final:
        return c, r, c.index[c.split == "train"], c.index[c.split == "val"], c.index[c.split == "test"]
    v = c.index[c.split == "val"]
    half = np.arange(len(v)) % 2 == 0
    return c, r, c.index[c.split == "train"], v[half], v[~half]


def recalibrate(p_val: np.ndarray, y_val: np.ndarray, p_ev: np.ndarray) -> np.ndarray:
    """Slope and intercept refit, used only for the diagnostic of what miscalibration costs."""
    z = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))[:, None]
    return LogisticRegression(C=1e6, max_iter=500).fit(z(p_val), y_val).predict_proba(z(p_ev))[:, 1]


def fit_fold(c, r, tr, va, ev, held_val, seed):
    """Predictions for validation and evaluation rows. `held_val` feeds the diagnostic only."""
    both = va.union(ev).union(held_val)
    out = pd.DataFrame(index=both)
    info = {}
    targets = [("p0", "t0", "closed", c, "correct"), ("p_before", "t_before", "closed", c, "correct1"), ("p_after", "t_after", "rag", r, "correct1")]
    for pname, tname, cond, frame, label in targets:
        view = lambda i: frame.loc[i].assign(correct=c.loc[i, label].values)
        probe, info[pname] = select_probe(view(tr), view(va), cond, seed)
        out[pname] = probe.predict(feats(cond, info[pname]["position"], info[pname]["layer_index"], frame.loc[both, "row"].values))
        scaler, clf = stat_probe(view(tr))
        out[tname] = clf.predict_proba(scaler.transform(frame.loc[both, STATS].values))[:, 1]
        if len(held_val):
            out.loc[ev, f"{pname}_rc"] = recalibrate(out.loc[held_val, pname].values, c.loc[held_val, label].values.astype(int), out.loc[ev, pname].values)
    out["role"] = "held_val"
    out.loc[va, "role"], out.loc[ev, "role"] = "val", "eval"
    for col in ("id", "dataset", "y0", "y1", "hit", "mean_logprob"):
        out[col] = c.loc[both, col]
    out["neg_entropy"] = -c.loc[both, "mean_entropy"]
    return out, info


def fit_all(c, r, train, val, test, seed):
    frames, chosen = [], {}
    for fold in ["in_domain"] + [f"without_{d}" for d in DATASETS]:
        t = time.time()
        held = fold.replace("without_", "")
        ds = c.dataset
        pick = lambda idx, keep: idx[keep.loc[idx].values]
        if fold == "in_domain":
            tr, va, ev, hv = train, val, test, val[:0]
        else:
            tr, va, ev, hv = pick(train, ds != held), pick(val, ds != held), pick(test, ds == held), pick(val, ds == held)
        p, chosen[fold] = fit_fold(c, r, tr, va, ev, hv, seed)
        frames.append(p.assign(fold=fold))
        print(fold, {k: (v["position"], v["layer"], v["c"], round(v["val_macro_auroc"], 3)) for k, v in chosen[fold].items()}, f"{time.time() - t:.0f}s", flush=True)
    return pd.concat(frames, ignore_index=True), chosen


def rows_for(fold, val, ev, costs):
    acts = policies(val, ev, costs)
    y0, y1, hit = ev.y0.values, ev.y1.values, ev.hit.values
    groups = [("all", np.ones(len(ev), bool))]
    if fold == "in_domain":
        groups += [(d, (ev.dataset == d).values) for d in DATASETS]
    rows = []
    for name, a in acts.items():
        for group, m in groups:
            o = score(a[m], y0[m], y1[m], costs)
            sub = {k: v[m] for k, v in acts.items()}
            rows.append(dict(fold=fold, group=group, wrong_cost=costs.wrong, retrieve_cost=costs.retrieve, policy=name, planned=name in PLANNED, n=int(m.sum()),
                             **vars(o), oracle_recovered=oracle_recovered(o.utility, sub, y0[m], y1[m], costs), **retrieval_stats(a[m], y0[m], y1[m], hit[m])))
    tests = []
    pairs = lambda name: list(zip(ev.id, per_item_utility(acts[name], y0, y1, costs)))
    for name in ("value", "value_token_stats"):
        for base in AGAINST:
            tests.append(dict(fold=fold, wrong_cost=costs.wrong, retrieve_cost=costs.retrieve, candidate=name, baseline=base, **paired_test(pairs(base), pairs(name))))
    return rows, tests, acts


def coverage_rows(fold, val, ev):
    """Exploratory: wrong answers at a set share of questions answered. Costs only affect the utility column."""
    y0, y1 = ev.y0.values, ev.y1.values
    rows, tests = [], []
    for target in COVERAGES:
        acts = coverage_policies(val, ev, target)
        for name, a in acts.items():
            rows.append(dict(fold=fold, target_coverage=target, policy=name, n=len(ev), **vars(score(a, y0, y1, HEADLINE))))
        wrong = lambda name: list(zip(ev.id, (resolve(acts[name], y0, y1)[0] == 0).astype(float)))
        for base in ("cov_closed_logprob", "cov_closed_token_stats", "cov_closed_probe", "cov_route_token_stats", "cov_rag_probe"):
            tests.append(dict(fold=fold, target_coverage=target, candidate="cov_route_probe", baseline=base, metric="wrong_answer", **paired_test(wrong(base), wrong("cov_route_probe"))))
    return rows, tests


def audit_fold(fold, ev, acts, c, r, costs):
    """Replays final answers through AuditKIT and checks them against the policy arithmetic."""
    closed = c.set_index("id").answer.loc[ev.id].values
    rag = r.set_index("id").answer.loc[ev.id].values
    gold = c.set_index("id").gold.loc[ev.id].tolist()
    runs, out = {}, []
    for name in PLANNED:
        a = acts[name]
        _, retrieved = resolve(a, ev.y0.values, ev.y1.values)
        text = np.where((a == ANSWER) | (a == RETRIEVE_FALLBACK), closed, np.where(a == RETRIEVE_ANSWER, rag, ABSTAIN_TEXT))
        runs[name] = replay(ev.id.tolist(), c.set_index("id").question.loc[ev.id].tolist(), gold, text.tolist(), retrieved.tolist(), costs, f"{fold}:{name}")
        mine = score(a, ev.y0.values, ev.y1.values, costs).utility
        theirs = runs[name].headline["utility"]
        if abs(mine - theirs) > 1e-9:
            raise SystemExit(f"AuditKIT and policy utility disagree on {fold} {name}: {theirs} against {mine}")
        out.append(dict(fold=fold, policy=name, **{k: float(v) for k, v in runs[name].headline.items() if isinstance(v, (int, float))}))
    tests = [dict(fold=fold, candidate="value", baseline=b, metric=m, **compare(runs[b], runs["value"], m))
             for b in ("closed_only", "two_cut_probe", "two_cut_logprob") for m in ("utility", "wrong_answer")]
    return out, tests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="dev")
    ap.add_argument("--reuse", action="store_true", help="skip fitting and use saved predictions for this tag and seed")
    args = ap.parse_args()
    c, r, train, val, test = load(args.final)
    print(f"train {len(train)} val {len(val)} eval {len(test)} final={args.final}", flush=True)
    stem = f"{'final' if args.final else 'dev'}_{args.tag}_seed{args.seed}"
    pred_dir, out_dir = inside_project(ARTIFACTS / "policy"), inside_project(RESULTS / "policy")
    pred_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.reuse:
        preds = pd.read_pickle(pred_dir / f"{stem}.pkl")
        preds["hit"] = preds["id"].map(dict(zip(c["id"], c["hit"])))
    else:
        preds, chosen = fit_all(c, r, train, val, test, args.seed)
        preds.to_pickle(pred_dir / f"{stem}.pkl")
        (out_dir / f"{stem}_chosen.json").write_text(json.dumps(chosen, indent=1))

    rows, tests, audit, audit_tests, matched, cover, cover_tests = [], [], [], [], [], [], []
    for fold, p in preds.groupby("fold", sort=False):
        v, e = p[p.role == "val"], p[p.role == "eval"]
        for w in WRONG:
            for k in RETRIEVE:
                costs = Costs(wrong=w, abstain=0.0, retrieve=k)
                a, b, acts = rows_for(fold, v, e, costs)
                rows += a
                tests += b
                if costs == HEADLINE:
                    x, y = audit_fold(fold, e, acts, c, r, costs)
                    audit += x
                    audit_tests += y
                    matched += [dict(fold=fold, **matched_retrieval(v, e, costs, m)) for m in ("utility", "correct_rate")]
        a, b = coverage_rows(fold, v, e)
        cover += a
        cover_tests += b
        print("scored", fold, flush=True)

    table = pd.DataFrame(rows)
    for name, frame in (("policies", table), ("tests", pd.DataFrame(tests)), ("audit", pd.DataFrame(audit)), ("audit_tests", pd.DataFrame(audit_tests)), ("matched_retrieval", pd.DataFrame(matched)),
                        ("coverage", pd.DataFrame(cover)), ("coverage_tests", pd.DataFrame(cover_tests))):
        frame.to_csv(out_dir / f"{stem}_{name}.csv", index=False)

    pd.set_option("display.width", 190, "display.max_columns", 30, "display.max_rows", 200)
    head = table[(table.group == "all") & (table.wrong_cost == HEADLINE.wrong) & (table.retrieve_cost == HEADLINE.retrieve)]
    print(head.pivot(index="policy", columns="fold", values="utility").round(3))
    print(head[head.fold == "in_domain"][["policy", "utility", "wrong_rate", "coverage", "retrieval_rate", "selective_accuracy", "oracle_recovered"]].round(3).to_string(index=False))
    print(pd.DataFrame(audit_tests).round(4).to_string(index=False))
    print(pd.DataFrame(matched).round(3).to_string(index=False))
    c = pd.DataFrame(cover)
    c = c[c.fold == "in_domain"]
    print(c[["target_coverage", "policy", "coverage", "wrong_rate", "correct_rate", "selective_accuracy", "retrieval_rate"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
