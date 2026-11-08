"""Multilingual transfer of the frozen closed book probe, added after the test run.

Follows the section "Multilingual transfer" in docs/analysis_plan.md, written before generation.
The in domain closed answer probe from artifacts/probes/<tag>/closed_answer.npz is applied with no
refit. Token statistics are refit on the original training rows, as in the main analysis.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from knowwhen.evaluation.data import load_frame
from knowwhen.paths import ARTIFACTS, DATA, ENV, RESULTS, inside_project
from knowwhen.probes.select import STATS, stat_probe

LANGS = ("en", "es", "de", "fr", "ru", "tr")
B = int(__import__("os").environ.get("KW_BOOT", 2000))


def probe_logits(npz, model_dir: str, rows: np.ndarray, condition: str = "closed") -> np.ndarray:
    meta = json.load(open(ENV / "acts" / model_dir / "meta.json"))
    li = meta["layers"].index(int(npz["hidden_state"]))
    acts = np.load(ENV / "acts" / model_dir / f"{condition}_{str(npz['position'])}.npy", mmap_mode="r")
    x = np.asarray(acts[rows, li, :], dtype=np.float32)
    return ((x - npz["mean"]) / npz["scale"]) @ npz["coef"] + float(npz["intercept"])


def online_keep(warm: np.ndarray, stream: np.ndarray, target: float, window: int = 100) -> np.ndarray:
    history, keep = list(warm[-window:]), np.empty(len(stream), bool)
    for i, c in enumerate(stream):
        keep[i] = c >= np.quantile(history[-window:], 1 - target)
        history.append(c)
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="MKQA run folder, e.g. tiny-aya-global-t4-mkqa")
    ap.add_argument("--base", required=True, help="run folder the probes were trained on, e.g. tiny-aya-global-t4-rerank")
    ap.add_argument("--tag", required=True, help="probe tag, e.g. t4_rerank")
    ap.add_argument("--dry", action="store_true", help="print only, write nothing; for checking on the smoke run")
    args = ap.parse_args()

    qs = {json.loads(l)["id"]: json.loads(l) for l in open(DATA / "processed" / "mkqa_questions.jsonl")}
    meta_ids = json.load(open(ENV / "acts" / args.run / "meta.json"))["ids"]
    pos = {i: k for k, i in enumerate(meta_ids)}
    runs = {}
    for l in open(ARTIFACTS / "runs" / args.run / "closed.jsonl"):
        r = json.loads(l)
        runs[r["id"]] = r
    m = pd.DataFrame([dict(id=i, row=pos[i], lang=qs[i]["lang"], qid=i.rsplit(":", 1)[0], **{k: runs[i][k] for k in ["correct", "refusal", *STATS]}) for i in runs])
    npz = np.load(ARTIFACTS / "probes" / args.tag / "closed_answer.npz")
    m["probe"] = probe_logits(npz, args.run, m.row.values)

    base = load_frame("closed", model=args.base)
    scaler, clf = stat_probe(base[base.split == "train"])
    m["token_stats"] = clf.predict_proba(scaler.transform(m[STATS].values))[:, 1]
    val = base[base.split == "val"]
    warm = probe_logits(npz, args.base, val.row.values)

    y = m.correct.astype(int).values
    qids = m.qid.unique()
    rng = np.random.default_rng(13)
    boot = [rng.choice(len(qids), len(qids)) for _ in range(B)]
    by_q = {l: m[m.lang == l].set_index("qid") for l in LANGS}

    def auroc(frame, col):
        yy = frame.correct.astype(int).values
        return roc_auc_score(yy, frame[col].values) if 0 < yy.sum() < len(yy) else np.nan

    rows = []
    for l in LANGS:
        f = by_q[l]
        rec = dict(lang=l, n=len(f), accuracy=f.correct.mean(), refusal_rate=f.refusal.mean())
        for col in ("probe", "token_stats", "mean_logprob"):
            point = auroc(f, col)
            bs = []
            for b in boot:
                bs.append(auroc(f.reindex(qids[b]).dropna(subset=[col]), col))
            rec[f"auroc_{col}"], rec[f"{col}_lo"], rec[f"{col}_hi"] = point, np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5)
            if l != "en":
                e = by_q["en"]
                diffs = [auroc(f.reindex(qids[b]).dropna(subset=[col]), col) - auroc(e.reindex(qids[b]).dropna(subset=[col]), col) for b in boot]
                rec[f"{col}_minus_en"] = point - auroc(e, col)
                rec[f"{col}_minus_en_lo"], rec[f"{col}_minus_en_hi"] = np.nanpercentile(diffs, 2.5), np.nanpercentile(diffs, 97.5)
        order = np.random.default_rng(1000).permutation(len(f))
        keep = np.empty(len(f), bool)
        keep[order] = online_keep(np.random.default_rng(1000).permutation(warm), f.probe.values[order], 0.7)
        cy = f.correct.values.astype(bool)
        answered_wrong = keep & ~cy & ~f.refusal.values.astype(bool)
        rec.update(coverage_online_70=(keep & ~f.refusal.values.astype(bool)).mean(), wrong_rate_online_70=answered_wrong.mean(),
                   wrong_rate_answer_all=(~cy & ~f.refusal.values.astype(bool)).mean())
        rows.append(rec)
    t = pd.DataFrame(rows)
    if not args.dry:
        d = inside_project(RESULTS / "final")
        t.to_csv(d / f"{args.tag}_multilingual.csv", index=False)
        m.drop(columns=["row"]).to_csv(d / f"{args.tag}_multilingual_rows.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(args.run, "probe", args.tag, "hidden state", int(npz["hidden_state"]), str(npz["position"]))
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
