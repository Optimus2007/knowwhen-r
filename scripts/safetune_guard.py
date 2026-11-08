"""Runs the prompt only probe through SafeTune's LinearProbeGuardModel, live, on validation questions.

SafeTune's guard refuses a flagged prompt or calls `on_refuse`. Here a flagged prompt means the
model is forecast to answer wrongly from memory, and `on_refuse` answers with retrieved passages.
Passages come from the saved reranked file. Test rows are not used.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from safetune.interventions.steer.probe_guard import LinearProbe, LinearProbeGuardModel

from knowwhen.evaluation.data import activations, layers, load_frame
from knowwhen.models.prompts import closed_prompt, rag_prompt
from knowwhen.models.runner import build_prefix, generate_batch, load_model, render
from knowwhen.paths import ARTIFACTS, DATA, RESULTS, inside_project
from knowwhen.text import contains_answer


class Tok:
    """The rendered chat text already has its special tokens, and SafeTune calls the tokenizer plainly."""

    def __init__(self, tok):
        self.tok = tok

    def __call__(self, batch, **kw):
        return self.tok(batch, add_special_tokens=False, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--tag", default="t4_rerank")
    ap.add_argument("--model", default="CohereLabs/tiny-aya-global")
    args = ap.parse_args()
    f = load_frame("closed")
    rag = load_frame("rag").set_index("id")
    v = f[f.split == "val"].sample(args.n, random_state=7)
    passages = {}
    for line in open(DATA / "processed" / "retrieved_reranked.jsonl"):
        q = json.loads(line)
        if q["id"] in set(v["id"]):
            passages[q["id"]] = q["passages"][:3]
    p = np.load(ARTIFACTS / "probes" / args.tag / "closed_forecast.npz")
    state = int(p["hidden_state"])
    # hidden_states[k] is the output of decoder layer k - 1. The sign is flipped so that flagged means likely wrong.
    guard_probe = LinearProbe(layer_idx=state - 1, weight=-torch.tensor(p["coef"]), bias=-float(p["intercept"]), mean=torch.tensor(p["mean"]), std=torch.tensor(p["scale"]), threshold=0.0, pool_method="last_token")
    model, tok = load_model(args.model)
    prefix = build_prefix(model, tok)
    ask = lambda prompt: generate_batch(model, tok, [prompt], [state], prefix=prefix).answers[0]
    rows = {render(tok, closed_prompt(q)): i for i, q in zip(v["id"], v["question"])}
    questions = dict(zip(v["id"], v["question"]))
    route = {}

    def closed(text):
        route[rows[text]] = "closed"
        return ask(closed_prompt(questions[rows[text]]))

    def retrieved(text):
        route[rows[text]] = "retrieved"
        return ask(rag_prompt(questions[rows[text]], passages[rows[text]]))

    guard = LinearProbeGuardModel(model, Tok(tok), guard_probe, on_pass=closed, on_refuse=retrieved)
    out = []
    for n, (text, i) in enumerate(rows.items()):
        score = float(guard.score([text])[0])
        answer = guard.guard(text)
        r = v.set_index("id").loc[i]
        out.append(dict(id=i, dataset=r.dataset, guard_score=score, route=route[i], answer=answer, correct=contains_answer(answer, r.gold), saved_closed=bool(r.correct), saved_rag=bool(rag.loc[i, "correct"]), row=int(r.row)))
        if n % 25 == 0:
            print(n, flush=True)
    t = pd.DataFrame(out)
    x = np.asarray(activations("closed", "pre")[t.row.values, layers().index(state)], dtype=np.float32)
    saved = -(((x - p["mean"]) / p["scale"]) @ p["coef"] + p["intercept"])
    print("rank correlation with saved T4 activations", round(float(spearmanr(t.guard_score, saved)[0]), 4), "same flag", round(float(((t.guard_score >= 0) == (saved >= 0)).mean()), 4))
    print("retrieved share", round(float((t.route == "retrieved").mean()), 3), "accuracy live", round(float(t.correct.mean()), 3), "saved closed", round(float(t.saved_closed.mean()), 3), "saved rag", round(float(t.saved_rag.mean()), 3))
    expected = np.where(t.route == "retrieved", t.saved_rag, t.saved_closed)
    print("live answer label matches saved label", round(float((t.correct == expected).mean()), 3))
    t.to_csv(inside_project(RESULTS / "final") / "safetune_guard_tiny.csv", index=False)


if __name__ == "__main__":
    main()
