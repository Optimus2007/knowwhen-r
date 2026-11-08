"""Writes the test traces used by the demo: answers, gold answers and saved probe scores."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from knowwhen.policies.sweep import _route

ROOT = Path(__file__).parents[1]
PAIRS = [("tiny-aya-global-t4-rerank", "t4_rerank", "Tiny Aya Global"), ("aya-expanse-8b-rerank", "aya8b_rerank", "Aya Expanse 8B")]


def main():
    frames = []
    for model, tag, name in PAIRS:
        os.environ["KW_MODEL"] = model
        from knowwhen.evaluation.data import load_frame
        c, r = load_frame("closed", model), load_frame("rag", model)
        c, r = c[c.split == "test"].set_index("id"), r[r.split == "test"].set_index("id")
        p = pd.read_pickle(ROOT / "artifacts" / "policy" / f"final_{tag}_seed0.pkl")
        v = p[(p.fold == "in_domain") & (p.role == "val")]
        e = p[(p.fold == "in_domain") & (p.role == "eval")].set_index("id")
        f = e[["dataset", "p0", "p_before", "p_after", "y0", "y1"]].copy()
        f["model"], f["question"], f["gold"] = name, c.question.loc[f.index], c.gold.loc[f.index].map(lambda g: json.dumps(g[:5], ensure_ascii=False))
        f["closed_answer"], f["rag_answer"] = c.answer.loc[f.index], r.answer.loc[f.index]
        frames.append(f.reset_index())
        frames[-1].attrs["val_conf"] = _route(v, "p0", "p_before", "p_after")[1].tolist()
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(ROOT / "demo" / "traces.parquet", index=False)
    cuts = {name: {str(q): float(pd.Series(f.attrs["val_conf"]).quantile(1 - q / 100)) for q in range(20, 100, 5)} for f, (_, _, name) in zip(frames, PAIRS)}
    (ROOT / "demo" / "cuts.json").write_text(json.dumps(cuts, indent=1))
    print(len(out))


if __name__ == "__main__":
    main()
