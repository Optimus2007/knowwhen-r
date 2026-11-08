"""Builds the static demo's data file from the same saved test traces as the Gradio demo.

Hugging Face hosts static Spaces for free, so the public demo is a plain page that replays these
traces in the browser. Nothing is generated live.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    t = pd.read_parquet(ROOT / "demo" / "traces.parquet")
    cuts = json.loads((ROOT / "demo" / "cuts.json").read_text())
    out = {"cuts": cuts, "models": {}}
    for m, f in t.groupby("model", sort=True):
        out["models"][m] = {c: f[c].tolist()
                            for c in ["dataset", "question", "gold", "closed_answer", "rag_answer", "y0", "y1", "p0", "p_before", "p_after"]}
    path = ROOT / "demo_static" / "data.json"
    path.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    print(path.name, round(path.stat().st_size / 1e6, 2), "MB,", {m: len(v["p0"]) for m, v in out["models"].items()})


if __name__ == "__main__":
    main()
