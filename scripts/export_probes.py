"""Saves the frozen in domain probes as plain arrays, for the model repository and the demo.

Three probes per model, fitted on train rows at the saved choice and calibrated on validation rows,
plus a prompt only probe that forecasts the closed book answer before anything is generated.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_policy import load  # noqa: E402

from knowwhen.evaluation.data import layers  # noqa: E402
from knowwhen.paths import ARTIFACTS, RESULTS, inside_project  # noqa: E402
from knowwhen.probes.linear import LinearProbe  # noqa: E402
from knowwhen.probes.select import feats, macro_auroc  # noqa: E402


def save(probe, k, path):
    np.savez(path, coef=probe.clf.coef_[0].astype(np.float32), intercept=np.float32(probe.clf.intercept_[0]), mean=probe.scaler.mean_.astype(np.float32),
             scale=probe.scaler.scale_.astype(np.float32), temperature=np.float32(probe.temperature.t), hidden_state=k["layer"], position=k["position"], c=k["c"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    c, r, train, val, _ = load(True)
    chosen = json.load(open(RESULTS / "policy" / f"final_{args.tag}_seed0_chosen.json"))["in_domain"]
    d = inside_project(ARTIFACTS / "probes" / args.tag)
    d.mkdir(parents=True, exist_ok=True)
    names = dict(p0="closed_answer", p_before="retrieved_forecast", p_after="retrieved_answer")
    for pname, cond, frame, label in (("p0", "closed", c, "correct"), ("p_before", "closed", c, "correct1"), ("p_after", "rag", r, "correct1")):
        k = chosen[pname]
        y = c[label].values.astype(int)
        x = lambda i: feats(cond, k["position"], k["layer_index"], frame.loc[i, "row"].values)
        save(LinearProbe(c=k["c"]).fit(x(train), y[train]).calibrate(x(val), y[val]), k, d / f"{names[pname]}.npz")
    y, g = c.correct.values.astype(int), c.dataset.values
    x = lambda li, i: feats("closed", "pre", li, c.loc[i, "row"].values)
    fit = lambda li: LinearProbe(c=1e-3).fit(x(li, train), y[train])
    best = max((macro_auroc(y[val], fit(li).logits(x(li, val)), g[val]), li) for li in range(1, len(layers()), 2))
    li = best[1]
    k = dict(layer=layers()[li], position="pre", c=1e-3)
    save(fit(li).calibrate(x(li, val), y[val]), k, d / "closed_forecast.npz")
    print(args.tag, "prompt only probe at hidden state", k["layer"], "validation macro AUROC", round(best[0], 3))
    print(sorted(p.name for p in d.iterdir() if not p.name.startswith("._")))


if __name__ == "__main__":
    main()
