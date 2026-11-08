"""Figures for the report, from saved test predictions and tables in results/final."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from knowwhen.paths import ARTIFACTS, RESULTS, inside_project
from knowwhen.policies.decision import Costs, score
from knowwhen.policies.sweep import coverage_policies

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
ROUTE, CLOSED, RAG, STATS = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MODELS = [("t4_rerank", "Tiny Aya Global (3.35B)"), ("aya8b_rerank", "Aya Expanse 8B")]
DATASETS = [("nq", "NQ"), ("triviaqa", "TriviaQA"), ("hotpotqa", "HotpotQA"), ("popqa", "PopQA")]
SERIES = [("cov_closed_probe", "Closed book, probe abstains", CLOSED, "s"), ("cov_route_token_stats", "Routed on token statistics", STATS, "^"),
          ("cov_rag_probe", "Always retrieve, probe abstains", RAG, "D"), ("cov_route_probe", "Routed on activation probes", ROUTE, "o")]
OUT = inside_project(RESULTS.parent / "reports" / "figures")

plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left"})


def curves(tag):
    p = pd.read_pickle(ARTIFACTS / "policy" / f"final_{tag}_seed0.pkl")
    p = p[p.fold == "in_domain"]
    v, e = p[p.role == "val"], p[p.role == "eval"]
    rows = []
    for target in np.arange(0.2, 1.0001, 0.05):
        for name, a in coverage_policies(v, e, min(target, 0.999)).items():
            o = score(a, e.y0.values, e.y1.values, Costs(wrong=2.0, abstain=0.0, retrieve=0.1))
            rows.append(dict(policy=name, coverage=o.coverage, wrong=o.wrong_rate))
    return pd.DataFrame(rows), float((e.y0 == 0).mean())


def fig_coverage():
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.1), sharey=True)
    for ax, (tag, title) in zip(axes, MODELS):
        c, closed = curves(tag)
        for name, label, color, marker in SERIES:
            d = c[c.policy == name].sort_values("coverage")
            ax.plot(d.coverage * 100, d.wrong * 100, color=color, lw=2, marker=marker, ms=4, markevery=4, label=label)
        ax.scatter([100], [closed * 100], s=60, color=INK, zorder=5, label="Closed book, no probe")
        ax.axvline(70, color=MUTED, lw=1, ls=":")
        ax.set_title(title)
        ax.set_xlabel("Questions answered (%)")
        ax.set_xlim(20, 102)
        ax.set_ylim(0, 85)
    axes[0].set_ylabel("Questions ending in a wrong answer (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(OUT / "wrong_vs_coverage.png", dpi=200)


def fig_unseen():
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True)
    for ax, (tag, title) in zip(axes, MODELS):
        c = pd.read_csv(RESULTS / "final" / f"{tag}_matched_unseen.csv")
        c = c[c.target_coverage == 0.7]
        x, w = np.arange(len(DATASETS)), 0.2
        for i, (name, label, color, _) in enumerate(SERIES):
            vals = [c[(c.fold == f"without_{d}") & (c.policy == name)].wrong_rate.item() * 100 for d, _ in DATASETS]
            ax.bar(x + (i - 1.5) * w, vals, w * 0.9, color=color, label=label)
        ax.set_xticks(x, [n for _, n in DATASETS])
        ax.grid(axis="x", visible=False)
        ax.set_title(title)
    axes[0].set_ylabel("Questions ending in a wrong answer (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT / "unseen_datasets_70.png", dpi=200)


def fig_detection():
    methods = [("token_stats_logistic", "Token statistics", STATS, "^"), ("activation_probe", "Activation probe", ROUTE, "o")]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 4.6), sharex=True, sharey=True)
    for col, (tag, title) in enumerate(MODELS):
        for row, (cond, name) in enumerate((("closed", "closed book answer"), ("rag", "retrieved answer"))):
            ax = axes[row, col]
            d = pd.read_csv(RESULTS / "final" / f"{tag}_probes_{cond}_mean.csv")
            for j, (ds, label) in enumerate(DATASETS):
                vals = [d[(d.fold == f"without_{ds}") & (d.method == m)].macro_auroc.item() for m, *_ in methods]
                ax.plot(vals, [j, j], color=GRID, lw=2, zorder=1)
            for m, label, color, marker in methods:
                ax.scatter([d[(d.fold == f"without_{ds}") & (d.method == m)].macro_auroc.item() for ds, _ in DATASETS], range(4), color=color, marker=marker, s=55, zorder=3, label=label)
            ax.set_yticks(range(4), [n for _, n in DATASETS])
            ax.set_title(f"{title}, {name}", fontsize=10)
            ax.set_xlim(0.65, 0.95)
    axes[0, 0].set_ylim(3.5, -0.5)
    for ax in axes[1]:
        ax.set_xlabel("AUROC on the unseen dataset")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / "detection_unseen.png", dpi=200)


if __name__ == "__main__":
    fig_coverage()
    fig_unseen()
    fig_detection()
