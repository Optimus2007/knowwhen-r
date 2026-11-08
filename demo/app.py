"""KnowWhen-R demo. Replays saved test traces, so it needs no GPU and shows real runs, not new generations."""
import json
import random
from pathlib import Path

import gradio as gr
import pandas as pd

HERE = Path(__file__).parent
T = pd.read_parquet(HERE / "traces.parquet")
CUTS = json.loads((HERE / "cuts.json").read_text())
MODELS = sorted(T.model.unique())
LABEL = {1: "right", 0: "wrong", -1: "declined"}


def decide(f, cut):
    routed = f.p_before > f.p0
    conf = routed * f.p_after + (~routed) * f.p0
    keep = conf >= cut
    y = routed * f.y1 + (~routed) * f.y0
    return routed, conf, keep & (y != -1), y


def summary(model, coverage):
    f = T[T.model == model]
    routed, _, answered, y = decide(f, CUTS[model][str(int(coverage))])
    rows = [("Closed book, answer everything", 100 * (f.y0 != -1).mean(), 100 * (f.y0 == 0).mean(), 100 * (f.y0 == 1).mean(), 0.0),
            ("Always retrieve, answer everything", 100 * (f.y1 != -1).mean(), 100 * (f.y1 == 0).mean(), 100 * (f.y1 == 1).mean(), 100.0),
            ("KnowWhen-R", 100 * answered.mean(), 100 * (answered & (y == 0)).mean(), 100 * (answered & (y == 1)).mean(), 100 * routed.mean())]
    return pd.DataFrame(rows, columns=["System", "Answered %", "Wrong %", "Correct %", "Retrieved %"]).round(1)


def show(model, coverage):
    f = T[T.model == model]
    i = random.randrange(len(f))
    routed, conf, answered, y = decide(f, CUTS[model][str(int(coverage))])
    r = f.iloc[i]
    went, final = bool(routed.iloc[i]), "retrieved passages" if routed.iloc[i] else "memory"
    verdict = f"Answered from {final}: **{r.rag_answer if went else r.closed_answer}** ({LABEL[int(y.iloc[i])]})" if answered.iloc[i] else f"Abstained. Confidence {conf.iloc[i]:.2f} was below the cut. Without the probe it would have been {LABEL[int(y.iloc[i])]}."
    return f"""**{r.question}** ({r.dataset})

Gold answers: {", ".join(json.loads(r.gold))}

| Step | Output | Probe |
|---|---|---|
| From memory | {r.closed_answer} ({LABEL[int(r.y0)]}) | chance it is right: {r.p0:.2f} |
| Forecast before reading passages | | chance a retrieved answer is right: {r.p_before:.2f} |
| With passages | {r.rag_answer} ({LABEL[int(r.y1)]}) | chance it is right: {r.p_after:.2f} |

{verdict}"""


with gr.Blocks(title="KnowWhen-R") as demo:
    gr.Markdown("# KnowWhen-R\nProbes on a frozen model's activations decide whether to answer from memory, use retrieved passages or abstain. This page replays recorded runs on the 3,968 test questions. Nothing is generated here, and the retrieved branch is shown for every question so you can compare it. Cuts come from validation rows, so coverage here is close to the target, not exact.")
    with gr.Row():
        model = gr.Dropdown(MODELS, value=MODELS[-1], label="Model")
        coverage = gr.Slider(20, 95, value=70, step=5, label="Target share of questions answered (%)")
    table = gr.Dataframe(label="All test questions")
    button = gr.Button("Show a random question")
    trace = gr.Markdown()
    for c in (model, coverage):
        c.change(summary, [model, coverage], table)
    button.click(show, [model, coverage], trace)
    demo.load(summary, [model, coverage], table)

if __name__ == "__main__":
    demo.launch()
