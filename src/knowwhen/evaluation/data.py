"""Loads finished generations with their labels and activations, using only the final rows."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from knowwhen.paths import ARTIFACTS, DATA, ENV

MODEL = os.environ.get("KW_MODEL", "tiny-aya-global")  # set to aya-expanse-8b to analyse the replication


def load_frame(condition: str = "closed", model: str = MODEL) -> pd.DataFrame:
    final = pd.DataFrame(json.load(open(DATA / "processed" / "final_rows.json")))
    questions = {}
    for line in open(DATA / "processed" / "questions.jsonl"):
        q = json.loads(line)
        questions[q["id"]] = q
    # A resumed or repeated ID can appear twice; the last record matches the last activation write.
    runs = {}
    for line in open(ARTIFACTS / "runs" / model / f"{condition}.jsonl"):
        r = json.loads(line)
        runs[r["id"]] = r
    frame = final[final["id"].isin(runs)].copy()
    run = pd.DataFrame([runs[i] for i in frame["id"]]).drop(columns=["id"])
    frame = pd.concat([frame.reset_index(drop=True), run.reset_index(drop=True)], axis=1)
    frame["dataset"] = [questions[i]["dataset"] for i in frame["id"]]
    frame["question"] = [questions[i]["question"] for i in frame["id"]]
    frame["gold"] = [questions[i]["answers"] for i in frame["id"]]
    frame["family"] = [questions[i]["meta"].get("family") for i in frame["id"]]
    frame["s_pop"] = [questions[i]["meta"].get("s_pop") for i in frame["id"]]
    return frame


def activations(condition: str, position: str, model: str = MODEL) -> np.ndarray:
    """Memory mapped array of shape (questions, layers, hidden)."""
    return np.load(ENV / "acts" / model / f"{condition}_{position}.npy", mmap_mode="r")


def layers(model: str = MODEL) -> list[int]:
    return json.load(open(ENV / "acts" / model / "meta.json"))["layers"]


def passage_file() -> str:
    """The passages this model's RAG answers were generated from."""
    return "retrieved_reranked.jsonl" if MODEL.endswith("-rerank") else "retrieved.jsonl"
