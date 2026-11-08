"""Writes the final list of questions used for analysis.

`prepare_data.py` filtered on question IDs, and TriviaQA repeats some rows under one ID, so exact
copies survived deduplication. This keeps one row per normalized question text, preferring test,
then validation, then train, so no text is shared between splits. The output is deterministic.
"""
from __future__ import annotations

import json

from knowwhen.paths import DATA
from knowwhen.text import normalize

PRIORITY = {"test": 0, "val": 1, "train": 2}
rows = [json.loads(l) for l in open(DATA / "processed" / "questions.jsonl")]
last_row = {r["id"]: i for i, r in enumerate(rows)}  # activations for a repeated ID live in its last row
best: dict[str, int] = {}
for i, r in enumerate(rows):
    key = normalize(r["question"])
    if key not in best or PRIORITY[r["split"]] < PRIORITY[rows[best[key]]["split"]]:
        best[key] = i
final = [dict(row=last_row[rows[i]["id"]], id=rows[i]["id"], split=rows[i]["split"]) for i in sorted(best.values())]
assert len({f["id"] for f in final}) == len(final), "IDs must be unique after filtering"
(DATA / "processed" / "final_rows.json").write_text(json.dumps(final))
print(f"{len(rows)} rows, {len(final)} kept, {len(rows) - len(final)} excluded")
